"""The shared bounce analysis: everything both the tables and the PDFs need.

A library.  `run_analysis` performs the whole reduction once — load, cuts, the
OFC SVD projections, the five recovery schemes, and the per-leg and per-night
paired Δ — and returns it.  `rubinwork.products.bounce_tables` turns the result
into the four product tables; the `bounce` study in `aos/` renders the nine PDFs
from the same result.

Why both halves call this rather than the study reading the tables back: three
of the nine PDF pages (Δ DZ, v-mode and DOF vs ordinal) plot the **per-visit**
arrays `C_all`, `DOF_all` and `DOF5_all`, which no product table carries.  The
cost is one extra run of the analysis (about 170 s of wall clock) when both the
tables and the PDFs are built.  Aaron's call, 2026-10-12; a later option is to
write the per-visit arrays as a fifth product table, after which the study could
read only the product.

The reduction itself is unchanged from the single script this was split out of —
the physics moved verbatim.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from astropy.table import QTable

from rubinwork.bounce import legs as bl
from rubinwork.bounce.legs import CAM_HEX_DOF, DOF22


@dataclass
class BounceAnalysis:
    """Everything one bounce run computes, before anything is written.

    Attributes
    ----------
    fit_table : `astropy.table.QTable`
        The per-visit DZ fit table after the bad-fit and quality cuts.
    bounce_results : `dict`
        Per bounce name: the reference stats and, per comparison leg, the paired
        Δ per (k, j), per v-mode and per DOF for each recovery scheme, pooled
        and per night.  The structure the table and plot halves both walk.
    df_kj : `pandas.DataFrame`
        Long-format per-(k, j) Δ, one row per (bounce, night, comparison, k, j).
        Becomes `bounce_kj_stats.parquet`.
    fwhm_rows, fwhm_bvalue_rows : `list` [`dict`]
        The per-leg and per-(night, leg) correctable-FWHM rows, in arcsec.
        Become `bounce_fwhm_metric.parquet` and `bounce_fwhm_vs_bvalue.parquet`.
    iZs : `list` [`int`]
        Pupil Noll indices of the DZ grid.
    k_list : `list` [`int`]
        Focal Zernike orders fitted.
    svd, svd5, svd22 : `OFCSvd` or `None`
        The 50/34, 5/5 camera-hexapod-only and 22/12 reduced decompositions.
    C_all, DOF_all : `numpy.ndarray` or `None`
        Per-visit v-mode amplitudes (dimensionless) and DOF (µm or arcsec),
        50/34.  The per-ordinal PDF pages need these.
    C5_all, DOF5_all, C22_all : `numpy.ndarray` or `None`
        The same for the 5/5 and 22/12 schemes.
    DOFSUM_all : `numpy.ndarray` or `None`
        Per-visit DOF plus the AOS Trim, µm or arcsec, when `add_dof_trim` is on.
    dof_ranges : `numpy.ndarray` or `None`
        Allowed range r_j per DOF, µm or arcsec, from the SVD normalization
        weights.  `None` when Range-Bounded Recovery is unavailable.
    oic_authority : `numpy.ndarray` or `None`
        The OIC authority vector over all 50 DOF.
    svd_ok : `bool`
        Whether the OFC SVD built.  Everything DOF-shaped is `None` if not.
    vmode_labels, vmode5_labels, vmode22_labels : `list` [`str`]
        v-mode names per scheme.
    """

    fit_table: Any
    bounce_results: dict
    df_kj: Any
    fwhm_rows: list
    fwhm_bvalue_rows: list
    iZs: list
    k_list: list
    svd: Any = None
    svd5: Any = None
    svd22: Any = None
    C_all: Any = None
    DOF_all: Any = None
    C5_all: Any = None
    DOF5_all: Any = None
    C22_all: Any = None
    DOFSUM_all: Any = None
    dof_ranges: Any = None
    oic_authority: Any = None
    svd_ok: bool = False
    oic_ok: bool = False
    vmode_labels: list = field(default_factory=list)
    vmode5_labels: list = field(default_factory=list)
    vmode22_labels: list = field(default_factory=list)


def _aos_fwhm():
    """The `aos_fwhm` module, which maps a residual DZ wavefront to a PSF FWHM.

    Returns
    -------
    mod : `module`
        ``aos/code/aos_fwhm.py``.

    Notes
    -----
    A cross-topic reach into `aos/code/`, done by path insert because the topic
    directories are not a package.  `aos_fwhm` is the one definition of the
    correctable-FWHM metric in this repository — `rubinwork.open_loop` reaches it
    the same way, and `rubinwork.smatrix.regularized_inversion` documents against
    it — so a copy here would be a second definition free to drift.  It is a
    library by the plan's section 2 rules and should move into `rubinwork/` when
    the flat `aos/code/` modules do.
    """
    import sys
    from pathlib import Path
    code = Path(__file__).resolve().parents[2] / 'aos' / 'code'
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import aos_fwhm
    return aos_fwhm


# ---- FWHM of the residual actually left by a recovered DOF correction ----
# `aos_fwhm.residual_dW` returns the *subspace projection* residual, which is
# independent of the recovered amplitudes and so cannot see a regularizer trade
# wavefront for amplitude.  Comparing the default recovery against RBR needs the
# achieved residual dW - S (d / w) instead.  The two agree exactly for the
# truncated solution, so the achieved-residual default series reproduces
# `fwhm_after_50_34`.
def _achieved_fwhm(dW, d, svd, iZs, grid, conv, afw):
    """Correctable FWHM in arcsec of the residual left by applying DOF `d`.

    Parameters
    ----------
    dW : `numpy.ndarray`
        Measured DZ wavefront, µm of wavefront, on `svd.kj_grid`.
    d : `numpy.ndarray`
        Recovered DOF, each in its own unit (µm or arcsec).
    svd : `OFCSvd`
        The decomposition supplying the forward operator and the weights.
    iZs : `list` [`int`]
        Pupil Noll indices of the DZ grid.
    grid : `numpy.ndarray`
        Focal-plane sample positions from `aos_fwhm.fp_grid`.
    conv : `callable`
        `lsst.ts.wep.utils.convertZernikesToPsfWidth`.
    afw : `module`
        `aos_fwhm`.

    Returns
    -------
    fwhm : `float`
        Median correctable FWHM over the focal plane, arcsec.
    """
    ri = bl.rbr_module()
    return afw.fp_fwhm(svd, iZs, ri.achieved_residual(dW, d, svd), grid, conv)


def _rbr_fwhm(dW, svd, ranges, cfg, iZs, grid, conv, afw):
    """Correctable FWHM in arcsec left after Range-Bounded Recovery.

    Notes
    -----
    Returns NaN rather than raising if the RBR solve fails, so one bad leg
    cannot take down the whole run.
    """
    ri = bl.rbr_module()
    try:
        d = ri.invert_range_penalty(dW, svd, ranges, kappa=cfg['rbr_kappa'],
                                    power=cfg['rbr_power'])
    except Exception as e:
        print(f'        (RBR FWHM failed [{type(e).__name__}: {e}])')
        return float('nan')
    return _achieved_fwhm(dW, d, svd, iZs, grid, conv, afw)


def _scheme_fwhm(dW, svd, solver, iZs, grid, conv, afw, tag):
    """Correctable FWHM in arcsec left after one recovery scheme.

    Parameters
    ----------
    dW : `numpy.ndarray`
        Measured DZ wavefront, µm of wavefront, on `svd.kj_grid`.
    svd : `OFCSvd`
        The scheme's own decomposition; the residual is scored in it.
    solver : `callable`
        Maps ``(dW, svd)`` to recovered DOF in each DOF's own unit (µm or arcsec).
    iZs : `list` [`int`]
        Pupil Noll indices of the DZ grid.
    grid : `numpy.ndarray`
        Focal-plane sample positions from `aos_fwhm.fp_grid`.
    conv : `callable`
        `lsst.ts.wep.utils.convertZernikesToPsfWidth`.
    afw : `module`
        `aos_fwhm`.
    tag : `str`
        Scheme name, used only in the failure message.

    Returns
    -------
    fwhm : `float`
        Median achieved correctable FWHM over the focal plane, arcsec, or NaN if
        the solve failed.

    Notes
    -----
    Always the *achieved* residual ``dW - S (d / w)``, for every scheme, so the
    schemes are comparable: the subspace projection cannot see a regularizer
    trading wavefront for amplitude.  For the unregularized truncated recoveries
    the two coincide.
    """
    try:
        d = solver(dW, svd)
    except Exception as e:
        print(f'        ({tag} FWHM failed [{type(e).__name__}: {e}])')
        return float('nan')
    return _achieved_fwhm(dW, d, svd, iZs, grid, conv, afw)


def run_analysis(fits_path, cfg, *, min_detectors=None):
    """Run the whole bounce reduction once.

    Parameters
    ----------
    fits_path : `str` or `pathlib.Path`
        The per-visit Double-Zernike fit table to read.
    cfg : `dict`
        The resolved bounce configuration: `rubinwork.bounce.legs.DEFAULT`
        overlaid with `analysis_config.yaml`'s `bounce:` section.
    min_detectors : `int`, optional
        Per-visit quality cut: keep visits with
        ``n_detectors_with_min_donuts >= min_detectors``.  Relaxes ONLY the
        CCD-count cut; the blur cut is kept.  `None` falls back to the
        precomputed ``visit_quality_pass`` column (nd >= 170), which preserves
        the selection the study originally ran.

    Returns
    -------
    result : `BounceAnalysis`
        Every quantity the tables and the PDFs are built from.
    """
    prefix = cfg['fit_prefix']
    bounces = cfg['bounces']
    print(f'[bounce] {fits_path}')
    # ---- load + drop bad fits + resolve j/k lists (cell 10) ----
    fit_table = QTable.read(str(fits_path))
    for bf in (f'{prefix}_bad_fit', 'bad_fit'):
        if bf in fit_table.colnames:
            bad = np.asarray(fit_table[bf]).astype(bool)
            if int(bad.sum()):
                print(f'  dropping {int(bad.sum())} bad-flagged visits ({bf})')
            fit_table = fit_table[~bad]
            break

    # ---- per-visit quality cut (fits.parquet now holds ALL visits) ----
    # --min-detectors relaxes ONLY the CCD-count cut (bounce uses 160 to recover
    # marginal low-CCD elevation-40 points); without it, fall back to the
    # precomputed visit_quality_pass (nd>=170), preserving the old selection.
    if min_detectors is not None and \
            'n_detectors_with_min_donuts' in fit_table.colnames:
        from lsst.ts.intrinsic.wavefront.intrinsics_lib import quality_visit_mask
        keep = np.asarray(quality_visit_mask(
            fit_table, min_detectors_per_visit=min_detectors, verbose=False),
            dtype=bool)
        print(f'  quality cut (min_detectors={min_detectors}): '
              f'{int(keep.sum())}/{len(fit_table)} visits kept')
        fit_table = fit_table[keep]
    elif 'visit_quality_pass' in fit_table.colnames:
        keep = np.asarray(fit_table['visit_quality_pass'], dtype=bool)
        print(f'  quality cut (visit_quality_pass): '
              f'{int(keep.sum())}/{len(fit_table)} visits kept')
        fit_table = fit_table[keep]
    if cfg['pupil_j_range'] is None:
        if 'nollIndices' not in fit_table.colnames:
            raise ValueError('no nollIndices column; set pupil_j_range in config')
        iZs = [int(j) for j in np.asarray(fit_table['nollIndices'][0]).tolist()]
    else:
        iZs = list(cfg['pupil_j_range'])
    k_list = list(cfg['focal_k_range'])
    print(f'  {len(fit_table)} visits; pupil j={iZs}; focal k={k_list}')

    # ---- OFC SVD + per-visit v-mode / DOF projection (cell 12) ----
    C_all = DOF_all = None
    svd = svd5 = C5_all = DOF5_all = None
    svd22 = C22_all = None
    vmode_labels = vmode5_labels = vmode22_labels = []
    try:
        from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd, project_dz_table
        svd = build_ofc_svd(iZs, int(min(k_list)), int(max(k_list)),
                            cfg['n_keep'], n_dof=cfg['n_dof'],
                            ofc_normalization_yaml=cfg['ofc_normalization_yaml'])
        _svd_ok = True
        vmode_labels = svd.vmode_labels
        C_all, DOF_all, _A, _W = project_dz_table(fit_table, prefix, svd)
        print(f'  OFC SVD: n_keep={svd.n_keep_eff}, n_dof={svd.n_dof}; '
              f'projected C_all{C_all.shape}, DOF_all{DOF_all.shape}')
        # 5/5 Camera-hexapod-only SVD (shares kj_grid) for the rotator bounce.
        svd5 = build_ofc_svd(iZs, int(min(k_list)), int(max(k_list)),
                             5, n_dof=CAM_HEX_DOF,
                             ofc_normalization_yaml=cfg['ofc_normalization_yaml'])
        vmode5_labels = svd5.vmode_labels
        C5_all, DOF5_all, _A5, _W5 = project_dz_table(fit_table, prefix, svd5)
        print(f'  5/5 Camera-hex SVD: n_keep={svd5.n_keep_eff}, n_dof={svd5.n_dof}')
        # 22/12 reduced-DOF SVD (shares kj_grid): the set the AOS is expected to
        # operate in, evaluated on every bounce so the cost of dropping the other
        # 28 DOF is visible.  n_dof is the explicit DOF22 index set.
        if cfg['reduced_enable']:
            svd22 = build_ofc_svd(iZs, int(min(k_list)), int(max(k_list)),
                                  cfg['n_keep_reduced'], n_dof=DOF22,
                                  ofc_normalization_yaml=cfg['ofc_normalization_yaml'])
            vmode22_labels = svd22.vmode_labels
            # The per-visit DOF matrix is not kept: the 22/12 DOF Δ is solved per
            # *pair* on the Δ wavefront, as every regularized scheme is, so that
            # all four share one reduction.  C22_all is kept because the v-mode
            # Δ is linear in the visit amplitudes and so can be paired directly.
            C22_all, _D22, _A22, _W22 = project_dz_table(fit_table, prefix, svd22)
            print(f'  22/{svd22.n_keep_eff} reduced SVD: '
                  f'n_keep={svd22.n_keep_eff}, n_dof={svd22.n_dof}; '
                  f'DOF indices {DOF22}; v-modes '
                  f'{vmode22_labels[0]}..{vmode22_labels[-1]}')
    except Exception as e:
        print(f'  (OFC SVD unavailable [{type(e).__name__}: {e}]; '
              f'v-mode/DOF sections skipped)')
        _svd_ok = False

    # ---- differential correctable-FWHM metric tooling (RSS of the median Δ-DZ
    # aberrations -> PSF FWHM; residual after each scheme's OFC projection) ----
    fwhm_conv = fwhm_grid = _afw = None
    try:
        from lsst.ts.wep.utils import convertZernikesToPsfWidth as fwhm_conv
        _afw = _aos_fwhm()
        fwhm_grid = _afw.fp_grid()
    except Exception as e:
        print(f'  (correctable-FWHM metric disabled: {type(e).__name__}: {e})')
    fwhm_rows = []
    fwhm_bvalue_rows = []

    # ---- Range-Bounded Recovery (RBR) tooling ----
    # The allowed range r_j per DOF is back-derived from the same SVD
    # normalization weights the recovery already uses, so no new input is
    # introduced.  RBR is reported alongside the default recovery everywhere
    # the per-night DOF Δ appears; if the solver or the range vector is
    # unavailable the default products are unaffected.
    dof_ranges = None
    if cfg['rbr_enable'] and _svd_ok and svd is not None:
        try:
            _ri = bl.rbr_module()
            dof_ranges = _ri.dof_range_vector(svd)
            print(f'  RBR: {bl.RBR_METHOD_NAME}, kappa='
                  f'{cfg["rbr_kappa"]} (dimensionless, |d_j|/r_j at unit '
                  f'penalty weight), power={cfg["rbr_power"]}; '
                  f'allowed range r_j spans '
                  f'{np.nanmin(dof_ranges):.4g} to {np.nanmax(dof_ranges):.4g} '
                  f'(µm or arcsec, per DOF unit)')
        except Exception as e:
            print(f'  (RBR unavailable [{type(e).__name__}: {e}]; '
                  f'default recovery only)')
            dof_ranges = None

    # ---- the OIC authority, and the reduced schemes' range consistency ----
    # Every DOF row of every scheme is reported against the same 50-DOF r_j,
    # indexed by its *global* DOF index.  That is only legitimate because
    # dof_range_vector indexes the shipped quadrature f_j by svd.dof_idx, so a
    # reduced scheme's own range vector is exactly the matching entries of the
    # 50-DOF one.  Assert it rather than assume it: a mismatch would silently
    # compare a 22/12 amplitude against the wrong DOF's range.
    oic_authority = None
    if _svd_ok:
        _ri_mod = None
        try:
            _ri_mod = bl.rbr_module()
        except Exception as e:
            print(f'  (regularized_inversion unavailable [{type(e).__name__}: {e}]; '
                  f'22/12 and OIC range reporting skipped)')
        if _ri_mod is not None:
            if dof_ranges is not None:
                for _tag, _s in (('22/12', svd22), ('5/5', svd5)):
                    if _s is None:
                        continue
                    _own = np.asarray(_ri_mod.dof_range_vector(_s), float)
                    _sub = np.asarray(dof_ranges, float)[
                        np.asarray(_s.dof_idx, int)]
                    if not np.allclose(_own, _sub, rtol=1e-12, atol=0.0):
                        raise RuntimeError(
                            f'{_tag} allowed range r_j does not match the '
                            f'50-DOF vector subset to its DOF indices; the '
                            f'per-DOF range reporting would be wrong')
            if cfg['oic_enable']:
                try:
                    _auth50, _auth_parts = _ri_mod.oic_authority()
                    oic_authority = _auth50
                    print(f'  OIC: motion penalty rho={cfg["oic_rho"]:g} '
                          f'(dimensionless; ts_ofc ships '
                          f'{_auth_parts["motion_penalty"]:g}, so the penalty is '
                          f'inactive as delivered). Value consumed from config; '
                          f'its derivation is in '
                          f'smatrix/regularized_inversion/regularized_inversion.md')
                except Exception as e:
                    print(f'  (OIC authority unavailable [{type(e).__name__}: '
                          f'{e}]; OIC scheme skipped)')
                    oic_authority = None

    # ---- the recovery schemes, as (dW, svd) -> DOF callables ----
    # One closure per scheme, so the per-pair Δ reduction and the achieved-FWHM
    # metric take the same object and differ only in which solver they call.
    def _solve_trunc(dW, s):
        """DOF from the truncated recovery in `s`'s own subspace, own units."""
        return bl.rbr_module().invert_truncated(dW, s)

    def _solve_oic(dW, s):
        """DOF from the OIC quadratic motion penalty at `cfg['oic_rho']`.

        Notes
        -----
        The authority vector is built over all 50 DOF, so it is subset to
        `s.dof_idx` before the solve; applied here only to the 50-DOF SVD, where
        that subset is the identity.
        """
        a = np.asarray(oic_authority, float)[np.asarray(s.dof_idx, int)]
        return bl.rbr_module().invert_oic(dW, s, a, float(cfg['oic_rho']))

    oic_ok = bool(cfg['oic_enable']) and oic_authority is not None

    # ---- optional AOS Trim (aggregatedDoF) (cell 14) ----
    DOFSUM_all = TRIM_segment = None
    if cfg['add_dof_trim'] and _svd_ok and DOF_all is not None:
        try:
            from rubinwork.common.dof_telemetry import fetch_aggregated_dof_for_visits
            TRIM_all, info = fetch_aggregated_dof_for_visits(
                fit_table, consdb_url=cfg['trim_consdb_url'],
                topic=cfg['trim_efd_topic'], n_dof=DOF_all.shape[1],
                mjd_fallback_col=cfg['trim_time_col'], mjd_scale=cfg['trim_mjd_scale'])
            DOFSUM_all = DOF_all + TRIM_all
            TRIM_segment = pd.factorize(info['event_id'])[0]
            print(f'  AOS Trim: {info["n_dof"]}/{len(fit_table)} visits resolved')
        except Exception as e:
            print(f'  (AOS Trim unavailable [{type(e).__name__}: {e}])')

    # ---- bounce analysis (cell 16) ----
    bounce_results, long_dfs = {}, []
    for b in bounces:
        name = b['name']
        combined = bl.run_bounce(fit_table, b, prefix, k_list, iZs,
                                 day_obs=None, trim_segment=TRIM_segment)
        nights = bl.bounce_nights(fit_table, b, prefix, k_list, iZs,
                                  min_visits=cfg['night_min_visits'],
                                  trim_segment=TRIM_segment)
        br = {'description': b.get('description', ''),
              'reference_label': b['reference']['label'],
              'reference_stats': combined['ref_stats'],
              'reference_n': combined['ref_n'],
              'reference_mask': combined['ref_mask'], 'comparisons': {}}
        print(f'  === {name} ===  ref "{b["reference"]["label"]}" '
              f'n={combined["ref_n"]}; nights={sorted(nights.keys())}')
        for comp in b['comparisons']:
            label = comp['label']
            cblock = combined['comparisons'][label]
            pairs_all = cblock['pairs']
            # Only the nights that actually populated *this* leg.  A qualifying
            # night carries a block for every leg the bounce defines, but with
            # zero pairs on the legs it never visited (the BLOCK-T720 sweep
            # throws to a different elevation on each July night), and an empty
            # leg must not count as a night with a result.
            leg_nights = [d for d in nights
                          if len(nights[d]['comparisons'][label]['pairs'])]
            deltas_by_night = {d: nights[d]['comparisons'][label]['deltas']
                               for d in leg_nights}
            cam_only = bool(b.get('camera_hexapod_only', False))
            vmode_deltas = dof_deltas = None
            vmode_deltas_by_night = dof_deltas_by_night = {}
            vmode5_deltas = dof5_deltas = None
            vmode5_deltas_by_night = dof5_deltas_by_night = {}
            rbr_deltas = None
            rbr_deltas_by_night = {}
            vmode22_deltas = dof22_deltas = None
            vmode22_deltas_by_night = dof22_deltas_by_night = {}
            dof_oic_deltas = None
            dof_oic_deltas_by_night = {}
            if _svd_ok and C_all is not None:
                vmode_deltas = bl.paired_deltas_matrix(C_all, pairs_all)
                dof_deltas = bl.paired_deltas_matrix(DOF_all, pairs_all)
                vmode_deltas_by_night = {
                    d: bl.paired_deltas_matrix(C_all, nights[d]['comparisons'][label]['pairs'])
                    for d in leg_nights}
                dof_deltas_by_night = {
                    d: bl.paired_deltas_matrix(DOF_all, nights[d]['comparisons'][label]['pairs'])
                    for d in leg_nights}
                if cam_only and C5_all is not None:      # 5/5 Camera-hexapod-only
                    # DOF5_all has 5 cols (Cam hex); key them by their global DOF
                    # indices so the DOF plots' 50-DOF panel layout places them right.
                    vmode5_deltas = bl.paired_deltas_matrix(C5_all, pairs_all)
                    dof5_deltas = bl.paired_deltas_matrix(DOF5_all, pairs_all, keys=CAM_HEX_DOF)
                    vmode5_deltas_by_night = {
                        d: bl.paired_deltas_matrix(C5_all, nights[d]['comparisons'][label]['pairs'])
                        for d in leg_nights}
                    dof5_deltas_by_night = {
                        d: bl.paired_deltas_matrix(DOF5_all, nights[d]['comparisons'][label]['pairs'],
                                                   keys=CAM_HEX_DOF)
                        for d in leg_nights}
            # RBR Δ, from the same pairs and the same median / median-SEM
            # reduction as the default recovery above.  It is applied per pair
            # (the inversion is nonlinear, so it does not commute with the
            # median) on the paired Δ wavefront.
            if dof_ranges is not None and _W is not None:
                rbr_deltas = bl.rbr_deltas(
                    _W, pairs_all, svd, dof_ranges,
                    kappa=cfg['rbr_kappa'], power=cfg['rbr_power'])
                rbr_deltas_by_night = {
                    d: bl.rbr_deltas(
                        _W, nights[d]['comparisons'][label]['pairs'], svd,
                        dof_ranges, kappa=cfg['rbr_kappa'],
                        power=cfg['rbr_power'])
                    for d in leg_nights}
            # 22/12 and OIC Δ, the same way: one solve per pair, then the same
            # median / median-SEM reduction.  The 22/12 result is keyed by its
            # *global* DOF index (svd22.dof_idx) so it lands on the right panels
            # of the 50-DOF layout, exactly as the 5/5 result is keyed by
            # CAM_HEX_DOF.
            if svd22 is not None and _W is not None:
                dof22_deltas = bl.solver_deltas(_W, pairs_all, svd22,
                                                _solve_trunc,
                                                keys=list(svd22.dof_idx))
                vmode22_deltas = bl.paired_deltas_matrix(C22_all, pairs_all)
                dof22_deltas_by_night = {
                    d: bl.solver_deltas(
                        _W, nights[d]['comparisons'][label]['pairs'], svd22,
                        _solve_trunc, keys=list(svd22.dof_idx))
                    for d in leg_nights}
                vmode22_deltas_by_night = {
                    d: bl.paired_deltas_matrix(
                        C22_all, nights[d]['comparisons'][label]['pairs'])
                    for d in leg_nights}
            if oic_ok and _svd_ok and svd is not None and _W is not None:
                dof_oic_deltas = bl.solver_deltas(_W, pairs_all, svd,
                                                  _solve_oic)
                dof_oic_deltas_by_night = {
                    d: bl.solver_deltas(
                        _W, nights[d]['comparisons'][label]['pairs'], svd,
                        _solve_oic)
                    for d in leg_nights}
            br['comparisons'][label] = {
                'comp_stats': cblock['comp_stats'], 'comp_n': cblock['comp_n'],
                'deltas': cblock['deltas'], 'pairs': pairs_all,
                # Kept so the DOF table can report the *measured* elevation and
                # rotator angle of each leg, per night, rather than the nominal
                # window from the config.
                'comp_mask': cblock['comp_mask'],
                'n_pairs_by_night': {
                    str(d): len(nights[d]['comparisons'][label]['pairs'])
                    for d in leg_nights},
                'deltas_by_night': deltas_by_night,
                'vmode_deltas': vmode_deltas, 'dof_deltas': dof_deltas,
                'vmode_deltas_by_night': vmode_deltas_by_night,
                'dof_deltas_by_night': dof_deltas_by_night,
                'cam_only': cam_only,
                'vmode5_deltas': vmode5_deltas, 'dof5_deltas': dof5_deltas,
                'vmode5_deltas_by_night': vmode5_deltas_by_night,
                'dof5_deltas_by_night': dof5_deltas_by_night,
                'rbr_deltas': rbr_deltas,
                'rbr_deltas_by_night': rbr_deltas_by_night,
                'vmode22_deltas': vmode22_deltas, 'dof22_deltas': dof22_deltas,
                'vmode22_deltas_by_night': vmode22_deltas_by_night,
                'dof22_deltas_by_night': dof22_deltas_by_night,
                'dof_oic_deltas': dof_oic_deltas,
                'dof_oic_deltas_by_night': dof_oic_deltas_by_night}
            print(f'      comp "{label}": n={cblock["comp_n"]}, '
                  f'{len(pairs_all)} pairs')

            # ---- differential correctable-FWHM metric ----
            # median Δ-DZ (comp-ref) over pairs -> FWHM before; residual after the
            # 50/34 (and, for the rotator, 5/5) OFC projection -> FWHM after.
            if fwhm_conv is not None and _svd_ok and _W is not None and len(pairs_all):
                pd_dz = bl.paired_deltas_matrix(_W, pairs_all)
                med_dW = np.array([pd_dz[i]['delta'] for i in range(_W.shape[1])], float)
                # Every "after" series is the *achieved* residual dW - S (d / w)
                # of its own scheme, so the five are directly comparable.  For
                # the unregularized truncated schemes (50/34, 22/12, 5/5) the
                # achieved residual equals the subspace projection, so this
                # reproduces the numbers the projection metric gave.
                row = {'bounce': name, 'comparison': label, 'n_pairs': len(pairs_all),
                       'fwhm_before': _afw.fp_fwhm(svd, iZs, med_dW, fwhm_grid, fwhm_conv),
                       'fwhm_after_50_34': _scheme_fwhm(
                           med_dW, svd, _solve_trunc, iZs, fwhm_grid, fwhm_conv,
                           _afw, '50/34')}
                if dof_ranges is not None:
                    row['fwhm_after_rbr'] = _rbr_fwhm(
                        med_dW, svd, dof_ranges, cfg, iZs, fwhm_grid, fwhm_conv,
                        _afw)
                if svd22 is not None:
                    row['fwhm_after_22_12'] = _scheme_fwhm(
                        med_dW, svd22, _solve_trunc, iZs, fwhm_grid, fwhm_conv,
                        _afw, '22/12')
                if oic_ok:
                    row['fwhm_after_oic'] = _scheme_fwhm(
                        med_dW, svd, _solve_oic, iZs, fwhm_grid, fwhm_conv,
                        _afw, 'OIC')
                if cam_only and svd5 is not None:
                    row['fwhm_after_5_5'] = _scheme_fwhm(
                        med_dW, svd5, _solve_trunc, iZs, fwhm_grid, fwhm_conv,
                        _afw, '5/5')
                fwhm_rows.append(row)
                _extra = ''.join(
                    f", {tag}={row[k]:.4f}" for k, tag in
                    (('fwhm_after_rbr', 'RBR'), ('fwhm_after_22_12', '22/12'),
                     ('fwhm_after_oic', 'OIC'), ('fwhm_after_5_5', '5/5'))
                    if k in row)
                print(f"        correctable FWHM [arcsec]: before={row['fwhm_before']:.4f}, "
                      f"after 50/34={row['fwhm_after_50_34']:.4f}{_extra}")

                # Per-(night, leg) FWHM, the form the vs-B-value plot needs.
                # Every series here is the *achieved* residual dW - S (d/w), so
                # the default and RBR numbers are directly comparable; for the
                # truncated solution the achieved residual equals the subspace
                # projection that `fwhm_after_50_34` reports.
                bval = bl.leg_b_value(label)
                for d in leg_nights:
                    npairs = nights[d]['comparisons'][label]['pairs']
                    if not npairs:
                        continue
                    _pd = bl.paired_deltas_matrix(_W, npairs)
                    _dW = np.array([_pd[i]['delta'] for i in range(_W.shape[1])],
                                   float)
                    if not np.all(np.isfinite(_dW)):
                        continue
                    frow = {'bounce': name, 'comparison': label,
                            'night': int(d), 'b_value': bval,
                            'n_pairs': len(npairs),
                            'fwhm_before': _afw.fp_fwhm(svd, iZs, _dW, fwhm_grid,
                                                        fwhm_conv),
                            'fwhm_after_default': _scheme_fwhm(
                                _dW, svd, _solve_trunc, iZs, fwhm_grid,
                                fwhm_conv, _afw, '50/34')}
                    if dof_ranges is not None:
                        frow['fwhm_after_rbr'] = _rbr_fwhm(
                            _dW, svd, dof_ranges, cfg, iZs, fwhm_grid,
                            fwhm_conv, _afw)
                    if svd22 is not None:
                        frow['fwhm_after_22_12'] = _scheme_fwhm(
                            _dW, svd22, _solve_trunc, iZs, fwhm_grid,
                            fwhm_conv, _afw, '22/12')
                    if oic_ok:
                        frow['fwhm_after_oic'] = _scheme_fwhm(
                            _dW, svd, _solve_oic, iZs, fwhm_grid, fwhm_conv,
                            _afw, 'OIC')
                    if cam_only and svd5 is not None:
                        frow['fwhm_after_5_5'] = _scheme_fwhm(
                            _dW, svd5, _solve_trunc, iZs, fwhm_grid, fwhm_conv,
                            _afw, '5/5')
                    fwhm_bvalue_rows.append(frow)
            long_dfs.append(bl.to_long_df(
                cblock['deltas'], name, br['reference_label'], label,
                combined['ref_stats'], cblock['comp_stats'], night='all'))
            for d in deltas_by_night:
                long_dfs.append(bl.to_long_df(
                    deltas_by_night[d], name, br['reference_label'], label,
                    nights[d]['ref_stats'],
                    nights[d]['comparisons'][label]['comp_stats'], night=d))
        bounce_results[name] = br
    df_kj = pd.concat(long_dfs, ignore_index=True) if long_dfs else pd.DataFrame()

    return BounceAnalysis(
        fit_table=fit_table, bounce_results=bounce_results, df_kj=df_kj,
        fwhm_rows=fwhm_rows, fwhm_bvalue_rows=fwhm_bvalue_rows,
        iZs=iZs, k_list=k_list,
        svd=svd, svd5=svd5, svd22=svd22,
        C_all=C_all, DOF_all=DOF_all, C5_all=C5_all, DOF5_all=DOF5_all,
        C22_all=C22_all, DOFSUM_all=DOFSUM_all,
        dof_ranges=dof_ranges, oic_authority=oic_authority,
        svd_ok=_svd_ok, oic_ok=oic_ok,
        vmode_labels=vmode_labels, vmode5_labels=vmode5_labels,
        vmode22_labels=vmode22_labels)
