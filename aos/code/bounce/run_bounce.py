#!/usr/bin/env python3
"""Bounce-test analysis (script port of study_bounce.ipynb), per (param_set,
mi_name) on the measured-intrinsic refit  output/miw/<P>_<M>/fits.parquet.

FAM-triplet telescope-position bounce tests (e.g. BLOCK-T720 elevation,
BLOCK-T724 rotator): time-ordered paired-difference Δ (comparison − reference)
per Double-Zernike (k, j), OFC v-mode and physical DOF, with robust errors;
significance / pass heatmaps; DZ / v-mode / DOF vs ordinal-image pages;
per-night cross-scatter; and DOF night-vs-night scatter.  All analysis/plot
logic lives in code/bounce_lib.py (verbatim from the notebook).

The recovered DOF are reported two ways: the default truncated 50-DOF /
34-v-mode recovery, and Range-Bounded Recovery (RBR), which adds a per-DOF
penalty that grows steeply as a recovered amplitude approaches the range the
telescope can physically apply.  The default recovery exceeds that range on
these bounce legs by up to a factor of 11.27 (dimensionless, recovered
amplitude over allowed range); RBR bounds it at a small cost in corrected
FWHM.  The method is derived in the smatrix `regularized_inversion` study and
imported from there.

Writes, under the directory given by --out-dir (the pipeline passes
output/bounce/<P>_<M>/) :
    bounce/bounce_summary.pdf             Δ / significance / pass heatmaps + cross-scatter
    bounce/bounce_dz_vs_ordinal.pdf       DZ_kj vs ordinal image (per bounce)
    bounce/bounce_vmode_vs_ordinal.pdf    v-mode amplitude vs ordinal
    bounce/bounce_dof_vs_ordinal.pdf      physical DOF vs ordinal (+ Trim sum if enabled)
    bounce/bounce_dof_night_scatter.pdf   DOF night-A vs night-B 5-panel scatter
    bounce/bounce_dof_night_values.pdf    paired-Δ DOF vs B-set position, default
                                         recovery with the RBR overlay and the
                                         allowed-range band
    bounce/bounce_5x5_camera_hexapod.pdf  5/5 Camera-hexapod-only v-mode + DOF plots
                                         (camera_hexapod_only bounces, e.g. rotator)
    bounce/bounce_fwhm_metric.pdf         differential correctable-FWHM bar (before vs
                                         50/34 [vs 5/5] [vs RBR]) per bounce
    bounce/bounce_fwhm_vs_bvalue.pdf      correctable FWHM vs B-set elevation or
                                         rotator angle — uncorrected, default
                                         50/34, and RBR — one point per (night, leg)
    bounce_kj_stats.parquet              long-format Δ table (combined + per night)
    bounce_fwhm_metric.parquet           correctable-FWHM metric per bounce comparison
    bounce_fwhm_vs_bvalue.parquet        the same three FWHM series per (night, leg)
    bounce_dof_stats.parquet             per-DOF and per-v-mode Δ, with the RBR Δ
                                         (delta_rbr, delta_rbr_err) and the allowed
                                         range beside the default recovery's

Knobs (bounce specs, thresholds, n_dof/n_keep, add_dof_trim) come from
analysis_config.yaml (section ``bounce``).  RSP-only (DOF recovery via
lsst.ts.ofc; optional EFD/ConsDB Trim).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import QTable

sys.path.insert(0, str(Path(__file__).resolve().parent))          # same-study siblings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # aos/code (shared + other studies)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))      # repo root (common/)

from lsst.ts.intrinsic.wavefront import mi_config as mc
import bounce_lib as bl
from lsst.ts.intrinsic.wavefront.ofc_svd import LABELS_50DOF, DOF_UNITS_50

DEFAULT_BOUNCES = [
    {'name': 'T720_elevation', 'description': 'Elevation 40 - 70 deg, rotator ~ 0',
     'program': 'BLOCK-T720',
     'reference': {'label': 'Elev=70', 'alt_range': [67.0, 73.0], 'rotator_range': [-3.0, 3.0]},
     'comparisons': [{'label': 'Elev=40', 'alt_range': [37.0, 43.0], 'rotator_range': [-3.0, 3.0]}]},
    {'name': 'T724_rotator', 'description': 'Rotator 60 - 0 deg, elevation ~ 70',
     'program': 'BLOCK-T724',
     # rotator bounce: only the camera hexapod moves, so also evaluate a 5-DOF /
     # 5-v-mode (Camera-hexapod-only) correction alongside the full 50/34.
     'camera_hexapod_only': True,
     'reference': {'label': 'Rot=0', 'rotator_range': [-3.0, 3.0], 'alt_range': [67.0, 73.0]},
     'comparisons': [{'label': 'Rot=60', 'rotator_range': [57.0, 63.0], 'alt_range': [67.0, 73.0]}]},
]
# Camera-hexapod DOF indices in LABELS_50DOF (Cam_dz/dx/dy/rx/ry) for the 5/5 scheme.
CAM_HEX_DOF = [5, 6, 7, 8, 9]
# The AOS reduced DOF set for the 22/12 scheme: 10 rigid-body, the first 7 M1M3
# bending modes, the first 5 M2 bending modes.  An *index set*, NOT the first 22
# contiguous indices -- see `aos-22dof-reduced-set` in notes/claude-memory/.
# Passed to build_ofc_svd as an explicit list, which carries it on OFCSvd.dof_idx.
DOF22 = list(range(0, 10)) + list(range(10, 17)) + list(range(30, 35))
DEFAULT = dict(
    fit_prefix='z1toz6', focal_k_range=[1, 2, 3, 4, 5, 6], pupil_j_range=None,
    bounces=DEFAULT_BOUNCES, n_dof=50, n_keep=34, ofc_normalization_yaml=None,
    heatmap_vlim_um=None, heatmap_cell_fontsize=7, sig_vlim=5.0,
    pass_nsigma_threshold=3.5, pass_delta_threshold_um=0.1,
    pass_sigma_only_threshold=5.0, cross_scatter_zoom_um=0.1,
    ordinal_j_per_page=7, night_min_visits=3,
    # A/B position identification tolerances, in deg. Elevation centers 70 and
    # 75 are only 5 deg apart, so the half-width must stay below 2.5 deg;
    # measured BLOCK-T720/T724 positions sit within 0.3 deg of a center.
    ab_elev_halfwidth_deg=2.0, ab_rot_halfwidth_deg=2.0,
    vmode_ncols=5, vmode_rows_per_page=7, dof_ncols=2, dof_rows_per_page=5,
    add_dof_trim=False, trim_efd_topic='lsst.sal.MTAOS.logevent_degreeOfFreedom',
    trim_consdb_url='http://consdb-pq.consdb:8080/consdb',
    trim_time_col='mjd', trim_mjd_scale='tai',
    # Range-Bounded Recovery (RBR): the range-penalized alternative to the
    # truncated recovery, reported alongside it.  kappa is the ratio
    # |d_j| / r_j (dimensionless, recovered amplitude over allowed range) at
    # which the penalty reaches unit weight; power sets how fast it climbs
    # (penalty ~ ratio ** (2 * power)).  The defaults are the setting the
    # smatrix `regularized_inversion` study found best or near-best on five of
    # the six bounce legs.
    rbr_enable=True, rbr_kappa=4.0, rbr_power=3,
    # The 22-DOF / 12-v-mode reduced scheme the AOS is expected to operate in.
    # Only the mode count is configurable: the 22 DOF are an *index set*, fixed in
    # code as `DOF22`, not a count that could be read from here.
    reduced_enable=True, n_keep_reduced=12,
    # The quadratic motion penalty the ts_ofc Optimal Integral Controller (OIC)
    # already carries, scored as a fourth scheme.  `oic_rho` is the OIC's
    # `motion_penalty`, dimensionless.  ts_ofc ships 0.0, i.e. the penalty is
    # inactive as delivered, so a value must be chosen; this one is *consumed*,
    # and the scan it comes from is in
    # smatrix/docs/studies/regularized_inversion.md.  Do not re-derive it here.
    oic_enable=True, oic_rho=1.0e-3)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--fits', default=None)
    # The caller (the Snakefile) owns the output layout and may name directories
    # differently from the param_set / mi_name keys, so it supplies the path
    # directly rather than having it derived from those keys.  Given, ALL
    # products land here — including the two tables that otherwise go one level
    # up (bounce_kj_stats and bounce_fwhm_metric).
    ap.add_argument('--out-dir', default=None,
                    help='directory to write into '
                         '(default: output/<ps>/<mi>/bounce)')
    ap.add_argument('--min-detectors', type=int, default=None,
                    help='Per-visit quality cut: keep visits with '
                         'n_detectors_with_min_donuts >= this (relaxes ONLY the '
                         'CCD-count cut; blur cut kept). fits.parquet now holds all '
                         'visits, so the cut is applied here. Default: None -> use '
                         'the precomputed visit_quality_pass (nd>=170).')
    ap.add_argument('--rbr-kappa', type=float, default=None,
                    help='Range-Bounded Recovery: ratio |d_j|/r_j '
                         '(dimensionless, recovered amplitude over allowed '
                         'range) at which the penalty reaches unit weight. '
                         'Default from config (4.0).')
    ap.add_argument('--rbr-power', type=int, default=None,
                    help='Range-Bounded Recovery: penalty exponent; the penalty '
                         'goes as the ratio to the 2*power. Default from '
                         'config (3).')
    ap.add_argument('--no-rbr', action='store_true',
                    help='Skip Range-Bounded Recovery and report only the '
                         'default truncated recovery.')
    args = ap.parse_args()

    cfg = {**DEFAULT, **mc.analysis_section(
        'bounce', args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    if args.rbr_kappa is not None:
        cfg['rbr_kappa'] = float(args.rbr_kappa)
    if args.rbr_power is not None:
        cfg['rbr_power'] = int(args.rbr_power)
    if args.no_rbr:
        cfg['rbr_enable'] = False
    prefix = cfg['fit_prefix']
    bounces = cfg['bounces']

    base = Path(args.output_root) / args.param_set / args.mi_name
    fits_path = Path(args.fits) if args.fits else base / 'fits.parquet'
    out_dir = Path(args.out_dir) if args.out_dir else base / 'bounce'
    out_dir.mkdir(parents=True, exist_ok=True)
    # The two long-format tables sit beside the PDFs when --out-dir is given,
    # and one level up in the legacy layout where out_dir is a 'bounce/' subdir.
    tbl_dir = out_dir if args.out_dir else out_dir.parent
    print(f'[bounce] {fits_path}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from lsst.ts.intrinsic.wavefront.intrinsics_lib import markers_legend_figure
        _marker_ok = True
    except Exception:
        _marker_ok = False

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
    if args.min_detectors is not None and \
            'n_detectors_with_min_donuts' in fit_table.colnames:
        from lsst.ts.intrinsic.wavefront.intrinsics_lib import quality_visit_mask
        keep = np.asarray(quality_visit_mask(
            fit_table, min_detectors_per_visit=args.min_detectors, verbose=False),
            dtype=bool)
        print(f'  quality cut (min_detectors={args.min_detectors}): '
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
    fwhm_conv = fwhm_grid = None
    try:
        from lsst.ts.wep.utils import convertZernikesToPsfWidth as fwhm_conv
        import aos_fwhm as _afw
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
                          f'smatrix/docs/studies/regularized_inversion.md')
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
            from common.dof_telemetry import fetch_aggregated_dof_for_visits
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

    # ---- DZ vs ordinal (cell 18) ----
    with PdfPages(str(out_dir / 'bounce_dz_vs_ordinal.pdf')) as pdf:
        if _marker_ok:
            leg = markers_legend_figure(show_iter_distinction=False)
            pdf.savefig(leg, bbox_inches='tight'); plt.close(leg)
        # A/B position table: what elevation/rotator each night's reference (A)
        # and throw (B) sets actually sat at, and whether every contiguous
        # block resolved to a grid position.
        for b in bounces:
            ab_rows = bl.ab_position_table(
                fit_table, b,
                elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg'),
                rot_halfwidth_deg=cfg.get('ab_rot_halfwidth_deg'))
            for f in bl.plot_ab_position_table(
                    ab_rows, title=f'{b["name"]}: A/B bounce positions'):
                pdf.savefig(f, bbox_inches='tight'); plt.close(f)
        for b in bounces:
            ftb = fit_table[bl.bounce_program_mask(fit_table, b)]
            if len(ftb) == 0:
                continue
            for f in bl.plot_dz_vs_ordinal_pages(ftb, prefix, k_list, iZs,
                                                 j_per_page=cfg['ordinal_j_per_page'],
                                                 title_prefix=f'{b["name"]}: ',
                                                 elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
                pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- v-mode + DOF vs ordinal (cell 20) ----
    if _svd_ok and C_all is not None:
        with PdfPages(str(out_dir / 'bounce_vmode_vs_ordinal.pdf')) as pdf:
            if _marker_ok:
                leg = markers_legend_figure(show_iter_distinction=False)
                pdf.savefig(leg, bbox_inches='tight'); plt.close(leg)
            for b in bounces:
                m = bl.bounce_program_mask(fit_table, b)
                if int(m.sum()) == 0:
                    continue
                for f in bl.plot_values_vs_ordinal_pages(
                        fit_table[m], C_all[m], vmode_labels, units=None,
                        title_root=f'{b["name"]}: OFC v-mode amplitude c_i',
                        ncols=cfg['vmode_ncols'], rows_per_page=cfg['vmode_rows_per_page'],
                        elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
                    pdf.savefig(f, bbox_inches='tight'); plt.close(f)
        with PdfPages(str(out_dir / 'bounce_dof_vs_ordinal.pdf')) as pdf:
            if _marker_ok:
                leg = markers_legend_figure(show_iter_distinction=False)
                pdf.savefig(leg, bbox_inches='tight'); plt.close(leg)
            for b in bounces:
                m = bl.bounce_program_mask(fit_table, b)
                if int(m.sum()) == 0:
                    continue
                for f in bl.plot_values_vs_ordinal_pages(
                        fit_table[m], DOF_all[m], LABELS_50DOF, units=DOF_UNITS_50,
                        title_root=f'{b["name"]}: Physical DOF (FAM analysis)',
                        ncols=cfg['dof_ncols'], rows_per_page=cfg['dof_rows_per_page'],
                        elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
                    pdf.savefig(f, bbox_inches='tight'); plt.close(f)
            if DOFSUM_all is not None:
                for b in bounces:
                    m = bl.bounce_program_mask(fit_table, b)
                    if int(m.sum()) == 0:
                        continue
                    for f in bl.plot_values_vs_ordinal_pages(
                            fit_table[m], DOFSUM_all[m], LABELS_50DOF, units=DOF_UNITS_50,
                            title_root=f'{b["name"]}: Physical DOF + AOS Trim',
                            ncols=cfg['dof_ncols'], rows_per_page=cfg['dof_rows_per_page'],
                        elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- summary: per-night Δ DZ_kj cross-comparison (cell 22) ----
    # The per-(k, j) Δ and significance heatmaps are dropped: the same numbers
    # are in bounce_kj_stats.parquet, and the night-vs-night cross-scatter is
    # the product that actually answers the repeatability question.  It is
    # drawn for every comparison leg that has 2 or more qualifying nights,
    # wide and zoomed.
    with PdfPages(str(out_dir / 'bounce_summary.pdf')) as pdf:
        cov = bl.leg_night_coverage(bounce_results)
        f = bl.plot_leg_night_coverage(
            cov, title='Nights with results per bounce leg (B set)')
        if f is not None:
            pdf.savefig(f, bbox_inches='tight'); plt.close(f)
        for name, br in bounce_results.items():
            for clabel, cb in br['comparisons'].items():
                dbn = cb.get('deltas_by_night', {})
                if len(dbn) < 2:
                    print(f'  (bounce_summary: {name} {clabel} has '
                          f'{len(dbn)} night(s) — no cross-scatter page)')
                    continue
                passing = bl.passing_terms(
                    cb['deltas'], cfg['pass_delta_threshold_um'],
                    cfg['pass_nsigma_threshold'],
                    sigma_only_th=cfg['pass_sigma_only_threshold'])
                for _d in dbn.values():
                    passing |= bl.passing_terms(_d, cfg['pass_delta_threshold_um'],
                                                cfg['pass_nsigma_threshold'],
                                                sigma_only_th=cfg['pass_sigma_only_threshold'])
                for f in bl.plot_night_cross_scatter(
                        dbn, sorted(passing),
                        title_root=f'{name}: per-night Δ DZ_kj cross-comparison '
                                   f'({clabel} − {br["reference_label"]})',
                        zoom_lim_um=cfg['cross_scatter_zoom_um']) or []:
                    pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- DOF night-vs-night scatter (cell 24) ----
    # One page per (comparison leg, night pair), for every leg with 2 or more
    # qualifying nights; a leg exercised on a single night gets no page.
    if _svd_ok and C_all is not None:
        with PdfPages(str(out_dir / 'bounce_dof_night_scatter.pdf')) as pdf:
            f = bl.plot_leg_night_coverage(
                bl.leg_night_coverage(bounce_results),
                title='Nights with results per bounce leg (B set)')
            if f is not None:
                pdf.savefig(f, bbox_inches='tight'); plt.close(f)
            for name, br in bounce_results.items():
                for clabel, cb in br['comparisons'].items():
                    dbn = cb.get('dof_deltas_by_night', {})
                    if len(dbn) < 2:
                        print(f'  (dof_night_scatter: {name} {clabel} has '
                              f'{len(dbn)} night(s) — no page)')
                        continue
                    for f in bl.plot_dof_night_scatter(
                            dbn, LABELS_50DOF, units=DOF_UNITS_50,
                            title_root=f'{name}: DOF Δ ({clabel} - {br["reference_label"]})'):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- DOF Δ per (night, B set), as per-DOF panels (cell 26) ----
    # x is the B-set position: elevation in deg for the BLOCK-T720 sweep,
    # camera-rotator angle in deg for BLOCK-T724.  Each point is one
    # (night, leg) paired Δ, labelled with its day_obs and B value.
    #
    # Every recovery scheme the run computed appears on the same panels, so the
    # DOF value a scheme asks for can be read against the allowed range r_j and
    # against what the other schemes ask for.  The base (filled-circle) series
    # is the 50/34 recovery on both bounces; on a camera_hexapod_only bounce the
    # 5/5 scheme — the one the result is actually used in, since with only the
    # rotator moving only camera-hexapod corrections are applied — is an overlay
    # on its own five panels.  A scheme that does not reach a DOF simply has no
    # marker there: 22/12 is absent on the 28 DOF outside its index set, and 5/5
    # on the 45 DOF outside the camera hexapod.
    if _svd_ok and DOF_all is not None:
        with PdfPages(str(out_dir / 'bounce_dof_night_values.pdf')) as pdf:
            for b in bounces:
                name = b['name']
                br = bounce_results.get(name)
                if br is None:
                    continue
                cam_only = bool(b.get('camera_hexapod_only', False))
                entries = []
                for clabel, cb in br['comparisons'].items():
                    bval = bl.leg_b_value(clabel)
                    _by = {k: (cb.get(f'{k}_by_night') or {})
                           for k in ('rbr_deltas', 'dof22_deltas',
                                     'dof_oic_deltas', 'dof5_deltas')}
                    for nt, dd in (cb.get('dof_deltas_by_night') or {}).items():
                        e = {'night': int(nt), 'b_value': bval,
                             'label': clabel, 'dof_deltas': dd}
                        for k, bn in _by.items():
                            e[k] = bn.get(nt)
                        entries.append(e)
                if not entries:
                    print(f'  (dof_night_values: {name} has no per-night DOF Δ)')
                    continue
                overlay = []
                if dof_ranges is not None and any(e.get('rbr_deltas')
                                                  for e in entries):
                    overlay.append(('rbr_deltas',
                                    f'RBR 50/34, kappa={cfg["rbr_kappa"]:g} '
                                    f'power={cfg["rbr_power"]} (both '
                                    f'dimensionless; open square)'))
                if any(e.get('dof22_deltas') for e in entries):
                    overlay.append(('dof22_deltas',
                                    f'{len(DOF22)}/{cfg["n_keep_reduced"]} '
                                    f'reduced DOF set (open triangle)'))
                if any(e.get('dof_oic_deltas') for e in entries):
                    overlay.append(('dof_oic_deltas',
                                    f'OIC 50/34, rho={cfg["oic_rho"]:g} '
                                    f'dimensionless (open diamond)'))
                if cam_only and any(e.get('dof5_deltas') for e in entries):
                    overlay.append(('dof5_deltas',
                                    '5/5 camera hexapod (open inverted '
                                    'triangle)'))
                axis = bl.leg_axis_name(b)
                sub = ''
                if dof_ranges is not None:
                    sub = ('\nShaded band is the allowed range ±r_j, in each '
                           'DOF\'s own unit')
                figs = bl.plot_dof_vs_b_value_panels(
                    entries, LABELS_50DOF, DOF_UNITS_50,
                    x_label=f'B set {axis} [deg]',
                    ncols=int(cfg['dof_ncols']),
                    rows_per_page=int(cfg['dof_rows_per_page']),
                    title=f'{name}: paired Δ DOF vs B-set {axis.lower()}, '
                          f'all recovery schemes\n'
                          f'Δ = comparison − {br["reference_label"]}, '
                          f'one point per (night, B set){sub}',
                    overlay_series=overlay,
                    ranges=dof_ranges,
                    base_label=f'default {cfg["n_dof"]}/{cfg["n_keep"]}')
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ---- 5/5 Camera-hexapod-only plots (camera_hexapod_only bounces, e.g. rotator) ----
    if _svd_ok and C5_all is not None:
        cam_labels = [LABELS_50DOF[i] for i in CAM_HEX_DOF]
        cam_units = [DOF_UNITS_50[i] for i in CAM_HEX_DOF]
        cam_bounces = [b for b in bounces if b.get('camera_hexapod_only')]
        if cam_bounces:
            dobs = np.asarray(fit_table['day_obs']).astype(int)
            with PdfPages(str(out_dir / 'bounce_5x5_camera_hexapod.pdf')) as pdf:
                if _marker_ok:
                    leg = markers_legend_figure(show_iter_distinction=False)
                    pdf.savefig(leg, bbox_inches='tight'); plt.close(leg)
                for b in cam_bounces:
                    name = b['name']; br = bounce_results[name]
                    m = bl.bounce_program_mask(fit_table, b)
                    if int(m.sum()) == 0:
                        continue
                    for f in bl.plot_values_vs_ordinal_pages(          # (1) v-mode vs ordinal
                            fit_table[m], C5_all[m], vmode5_labels, units=None,
                            title_root=f'{name} [5/5 Cam-hex]: OFC v-mode amplitude c_i',
                            ncols=cfg['vmode_ncols'], rows_per_page=cfg['vmode_rows_per_page'],
                        elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)
                    for f in bl.plot_values_vs_ordinal_pages(          # (2) DOF vs ordinal
                            fit_table[m], DOF5_all[m], cam_labels, units=cam_units,
                            title_root=f'{name} [5/5 Cam-hex]: Camera-hexapod DOF',
                            ncols=cfg['dof_ncols'], rows_per_page=cfg['dof_rows_per_page'],
                        elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)
                    for clabel, cb in br['comparisons'].items():       # (3) DOF night-vs-night scatter
                        dbn5 = cb.get('dof5_deltas_by_night', {})
                        if len(dbn5) >= 2:
                            for f in bl.plot_dof_night_scatter(
                                    dbn5, LABELS_50DOF, units=DOF_UNITS_50,
                                    title_root=f'{name} [5/5]: Cam-hex DOF Δ '
                                               f'({clabel} - {br["reference_label"]})'):
                                pdf.savefig(f, bbox_inches='tight'); plt.close(f)
                    dof50_by_night = {}                                # (4) DOF per-night median
                    for nt in sorted(set(dobs[m].tolist())):
                        sel = m & (dobs == nt)
                        if int(sel.sum()) == 0:
                            continue
                        full = np.full((int(sel.sum()), len(LABELS_50DOF)), np.nan)
                        full[:, CAM_HEX_DOF] = DOF5_all[sel]
                        dof50_by_night[int(nt)] = full
                    if dof50_by_night:
                        pdf.savefig(bl.plot_dof_per_night_summary(
                            dof50_by_night, LABELS_50DOF, DOF_UNITS_50,
                            title=f'{name} [5/5 Cam-hex]: DOF median per night'),
                            bbox_inches='tight')
                        plt.close('all')
            print('  wrote bounce_5x5_camera_hexapod.pdf')

    # ---- differential correctable-FWHM metric ----
    if fwhm_rows:
        fdf = pd.DataFrame(fwhm_rows)
        fdf.to_parquet(tbl_dir / 'bounce_fwhm_metric.parquet')
        print('  correctable-FWHM metric [arcsec, median over focal plane]:')
        print('   ' + fdf.to_string(index=False).replace('\n', '\n   '))
        # Every "after" bar is the achieved residual dW − S·(d/w) of its own
        # scheme, so the bars within a group are a like-for-like comparison.
        bar_cols = [c for c in ('fwhm_before', 'fwhm_after_50_34',
                                'fwhm_after_rbr', 'fwhm_after_22_12',
                                'fwhm_after_oic', 'fwhm_after_5_5')
                    if c in fdf.columns]
        lab = {'fwhm_before': 'no correction',
               'fwhm_after_50_34': '50/34 truncated',
               'fwhm_after_rbr': f'RBR 50/34 (kappa={cfg["rbr_kappa"]:g}, '
                                 f'power={cfg["rbr_power"]}; both dimensionless)',
               'fwhm_after_22_12': f'{len(DOF22)}/{cfg["n_keep_reduced"]} '
                                   f'reduced DOF set',
               'fwhm_after_oic': f'OIC 50/34 (rho={cfg["oic_rho"]:g} '
                                 f'dimensionless)',
               'fwhm_after_5_5': '5/5 camera hexapod'}
        with PdfPages(str(out_dir / 'bounce_fwhm_metric.pdf')) as pdf:
            fig, ax = plt.subplots(figsize=(1.8 * len(fdf) + 3, 5), constrained_layout=True)
            x = np.arange(len(fdf)); w = 0.8 / max(len(bar_cols), 1)
            for i, c in enumerate(bar_cols):
                ax.bar(x + i * w, fdf[c].to_numpy(float), w, label=lab[c])
            ax.set_xticks(x + w * (len(bar_cols) - 1) / 2)
            ax.set_xticklabels([f'{r.bounce}\n{r.comparison}' for r in fdf.itertuples()],
                               fontsize=8)
            ax.set_ylabel('differential FWHM [arcsec]  (median over FP)')
            ax.set_title('Bounce optical-state change — correctable FWHM\n'
                         'RSS of median Δ-DZ aberrations; every "after" bar is '
                         'the achieved residual dW − S·(d/w) of its own scheme')
            ax.legend(); ax.grid(axis='y', alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print('  wrote bounce_fwhm_metric.pdf + .parquet')

    # ---- correctable FWHM vs B-set position, per (night, leg) ----
    # The image-quality question the scheme comparison exists to answer: what
    # does each recovery cost in delivered FWHM as a function of the throw — for
    # RBR the cost of bounding the recovered DOF to the allowed range, for 22/12
    # the cost of dropping to the reduced operational DOF set.  One page per
    # bounce, one point per (night, B set), one series per scheme.
    if fwhm_bvalue_rows:
        bdf = pd.DataFrame(fwhm_bvalue_rows)
        bdf.to_parquet(tbl_dir / 'bounce_fwhm_vs_bvalue.parquet')
        with PdfPages(str(out_dir / 'bounce_fwhm_vs_bvalue.pdf')) as pdf:
            for b in bounces:
                name = b['name']
                sub = bdf[bdf['bounce'] == name]
                if sub.empty:
                    continue
                axis = bl.leg_axis_name(b)
                ttl = (f'{name}: correctable FWHM of the bounce optical-state '
                       f'change vs B-set {axis.lower()}\n'
                       f'one point per (night, B set); residuals are the '
                       f'achieved dW − S·(d/w)')
                if dof_ranges is not None:
                    ttl += (f'\nRBR: kappa={cfg["rbr_kappa"]:g}, '
                            f'power={cfg["rbr_power"]} '
                            f'(penalty ~ (|d_j|/(kappa·r_j))^{2 * cfg["rbr_power"]})')
                if oic_ok:
                    ttl += (f'; OIC: rho={cfg["oic_rho"]:g} (dimensionless '
                            f'motion-penalty weight)')
                fig = bl.plot_fwhm_vs_b_value(
                    sub.to_dict('records'), x_label=f'B set {axis} [deg]',
                    title=ttl)
                if fig is not None:
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print(f'  wrote bounce_fwhm_vs_bvalue.pdf + .parquet '
              f'({len(bdf)} rows)')
        # Image-quality cost of each scheme relative to the unregularized 50/34
        # recovery, as the difference of two achieved-residual FWHM in arcsec.
        for _col, _tag in (('fwhm_after_rbr',
                            f'RBR 50/34 (kappa={cfg["rbr_kappa"]:g}, '
                            f'power={cfg["rbr_power"]}; both dimensionless)'),
                           ('fwhm_after_22_12',
                            f'{len(DOF22)}/{cfg["n_keep_reduced"]} reduced'),
                           ('fwhm_after_oic',
                            f'OIC 50/34 (rho={cfg["oic_rho"]:g} dimensionless)'),
                           ('fwhm_after_5_5', '5/5 camera hexapod')):
            if _col not in bdf.columns:
                continue
            _c = bdf.dropna(subset=['fwhm_after_default', _col])
            if not len(_c):
                continue
            _cost = _c[_col] - _c['fwhm_after_default']
            print(f'  {_tag} FWHM cost over the 50/34 truncated recovery '
                  f'[arcsec]: median={_cost.median():.5f}, '
                  f'min={_cost.min():.5f}, max={_cost.max():.5f} '
                  f'over {len(_c)} (night, leg) points')

    # ---- per-DOF and per-v-mode Δ table ----
    # The DOF Δ was previously visible only inside the PDFs, so any number
    # quoted from it had to be read off a plot.  One row per (bounce, leg,
    # night, quantity); `unit` is µm for translations and bending-mode
    # amplitudes, arcsec for hexapod rotations, and dimensionless for v-modes.
    # This table is what reproduces bounce_dof_night_values.pdf, and is the
    # form shared externally, so it carries the *measured* elevation and
    # rotator angle of both legs rather than the nominal config window.
    if _svd_ok and DOF_all is not None:
        dof_rows = []
        for name, br in bounce_results.items():
            ref_pt_all = bl.leg_pointing(fit_table, br['reference_mask'])
            for clabel, cb in br['comparisons'].items():
                # Pointing per (leg, night), plus the pooled 'all' row.
                pt_cache = {}

                def _pointing(night, _cb=cb, _br=br, _cache=pt_cache,
                              _ref_all=ref_pt_all):
                    if night not in _cache:
                        d = None if night == 'all' else int(night)
                        cp = bl.leg_pointing(fit_table, _cb['comp_mask'], day_obs=d)
                        rp = (_ref_all if night == 'all'
                              else bl.leg_pointing(fit_table, _br['reference_mask'],
                                                   day_obs=d))
                        _cache[night] = (cp, rp)
                    return _cache[night]

                # RBR Δ, keyed the same way, so the `dof` rows can carry it as
                # two extra columns beside the default recovery's Δ rather than
                # as a separate `kind` a reader would have to pivot.
                rbr_blocks = {'all': cb.get('rbr_deltas') or {}}
                rbr_blocks.update({str(int(d)): (v or {}) for d, v
                                   in (cb.get('rbr_deltas_by_night') or {}).items()})

                for kind, pooled, by_night, labels, units in (
                        ('dof', cb.get('dof_deltas'), cb.get('dof_deltas_by_night'),
                         LABELS_50DOF, DOF_UNITS_50),
                        ('vmode', cb.get('vmode_deltas'), cb.get('vmode_deltas_by_night'),
                         None, None),
                        # The 5-DOF / 5-v-mode camera-hexapod-only scheme, present
                        # only on a camera_hexapod_only bounce.  Its DOF indices are
                        # the camera-hexapod entries of the same 50-DOF labelling.
                        ('dof5', cb.get('dof5_deltas'), cb.get('dof5_deltas_by_night'),
                         LABELS_50DOF, DOF_UNITS_50),
                        ('vmode5', cb.get('vmode5_deltas'),
                         cb.get('vmode5_deltas_by_night'), None, None),
                        # The 22-DOF / 12-v-mode reduced set, the scheme the AOS
                        # is expected to operate in.  Its DOF rows are keyed by
                        # global DOF index (DOF22), so they share the 50-DOF
                        # labelling; its v-mode rows are its own 12 v-modes and
                        # are *not* the first 12 of the 50/34 scheme's.
                        ('dof22', cb.get('dof22_deltas'),
                         cb.get('dof22_deltas_by_night'),
                         LABELS_50DOF, DOF_UNITS_50),
                        ('vmode22', cb.get('vmode22_deltas'),
                         cb.get('vmode22_deltas_by_night'), None, None),
                        # The 50-DOF recovery with the ts_ofc OIC quadratic
                        # motion penalty at cfg['oic_rho'].  No v-mode kind:
                        # like RBR it is solved in the retained-mode
                        # coefficients but reported as DOF.
                        ('dof_oic', cb.get('dof_oic_deltas'),
                         cb.get('dof_oic_deltas_by_night'),
                         LABELS_50DOF, DOF_UNITS_50)):
                    blocks = {'all': pooled} if pooled else {}
                    blocks.update({str(int(d)): v for d, v in (by_night or {}).items()})
                    for night, block in blocks.items():
                        cp, rp = _pointing(night)
                        # Only the 50-DOF `dof` kind has an RBR counterpart:
                        # RBR is a 50-DOF recovery, and v-modes are the step it
                        # replaces rather than a quantity it produces.
                        rb = rbr_blocks.get(night, {}) if kind == 'dof' else {}
                        for q, v in (block or {}).items():
                            rv = rb.get(int(q)) or {}
                            dof_rows.append({
                                'bounce': name, 'comparison': clabel,
                                'reference': br['reference_label'],
                                'block': cp['block'] or rp['block'],
                                'night': night,
                                'day_obs': (pd.NA if night == 'all' else int(night)),
                                'elevation_deg': cp['elevation_deg'],
                                'rot_angle_deg': cp['rot_angle_deg'],
                                'ref_elevation_deg': rp['elevation_deg'],
                                'ref_rot_angle_deg': rp['rot_angle_deg'],
                                'n_visits': cp['n_visits'],
                                'n_pairs': (len(cb.get('pairs', [])) if night == 'all'
                                            else cb.get('n_pairs_by_night', {}).get(
                                                night, 0)),
                                'kind': kind, 'index': int(q),
                                'label': (labels[int(q)] if labels is not None
                                          else f'v{int(q) + 1}'),
                                'unit': (units[int(q)] if units is not None
                                         else 'dimensionless'),
                                'delta': float(v.get('delta', np.nan)),
                                'delta_err': float(v.get('err', np.nan)),
                                # Range-Bounded Recovery Δ for the same DOF,
                                # same unit and same median / median-SEM error
                                # definition; NaN for the v-mode kinds.
                                'delta_rbr': float(rv.get('delta', np.nan)),
                                'delta_rbr_err': float(rv.get('err', np.nan)),
                            })
        if dof_rows:
            ddf = pd.DataFrame(dof_rows)
            ddf['significance'] = (ddf['delta'].abs() / ddf['delta_err']).replace(
                [np.inf, -np.inf], np.nan)
            # Allowed range and the two amplitude-over-range ratios, so a reader
            # can see which recovery exceeds what the telescope can apply
            # without having to re-derive r_j from the normalization weights.
            if dof_ranges is not None:
                is_dof = ddf['kind'].isin(['dof', 'dof5', 'dof22', 'dof_oic'])
                rj = ddf['index'].map(
                    lambda i: (float(dof_ranges[int(i)])
                               if 0 <= int(i) < len(dof_ranges) else np.nan))
                ddf['dof_range'] = rj.where(is_dof)
                ddf['ratio_to_range'] = (ddf['delta'].abs() / ddf['dof_range']
                                         ).replace([np.inf, -np.inf], np.nan)
                ddf['ratio_to_range_rbr'] = (
                    ddf['delta_rbr'].abs() / ddf['dof_range']
                ).replace([np.inf, -np.inf], np.nan)
            ddf.to_parquet(tbl_dir / 'bounce_dof_stats.parquet')
            print(f'  wrote bounce_dof_stats.parquet ({len(ddf)} rows)')
            if 'ratio_to_range' in ddf.columns:
                # One line per recovery scheme: how many of its DOF rows ask for
                # more than the telescope can apply.  Per-night and leg-pooled
                # rows are split because they describe the same physics and a
                # single total would double-count it.  The `dof` kind also
                # carries RBR as columns, so RBR is counted from
                # `ratio_to_range_rbr` on those rows.
                print('    DOF rows over the allowed range (|Δ|/r_j > 1, '
                      'dimensionless), per scheme:')
                _schemes = [('50/34 truncated', 'dof', 'ratio_to_range'),
                            (f'RBR 50/34 (kappa={cfg["rbr_kappa"]:g}, '
                             f'power={cfg["rbr_power"]}; both dimensionless)',
                             'dof', 'ratio_to_range_rbr'),
                            (f'{len(DOF22)}/{cfg["n_keep_reduced"]} reduced',
                             'dof22', 'ratio_to_range'),
                            (f'OIC 50/34 (rho={cfg["oic_rho"]:g} '
                             f'dimensionless)',
                             'dof_oic', 'ratio_to_range'),
                            ('5/5 camera hexapod', 'dof5', 'ratio_to_range')]
                for _tag, _kind, _col in _schemes:
                    if _col not in ddf.columns:
                        continue
                    _d = ddf[(ddf['kind'] == _kind) & ddf[_col].notna()]
                    if not len(_d):
                        continue
                    _parts = []
                    for _lab, _s in (
                            ('per (leg, night)',
                             _d[_d['night'].astype(str) != 'all']),
                            ('leg-pooled',
                             _d[_d['night'].astype(str) == 'all'])):
                        if len(_s):
                            _parts.append(f'{_lab} {int((_s[_col] > 1).sum())} '
                                          f'of {len(_s)}')
                    print(f'      {_tag:42s} ' + '; '.join(_parts))

    # ---- long-format table (cell 28) ----
    df_kj.to_parquet(tbl_dir / 'bounce_kj_stats.parquet')
    print(f'  wrote bounce_*.pdf + bounce_kj_stats.parquet ({len(df_kj)} rows)')
    print('[bounce] done.')


if __name__ == '__main__':
    main()
