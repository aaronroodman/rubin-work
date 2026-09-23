#!/usr/bin/env python3
"""Compare regularized sensitivity-matrix inversions against the truncated 50/34
recovery, on the measured bounce-test wavefronts, scored by image quality (IQ).

Three inversions of the same measured Double Zernike (DZ) wavefront are run on every
bounce leg: the current truncated singular value decomposition (SVD) at 34 modes
(method 0), a damped SVD (method A), and a superlinear per-degree-of-freedom (DOF)
range penalty solved by iteratively reweighted least squares (method B). Each is
scored on two axes:

* **image quality** — the correctable full width at half maximum (FWHM) in arcsec
  that the recovered DOF correction actually leaves on the focal plane, via
  ``aos_fwhm``;
* **feasibility** — the largest recovered ``|d_j| / r_j`` against the allowed range
  ``r_j``, dimensionless, and how many DOF exceed the range.

The point of the comparison is to find out what image quality a feasible solution
costs. A truncated inversion can drive a bending-mode amplitude many times past the
actuator-force-limited stroke the mirror can physically reach; if bounding it back
inside range costs little FWHM, the excursion was ill-conditioning rather than signal.

The IQ metric here is the **achieved** residual ``dW - S (d / w)``, not the subspace
projection ``(I - U_eff U_eff^T) dW`` that ``run_bounce.py`` reports as
``fwhm_after_50_34``. The projection is blind to the recovered amplitudes — it is what
an ideal correction in the kept subspace would leave — so it cannot see a regularizer
trading wavefront for amplitude at all. The two coincide exactly for method 0, which
this script checks and reports, so the new numbers remain traceable to the committed
``bounce_fwhm_metric.parquet``.

Usage, with the lead MIW-referenced bounce fit table::

    python run_regularized_compare.py \
        --fits /sdf/home/r/roodman/notebooks/rubin-work/aos/output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet \
        --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
        --mi-name pathA_50_34_i_5rot \
        --out-dir /sdf/home/r/roodman/notebooks/rubin-work/smatrix/output/regularized_inversion/danish_1_2_A_50_34_i_5rot_july \
        --min-detectors 160

Writes ``regularized_inversion_metrics.parquet`` (one row per leg per method
setting), ``regularized_inversion_dof.parquet`` (per-DOF recovered amplitude per
method) and ``regularized_inversion_compare.pdf``.

Needs `lsst.ts.ofc` and `lsst.ts.wep` (the FWHM conversion), so it is RSP/USDF-only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import QTable

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))                 # same-study siblings
sys.path.insert(0, str(_HERE.parents[1]))             # smatrix/code (normalization_weights)
sys.path.insert(0, str(_HERE.parents[3]))             # repo root (common/)
# The bounce leg definitions and the pairing live in the aos topic; this study reuses
# them rather than restating the legs, so that the Δ scored here is byte-identical to
# the Δ in the committed bounce products.
_AOS = _HERE.parents[3] / 'aos' / 'code'
sys.path.insert(0, str(_AOS))
sys.path.insert(0, str(_AOS / 'bounce'))

import regularized_inversion as RI                                    # noqa: E402
import bounce_lib as bl                                               # noqa: E402
import run_bounce                                                     # noqa: E402
import aos_fwhm as afw                                                # noqa: E402
from lsst.ts.intrinsic.wavefront import mi_config as mc               # noqa: E402
from lsst.ts.intrinsic.wavefront.ofc_svd import (                     # noqa: E402
    build_ofc_svd, dz_table_to_W, LABELS_50DOF, DOF_UNITS_50)

# Damping lambda for method A, in units of a singular value (µm of wavefront per unit
# normalized DOF). The 50/34 spectrum runs from sigma_1 = 67.7 down to
# sigma_34 = 0.0563, so this grid brackets the retained tail.
DEFAULT_LAMBDAS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 1.0]
# Knee position for method B, dimensionless, as a fraction of the allowed range r_j.
# The grid runs well past 1 because the penalty only has to make the *largest*
# recovered amplitude land near its range, and a knee at the range itself
# over-shrinks the other 49 DOF: measured on the elevation 40 deg leg, kappa = 1
# gives a largest |d_j|/r_j of 0.208, far inside the feasible set and at needless
# FWHM cost, whereas kappa = 8 lands at 0.990.
DEFAULT_KAPPAS = [1.0, 2.0, 4.0, 8.0, 16.0]


def leg_median_dz(fit_table, bounces, prefix, k_list, iZs, svd, min_visits):
    """Median paired-difference DZ wavefront per bounce leg.

    Parameters
    ----------
    fit_table : `astropy.table.Table` or `pandas.DataFrame`
        The DZ fit table, one row per visit.
    bounces : `list`
        Bounce specifications, as `run_bounce.py` resolves them from
        ``analysis_config.yaml``.
    prefix : `str`
        DZ column prefix, e.g. ``'z1toz6'``.
    k_list, iZs : `list`
        Focal orders and pupil Noll indices defining the DZ grid.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        Supplies ``kj_grid``, so the packed vector matches the design matrix.
    min_visits : `int`
        Per-leg minimum visit count, passed through to the pairing.

    Returns
    -------
    legs : `list` of `dict`
        One entry per populated leg, with keys ``bounce`` (`str`), ``leg`` (`str`),
        ``n_pairs`` (`int`), ``med_dW`` (`numpy.ndarray`, µm of wavefront over
        ``svd.kj_grid``) and ``cam_only`` (`bool`).
    """
    W = dz_table_to_W(fit_table, prefix, svd.kj_grid)
    legs = []
    for b in bounces:
        combined = bl.run_bounce(fit_table, b, prefix, k_list, iZs, day_obs=None)
        for comp in b['comparisons']:
            label = comp['label']
            pairs = combined['comparisons'][label]['pairs']
            if not len(pairs):
                continue
            pd_dz = bl.paired_deltas_matrix(W, pairs)
            med = np.array([pd_dz[i]['delta'] for i in range(W.shape[1])], float)
            legs.append(dict(bounce=b['name'], leg=label, n_pairs=len(pairs),
                             med_dW=med,
                             cam_only=bool(b.get('camera_hexapod_only', False))))
    return legs


def score(dW, d, svd, r_j, grid, conv, iZs):
    """Image-quality and feasibility scores for one recovered DOF vector.

    Parameters
    ----------
    dW : `numpy.ndarray`
        Measured DZ wavefront, µm of wavefront, over ``svd.kj_grid``.
    d : `numpy.ndarray`
        Recovered physical DOF, µm and arcsec per ``DOF_UNITS_50``.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        The sensitivity-matrix SVD.
    r_j : `numpy.ndarray`
        Allowed range per DOF, in each DOF's own unit.
    grid : `numpy.ndarray`
        Field points in deg, from ``aos_fwhm.fp_grid``.
    conv : `callable`
        ``lsst.ts.wep.utils.convertZernikesToPsfWidth``.
    iZs : `list`
        Pupil Noll indices.

    Returns
    -------
    out : `dict`
        ``fwhm_achieved`` (`float`, arcsec FWHM left by this correction),
        ``res_rms_um`` (`float`, µm of wavefront RMS of the achieved residual),
        ``max_ratio`` (`float`, largest ``|d_j| / r_j``, dimensionless),
        ``n_over_range`` (`int`, DOF with ``|d_j| > r_j``),
        ``bend_l2_um`` (`float`, L2 norm of the 40 bending-mode amplitudes, µm of
        mode amplitude).
    """
    res = RI.achieved_residual(dW, d, svd)
    ratio = np.abs(d) / r_j
    idx = list(svd.dof_idx) if svd.dof_idx else list(range(len(d)))
    bend = np.array([d[i] for i, g in enumerate(idx) if g >= 10], float)
    return dict(
        fwhm_achieved=afw.fp_fwhm(svd, iZs, res, grid, conv),
        res_rms_um=float(np.sqrt(np.mean(res ** 2))),
        max_ratio=float(np.max(ratio)),
        n_over_range=int(np.sum(ratio > 1.0)),
        bend_l2_um=float(np.linalg.norm(bend)) if len(bend) else 0.0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fits', required=True, help='DZ fit table (parquet)')
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--min-detectors', type=int, default=None,
                    help='Relax the CCD-count quality cut to this many detectors '
                         'with enough donuts; the blur cut is kept regardless.')
    ap.add_argument('--lambdas', type=float, nargs='*', default=DEFAULT_LAMBDAS,
                    help='Method A damping values, in singular-value units '
                         '(µm of wavefront per unit normalized DOF).')
    ap.add_argument('--kappas', type=float, nargs='*', default=DEFAULT_KAPPAS,
                    help='Method B knee positions, dimensionless fraction of the '
                         'allowed range r_j.')
    ap.add_argument('--powers', type=int, nargs='*', default=[2, 3],
                    help='Method B penalty exponents p; the penalty grows as '
                         '|d_j|^(2p), so p=2 is quartic. Each is crossed with every '
                         '--kappas value.')
    ap.add_argument('--full-rank', action='store_true',
                    help='Let method A use the full rank instead of the 34 retained '
                         'modes, so damping replaces truncation entirely.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from lsst.ts.wep.utils import convertZernikesToPsfWidth as conv
    # The leg definitions come from the aos topic's bounce config, merged onto
    # run_bounce.py's own defaults exactly as that script does -- analysis_section
    # returns only the overrides, so the defaults carry the unoverridden legs.
    cfg_path = (Path(args.analysis_config) if args.analysis_config
                else _HERE.parents[3] / 'aos' / 'analysis_config.yaml')
    cfg = {**run_bounce.DEFAULT, **mc.analysis_section(
        'bounce', args.param_set, args.mi_name, config_path=cfg_path)}
    prefix = cfg.get('fit_prefix', 'z1toz6')
    k_list = list(cfg.get('focal_k_range', [1, 2, 3, 4, 5, 6]))
    bounces = cfg['bounces']
    n_dof, n_keep = int(cfg.get('n_dof', 50)), int(cfg.get('n_keep', 34))

    # Load and cut exactly as run_bounce.py does, so the leg selection and therefore
    # the median Δ scored here match the committed bounce products.
    fit_table = QTable.read(str(args.fits))
    for bf in (f'{prefix}_bad_fit', 'bad_fit'):
        if bf in fit_table.colnames:
            fit_table = fit_table[~np.asarray(fit_table[bf]).astype(bool)]
            break
    if args.min_detectors is not None and \
            'n_detectors_with_min_donuts' in fit_table.colnames:
        from lsst.ts.intrinsic.wavefront.intrinsics_lib import quality_visit_mask
        fit_table = fit_table[np.asarray(quality_visit_mask(
            fit_table, min_detectors_per_visit=args.min_detectors, verbose=False),
            dtype=bool)]
    elif 'visit_quality_pass' in fit_table.colnames:
        fit_table = fit_table[np.asarray(fit_table['visit_quality_pass'], dtype=bool)]
    if cfg.get('pupil_j_range') is None:
        iZs = [int(j) for j in np.asarray(fit_table['nollIndices'][0]).tolist()]
    else:
        iZs = [int(j) for j in cfg['pupil_j_range']]
    print(f'[regularized-compare] {args.fits}: {len(fit_table)} visits after cuts')

    svd = build_ofc_svd(iZs, min(k_list), max(k_list), n_keep, n_dof=n_dof,
                        ofc_normalization_yaml=cfg.get('ofc_normalization_yaml'))
    r_j = RI.dof_range_vector(svd)
    grid = afw.fp_grid()
    full_rank = int(svd.U_eff.shape[1])
    sig = np.asarray(svd.Sigma, float)
    print(f'  SVD: {len(svd.kj_grid)} DZ terms, {svd.n_dof} DOF, '
          f'{svd.n_keep_eff} modes kept; sigma_1 = {sig[0]:.4f}, '
          f'sigma_{svd.n_keep_eff} = {sig[svd.n_keep_eff - 1]:.5f} '
          f'(µm of wavefront per unit normalized DOF)')

    legs = leg_median_dz(fit_table, bounces, prefix, k_list, iZs, svd,
                         int(cfg.get('night_min_visits', 3)))
    print(f'  {len(legs)} populated legs')

    labels = [LABELS_50DOF[i] for i in (svd.dof_idx or range(50))]
    units = [DOF_UNITS_50[i] for i in (svd.dof_idx or range(50))]
    rows, dof_rows = [], []
    for leg in legs:
        dW = leg['med_dW']
        fwhm_before = afw.fp_fwhm(svd, iZs, dW, grid, conv)
        # The projection metric run_bounce.py reports, recomputed here so the two
        # tables can be compared directly.
        fwhm_proj = afw.fp_fwhm(svd, iZs, afw.residual_dW(svd, dW), grid, conv)

        settings = [('truncated_50_34', dict())]
        settings += [(f'damped_lam{lam:g}', dict(lam=lam)) for lam in args.lambdas]
        settings += [(f'range_p{p}_kappa{kap:g}', dict(kappa=kap, power=p))
                     for p in args.powers for kap in args.kappas]

        for name, kw in settings:
            info = {}
            if name.startswith('truncated'):
                d = RI.invert_truncated(dW, svd)
            elif name.startswith('damped'):
                d = RI.invert_damped(dW, svd, kw['lam'],
                                     rank=(full_rank if args.full_rank else None))
            else:
                d, info = RI.invert_range_penalty(
                    dW, svd, r_j, kappa=kw['kappa'], power=kw['power'],
                    return_info=True)
            sc = score(dW, d, svd, r_j, grid, conv, iZs)
            row = dict(bounce=leg['bounce'], leg=leg['leg'],
                       n_pairs=leg['n_pairs'], method=name,
                       fwhm_before_arcsec=fwhm_before,
                       fwhm_projection_arcsec=fwhm_proj,
                       fwhm_achieved_arcsec=sc['fwhm_achieved'],
                       residual_rms_um=sc['res_rms_um'],
                       max_dof_over_range=sc['max_ratio'],
                       n_dof_over_range=sc['n_over_range'],
                       bending_l2_um=sc['bend_l2_um'],
                       lam=kw.get('lam', np.nan),
                       kappa=kw.get('kappa', np.nan),
                       power=kw.get('power', np.nan),
                       irls_iter=info.get('n_iter', np.nan),
                       irls_converged=info.get('converged', True))
            rows.append(row)
            for i, (lab, un) in enumerate(zip(labels, units)):
                dof_rows.append(dict(
                    bounce=leg['bounce'], leg=leg['leg'], method=name,
                    dof_index=(svd.dof_idx or list(range(50)))[i], label=lab,
                    unit=un, value=float(d[i]), range=float(r_j[i]),
                    ratio=float(abs(d[i]) / r_j[i])))
        # The truncated method's achieved residual must equal the projection the
        # committed bounce table reports; anything else means the design matrix and
        # the recovery have drifted apart.
        t = [r for r in rows if r['leg'] == leg['leg']
             and r['bounce'] == leg['bounce'] and r['method'] == 'truncated_50_34'][0]
        dchk = abs(t['fwhm_achieved_arcsec'] - fwhm_proj)
        print(f"  {leg['bounce']}/{leg['leg']}: {leg['n_pairs']} pairs, "
              f"FWHM before = {fwhm_before:.4f} arcsec, "
              f"truncated achieved = {t['fwhm_achieved_arcsec']:.4f} arcsec "
              f"(projection {fwhm_proj:.4f} arcsec, "
              f"|difference| = {dchk:.2e} arcsec)")

    df = pd.DataFrame(rows)
    ddf = pd.DataFrame(dof_rows)

    # ---- the headline: cheapest feasible solution per leg and family ----
    # "Feasible" means every recovered DOF sits inside its allowed range. Among the
    # feasible settings the one with the lowest achieved FWHM is the best that family
    # can do, and its excess over the truncated FWHM is what insisting on a
    # physically reachable solution costs in image quality.
    print('\n  Cheapest feasible setting per leg (all |d_j| <= r_j):')
    print(f"    {'leg':<10} {'family':<7} {'setting':<18} {'FWHM [arcsec]':>13} "
          f"{'cost [arcsec]':>13} {'max |d_j|/r_j':>13}")
    best = []
    for bname, lname in dict.fromkeys(zip(df['bounce'], df['leg'])):
        sub = df[(df['bounce'] == bname) & (df['leg'] == lname)]
        f_trunc = float(sub[sub['method'] == 'truncated_50_34']
                        ['fwhm_achieved_arcsec'].iloc[0])
        for fam, pref in (('damped', 'damped'), ('range', 'range')):
            feas = sub[(sub['method'].str.startswith(pref))
                       & (sub['n_dof_over_range'] == 0)]
            if not len(feas):
                print(f'    {lname:<10} {fam:<7} {"none feasible":<18}')
                continue
            row = feas.loc[feas['fwhm_achieved_arcsec'].idxmin()]
            cost = float(row['fwhm_achieved_arcsec']) - f_trunc
            print(f"    {lname:<10} {fam:<7} {row['method']:<18} "
                  f"{row['fwhm_achieved_arcsec']:>13.5f} {cost:>13.5f} "
                  f"{row['max_dof_over_range']:>13.4f}")
            best.append(dict(bounce=bname, leg=lname, family=fam,
                             method=row['method'], n_pairs=int(row['n_pairs']),
                             fwhm_before_arcsec=float(row['fwhm_before_arcsec']),
                             fwhm_truncated_arcsec=f_trunc,
                             fwhm_feasible_arcsec=float(row['fwhm_achieved_arcsec']),
                             fwhm_cost_arcsec=cost,
                             max_dof_over_range=float(row['max_dof_over_range']),
                             truncated_max_over_range=float(
                                 sub[sub['method'] == 'truncated_50_34']
                                 ['max_dof_over_range'].iloc[0])))
    bdf = pd.DataFrame(best)
    bdf.to_parquet(out_dir / 'regularized_inversion_best_feasible.parquet')
    print()
    df.to_parquet(out_dir / 'regularized_inversion_metrics.parquet')
    ddf.to_parquet(out_dir / 'regularized_inversion_dof.parquet')
    print(f'  wrote regularized_inversion_metrics.parquet ({len(df)} rows) '
          f'and regularized_inversion_dof.parquet ({len(ddf)} rows)')

    _plot(df, ddf, out_dir, svd, r_j, labels, bdf)
    print('  wrote regularized_inversion_compare.pdf')


def _plot(df, ddf, out_dir, svd, r_j, labels, bdf=None):
    """Write the comparison PDF: IQ-vs-feasibility trade-off and per-DOF amplitudes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    legs = list(dict.fromkeys(zip(df['bounce'], df['leg'])))
    with PdfPages(str(out_dir / 'regularized_inversion_compare.pdf')) as pdf:
        # ---- Page 1: the trade-off curve, achieved FWHM against feasibility ----
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=150)
        ax = axes[0]
        for bname, lname in legs:
            sub = df[(df['bounce'] == bname) & (df['leg'] == lname)]
            for meth, mk in (('damped', 'o-'), ('range', 's--')):
                s = sub[sub['method'].str.startswith(meth)].sort_values(
                    'max_dof_over_range')
                if len(s):
                    ax.plot(s['max_dof_over_range'], s['fwhm_achieved_arcsec'], mk,
                            ms=4, lw=1.0, label=f'{lname} {meth}')
            t = sub[sub['method'] == 'truncated_50_34']
            if len(t):
                ax.plot(t['max_dof_over_range'], t['fwhm_achieved_arcsec'], 'k*',
                        ms=13)
        ax.axvline(1.0, color='red', lw=1.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'largest recovered $|d_j| / r_j$  (dimensionless);'
                      ' red line is the allowed range')
        ax.set_ylabel('achieved correctable FWHM [arcsec]')
        ax.set_title('Image quality against feasibility, per bounce leg\n'
                     'black stars: the current truncated 50/34 recovery')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5, ncol=4, loc='upper right')

        # Only three bars per leg: the current recovery and the cheapest feasible
        # setting of each family. Drawing every swept setting makes the panel
        # unreadable and says nothing the trade-off curve above does not.
        ax = axes[1]
        x = np.arange(len(legs))
        series = [('truncated_50_34', 'the current truncated 50/34', 'tab:gray')]
        series += [('damped', 'cheapest feasible damped (A)', 'tab:blue'),
                   ('range', 'cheapest feasible range penalty (B)', 'tab:red')]
        wdt = 0.8 / len(series)
        for mi, (key, lab, col) in enumerate(series):
            ys, ann = [], []
            for b, l in legs:
                if key == 'truncated_50_34':
                    s = df[(df['bounce'] == b) & (df['leg'] == l)
                           & (df['method'] == key)]
                    ys.append(float(s['fwhm_achieved_arcsec'].iloc[0])
                              if len(s) else np.nan)
                    ann.append('')
                    continue
                s = bdf[(bdf['bounce'] == b) & (bdf['leg'] == l)
                        & (bdf['family'] == key)] if bdf is not None else []
                ys.append(float(s['fwhm_feasible_arcsec'].iloc[0])
                          if len(s) else np.nan)
                ann.append(s['method'].iloc[0].replace(f'{key}_', '')
                           if len(s) else '')
            bars = ax.bar(x + mi * wdt, ys, wdt, label=lab, color=col)
            for bar, txt in zip(bars, ann):
                if txt:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), txt, rotation=90, fontsize=5,
                            ha='center', va='bottom')
        befores = [float(df[(df['bounce'] == b) & (df['leg'] == l)]
                         ['fwhm_before_arcsec'].iloc[0]) for b, l in legs]
        ax.plot(x + 0.4, befores, 'k_', ms=26, label='no correction')
        ax.set_xticks(x + 0.4)
        ax.set_xticklabels([l for _, l in legs], rotation=30, ha='right')
        ax.set_ylabel('achieved correctable FWHM [arcsec]')
        ax.set_title('Achieved FWHM: the current recovery against the cheapest '
                     'solution of each family that stays inside the allowed range')
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2+: per-DOF amplitudes against the range, per leg ----
        for bname, lname in legs:
            sub = ddf[(ddf['bounce'] == bname) & (ddf['leg'] == lname)]
            # Only the truncated solution and the cheapest feasible setting of each
            # family; drawing every setting makes the 50-DOF trace unreadable.
            keep = ['truncated_50_34']
            if bdf is not None and len(bdf):
                keep += list(bdf[(bdf['bounce'] == bname)
                                 & (bdf['leg'] == lname)]['method'])
            fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)
            xs = np.arange(len(labels))
            for meth in keep:
                s = sub[sub['method'] == meth].sort_values('dof_index')
                ax.plot(xs, s['value'].to_numpy(), '.-', ms=4, lw=0.9, label=meth)
            ax.plot(xs, r_j, 'r-', lw=1.5, label=r'$+r_j$ (allowed range)')
            ax.plot(xs, -r_j, 'r-', lw=1.5)
            ax.fill_between(xs, -r_j, r_j, color='red', alpha=0.10)
            ax.set_yscale('symlog', linthresh=1e-4)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
            ax.set_ylabel('recovered DOF  [µm or arcsec per DOF]')
            ax.set_title(f'{bname} / {lname}: recovered DOF against the allowed '
                         'range\n(µm for translations and bending amplitudes, '
                         'arcsec for hexapod rotations)')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, ncol=3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == '__main__':
    main()
