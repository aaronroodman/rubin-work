"""Compare the ts_ofc OIC controller's motion penalty against Range-Bounded Recovery.

The Optimal Integral Controller in `lsst.ts.ofc.controllers.oic_controller` already
carries a per-degree-of-freedom (DOF) penalty, built as
`motion_penalty**2 * diag(authority**2)`. This script asks whether that penalty is
functionally equivalent to the superlinear range penalty of the `regularized_inversion`
study (method B, "Range-Bounded Recovery" / RBR), and quantifies the difference on the
full-array-mode (FAM) bounce legs.

Three comparisons are made:

1. The OIC `authority` against this study's allowed-range vector `r_j`, per DOF. The
   rigid-body block agrees exactly; the bending-mode blocks differ by a factor that
   factorizes as `PREFACTOR_block * (max_j / std_j)`.
2. The effective quadratic penalty curvature each scheme applies as a function of
   `t_j = |d_j| / r_j` (dimensionless, recovered amplitude over allowed range). The OIC's
   is flat in `t`; RBR's goes as `t**(2p-2)`.
3. Both inversions run on the same per-pair paired-difference wavefronts from every
   bounce leg, in the same retained-mode subspace, so only the penalty differs. Reports
   the achieved residual in µm of wavefront and the max `|d_j| / r_j` attained.

Invocation (needs `lsst.ts.ofc`, `lsst.ts.intrinsic.wavefront` and a FAM fit table, so
RSP/USDF-only):

    python run_oic_compare.py \
      --fits /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet \
      --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
      --mi-name pathA_50_34_i_5rot \
      --min-detectors 160

With `--rho-scan` it instead reports one compact row per bounce leg and recovery giving,
for the truncated and RBR recoveries and for each `rho`, the median over pairs of
`max_j |d_j| / r_j` (dimensionless), the achieved-residual full width at half maximum
(FWHM) in arcsec, and the **amplitude retention** of the rigid-body and bending-mode
blocks against the unregularized truncated recovery (dimensionless). That is the scan the
adopted `rho` is chosen from; it is recorded in
`smatrix/docs/studies/regularized_inversion.md`, and with `--out-dir` it is also written
as `oic_rho_scan.parquet` and `oic_rho_scan.pdf`.

The retention columns are what make the scan a validation rather than a feasibility
check: the rigid-body degrees of freedom sit far inside their ranges, so a penalty that
suppresses them twentyfold does not move `max_j |d_j| / r_j` at all.

The solvers themselves live in `smatrix/code/regularized_inversion.py`, shared with the
`aos` bounce study; this script only exercises and reports on them.

Key arguments: `--kappa` and `--power` set the RBR penalty shape (defaults 4.0 and 3, the
study's candidate operational setting); `--rhos` sets the OIC `motion_penalty` values
scanned; `--rho-scan` selects the compact per-leg table.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # smatrix/code
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))       # this study
_AOS = pathlib.Path(__file__).resolve().parents[3] / 'aos' / 'code'
sys.path.insert(0, str(_AOS))              # aos_fwhm
sys.path.insert(0, str(_AOS / 'bounce'))   # bounce_lib

import regularized_inversion as ri   # noqa: E402
from regularized_inversion import invert_oic, oic_authority   # noqa: E402


def residual_rms(dW, d, svd, rank=None):
    """RMS of the achieved residual `dW - S d`, in µm of wavefront.

    Scores what the recovered state actually corrects, rather than the subspace
    projection, which is blind to the recovered amplitudes.

    Parameters
    ----------
    dW : `numpy.ndarray`
        Measured DZ wavefront, µm of wavefront, over `svd.kj_grid`.
    d : `numpy.ndarray`
        Recovered DOF, `(n_dof,)`, µm and arcsec per `DOF_UNITS_50`.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        The sensitivity-matrix SVD.
    rank : `int`, optional
        Modes retained; default `svd.U_eff.shape[1]`.

    Returns
    -------
    rms : `float`
        L2 norm of the residual over the (k, j) grid, µm of wavefront.

    Notes
    -----
    `regularized_inversion.achieved_residual` returns the residual *vector*; this is
    the scalar norm of it, computed in the retained-mode coefficients so the two agree
    for any `d` reachable in that subspace.
    """
    w = np.asarray(svd.normalization_weights, float)
    n_keep = int(svd.U_eff.shape[1])
    r = n_keep if rank is None else int(rank)
    U = np.asarray(svd.U_eff, float)[:, :r]
    sig = np.asarray(svd.Sigma, float)[:r]
    V_r = np.asarray(svd.V, float)[:, :r]
    WV = w[:, None] * V_r
    b = np.linalg.lstsq(WV, d, rcond=None)[0]
    mis = np.nan_to_num(dW) - U @ (sig * b)
    return float(np.sqrt(mis @ mis))


def report_range_comparison(ranges, authority, parts, labels):
    """Print the per-DOF comparison of the OIC authority against `r_j`."""
    inv = 1.0 / authority
    scale = ranges[0] / inv[0]          # match on M2_dz, the OIC's own normalization
    ratio = ranges / (scale * inv)
    print('=== OIC authority as an implied allowed range, against r_j ===')
    print('  scaled so M2_dz matches; ratio 1.000 means the two schemes agree\n')
    print(f'  {"DOF":8s} {"r_j (ours)":>12s} {"r_j (OIC)":>12s} {"ratio":>9s}')
    for j in range(10):
        print(f'  {labels[j]:8s} {ranges[j]:12.4g} {scale * inv[j]:12.4g} '
              f'{ratio[j]:9.4f}')
    for name, lo, hi in [('M1M3 bending', 10, 30), ('M2 bending', 30, 50)]:
        blk = ratio[lo:hi]
        print(f'  {name}: ratio min={blk.min():.4g} max={blk.max():.4g} '
              f'median={np.median(blk):.4g}  (ours over OIC-implied)')
    print()
    # The disagreement factorizes exactly: PREFACTOR_block * (max_j / std_j).
    for name, force, pen, force_range, lo, hi in [
            ('M1M3', parts['m1m3_force_per_um'], parts['m1m3_actuator_penalty'],
             134.0, 10, 30),
            ('M2', parts['m2_force_per_um'], parts['m2_actuator_penalty'],
             45.0, 30, 50)]:
        mx = np.max(np.abs(force), axis=0)
        sd = np.std(force, axis=0)
        prefactor = parts['rb_stroke'][0] * 20.0 / (pen * force_range)
        implied = prefactor * (mx / sd)
        actual = (scale * inv[lo:hi]) / ranges[lo:hi]
        print(f'  {name}: OIC-implied / ours = {prefactor:.4g} * (max/std), '
              f'exact={np.allclose(implied, actual)}; '
              f'max/std median={np.median(mx / sd):.3f} '
              f'(range {np.min(mx / sd):.3f} to {np.max(mx / sd):.3f})')
    print()


def report_curvature(kappa, power):
    """Print the effective penalty curvature of each scheme against `t = |d_j|/r_j`."""
    print('=== Effective quadratic penalty curvature, in units of (kappa*r_j)^-2 ===')
    print(f'  RBR kappa={kappa:g} power={power:d} gives q_j*(kappa r_j)^2 = '
          f'(t/kappa)^(2p-2); the OIC is flat at 1 by construction\n')
    print(f'  {"t = |d_j|/r_j":>14s} {"RBR weight":>14s} {"OIC weight":>12s}')
    ts = [0.10, 0.25, 0.50, 1.00, 2.00, 4.00]
    for t in ts:
        print(f'  {t:14.2f} {(t / kappa) ** (2 * power - 2):14.3e} {1:12d}')
    dyn = ((ts[-1] / kappa) ** (2 * power - 2)) / ((ts[0] / kappa) ** (2 * power - 2))
    print(f'\n  RBR dynamic range over t={ts[0]:g} to {ts[-1]:g}: {dyn:.3g} '
          f'(dimensionless); the OIC is exactly 1.0\n')


#: Index groups the rho scan reports amplitude retention over, because the OIC
#: authority `a_j` spans five orders of magnitude between them: the 10 rigid-body
#: DOF sit at `a_j` of order 1 while the 40 bending modes run to 9.1e4.  A single
#: scalar rho therefore cannot be read as "the penalty strength" — it acts on the
#: two groups with wildly different force, which is the whole point of the scan.
SCAN_GROUPS = (('rigid body', list(range(0, 10))),
               ('bending', list(range(10, 50))))


def report_rho_scan(tab, W, svd, ranges, authority, cfg, k_list, pupil_j,
                    kappa, power, rhos, out_dir=None, adopted_rho=None):
    """Print the table behind the adopted OIC `rho`: feasibility, image quality *and*
    amplitude retention.

    One row per bounce leg and recovery, giving the median over the leg's pairs of
    `max_j |d_j| / r_j` (dimensionless, recovered amplitude over allowed range), the
    achieved-residual full width at half maximum (FWHM) in arcsec, and the **amplitude
    retention** of each `SCAN_GROUPS` group against the unregularized truncated
    recovery.

    Retention is the regression slope of the penalized amplitudes on the truncated ones
    over all of the leg's pairs and the group's DOF, `sum(d_pen · d_trunc) /
    sum(d_trunc²)`, dimensionless. 1.0 means the penalty left the group's amplitudes
    alone; 0.05 means it suppressed them twentyfold. It is reported per group because
    the OIC authority differs by five orders of magnitude between rigid body and
    bending modes (see `SCAN_GROUPS`), so one rho can leave the bending modes barely
    touched while crushing the rigid body, or the reverse.

    Feasibility and FWHM alone cannot detect that: the rigid-body DOF sit far inside
    their ranges, so suppressing them does not move `max |d_j| / r_j`, and they are
    partly degenerate in wavefront, so it moves the FWHM much less than it moves the
    amplitudes. Choosing rho on feasibility alone selects a value that silently
    destroys the rigid-body solution, which is why retention is scanned alongside.

    Parameters
    ----------
    tab : `astropy.table.Table`
        The quality-selected FAM fit table.
    W : `numpy.ndarray`
        Per-visit DZ wavefronts, µm of wavefront, over `svd.kj_grid`.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        The 50-DOF / 34-mode sensitivity-matrix SVD.
    ranges : `numpy.ndarray`
        `(50,)` allowed range `r_j`, µm and arcsec per `DOF_UNITS_50`.
    authority : `numpy.ndarray`
        `(50,)` OIC authority, inverse DOF units, from `oic_authority`.
    cfg : `dict`
        The resolved `bounce` analysis section, supplying `bounces`.
    k_list, pupil_j : `list` of `int`
        Focal orders k and pupil Noll indices j defining the DZ grid.
    kappa : `float`
        RBR knee, dimensionless `|d_j| / r_j` at unit penalty weight.
    power : `int`
        RBR penalty exponent p, dimensionless.
    rhos : `list` of `float`
        OIC `motion_penalty` values to scan, dimensionless.
    out_dir : `str` or `pathlib.Path`, optional
        When given, also write the scan as `oic_rho_scan.parquet` (one row per
        leg and recovery) and `oic_rho_scan.pdf` (feasibility and retention
        against rho, one panel per leg) — the durable record validating the
        adopted value, rather than a table that exists only in a terminal.
    adopted_rho : `float`, optional
        The rho carried in `aos/analysis_config.yaml`, dimensionless, marked on
        the plot so the trade-off at the adopted value is visible.

    Returns
    -------
    rows : `list` [`dict`]
        One row per (leg, recovery), with `leg`, `n_pairs`, `recovery`, `rho`
        (dimensionless, NaN for the non-OIC recoveries), `ratio` (dimensionless),
        `fwhm_arcsec`, and one retention column per `SCAN_GROUPS` group
        (dimensionless).
    """
    import aos_fwhm as _afw
    import bounce_lib as _bl
    from lsst.ts.wep.utils import convertZernikesToPsfWidth as _conv

    grid = _afw.fp_grid()

    def _retention(sols, ref):
        """Regression slope of penalized on truncated amplitudes, per group."""
        out = []
        for _nm, idx in SCAN_GROUPS:
            a = np.asarray([s[idx] for s in sols], float).ravel()
            b = np.asarray([s[idx] for s in ref], float).ravel()
            den = float(b @ b)
            out.append(float(a @ b) / den if den > 0 else np.nan)
        return out

    def _score(dWs, sols, ref):
        mr = np.median([np.max(np.abs(s) / ranges) for s in sols])
        fw = np.median([_afw.fp_fwhm(svd, pupil_j,
                                     ri.achieved_residual(d, s, svd), grid, _conv)
                        for d, s in zip(dWs, sols)])
        return (mr, fw, *_retention(sols, ref))

    gnames = [nm for nm, _ in SCAN_GROUPS]
    print('=== OIC motion penalty rho: feasibility, image quality and amplitude '
          'retention ===')
    print('  ratio  = median over the leg pairs of max_j |d_j|/r_j, dimensionless '
          '(<= 1 is feasible)')
    print('  FWHM   = median achieved-residual FWHM, arcsec')
    for nm, idx in SCAN_GROUPS:
        print(f'  {nm:10s} = amplitude retention over DOF {idx[0]}-{idx[-1]}, '
              f'dimensionless: regression slope of the penalized amplitudes on the')
        print('               unregularized truncated ones (1.0 = untouched, '
              '0.05 = suppressed 20x)')
    print('  The OIC authority a_j spans 0.68 to 9.1e4 in inverse DOF units, so one '
          'rho acts')
    print('  on the two groups with vastly different strength — read both retention '
          'columns.\n')
    head = (f'  {"leg":16s} {"n":>3s} {"recovery":>22s} {"ratio":>8s} {"FWHM":>7s}'
            + ''.join(f' {nm:>11s}' for nm in gnames))
    print(head)
    out_rows = []
    for bounce in cfg['bounces']:
        res = _bl.run_bounce(tab, bounce, 'z1toz6', k_list, pupil_j)
        for label, comp in res['comparisons'].items():
            pairs = comp['pairs']
            if not pairs:
                continue
            dWs = [np.nan_to_num(W[c] - W[r]) for (r, c) in pairs]
            trunc = [ri.invert_truncated(d, svd) for d in dWs]
            recs = [('truncated', np.nan, trunc),
                    (f'RBR k={kappa:g} p={power:d}', np.nan,
                     [ri.invert_range_penalty(d, svd, ranges, kappa=kappa,
                                              power=power) for d in dWs])]
            recs += [(f'OIC rho={rho:.2e}', rho,
                      [invert_oic(d, svd, authority, rho) for d in dWs])
                     for rho in rhos]
            for i, (tag, rho, sols) in enumerate(recs):
                mr, fw, *ret = _score(dWs, sols, trunc)
                lead = (f'  {label:16s} {len(pairs):3d}' if i == 0
                        else f'  {"":16s} {"":3s}')
                print(f'{lead} {tag:>22s} {mr:8.3f} {fw:7.4f}'
                      + ''.join(f' {v:11.4f}' for v in ret))
                row = {'bounce': bounce['name'], 'leg': label,
                       'n_pairs': len(pairs), 'recovery': tag, 'rho': rho,
                       'ratio': mr, 'fwhm_arcsec': fw}
                row.update({f'retention_{nm.replace(" ", "_")}': v
                            for nm, v in zip(gnames, ret)})
                out_rows.append(row)
    print('\n  The adopted value is recorded in '
          'smatrix/docs/studies/regularized_inversion.md and consumed by '
          'aos/analysis_config.yaml as bounce: oic_rho.\n')
    if out_dir is not None:
        _write_rho_scan(out_rows, gnames, pathlib.Path(out_dir), adopted_rho)
    return out_rows


def _write_rho_scan(rows, gnames, out_dir, adopted_rho):
    """Write the rho scan as a parquet table and a per-leg plot.

    Parameters
    ----------
    rows : `list` [`dict`]
        The scan rows from `report_rho_scan`.
    gnames : `list` [`str`]
        `SCAN_GROUPS` names, naming the retention columns.
    out_dir : `pathlib.Path`
        Directory to write `oic_rho_scan.parquet` and `oic_rho_scan.pdf` into.
    adopted_rho : `float` or `None`
        Adopted `rho`, dimensionless, marked with a vertical line when given.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / 'oic_rho_scan.parquet')

    rcols = [f'retention_{nm.replace(" ", "_")}' for nm in gnames]
    legs = list(dict.fromkeys(df['leg']))
    oic = df[np.isfinite(df['rho'])]
    with PdfPages(str(out_dir / 'oic_rho_scan.pdf')) as pdf:
        ncols = 3
        nrows = int(np.ceil(len(legs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, layout='constrained',
                                 figsize=(4.6 * ncols, 3.6 * nrows),
                                 squeeze=False)
        for i, leg in enumerate(legs):
            ax = axes[i // ncols][i % ncols]
            g = oic[oic['leg'] == leg].sort_values('rho')
            t = df[(df['leg'] == leg) & (df['recovery'] == 'truncated')]
            ax.plot(g['rho'], g['ratio'], 'o-', color='#1f77b4',
                    label='max_j |d_j|/r_j [dimensionless]')
            for rc, col, mk in zip(rcols, ('#d62728', '#2ca02c'), ('s', '^')):
                ax.plot(g['rho'], g[rc], mk + '-', color=col,
                        label=f'{rc.replace("retention_", "").replace("_", " ")} '
                              f'retention [dimensionless]')
            ax.axhline(1.0, color='gray', lw=0.6, ls=':')
            if len(t):
                ax.axhline(float(t['ratio'].iloc[0]), color='#1f77b4', lw=0.6,
                           ls='--', alpha=0.6)
            if adopted_rho is not None:
                ax.axvline(adopted_rho, color='black', lw=1.0, ls='-.',
                           alpha=0.7)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_ylim(1e-4, 50.0)
            ax.set_xlabel('OIC rho [dimensionless]', fontsize=9)
            ax.set_title(f'{leg}  (n_pairs = {int(g["n_pairs"].iloc[0])})',
                         fontsize=10)
            ax.grid(alpha=0.3, which='both')
            ax.tick_params(labelsize=8)
        for i in range(len(legs), nrows * ncols):
            axes[i // ncols][i % ncols].axis('off')
        h, l = axes[0][0].get_legend_handles_labels()
        extra = [plt.Line2D([], [], color='gray', ls=':',
                            label='unity: feasibility limit and untouched '
                                  'retention'),
                 plt.Line2D([], [], color='#1f77b4', ls='--', alpha=0.6,
                            label='unregularized truncated ratio')]
        if adopted_rho is not None:
            extra.append(plt.Line2D([], [], color='black', ls='-.', alpha=0.7,
                                    label=f'adopted rho = {adopted_rho:g} '
                                          f'[dimensionless]'))
        fig.legend(handles=h + extra, loc='outside lower center', ncol=2,
                   fontsize=9, frameon=False)
        fig.suptitle(
            'OIC motion penalty: feasibility against amplitude retention '
            'versus rho\n'
            'Feasibility (blue) reaches unity only where rigid-body retention '
            '(red) has already collapsed,\n'
            'so no rho is both feasible and faithful — the adopted value buys '
            'feasibility by suppressing the rigid body',
            fontsize=12)
        pdf.savefig(fig); plt.close(fig)
    print(f'  wrote {out_dir / "oic_rho_scan.parquet"}')
    print(f'  wrote {out_dir / "oic_rho_scan.pdf"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fits', required=True, help='FAM DZ fit table (parquet)')
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--min-detectors', type=int, default=160,
                    help='CCDs with enough donuts per visit')
    ap.add_argument('--kappa', type=float, default=4.0,
                    help='RBR knee, dimensionless |d_j|/r_j at unit penalty weight')
    ap.add_argument('--power', type=int, default=3, help='RBR penalty exponent p')
    ap.add_argument('--rhos', type=float, nargs='+',
                    default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                    help='OIC motion_penalty values to scan, dimensionless')
    ap.add_argument('--rho-scan', action='store_true',
                    help='one compact row per leg: feasibility, FWHM and amplitude '
                         'retention against rho, the table behind the adopted rho')
    ap.add_argument('--out-dir',
                    help='with --rho-scan, also write oic_rho_scan.parquet and '
                         'oic_rho_scan.pdf here')
    ap.add_argument('--adopted-rho', type=float, default=1e-3,
                    help='rho carried in aos/analysis_config.yaml, dimensionless, '
                         'marked on the scan plot')
    args = ap.parse_args()

    from astropy.table import Table
    from lsst.ts.intrinsic.wavefront.intrinsics_lib import quality_visit_mask
    from lsst.ts.intrinsic.wavefront.mi_config import analysis_section
    from lsst.ts.intrinsic.wavefront.ofc_svd import (LABELS_50DOF, build_ofc_svd,
                                                     project_dz_table)
    import bounce_lib as bl

    pupil_j = list(range(4, 20)) + list(range(22, 27))
    k_list = list(range(1, 7))
    svd = build_ofc_svd(pupil_j, min(k_list), max(k_list), 34, n_dof=50)
    ranges = ri.dof_range_vector(svd)
    authority, parts = oic_authority()

    print(f'OIC shipped motion_penalty rho = {parts["motion_penalty"]:g} '
          f'(dimensionless); the penalty is inactive at this value\n')
    report_range_comparison(ranges, authority, parts, LABELS_50DOF)
    report_curvature(args.kappa, args.power)

    # The bounce legs are defined in aos/analysis_config.yaml; pass its path
    # explicitly so this runs from any working directory.
    cfg = analysis_section('bounce', args.param_set, args.mi_name,
                           config_path=str(pathlib.Path(__file__).resolve().parents[3]
                                           / 'aos' / 'analysis_config.yaml'))
    if not cfg.get('bounces'):
        raise RuntimeError(
            f'no bounce legs resolved for param_set={args.param_set!r} '
            f'mi_name={args.mi_name!r} in aos/analysis_config.yaml')
    # Exactly the selection run_bounce.py applies, so the pair counts here match the
    # bounce study's. A bare n_detectors_with_min_donuts threshold does not: the blur
    # cut inside quality_visit_mask removes visits the donut-count floor keeps.
    tab = Table.read(args.fits)
    if 'z1toz6_bad_fit' in tab.colnames:
        tab = tab[~np.asarray(tab['z1toz6_bad_fit']).astype(bool)]
    tab = tab[np.asarray(quality_visit_mask(
        tab, min_detectors_per_visit=args.min_detectors, verbose=False), bool)]
    _, _, _, W = project_dz_table(tab, 'z1toz6', svd)

    if args.rho_scan:
        report_rho_scan(tab, W, svd, ranges, authority, cfg, k_list, pupil_j,
                        args.kappa, args.power, args.rhos,
                        out_dir=args.out_dir, adopted_rho=args.adopted_rho)
        return 0

    print('=== Both inversions on the same per-pair bounce Delta wavefronts ===')
    print('  residual is the achieved ||dW - S d||, um of wavefront, median over pairs')
    print('  max ratio is max_j |d_j| / r_j, dimensionless\n')
    for bounce in cfg['bounces']:
        res = bl.run_bounce(tab, bounce, 'z1toz6', k_list, pupil_j)
        for label, comp in res['comparisons'].items():
            pairs = comp['pairs']
            if not pairs:
                continue
            dWs = [np.nan_to_num(W[c] - W[r]) for (r, c) in pairs]
            trunc = [ri.invert_truncated(d, svd) for d in dWs]
            rbr = [ri.invert_range_penalty(d, svd, ranges, kappa=args.kappa,
                                           power=args.power) for d in dWs]
            print(f'{bounce["name"]:16s} {label:9s} n_pairs={len(pairs):3d}')
            for name, sols in [('truncated', trunc),
                               (f'RBR k={args.kappa:g} p={args.power:d}', rbr)]:
                mr = np.median([np.max(np.abs(s) / ranges) for s in sols])
                rs = np.median([residual_rms(d, s, svd)
                                for d, s in zip(dWs, sols)])
                print(f'   {name:18s} max|d|/r = {mr:8.3f}   resid = {rs:.4f} um')
            for rho in args.rhos:
                sols = [invert_oic(d, svd, authority, rho) for d in dWs]
                mr = np.median([np.max(np.abs(s) / ranges) for s in sols])
                rs = np.median([residual_rms(d, s, svd)
                                for d, s in zip(dWs, sols)])
                print(f'   OIC rho={rho:<10g} max|d|/r = {mr:8.3f}   '
                      f'resid = {rs:.4f} um')
            print()


if __name__ == '__main__':
    main()
