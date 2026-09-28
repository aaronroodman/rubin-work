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

Key arguments: `--kappa` and `--power` set the RBR penalty shape (defaults 4.0 and 3, the
study's candidate operational setting); `--rhos` sets the OIC `motion_penalty` values
scanned.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # smatrix/code
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))       # this study
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]
                       / 'aos' / 'code' / 'bounce'))

import regularized_inversion as ri   # noqa: E402


def oic_authority():
    """The OIC controller's authority vector, reproduced from ts_ofc.

    Mirrors `OICController.authority`: a reciprocal rigid-body stroke normalized to
    `M2_dz`, then the actuator-force standard deviation per bending mode scaled by the
    per-mirror actuator penalty.

    Returns
    -------
    authority : `numpy.ndarray`
        `(50,)` authority `a_j`, in inverse DOF units (per µm for translations and
        bending amplitudes, per arcsec for hexapod rotations). `H = diag(a**2)`.
    parts : `dict`
        The pieces needed for the range comparison: `rb_stroke` (µm and arcsec),
        `m1m3_force_per_um` and `m2_force_per_um` (N per µm, actuators by mode), and the
        two actuator penalties (dimensionless).
    """
    from lsst.ts.ofc import BendModeToForce, OFCData

    data = OFCData('lsst')
    m1m3 = BendModeToForce('M1M3', data)
    m2 = BendModeToForce('M2', data)
    rbs = data.rb_stroke[0] / data.rb_stroke
    a1 = data.m1m3_actuator_penalty * np.std(m1m3.rot_mat, axis=0)
    a2 = data.m2_actuator_penalty * np.std(m2.rot_mat, axis=0)
    parts = {'rb_stroke': np.asarray(data.rb_stroke, float),
             'm1m3_force_per_um': np.asarray(m1m3.rot_mat, float),
             'm2_force_per_um': np.asarray(m2.rot_mat, float),
             'm1m3_actuator_penalty': float(data.m1m3_actuator_penalty),
             'm2_actuator_penalty': float(data.m2_actuator_penalty),
             'motion_penalty': float(data.motion_penalty)}
    return np.concatenate((rbs, a1, a2)), parts


def invert_oic(dW, svd, authority, rho, rank=None):
    """Quadratic OIC-style penalized inversion, in the retained-mode subspace.

    Solves `min ||dW - S x||^2 + rho^2 * d^T diag(authority^2) d` with `d = w x`,
    restricted to `x = V_r b` exactly as `invert_range_penalty` does, so that the only
    difference from RBR is the penalty itself.

    Parameters
    ----------
    dW : `numpy.ndarray`
        Measured Double Zernike wavefront, µm of wavefront, over `svd.kj_grid`.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        The sensitivity-matrix SVD.
    authority : `numpy.ndarray`
        `(n_dof,)` OIC authority `a_j`, in inverse DOF units.
    rho : `float`
        The OIC `motion_penalty`, dimensionless.
    rank : `int`, optional
        Modes retained; default `svd.U_eff.shape[1]`, i.e. the operational 34.

    Returns
    -------
    d : `numpy.ndarray`
        `(n_dof,)` recovered DOF, µm and arcsec per `DOF_UNITS_50`.
    """
    w = np.asarray(svd.normalization_weights, float)
    n_keep = int(svd.U_eff.shape[1])
    r = n_keep if rank is None else int(rank)
    U = np.asarray(svd.U_eff, float)[:, :r]
    sig = np.asarray(svd.Sigma, float)[:r]
    V_r = np.asarray(svd.V, float)[:, :r]
    WV = w[:, None] * V_r
    a2 = np.asarray(authority, float) ** 2
    mat = np.diag(sig ** 2) + rho ** 2 * (WV.T @ (a2[:, None] * WV))
    return WV @ np.linalg.solve(mat, sig * (U.T @ np.nan_to_num(dW)))


def achieved_residual(dW, d, svd, rank=None):
    """RMS of the achieved residual `dW - S d`, in µm of wavefront.

    Scores what the recovered state actually corrects, rather than the subspace
    projection, which is blind to the recovered amplitudes.
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
                    default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    help='OIC motion_penalty values to scan, dimensionless')
    args = ap.parse_args()

    from astropy.table import Table
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
    tab = Table.read(args.fits)
    keep = np.asarray(tab['n_detectors_with_min_donuts'], float) >= args.min_detectors
    tab = tab[keep]
    _, _, _, W = project_dz_table(tab, 'z1toz6', svd)

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
                rs = np.median([achieved_residual(d, s, svd)
                                for d, s in zip(dWs, sols)])
                print(f'   {name:18s} max|d|/r = {mr:8.3f}   resid = {rs:.4f} um')
            for rho in args.rhos:
                sols = [invert_oic(d, svd, authority, rho) for d in dWs]
                mr = np.median([np.max(np.abs(s) / ranges) for s in sols])
                rs = np.median([achieved_residual(d, s, svd)
                                for d, s in zip(dWs, sols)])
                print(f'   OIC rho={rho:<10g} max|d|/r = {mr:8.3f}   '
                      f'resid = {rs:.4f} um')
            print()


if __name__ == '__main__':
    main()
