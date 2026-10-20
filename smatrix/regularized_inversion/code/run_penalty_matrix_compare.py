"""Compare RBR against a conventional quadratic penalty matrix in the SVD inversion.

RSO-217 section 3.5 shows that truncating the normalized SVD at k modes is the
penalized solve ``min ||y - A d||^2 + rho d^T H d`` with ``Gamma = N^T H N =
V diag(0, ..., 0, inf, ..., inf) V^T``: zero penalty on the kept v-modes, infinite on
the discarded ones. That ``H`` is dense in the DOF basis. This script asks two things:

1. Why does RBR on the 50/34 subspace hold the hexapods back, and does 50/50 fix it?
2. Is 50/50 + RBR the only way to get a feasible state and a good correction, or does a
   conventional full-rank diagonal penalty ``H = diag(1 / r_j^2)`` do as well?

Solvers, all in normalized DOF ``x = d / w`` with ``B = A diag(w)``:

``T34`` / ``T50``
    truncated SVD at 34 / 50 modes.
``R34`` / ``R50``
    RBR (``invert_range_penalty``, ``kappa = 4``, ``p = 3``) restricted to the first
    34 / 50 v-modes.
``Q50(rho)``
    full-rank quadratic range penalty, ``||y - B x||^2 + rho^2 sum_j (d_j / r_j)^2``.
``Q34x(rho)``
    the same penalty on the rank-34 operator but solved over all 50 DOF: the truncated
    wavefront model, with the 16 discarded directions used only to re-partition
    amplitude at minimum range cost.

Everything is scored with the full-rank operator, which is the physical prediction.

Inputs are local: the ts_ofc sensitivity matrix and range-fwhm weights from a
``ts_ofc`` checkout, the ranges from ``smatrix/output/normalization_50dof.npz``, and the
two cached bounce legs in ``bounce_kj_stats.parquet``. numpy only, no LSST stack.

    python run_penalty_matrix_compare.py --ts-ofc ~/Astrophysics/Claude/packages/ts_ofc
"""
import argparse
import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from rubinwork.smatrix import regularized_inversion as ri

REPO = pathlib.Path(__file__).resolve().parents[3]
BOUNCE = (REPO / 'aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/'
          'pathA_50_34_i_5rot/bounce_kj_stats.parquet')
K_LIST = list(range(1, 7))
J_LIST = list(range(4, 20)) + list(range(22, 27))
KAPPA, POWER = 4.0, 3
HEX = slice(0, 10)
BEND = slice(10, 50)


def load_inputs(ts_ofc, norm_yaml, cache):
    """Return (A, w, r, labels): A is (126, 50) µm of wavefront per DOF unit.

    ``r`` is the physical range (rigid-body stroke, force-limited bending), taken
    from ``normalization_50dof.npz`` and independent of the weights ``w``, so the
    normalization can be swapped without moving the feasibility target.
    """
    if cache.exists():
        S_full = np.load(cache)
    else:
        p = ts_ofc / 'policy/sensitivity_matrix/lsst_sensitivity_dz_31_29_50.yaml'
        S_full = np.asarray(yaml.safe_load(p.read_text()), float)
        np.save(cache, S_full)
    w = np.asarray(yaml.safe_load(norm_yaml.read_text()), float)
    nz = np.load(REPO / 'smatrix/output/normalization_50dof.npz', allow_pickle=True)
    r = nz['w_official'] ** 2 * nz['f']     # back-derived exactly as dof_range_vector
    A = S_full[K_LIST[0]:K_LIST[-1] + 1][:, J_LIST, :].reshape(-1, 50)
    return A, w, r, [str(s) for s in nz['labels']]


def load_legs():
    """Measured (k, j) deltas and their errors per bounce leg, µm of wavefront."""
    d = pd.read_parquet(BOUNCE)
    d = d[d.night == 'all']
    legs = {}
    for comp, g in d.groupby('comparison'):
        g = g.set_index(['k', 'j'])
        idx = [(k, j) for k in K_LIST for j in J_LIST]
        legs[comp] = (g.loc[idx, 'delta'].to_numpy(float),
                      g.loc[idx, 'delta_err'].to_numpy(float))
    return legs


class Solvers:
    """All inversions over one normalized design matrix ``B = A diag(w)``."""

    def __init__(self, A, w, r):
        self.A, self.w, self.r = A, w, r
        self.B = A * w[None, :]
        self.U, self.s, Vh = np.linalg.svd(self.B, full_matrices=False)
        self.V = Vh.T
        self.D2 = (w / r) ** 2               # range penalty in x: (d_j / r_j)^2

    def svd_ns(self, rank):
        return SimpleNamespace(U_eff=self.U[:, :rank], Sigma=self.s, V=self.V,
                               normalization_weights=self.w)

    def trunc(self, y, rank):
        return ri.invert_truncated(y, self.svd_ns(rank))

    def rbr(self, y, rank):
        return ri.invert_range_penalty(y, self.svd_ns(rank), self.r, kappa=KAPPA,
                                       power=POWER)

    def quad(self, y, rho, rank=50):
        Br = (self.U[:, :rank] * self.s[:rank]) @ self.V[:, :rank].T
        x = np.linalg.solve(Br.T @ Br + rho ** 2 * np.diag(self.D2), Br.T @ y)
        return self.w * x

    def resid(self, y, d):
        return float(np.linalg.norm(y - self.A @ d))

    def tmax(self, d, sl=slice(None)):
        return float(np.max(np.abs(d[sl]) / self.r[sl]))


def quad_rho_for(sol, y, target=1.0, rank=50):
    """Smallest rho (µm of wavefront per unit |d_j|/r_j) giving max |d_j|/r_j <=
    target, by bisection in log rho."""
    lo, hi = 1e-8, 1e3
    if sol.tmax(sol.quad(y, lo, rank)) <= target:
        return lo
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if sol.tmax(sol.quad(y, mid, rank)) > target:
            lo = mid
        else:
            hi = mid
    return hi


def report_structure(sol, labels):
    """Why 50/34 couples hexapods to bending: hexapod moves are not in span(V_34)."""
    print('=== Singular values of B (µm of wavefront per unit normalized DOF) ===')
    print('  ' + ' '.join(f'{v:.3g}' for v in sol.s))
    print(f'  s34/s1 = {sol.s[33] / sol.s[0]:.3g}, s50/s1 = {sol.s[49] / sol.s[0]:.3g}'
          ' (dimensionless)\n')

    # Principal angles between the 10-D hexapod coordinate subspace and span(V_34).
    E = np.zeros((50, 10))
    E[np.arange(10), np.arange(10)] = 1
    cosang = np.linalg.svd(sol.V[:, :34].T @ E, compute_uv=False)
    print('=== Principal angles, hexapod subspace vs span(V_34), normalized x ===')
    print('  ' + ' '.join(f'{np.degrees(np.arccos(min(c, 1))):.1f}' for c in cosang)
          + '  deg\n')

    # The RSO-217 equivalent H for 50/34, in physical units scaled by the range:
    # Gamma = V diag(0, L) V^T, H = N^-1 Gamma N^-1; in t = d/r units H_t = R H R.
    # Report how much of the discarded-subspace weight couples hexapod and bending.
    Vd = sol.V[:, 34:]
    P = Vd @ Vd.T                             # projector onto discarded modes in x
    hh = np.linalg.norm(P[HEX, HEX])
    bb = np.linalg.norm(P[BEND, BEND])
    hb = np.linalg.norm(P[HEX, BEND])
    print('=== Discarded-mode projector V_d V_d^T (the infinite-penalty block) ===')
    print(f'  Frobenius norm: hex-hex {hh:.3f}, bend-bend {bb:.3f}, hex-bend {hb:.3f}'
          ' (dimensionless)')
    print('  diag(P) per DOF = fraction of each normalized DOF that 50/34 forbids:')
    for j in range(50):
        print(f'    {labels[j]:9s} {P[j, j]:.3f}', end='\n' if j % 5 == 4 else '')
    print()


def synthetic_hexapod(sol, labels, frac=0.05):
    """Pure hexapod truth at ``frac`` of range, one DOF at a time: what each solver
    recovers for that hexapod DOF, the bending range ratio it invents, and residual."""
    print(f'=== Synthetic pure-hexapod state, d_h = {frac:g} r_h, no noise ===')
    print('  retention = d_h(recovered) / d_h(true), dimensionless; '
          'bend tmax = max_bending |d_j|/r_j; resid in µm of wavefront RMS-norm\n')
    hdr = f'  {"DOF":8s} {"|y| um":>7s}'
    names = ['T34', 'R34', 'R50', 'Q50', 'Q34x']
    for n in names:
        hdr += f' | {n + " ret":>8s} {"btmax":>6s} {"resid":>7s}'
    print(hdr)
    rows = []
    for h in range(10):
        d_true = np.zeros(50)
        d_true[h] = frac * sol.r[h]
        y = sol.A @ d_true
        ds = {'T34': sol.trunc(y, 34), 'R34': sol.rbr(y, 34), 'R50': sol.rbr(y, 50),
              'Q50': sol.quad(y, RHO_REF), 'Q34x': sol.quad(y, RHO_REF, rank=34)}
        line = f'  {labels[h]:8s} {np.linalg.norm(y):7.3f}'
        row = {'dof': labels[h], 'y_norm': np.linalg.norm(y)}
        for n in names:
            d = ds[n]
            ret = d[h] / d_true[h]
            bt = sol.tmax(d, BEND)
            res = sol.resid(y, d)
            line += f' | {ret:8.3f} {bt:6.2f} {res:7.4f}'
            row.update({f'{n}_ret': ret, f'{n}_btmax': bt, f'{n}_resid': res})
        rows.append(row)
        print(line)
    print()
    return pd.DataFrame(rows)


def hexapod_amplitude_scan(sol, labels, fracs=(0.02, 0.05, 0.1, 0.2, 0.4)):
    """Pure single-DOF hexapod truth versus amplitude: where RBR on 34 modes starts
    to pull the hexapod back because its span(V_34) image carries bending."""
    print('=== Hexapod amplitude scan: retention d_h(rec)/d_h(true) [bend tmax], '
          'dimensionless ===')
    rows = []
    for h in (0, 1, 5, 6):
        for frac in fracs:
            d_true = np.zeros(50)
            d_true[h] = frac * sol.r[h]
            y = sol.A @ d_true
            row = dict(dof=labels[h], frac=frac)
            for n, d in (('T34', sol.trunc(y, 34)), ('R34', sol.rbr(y, 34)),
                         ('R50', sol.rbr(y, 50)), ('Q50', sol.quad(y, RHO_REF))):
                row[f'{n}_ret'] = d[h] / d_true[h]
                row[f'{n}_btmax'] = sol.tmax(d, BEND)
                row[f'{n}_resid'] = sol.resid(y, d)
            rows.append(row)
            print(f'  {labels[h]:7s} {frac:5.2f} r  ' + '  '.join(
                f'{n} {row[n + "_ret"]:.3f} [{row[n + "_btmax"]:.2f}] '
                f'{row[n + "_resid"]:.4f}um' for n in ('T34', 'R34', 'R50', 'Q50')))
    print()
    return pd.DataFrame(rows)


def real_legs(sol, legs, labels, n_mc=300, seed=1):
    """Both measured legs: residual, feasibility, hexapod DOF, and noise by MC."""
    rng = np.random.default_rng(seed)
    out = []
    for leg, (y, err) in legs.items():
        rho1 = quad_rho_for(sol, y, 1.0)
        rho1_34 = quad_rho_for(sol, y, 1.0, rank=34)
        meths = {
            'T34': lambda v: sol.trunc(v, 34),
            'T50': lambda v: sol.trunc(v, 50),
            'R34': lambda v: sol.rbr(v, 34),
            'R50': lambda v: sol.rbr(v, 50),
            f'Q50(rho={RHO_REF:g})': lambda v: sol.quad(v, RHO_REF),
            f'Q50(rho*={rho1:.3g})': lambda v, r_=rho1: sol.quad(v, r_),
            f'Q34x(rho*={rho1_34:.3g})': lambda v, r_=rho1_34: sol.quad(v, r_, 34),
        }
        print(f'=== Leg {leg}: |y| = {np.linalg.norm(y):.4f} µm of wavefront, '
              f'noise |err| = {np.linalg.norm(err):.4f} µm ===')
        print('  rho* = smallest Q penalty giving max |d_j|/r_j <= 1 on this leg')
        print(f'  {"method":22s} {"resid":>7s} {"tmax":>7s} {"hex tmax":>8s} '
              f'{"bend tmax":>9s} {"M2dz":>8s} {"M2dx":>8s} {"M2dy":>8s} '
              f'{"Camdx":>8s} {"Camdy":>8s} | {"sd tmax":>7s} {"sd M2dx":>8s} '
              f'{"sd Camdx":>8s}')
        noise = rng.standard_normal((n_mc, len(y))) * err[None, :]
        for name, f in meths.items():
            d = f(y)
            mc = np.array([f(y + e) for e in noise])
            tm_mc = np.max(np.abs(mc) / sol.r[None, :], axis=1)
            row = dict(leg=leg, method=name, resid=sol.resid(y, d),
                       tmax=sol.tmax(d), hex_tmax=sol.tmax(d, HEX),
                       bend_tmax=sol.tmax(d, BEND),
                       sd_tmax=float(np.std(tm_mc)),
                       sd_M2dx=float(np.std(mc[:, 1])),
                       sd_Camdx=float(np.std(mc[:, 6])))
            for j, lab in [(0, 'M2dz'), (1, 'M2dx'), (2, 'M2dy'), (6, 'Camdx'),
                           (7, 'Camdy')]:
                row[lab] = d[j]
            # wavefront made by the hexapod block alone
            dh = np.zeros(50)
            dh[HEX] = d[HEX]
            row['hex_wf'] = float(np.linalg.norm(sol.A @ dh))
            out.append(row)
            print(f'  {name:22s} {row["resid"]:7.4f} {row["tmax"]:7.2f} '
                  f'{row["hex_tmax"]:8.3f} {row["bend_tmax"]:9.2f} '
                  f'{d[0]:8.1f} {d[1]:8.1f} {d[2]:8.1f} {d[6]:8.1f} {d[7]:8.1f} | '
                  f'{row["sd_tmax"]:7.2f} {row["sd_M2dx"]:8.1f} '
                  f'{row["sd_Camdx"]:8.1f}')
        print()
    return pd.DataFrame(out)


def rho_scan(sol, legs, rhos):
    """Q50 and Q34x residual and feasibility versus rho, both legs."""
    print('=== Q penalty scan: resid (µm of wavefront) / max |d_j|/r_j ===')
    rows = []
    for leg, (y, _) in legs.items():
        for rank in (50, 34):
            for rho in rhos:
                d = sol.quad(y, rho, rank)
                rows.append(dict(leg=leg, rank=rank, rho=rho, resid=sol.resid(y, d),
                                 tmax=sol.tmax(d), hex_wf=float(np.linalg.norm(
                                     sol.A[:, HEX] @ d[HEX]))))
    df = pd.DataFrame(rows)
    print(df.pivot_table(index='rho', columns=['leg', 'rank'],
                         values=['resid', 'tmax']).round(4).to_string())
    print()
    return df


RHO_REF = 0.01


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ts-ofc', type=pathlib.Path,
                    default=pathlib.Path.home() / 'Astrophysics/Claude/packages/ts_ofc')
    ap.add_argument('--norm-yaml', type=pathlib.Path,
                    default=pathlib.Path.home() / 'Astrophysics/Claude/packages/'
                    'ts_config_mttcs/MTAOS/ofc/normalization_weights/'
                    'range0.5_fwhm-0.15.yaml',
                    help='normalization weights; default is the deployed init.yaml '
                    'choice, which the bounce and MIW builds also use')
    ap.add_argument('--out-dir', type=pathlib.Path,
                    default=REPO / 'smatrix/output/regularized_inversion/penalty_matrix')
    args = ap.parse_args()
    out = args.out_dir / args.norm_yaml.stem
    out.mkdir(parents=True, exist_ok=True)
    A, w, r, labels = load_inputs(args.ts_ofc, args.norm_yaml,
                                  args.out_dir / 'S_full_31_29_50.npy')
    print(f'normalization: {args.norm_yaml.name}\n')
    sol = Solvers(A, w, r)
    legs = load_legs()

    # Self-checks: R at kappa -> inf equals truncation; Q at rho -> 0 equals T50.
    y = legs['Elev=40'][0]
    assert np.allclose(sol.quad(y, 1e-9), sol.trunc(y, 50), rtol=1e-6, atol=1e-6)
    print(f'check: T34 resid via full-rank A = {sol.resid(y, sol.trunc(y, 34)):.6f}, '
          f'via subspace = {np.linalg.norm(y - sol.U[:, :34] @ (sol.U[:, :34].T @ y)):.6f}'
          ' µm of wavefront\n')

    report_structure(sol, labels)
    syn = synthetic_hexapod(sol, labels)
    amp = hexapod_amplitude_scan(sol, labels)
    amp.to_csv(out / 'hexapod_amplitude_scan.csv', index=False)
    real = real_legs(sol, legs, labels)
    scan = rho_scan(sol, legs, np.logspace(-4, 0, 9))
    syn.to_csv(out / 'synthetic_hexapod.csv', index=False)
    real.to_csv(out / 'bounce_legs.csv', index=False)
    scan.to_csv(out / 'rho_scan.csv', index=False)


if __name__ == '__main__':
    main()
