#!/usr/bin/env python3
"""wfs_dof_compare — corner-WFS vs FAM optical-state (v-modes + DoF) per image.

Extract the optical state two ways per FAM triplet and compare:
  FAM  (reference) — project the per-visit full-focal-plane double-Zernike fit
        (<mi>/fits.parquet, MIW-subtracted) onto the OFC sensitivity SVD:
        A_fam = W @ U_eff  ->  v-modes = A/sigma,  DoF = V-recon(A).
  CWFS            — from the 4 corner medians only: subtract the MIW (5rot
        intrinsic_split OCS maps, interpolated to each corner) and the per-(Zj,corner)
        offset, then the corner OFC inverse  A_cwfs = pinv(B·U_eff, rcond)·z_corner_dev
        (B = focal basis at the 4 corner OCS positions).  Same SVD/normalization/
        truncation as FAM, so the two are directly comparable (only the measurement
        differs: 4 corners vs full field).

Both 50-DOF/34-vmode and 22-DOF/12-vmode are produced.  Pages per SVD: v-mode time
history (vs image ordinal) and CWFS-vs-FAM scatter; then DoF time history and scatter.
Scatter fits reject points far (>K·nMAD) from the mass.

Needs ts_ofc (build_ofc_svd) + TS_CONFIG_MTTCS_DIR; runs in the LSST stack env.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
CORNERS = ['R00_SW0', 'R04_SW0', 'R40_SW0', 'R44_SW0']
FP_RADIUS = 1.75
DOF22 = list(range(0, 10)) + list(range(10, 17)) + list(range(30, 35))
# CWFS - FAM offsets (OCS, µm) from the corner-compare study; subtracted from CWFS.
DEFAULT_OFFSETS = {4: {'R00_SW0': -0.11, 'R04_SW0': -0.11, 'R40_SW0': -0.11, 'R44_SW0': -0.19},
                   11: {c: -0.07 for c in CORNERS},
                   14: {c: +0.13 for c in CORNERS}}


def nmad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size >= 3 else np.nan


def corner_matrix_at(svd, noll, pos_deg):
    """B (ncorner*nj, n_kj): focal basis at the corner OCS positions (deg)."""
    from ofc_svd import focal_zernike_at_points
    jpos = {j: i for i, j in enumerate(noll)}; nj = len(noll)
    B = np.zeros((len(pos_deg) * nj, len(svd.kj_grid)))
    for ci, (tx, ty) in enumerate(pos_deg):
        rho = np.hypot(tx, ty) / FP_RADIUS; th = np.arctan2(ty, tx)
        for k_i, (k, j) in enumerate(svd.kj_grid):
            if j in jpos:
                B[ci * nj + jpos[j], k_i] = float(focal_zernike_at_points(k, rho, th))
    return B


def robust_fit(x, y, K):
    """OLS fit + Pearson r + robust RMS after dropping points >K·nMAD from the mass."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 3:
        return np.zeros(len(m), bool), dict(n=len(x), ndrop=0, r=np.nan, slope=np.nan, off=np.nan, rms=np.nan)
    keep = ((np.abs(x - np.median(x)) <= K * nmad(x) + 1e-12)
            & (np.abs(y - np.median(y)) <= K * nmad(y) + 1e-12))
    xk, yk = x[keep], y[keep]
    if keep.sum() < 3:
        return keep, dict(n=len(x), ndrop=int((~keep).sum()), r=np.nan, slope=np.nan, off=np.nan, rms=np.nan)
    s, b = np.polyfit(xk, yk, 1); res = yk - (s * xk + b)
    return keep, dict(n=len(x), ndrop=int((~keep).sum()), r=float(np.corrcoef(xk, yk)[0, 1]),
                      slope=float(s), off=float(b), rms=float(nmad(res)))


def th_pages(labels, ordn, fam, cwfs, title, pdf, per=20):
    import matplotlib.pyplot as plt
    n = len(labels)
    for p0 in range(0, n, per):
        idx = range(p0, min(p0 + per, n)); nr = int(np.ceil(len(list(idx)) / 4))
        fig, axes = plt.subplots(nr, 4, figsize=(15, 2.2 * nr), constrained_layout=True, squeeze=False)
        for ax, i in zip(axes.ravel(), idx):
            ax.plot(ordn, fam[:, i], '-', lw=0.6, color='steelblue', label='FAM')
            ax.plot(ordn, cwfs[:, i], '-', lw=0.6, color='crimson', label='CWFS')
            ax.axhline(0, color='k', lw=0.3); ax.set_title(labels[i], fontsize=7); ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(list(idx)):]:
            ax.axis('off')
        axes.ravel()[0].legend(fontsize=6)
        fig.suptitle(f'{title} — time history vs image ordinal  [FAM blue, CWFS red]', fontsize=11)
        pdf.savefig(fig); plt.close(fig)


def scatter_pages(labels, fam, cwfs, title, pdf, K, per=20):
    import matplotlib.pyplot as plt
    n = len(labels)
    for p0 in range(0, n, per):
        idx = list(range(p0, min(p0 + per, n))); nr = int(np.ceil(len(idx) / 4))
        fig, axes = plt.subplots(nr, 4, figsize=(15, 3.0 * nr), constrained_layout=True, squeeze=False)
        for ax, i in zip(axes.ravel(), idx):
            x, y = fam[:, i], cwfs[:, i]
            keep, m = robust_fit(x, y, K)
            fm = np.isfinite(x) & np.isfinite(y)
            xf, yf = x[fm], y[fm]
            ax.scatter(xf[keep], yf[keep], s=6, alpha=0.4)
            if (~keep).any():
                ax.scatter(xf[~keep], yf[~keep], s=6, alpha=0.4, color='lightgray')
            if np.isfinite(m['slope']):
                lo, hi = np.nanpercentile(np.concatenate([xf[keep], yf[keep]]), [1, 99])
                ax.plot([lo, hi], [lo, hi], 'k--', lw=0.6)
                ax.plot([lo, hi], [m['slope'] * lo + m['off'], m['slope'] * hi + m['off']], 'r-', lw=0.8)
                ax.text(0.04, 0.96, f"r={m['r']:.2f} s={m['slope']:.2f}\nrms={m['rms']:.3f} drop={m['ndrop']}",
                        transform=ax.transAxes, va='top', fontsize=6)
            ax.set_title(labels[i], fontsize=7); ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(idx):]:
            ax.axis('off')
        fig.suptitle(f'{title} — CWFS (y) vs FAM (x)', fontsize=11)
        pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', default='pathA_50_34_i_5rot')
    ap.add_argument('--coord', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--dz-prefix', default='z1toz6')
    ap.add_argument('--rcond', type=float, default=1e-2, help='pinv cutoff for the corner OFC inverse')
    ap.add_argument('--reject-k', type=float, default=5.0, help='scatter outlier: drop >K·nMAD from the mass')
    ap.add_argument('--no-offsets', action='store_true', help='disable the CWFS-FAM per-(Zj,corner) offsets')
    ap.add_argument('--max-triplets', type=int, default=0, help='cap triplets (quick test); 0=all')
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()
    from ofc_svd import build_ofc_svd
    from scipy.interpolate import LinearNDInterpolator
    coord = args.coord
    base = Path(args.output_root) / args.param_set
    bmi = base / args.mi_name
    zc, txc, tyc = f'zk_{coord}', f'thx_{coord}', f'thy_{coord}'
    offsets = {} if args.no_offsets else DEFAULT_OFFSETS

    noll = [int(x) for x in np.asarray(
        pq.read_table(str(base / 'visits.parquet'), columns=['nollIndices']).to_pandas()['nollIndices'].iloc[0])]

    # ---- MIW (5rot OCS maps) interpolators per Zj ----
    M = pd.read_parquet(bmi / 'intrinsic_split_maps.parquet')
    pts = np.column_stack([M.thx_deg, M.thy_deg])
    miw = {j: LinearNDInterpolator(pts, M[f'Z{j}_OCS'].values) for j in noll if f'Z{j}_OCS' in M.columns}

    # ---- FAM per-visit DZ fits (W built per SVD from its kj_grid below) ----
    fits = pd.read_parquet(bmi / 'fits.parquet')
    fam_key = {(int(d), int(s)): i for i, (d, s) in enumerate(zip(fits.day_obs, fits.seq_num))}

    # ---- CWFS corner donuts ----
    cw = pq.read_table(str(base / 'wfs' / 'donuts.parquet'),
                       columns=['detector', 'day_obs', 'seq_num', 'fam_seq_num', zc, txc, tyc]).to_pandas()
    cw_zk = np.stack(cw[zc].values).astype(float)
    cw_thx = np.rad2deg(cw[txc].astype(float).values); cw_thy = np.rad2deg(cw[tyc].astype(float).values)
    cw_det = cw.detector.astype(str).values
    cw_grp = cw.groupby(['day_obs', 'fam_seq_num']).indices

    triplets = sorted(set(cw_grp.keys()) & set(fam_key.keys()), key=lambda k: (k[0], k[1]))
    if args.max_triplets:
        triplets = triplets[:args.max_triplets]
    print(f'[wfs_dof_compare] {args.param_set}/{args.mi_name}: {len(triplets)} matched triplets, '
          f'coord={coord}, offsets={"off" if args.no_offsets else "on"}, rcond={args.rcond}')

    # per-corner CWFS deviation (MIW- and offset-subtracted) + corner positions, per triplet
    jpos = {j: i for i, j in enumerate(noll)}
    zdev = np.full((len(triplets), 4, len(noll)), np.nan)
    pos = np.full((len(triplets), 4, 2), np.nan)
    for t, key in enumerate(triplets):
        ci = np.asarray(cw_grp[key], int)
        for c, det in enumerate(CORNERS):
            cc = ci[cw_det[ci] == det]
            if len(cc) == 0:
                continue
            txm, tym = np.median(cw_thx[cc]), np.median(cw_thy[cc])
            pos[t, c] = [txm, tym]
            zmed = np.nanmedian(cw_zk[cc], axis=0)
            for j, ip in jpos.items():
                mij = float(miw[j](txm, tym)) if j in miw else np.nan
                off = offsets.get(j, {}).get(det, 0.0)
                zdev[t, c, ip] = zmed[ip] - mij - off

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    out = bmi / 'wfs_dof_compare.pdf'
    ordn = np.arange(len(triplets))
    with PdfPages(str(out)) as pdf:
        for name, (n_keep, n_dof) in [('50DOF-34vmode', (34, None)), ('22DOF-12vmode', (12, DOF22))]:
            svd = build_ofc_svd(list(noll), k_min=1, k_max=6, n_keep=n_keep, n_dof=n_dof)
            W = np.column_stack([fits[f'{args.dz_prefix}_z{j}_c{k}'].values if
                                 f'{args.dz_prefix}_z{j}_c{k}' in fits.columns else np.full(len(fits), np.nan)
                                 for (k, j) in svd.kj_grid])
            A_fam_all = svd.project_amplitudes(W)                      # (nfits, n_keep)
            fam_rows = np.array([fam_key[key] for key in triplets])
            A_fam = A_fam_all[fam_rows]
            A_cwfs = np.full_like(A_fam, np.nan)
            for t in range(len(triplets)):
                if not np.all(np.isfinite(zdev[t])):
                    continue
                B = corner_matrix_at(svd, noll, pos[t])
                A_cwfs[t] = np.linalg.pinv(B @ svd.U_eff, rcond=args.rcond) @ zdev[t].ravel()
            vmode_lab = [f'v{m+1}' for m in range(n_keep)]
            dof_lab = svd.dof_labels()[0]
            vfam, vcw = svd.vmodes(A_fam), svd.vmodes(A_cwfs)
            dfam, dcw = svd.dof(A_fam), svd.dof(A_cwfs)
            th_pages(vmode_lab, ordn, vfam, vcw, f'{name} v-modes', pdf)
            scatter_pages(vmode_lab, vfam, vcw, f'{name} v-modes', pdf, args.reject_k)
            th_pages(dof_lab, ordn, dfam, dcw, f'{name} DoF', pdf)
            scatter_pages(dof_lab, dfam, dcw, f'{name} DoF', pdf, args.reject_k)
            med_r = np.nanmedian([robust_fit(vfam[:, i], vcw[:, i], args.reject_k)[1]['r'] for i in range(n_keep)])
            print(f'  {name}: median v-mode CWFS-vs-FAM r = {med_r:.3f}')
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
