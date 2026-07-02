#!/usr/bin/env python3
"""wfs_corner_compare — corner-by-corner CWFS vs FAM measured-OPD comparison.

For each corner raft (R00/R04/R40/R44) and each FAM triplet, compare the median
in-focus corner-WFS Zernike (zk_<coord>, y-axis) against the FAM measured OPD
interpolated to that corner (x-axis).  The interpolation is a plain azimuthal-wedge
median: FAM donuts in the outer annulus r in [r_min, r_max] within +/- wedge_half deg
of the corner azimuth (>= min_fam donuts).  Both are raw OPD in <coord> (default OCS),
from the same triplet (matched CWFS fam_seq_num <-> FAM seq_num).

Pages (output/<ps>/wfs/wfs_corner_compare.pdf):
  - per corner: FAM-vs-CWFS scatter, all Zj (Pearson r, OLS slope/offset, robust RMS
    of residuals about the fit);
  - per corner: time history (FAM-interp + CWFS-median vs image ordinal), all Zj;
  - summary: corr / slope / offset / robust-RMS vs Zj, all 4 corners;
  - validation: every val_stride-th triplet, Z5-Z8 vs focal-plane azimuth showing the
    individual FAM (annulus) and CWFS donuts plus the FAM interpolation and CWFS median.

Pure parquet (CWFS wfs/donuts.parquet + FAM donuts.parquet) — runs anywhere.
Replaces the old wfs_build / study_wfs_radial continuity plots.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CORNERS = ['R00_SW0', 'R04_SW0', 'R40_SW0', 'R44_SW0']


def nmad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size >= 3 else np.nan


def ang_dist(a, b):
    """Smallest |a-b| on the circle (deg)."""
    return np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)


def _fourier_curve(az, z, M):
    """Least-squares truncated Fourier series Z(az) with M harmonics -> callable."""
    def design(a):
        a = np.atleast_1d(np.asarray(a, float)); cols = [np.ones_like(a)]
        for m in range(1, M + 1):
            cols += [np.cos(m * np.radians(a)), np.sin(m * np.radians(a))]
        return np.column_stack(cols)
    coef = np.linalg.lstsq(design(az), z, rcond=None)[0]
    return lambda a: design(a) @ coef


def _gp_curve(az, z, bin_deg):
    """GP on the cos/sin azimuth circle, fit to bin_deg-binned medians -> callable."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    bc, bm = [], []
    for lo in np.arange(0, 360, bin_deg):
        m = (az >= lo) & (az < lo + bin_deg)
        if m.any():
            bc.append(lo + bin_deg / 2); bm.append(np.nanmedian(z[m]))
    bc, bm = np.array(bc), np.array(bm)
    circ = lambda a: np.c_[np.cos(np.radians(np.atleast_1d(a))), np.sin(np.radians(np.atleast_1d(a)))]
    k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(0.5, (0.05, 3.0)) + WhiteKernel(0.01, (1e-4, 0.2))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1).fit(circ(bc), bm)
    return lambda a: gp.predict(circ(a))


def azimuth_interp(az, z, corner_az, method, fourier_m, gp_bin, wedge_half, min_fam):
    """FAM Zj interpolated to the corner azimuths. Returns (values[len(corner_az)], curve|None).
    'wedge-median' = local median (curve=None); 'fourier'/'gp' = smooth ring fit + curve callable."""
    ca = np.atleast_1d(np.asarray(corner_az, float))
    fin = np.isfinite(z)
    if method == 'wedge-median':
        vals = np.array([np.nanmedian(z[ang_dist(az, c) <= wedge_half])
                         if (ang_dist(az, c) <= wedge_half).sum() >= min_fam else np.nan for c in ca])
        return vals, None
    if fin.sum() < max(min_fam, fourier_m * 2 + 2):
        return np.full(len(ca), np.nan), None
    try:
        curve = (_gp_curve(az[fin], z[fin], gp_bin) if method == 'gp'
                 else _fourier_curve(az[fin], z[fin], fourier_m))
    except Exception:                              # e.g. sklearn missing -> fall back to fourier
        curve = _fourier_curve(az[fin], z[fin], fourier_m)
    return np.asarray(curve(ca), float).ravel(), curve


def fit_metrics(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return dict(n=int(m.sum()), r=np.nan, slope=np.nan, off=np.nan, rms=np.nan)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    s, b = np.polyfit(x, y, 1)
    res = y - (s * x + b)
    return dict(n=len(x), r=float(np.corrcoef(x, y)[0, 1]), slope=float(s),
                off=float(b), rms=float(nmad(res)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--wfs-name', required=True,
                    help='CWFS variant key; reads/writes under output/<ps>/wfs/<wfs-name>/')
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--coord', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--r-min', type=float, default=1.5178)
    ap.add_argument('--r-max', type=float, default=1.725)
    ap.add_argument('--interp', default='gp', choices=['gp', 'fourier', 'wedge-median'],
                    help='FAM azimuth interpolation to the corner: gp (smooth ring fit, default), '
                         'fourier (truncated harmonics), wedge-median (local flat median)')
    ap.add_argument('--fourier-m', type=int, default=4, help='harmonics for --interp fourier')
    ap.add_argument('--gp-bin-deg', type=float, default=15.0, help='azimuth bin (deg) for the GP medians')
    ap.add_argument('--wedge-half', type=float, default=30.0, help='FAM azimuth half-width (deg), --interp wedge-median')
    ap.add_argument('--min-fam', type=int, default=3, help='min FAM donuts (wedge) / ring donuts (gp,fourier)')
    ap.add_argument('--val-stride', type=int, default=20, help='validation page every Nth triplet')
    ap.add_argument('--val-zernikes', default='5,6,7,8')
    args = ap.parse_args()
    coord = args.coord
    base = Path(args.output_root) / args.param_set
    zc, txc, tyc = f'zk_{coord}', f'thx_{coord}', f'thy_{coord}'

    noll = [int(x) for x in np.asarray(
        pq.read_table(str(base / 'visits.parquet'), columns=['nollIndices']).to_pandas()['nollIndices'].iloc[0])]
    nZk = len(noll)
    vjs = [int(j) for j in args.val_zernikes.split(',') if int(j) in noll]

    # ---- CWFS (in-focus corner donuts) ----
    cw = pq.read_table(str(base / 'wfs' / args.wfs_name / 'donuts.parquet'),
                       columns=['detector', 'day_obs', 'seq_num', 'fam_seq_num', zc, txc, tyc]).to_pandas()
    cw_zk = np.stack(cw[zc].values).astype(float)
    cw_az = np.degrees(np.arctan2(cw[tyc].astype(float), cw[txc].astype(float))) % 360.0
    cw_det = cw.detector.astype(str).values

    # ---- FAM (science array), outer annulus only ----
    fam = pq.read_table(str(base / 'donuts.parquet'), columns=['day_obs', 'seq_num', zc, txc, tyc]).to_pandas()
    fr = np.degrees(np.hypot(fam[txc].astype(float), fam[tyc].astype(float)))
    ann = (fr >= args.r_min) & (fr <= args.r_max)
    fam = fam[ann].reset_index(drop=True)
    fam_zk = np.stack(fam[zc].values).astype(float)
    fam_az = np.degrees(np.arctan2(fam[tyc].astype(float), fam[txc].astype(float))) % 360.0
    fam_grp = fam.groupby(['day_obs', 'seq_num']).indices            # (day,seq) -> row positions
    cw_grp = cw.groupby(['day_obs', 'fam_seq_num']).indices          # (day,fam_seq) -> row positions
    triplets = sorted(set(cw_grp.keys()) & set(fam_grp.keys()))
    print(f'[wfs_corner_compare] {args.param_set}: {len(triplets)} matched triplets, '
          f'{nZk} Zernikes, coord={coord}, wedge=±{args.wedge_half}°')
    nT = len(triplets)
    FAMI = np.full((nT, 4, nZk), np.nan)        # FAM interp at each corner
    CWM = np.full((nT, 4, nZk), np.nan)         # CWFS median at each corner
    AZC = np.full((nT, 4), np.nan)              # corner azimuth (deg)
    val_idx = set(range(0, nT, args.val_stride))
    val = {}
    print(f'  FAM->corner interpolation: {args.interp}'
          + (f' (M={args.fourier_m})' if args.interp == 'fourier' else
             f' (bin={args.gp_bin_deg}°)' if args.interp == 'gp' else f' (±{args.wedge_half}°)'))
    ikw = dict(method=args.interp, fourier_m=args.fourier_m, gp_bin=args.gp_bin_deg,
               wedge_half=args.wedge_half, min_fam=args.min_fam)
    for t, key in enumerate(triplets):
        ci = np.asarray(cw_grp[key], int)
        fi = np.asarray(fam_grp[key], int)
        faz_t, fzk_t = fam_az[fi], fam_zk[fi]
        for c, det in enumerate(CORNERS):       # CWFS median + corner azimuth
            cc = ci[cw_det[ci] == det]
            if len(cc) == 0:
                continue
            CWM[t, c] = np.nanmedian(cw_zk[cc], axis=0)
            AZC[t, c] = np.degrees(np.arctan2(np.nanmedian(np.sin(np.radians(cw_az[cc]))),
                                              np.nanmedian(np.cos(np.radians(cw_az[cc]))))) % 360.0
        cok = np.isfinite(AZC[t])               # FAM interp to those corner azimuths, per Zj
        if cok.any():
            for jp in range(nZk):
                FAMI[t, cok, jp] = azimuth_interp(faz_t, fzk_t[:, jp], AZC[t, cok], **ikw)[0]
        if t in val_idx:
            val[t] = (key, faz_t, fzk_t, cw_az[ci], cw_zk[ci], cw_det[ci])

    out = base / 'wfs' / args.wfs_name / 'wfs_corner_compare.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}
    nr = int(np.ceil(np.sqrt(nZk))); ncpanel = int(np.ceil(nZk / nr))
    metrics = {}        # (corner, j) -> fit dict

    with PdfPages(str(out)) as pdf:
        # ---- per-corner scatter (all Zj) ----
        for c, det in enumerate(CORNERS):
            fig, axes = plt.subplots(nr, ncpanel, figsize=(2.5 * ncpanel, 2.5 * nr),
                                     constrained_layout=True, squeeze=False)
            for p, j in enumerate(noll):
                ax = axes.ravel()[p]
                x, y = FAMI[:, c, p], CWM[:, c, p]
                m = fit_metrics(x, y); metrics[(det, j)] = m
                ax.scatter(x, y, s=8, alpha=0.4)
                fin = np.isfinite(x) & np.isfinite(y)
                if fin.sum() >= 3:
                    lo, hi = np.nanpercentile(np.concatenate([x[fin], y[fin]]), [1, 99])
                    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.6)
                    xx = np.array([lo, hi]); ax.plot(xx, m['slope'] * xx + m['off'], 'r-', lw=0.8)
                    ax.text(0.04, 0.96, f"r={m['r']:.2f}\ns={m['slope']:.2f} b={m['off']:+.3f}\n"
                            f"rms={m['rms']:.3f}", transform=ax.transAxes, va='top', fontsize=6)
                ax.set_title(f'Z{j}', fontsize=8); ax.tick_params(labelsize=6)
            for ax in axes.ravel()[nZk:]:
                ax.axis('off')
            fig.suptitle(f'{det}: CWFS median (y) vs FAM-interp (x) [{coord}, µm] — {args.param_set}', fontsize=11)
            pdf.savefig(fig); plt.close(fig)

        # ---- per-corner time history (all Zj) ----
        ordn = np.arange(nT)
        for c, det in enumerate(CORNERS):
            fig, axes = plt.subplots(nr, ncpanel, figsize=(2.5 * ncpanel, 2.2 * nr),
                                     constrained_layout=True, squeeze=False)
            for p, j in enumerate(noll):
                ax = axes.ravel()[p]
                ax.plot(ordn, FAMI[:, c, p], '.-', ms=2, lw=0.5, color='steelblue', label='FAM')
                ax.plot(ordn, CWM[:, c, p], '.-', ms=2, lw=0.5, color='crimson', label='CWFS')
                ax.set_title(f'Z{j}', fontsize=8); ax.tick_params(labelsize=6)
            for ax in axes.ravel()[nZk:]:
                ax.axis('off')
            axes.ravel()[0].legend(fontsize=6)
            fig.suptitle(f'{det}: time history vs image ordinal [{coord}, µm] — {args.param_set}', fontsize=11)
            pdf.savefig(fig); plt.close(fig)

        # ---- summary: metric vs Zj, all corners ----
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for ax, (key, lab) in zip(axes.ravel(),
                                  [('r', 'correlation r'), ('slope', 'slope'),
                                   ('off', 'offset [µm]'), ('rms', 'robust RMS [µm]')]):
            for c, det in enumerate(CORNERS):
                ax.plot(noll, [metrics[(det, j)][key] for j in noll], '-o', ms=4, label=det)
            ax.set_xlabel('Noll j'); ax.set_ylabel(lab); ax.grid(alpha=0.3)
            if key == 'slope':
                ax.axhline(1, color='k', lw=0.5)
            if key in ('off',):
                ax.axhline(0, color='k', lw=0.5)
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle(f'CWFS vs FAM-interp summary — {args.param_set} ({coord})', fontsize=12)
        pdf.savefig(fig); plt.close(fig)

        # ---- validation: Z5-Z8 vs azimuth, every val_stride-th triplet ----
        for t in sorted(val):
            key, faz, fzk, caz, czk, cdet = val[t]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            agrid = np.arange(0, 360.5, 2.0)
            for ax, j in zip(axes.ravel(), vjs):
                jp = noll.index(j)
                ax.scatter(faz, fzk[:, jp], s=8, alpha=0.25, color='steelblue', label='FAM donuts')
                ax.scatter(caz, czk[:, jp], s=18, alpha=0.6, color='crimson', marker='s', label='CWFS donuts')
                _, curve = azimuth_interp(faz, fzk[:, jp], [0.0], **ikw)
                if curve is not None:
                    ax.plot(agrid, np.asarray(curve(agrid), float).ravel(), '-', color='navy', lw=1,
                            alpha=0.7, label=f'FAM {args.interp} fit')
                ax.scatter(AZC[t], FAMI[t, :, jp], s=80, color='navy', marker='P',
                           edgecolors='w', zorder=5, label='FAM interp')
                ax.scatter(AZC[t], CWM[t, :, jp], s=80, color='darkred', marker='X',
                           edgecolors='w', zorder=5, label='CWFS median')
                ax.set_title(f'Z{j} {NOLL_NAMES.get(j, "")}', fontsize=9)
                ax.set_xlabel('focal-plane azimuth [deg]'); ax.set_ylabel(f'Z{j} [µm]')
                ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90)); ax.grid(alpha=0.3)
            axes.ravel()[0].legend(fontsize=7, loc='best')
            fig.suptitle(f'Validation triplet #{t} day {key[0]} fam_seq {key[1]} '
                         f'(annulus {args.r_min}-{args.r_max}°) — {args.param_set}', fontsize=11)
            pdf.savefig(fig); plt.close(fig)

    # stdout summary
    print('  per-corner median (over Zj) correlation / robust-RMS:')
    for det in CORNERS:
        rr = np.nanmedian([metrics[(det, j)]['r'] for j in noll])
        rm = np.nanmedian([metrics[(det, j)]['rms'] for j in noll])
        print(f'    {det}: median r={rr:.3f}  median robRMS={rm:.3f} µm')
    print(f'  wrote {out} ({len(triplets)} triplets, {len(val)} validation pages)')


if __name__ == '__main__':
    main()
