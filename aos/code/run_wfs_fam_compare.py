#!/usr/bin/env python3
"""wfs_fam_compare — per-triplet FAM vs in-focus corner-WFS wavefront, vs azimuth.

A foundational, PER-IMAGE FAM<->WFS consistency check — deliberately simpler than
the corrected/binned continuity study: no per-visit field subtraction and no grid.
For each FAM triplet (one PDF page):

  * FAM  — every science-array donut in the WFS annulus (r in [r_min, r_max]),
           its measured Zernike median per focal-plane azimuth bin.
  * WFS  — the corner-WFS donuts of the paired in-focus exposure (matched by
           fam_seq_num), measured Zernike median per azimuth bin.

Z5, Z6, Z7, Z8 in four panels.  Runs only over FAM visits near a chosen rotator
angle and elevation (default rotator 0 deg, elevation 70 deg).

Both series are RAW measured wavefronts (total = intrinsic + per-visit aberration)
at the same field radius, so per-image agreement tests whether the two estimators
are mutually consistent BEFORE any DOF / v-mode correction.

Reads existing parquets only:
    output/<ps>/donuts.parquet        FAM per-donut zk_<coord> + thx/thy_<coord>
    output/<ps>/visits.parquet        rotator_angle, alt, nollIndices per visit
    output/<ps>/wfs/donuts.parquet    corner-WFS zk_<coord> + thx/thy_<coord>
                                      + fam_seq_num (+ nollIndices in .meta)
Writes output/<ps>/wfs/fam_wfs_triplet_compare.pdf .  numpy / pandas / astropy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.table import QTable

DEFAULT_ZK = [5, 6, 7, 8]


def _alt_to_deg(a):
    """Robustly express altitude in degrees (auto-detect radians)."""
    a = np.asarray(a, dtype=float)
    return np.rad2deg(a) if np.nanmax(np.abs(a)) < 2 * np.pi + 1e-3 else a


def _az_medians(az, vals, azedges, min_n):
    """Median + count of ``vals`` per azimuth bin (NaN-aware)."""
    from scipy.stats import binned_statistic
    med = np.full(len(azedges) - 1, np.nan)
    cnt = np.zeros(len(azedges) - 1, dtype=int)
    fin = np.isfinite(vals)
    if fin.sum() == 0:
        return med, cnt
    med, _, _ = binned_statistic(az[fin], vals[fin], 'median', bins=azedges)
    cnt, _, _ = binned_statistic(az[fin], vals[fin], 'count', bins=azedges)
    cnt = cnt.astype(int)
    med[cnt < min_n] = np.nan
    return med, cnt


def _annulus(thx_deg, thy_deg, r_min, r_max):
    r = np.hypot(thx_deg, thy_deg)
    az = np.degrees(np.arctan2(thy_deg, thx_deg)) % 360.0
    return (r >= r_min) & (r <= r_max), az


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--r-min', type=float, default=1.5178,
                    help='WFS-annulus inner radius (deg)')
    ap.add_argument('--r-max', type=float, default=1.725,
                    help='WFS-annulus outer radius (deg)')
    ap.add_argument('--rot-center', type=float, default=0.0)
    ap.add_argument('--rot-tol', type=float, default=3.0,
                    help='keep FAM visits with |rotator - rot_center| <= rot_tol (deg)')
    ap.add_argument('--alt-center', type=float, default=70.0)
    ap.add_argument('--alt-tol', type=float, default=2.0,
                    help='keep FAM visits with |alt - alt_center| <= alt_tol (deg)')
    ap.add_argument('--az-bin-deg', type=float, default=30.0)
    ap.add_argument('--zernikes', default='5,6,7,8',
                    help='comma-separated Noll indices (4 panels)')
    ap.add_argument('--min-donuts-per-bin', type=int, default=1)
    ap.add_argument('--max-pages', type=int, default=40)
    args = ap.parse_args()
    coord = args.coord_sys
    zks = [int(x) for x in args.zernikes.split(',')]
    base = Path(args.output_root) / args.param_set

    # ---- FAM per-donut measured wavefront ----
    dd = pq.read_table(str(base / 'donuts.parquet')).to_pandas()
    zk_fam = np.stack(dd[f'zk_{coord}'].values).astype(float)
    fam_day = np.asarray(dd['day_obs']).astype(int)
    fam_seq = np.asarray(dd['seq_num']).astype(int)
    fam_thx = np.rad2deg(np.asarray(dd[f'thx_{coord}'], dtype=float))
    fam_thy = np.rad2deg(np.asarray(dd[f'thy_{coord}'], dtype=float))

    # ---- visits: rotator/alt + measured Noll ordering ----
    vt = pq.read_table(str(base / 'visits.parquet')).to_pandas()
    noll_fam = ([int(x) for x in np.asarray(vt['nollIndices'].iloc[0])]
                if 'nollIndices' in vt.columns
                else list(range(4, 4 + zk_fam.shape[1])))
    vt = vt.copy()
    vt['alt_deg'] = _alt_to_deg(vt['alt']) if 'alt' in vt.columns else np.nan

    # ---- corner-WFS measured wavefront ----
    wd = QTable.read(str(base / 'wfs' / 'donuts.parquet'))
    noll_wfs = [int(x) for x in wd.meta.get('nollIndices')]
    zk_wfs = np.array(wd[f'zk_{coord}'], dtype=float)
    wfs_day = np.asarray(wd['day_obs']).astype(int)
    wfs_fam_seq = np.asarray(wd['fam_seq_num']).astype(int)
    wfs_thx = np.rad2deg(np.asarray(wd[f'thx_{coord}'], dtype=float))
    wfs_thy = np.rad2deg(np.asarray(wd[f'thy_{coord}'], dtype=float))

    # column index of each requested Noll in each table
    jf = {j: noll_fam.index(j) for j in zks if j in noll_fam}
    jw = {j: noll_wfs.index(j) for j in zks if j in noll_wfs}
    missing = [j for j in zks if j not in jf or j not in jw]
    if missing:
        print(f'  WARNING: Z{missing} absent from FAM or WFS Noll set '
              f'(FAM={noll_fam}, WFS={noll_wfs}); those panels will be blank')

    # ---- select FAM triplets near (rot_center, alt_center) ----
    sel = vt[(np.abs(vt['rotator_angle'] - args.rot_center) <= args.rot_tol)
             & (np.abs(vt['alt_deg'] - args.alt_center) <= args.alt_tol)]
    sel = sel.sort_values(['day_obs', 'seq_num'])
    print(f'[wfs_fam_compare] {args.param_set}: {len(sel)} FAM visits near '
          f'rotator {args.rot_center}+/-{args.rot_tol} deg, '
          f'alt {args.alt_center}+/-{args.alt_tol} deg')

    azedges = np.arange(0.0, 360.0 + 1e-6, args.az_bin_deg)
    azc = 0.5 * (azedges[:-1] + azedges[1:])
    out = base / 'wfs' / 'fam_wfs_triplet_compare.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}

    n_pages = 0
    with PdfPages(str(out)) as pdf:
        for row in sel.itertuples():
            d, s = int(row.day_obs), int(row.seq_num)
            fmask = (fam_day == d) & (fam_seq == s)
            wmask = (wfs_day == d) & (wfs_fam_seq == s)
            n_fam, n_wfs = int(fmask.sum()), int(wmask.sum())
            if n_fam == 0 and n_wfs == 0:
                continue
            fa_in, fa_az = _annulus(fam_thx[fmask], fam_thy[fmask], args.r_min, args.r_max)
            wa_in, wa_az = _annulus(wfs_thx[wmask], wfs_thy[wmask], args.r_min, args.r_max)
            n_fam_ann, n_wfs_ann = int(fa_in.sum()), int(wa_in.sum())
            if n_fam_ann == 0 and n_wfs_ann == 0:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained',
                                     sharex=True)
            axes = axes.ravel()
            for p, j in enumerate(zks):
                ax = axes[p]
                if j in jf and n_fam_ann:
                    fv = zk_fam[fmask][fa_in, jf[j]]
                    ax.scatter(fa_az[fa_in], fv, s=10, color='steelblue', alpha=0.25,
                               edgecolors='none')
                    med, cnt = _az_medians(fa_az[fa_in], fv, azedges, args.min_donuts_per_bin)
                    ax.plot(azc, med, '-o', color='steelblue', ms=5, lw=1.3,
                            label=f'FAM (n={n_fam_ann})')
                if j in jw and n_wfs_ann:
                    wv = zk_wfs[wmask][wa_in, jw[j]]
                    ax.scatter(wa_az[wa_in], wv, s=28, color='crimson', alpha=0.5,
                               marker='s', edgecolors='none')
                    med, cnt = _az_medians(wa_az[wa_in], wv, azedges, args.min_donuts_per_bin)
                    ax.plot(azc, med, '-s', color='crimson', ms=6, lw=1.3,
                            label=f'WFS (n={n_wfs_ann})')
                ax.axhline(0, color='k', lw=0.4, alpha=0.5)
                ax.grid(alpha=0.3)
                ax.set_title(f'Z{j} {NOLL_NAMES.get(j, "")}', fontsize=10)
                ax.set_ylabel(f'Z{j} [μm]', fontsize=9)
                ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 60))
                if p >= 2:
                    ax.set_xlabel('focal-plane azimuth [deg] (OCS)', fontsize=9)
                ax.legend(fontsize=8, loc='upper right')
            fig.suptitle(
                f'FAM vs corner-WFS — day_obs {d}, FAM seq {s} '
                f'(rotator {row.rotator_angle:.1f} deg, alt {row.alt_deg:.1f} deg)   '
                f'annulus r=[{args.r_min}, {args.r_max}] deg',
                fontsize=12)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            n_pages += 1
            if n_pages >= args.max_pages:
                print(f'  reached --max-pages={args.max_pages}; stopping')
                break

    print(f'  wrote wfs/fam_wfs_triplet_compare.pdf ({n_pages} triplet pages)')


if __name__ == '__main__':
    main()
