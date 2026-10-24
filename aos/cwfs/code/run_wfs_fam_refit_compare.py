#!/usr/bin/env python3
"""wfs_fam_refit_compare — FAM vs Danish-1.2-REFIT corner-WFS wavefront, vs azimuth.

Like run_wfs_fam_compare, but the WFS Zernikes come from the Danish-1.2 ensemble refit
(run_wfs_refit_ensemble -> wfs_refit_ensemble_<pupil>.parquet) instead of the stored
Danish-1.0 pipeline values.  Use pupil='old' to match the pupil the FAM was generated
with (the FAM is the high-statistics wavefront reference).

Frame: CCS.  The refit Zernikes are native CCS (danish), and the FAM zk_CCS is compared
directly — no CCS->OCS rotation guess.  The comparison set is rotator~0, so CCS ~ OCS and
the plots look like the OCS fam_wfs_triplet_compare.

Per FAM triplet (one page): Z5/Z6/Z7/Z8 median measured Zernike vs focal-plane azimuth:
  FAM  — donuts.parquet zk_CCS in the WFS annulus (r in [r_min, r_max]).
  WFS  — the refit corner-WFS pairs (zk_refit) of the paired in-focus exposure.

Pure parquet (numpy/pandas/astropy/scipy) — no danish/butler; runs anywhere.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _az_median(az, vals, azedges, min_n):
    from scipy.stats import binned_statistic
    fin = np.isfinite(vals)
    if fin.sum() == 0:
        return np.full(len(azedges) - 1, np.nan)
    med, _, _ = binned_statistic(az[fin], vals[fin], 'median', bins=azedges)
    cnt, _, _ = binned_statistic(az[fin], vals[fin], 'count', bins=azedges)
    med[cnt.astype(int) < min_n] = np.nan
    return med


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--pupil', default='old', choices=['new', 'old'])
    ap.add_argument('--alpha', type=float, default=0.0, help='which refit alpha rows to plot')
    ap.add_argument('--r-min', type=float, default=1.5178)
    ap.add_argument('--r-max', type=float, default=1.725)
    ap.add_argument('--az-bin-deg', type=float, default=30.0)
    ap.add_argument('--zernikes', default='5,6,7,8')
    ap.add_argument('--min-donuts-per-bin', type=int, default=1)
    ap.add_argument('--max-pages', type=int, default=40)
    args = ap.parse_args()
    zks = [int(x) for x in args.zernikes.split(',')]
    base = Path(args.output_root) / args.param_set

    # ---- WFS refit (CCS) ----
    rf = pd.read_parquet(base / 'wfs' / f'wfs_refit_ensemble_{args.pupil}.parquet')
    rf = rf[np.isclose(rf['alpha'].astype(float), args.alpha)]
    if len(rf) == 0:
        raise SystemExit(f'no refit rows at alpha={args.alpha} in wfs_refit_ensemble_{args.pupil}.parquet')
    noll_w = [int(x) for x in rf['nollIndices'].iloc[0]]
    zk_w = np.array(rf['zk_refit'].tolist())
    rf = rf.reset_index(drop=True)
    w_thx = np.rad2deg(np.asarray(rf['thx_ccs'], float))
    w_thy = np.rad2deg(np.asarray(rf['thy_ccs'], float))

    # ---- FAM donuts (CCS) ----
    dd = pq.read_table(str(base / 'donuts.parquet')).to_pandas()
    zk_f = np.stack(dd['zk_CCS'].values).astype(float)
    f_day = np.asarray(dd['day_obs']).astype(int)
    f_seq = np.asarray(dd['seq_num']).astype(int)
    f_thx = np.rad2deg(np.asarray(dd['thx_CCS'], float))
    f_thy = np.rad2deg(np.asarray(dd['thy_CCS'], float))
    vt = pq.read_table(str(base / 'visits.parquet')).to_pandas()
    noll_f = ([int(x) for x in np.asarray(vt['nollIndices'].iloc[0])]
              if 'nollIndices' in vt.columns else list(range(4, 4 + zk_f.shape[1])))

    jf = {j: noll_f.index(j) for j in zks if j in noll_f}
    jw = {j: noll_w.index(j) for j in zks if j in noll_w}
    azedges = np.arange(0.0, 360.0 + 1e-6, args.az_bin_deg)
    azc = 0.5 * (azedges[:-1] + azedges[1:])

    triplets = (rf[['day_obs', 'fam_seq', 'infocus_seq', 'gal_b']]
                .drop_duplicates().sort_values(['day_obs', 'fam_seq']))
    out = base / 'wfs' / f'fam_wfs_refit_compare_{args.pupil}_a{args.alpha:g}.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'[wfs_fam_refit_compare] pupil={args.pupil} alpha={args.alpha}: '
          f'{len(triplets)} triplets, CCS frame')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from lsst.ts.intrinsic.wavefront.common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}

    n = 0
    with PdfPages(str(out)) as pdf:
        for t in triplets.itertuples():
            day, fseq, iseq, gb = int(t.day_obs), int(t.fam_seq), int(t.infocus_seq), float(t.gal_b)
            fm = (f_day == day) & (f_seq == fseq)
            r = np.hypot(f_thx[fm], f_thy[fm])
            ann = (r >= args.r_min) & (r <= args.r_max)
            faz = np.arctan2(f_thy[fm][ann], f_thx[fm][ann]); faz = np.degrees(faz) % 360.0
            wm = (np.asarray(rf['day_obs']) == day) & (np.asarray(rf['fam_seq']) == fseq)
            waz = np.degrees(np.arctan2(w_thy[wm], w_thx[wm])) % 360.0
            if ann.sum() == 0 and wm.sum() == 0:
                continue
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained', sharex=True)
            axes = axes.ravel()
            for p, j in enumerate(zks):
                ax = axes[p]
                if j in jf and ann.sum():
                    fv = zk_f[fm][ann, jf[j]]
                    ax.scatter(faz, fv, s=10, color='steelblue', alpha=0.25, edgecolors='none')
                    ax.plot(azc, _az_median(faz, fv, azedges, args.min_donuts_per_bin), '-o',
                            color='steelblue', ms=5, lw=1.3, label=f'FAM (n={int(ann.sum())})')
                if j in jw and wm.sum():
                    wv = zk_w[wm][:, jw[j]]
                    ax.scatter(waz, wv, s=28, color='crimson', alpha=0.5, marker='s', edgecolors='none')
                    ax.plot(azc, _az_median(waz, wv, azedges, args.min_donuts_per_bin), '-s',
                            color='crimson', ms=6, lw=1.3, label=f'WFS refit (n={int(wm.sum())})')
                ax.axhline(0, color='k', lw=0.4, alpha=0.5); ax.grid(alpha=0.3)
                ax.set_title(f'Z{j} {NOLL_NAMES.get(j, "")}', fontsize=10)
                ax.set_ylabel(f'Z{j} [μm]', fontsize=9); ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 60))
                if p >= 2:
                    ax.set_xlabel('focal-plane azimuth [deg] (CCS)', fontsize=9)
                ax.legend(fontsize=8, loc='upper right')
            fig.suptitle(f'FAM vs Danish-1.2 refit WFS ({args.pupil} pupil, α={args.alpha:g}) — '
                         f'day_obs {day}, FAM seq {fseq} (b={gb:+.1f} deg)', fontsize=12)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            n += 1
            if n >= args.max_pages:
                break
    print(f'  wrote {out} ({n} triplet pages)')


if __name__ == '__main__':
    main()
