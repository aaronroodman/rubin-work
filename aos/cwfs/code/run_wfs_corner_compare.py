#!/usr/bin/env python3
"""wfs_corner_compare plots: corner-by-corner CWFS against FAM measured wavefront.

Reads one ``wfs_corner_compare`` product build (the matched table and the per-corner
stats, `rubinwork.corner_compare`) and writes ``wfs_corner_compare.pdf``:

  - per corner: FAM-against-CWFS scatter, all Zj (Pearson r, Spearman rho, Huber
    slope and offset, nMAD of the residuals about the Huber fit);
  - per corner: time history (FAM interpolation and CWFS median against image ordinal);
  - summary: Pearson r, Spearman rho, Huber slope, offset and robust RMS against Zj,
    all corners;
  - validation: every val_stride-th triplet, Z5-Z8 against focal-plane azimuth, showing
    the FAM annulus donuts and CWFS donuts with the FAM ring fit, the interpolation and
    the CWFS median (reads those triplets' donuts from the product's two upstreams).

Usage::

    python code/run_wfs_corner_compare.py --variant d12-refitWcs
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from rubinwork import corner_compare as cc
from rubinwork.common import study_run
from rubinwork.products import catalog, wfs_corner_compare as wcc


def validation_donuts(variant, build_dir, keys, coord, r_min, r_max):
    """FAM annulus and CWFS donuts of the validation triplets, from the build's inputs."""
    inputs = catalog.manifest('wfs_corner_compare', variant, build_dir.name)['inputs']
    cv, cb = inputs['cwfs_tables'].split('@')
    fv, fb = inputs['fam_tables'].split('@')
    zc, txc, tyc = f'zk_{coord}', f'thx_{coord}', f'thy_{coord}'
    nights = sorted({k[0] for k in keys})
    fam = pads.dataset(catalog.path('fam_tables', fv, fb) / 'donuts.parquet').to_table(
        columns=['day_obs', 'seq_num', zc, txc, tyc],
        filter=pads.field('day_obs').isin(nights)).to_pandas()
    cw = pq.read_table(catalog.path('cwfs_tables', cv, cb) / 'donuts.parquet',
                       columns=['detector', 'day_obs', 'fam_seq_num', zc, txc, tyc]).to_pandas()
    fr = np.degrees(np.hypot(fam[txc], fam[tyc]))
    fam = fam[(fr >= r_min) & (fr <= r_max)]
    out = {}
    for k in keys:
        f = fam[(fam.day_obs == k[0]) & (fam.seq_num == k[1])]
        c = cw[(cw.day_obs == k[0]) & (cw.fam_seq_num == k[1])]
        out[k] = (np.degrees(np.arctan2(f[tyc], f[txc])) % 360, np.stack(f[zc].values),
                  np.degrees(np.arctan2(c[tyc], c[txc])) % 360, np.stack(c[zc].values))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', required=True, help='wfs_corner_compare variant')
    ap.add_argument('--build', default=catalog.CURRENT)
    ap.add_argument('--out-dir', default=None,
                    help='default: a dated run directory in the study output')
    ap.add_argument('--val-stride', type=int, default=20, help='validation page every Nth triplet')
    ap.add_argument('--val-zernikes', default='5,6,7,8')
    args = ap.parse_args()

    build_dir = catalog.path('wfs_corner_compare', args.variant, args.build)
    table, stats = wcc.load(args.variant, build_dir.name)
    s = catalog.manifest('wfs_corner_compare', args.variant, build_dir.name)['config']['settings']
    coord = s['coord']
    label = f'{args.variant} @ {build_dir.name}'
    if args.out_dir:
        outdir = Path(args.out_dir)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = study_run('aos/cwfs', 'wfs_corner_compare').path / args.variant
        outdir.mkdir(parents=True, exist_ok=True)

    FAMI, triplets, CORNERS, noll = cc.pivot(table, 'fam_interp')
    CWM = cc.pivot(table, 'cwfs_median')[0]
    AZC = np.nanmedian(cc.pivot(table, 'corner_az_deg')[0], axis=2)
    defocus_of = dict(zip(table.corner, table.defocus.fillna('')))
    metrics = {(r.corner, int(r.j)): r._asdict() for r in stats.itertuples(index=False)}
    nZk, nT = len(noll), len(triplets)
    vjs = [int(j) for j in args.val_zernikes.split(',') if int(j) in noll]
    val_keys = [triplets[t] for t in range(0, nT, args.val_stride)]
    val = validation_donuts(args.variant, build_dir, val_keys, coord, s['r_min_deg'], s['r_max_deg'])
    ikw = {k: s[k] for k in ('interp', 'fourier_m', 'gp_bin_deg', 'gp_restarts',
                             'wedge_half_deg', 'min_fam') if k in s}
    print(f'[wfs_corner_compare] {label}: {nT} triplets, corners {CORNERS}')

    out = outdir / 'wfs_corner_compare.pdf'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from lsst.ts.intrinsic.wavefront.common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}
    nr = int(np.ceil(np.sqrt(nZk))); ncpanel = int(np.ceil(nZk / nr))

    with PdfPages(str(out)) as pdf:
        # ---- per-corner scatter (all Zj) ----
        for c, det in enumerate(CORNERS):
            fig, axes = plt.subplots(nr, ncpanel, figsize=(2.5 * ncpanel, 2.5 * nr),
                                     constrained_layout=True, squeeze=False)
            for p, j in enumerate(noll):
                ax = axes.ravel()[p]
                x, y = FAMI[:, c, p], CWM[:, c, p]
                m = metrics[(det, j)]
                ax.scatter(x, y, s=8, alpha=0.4)
                fin = np.isfinite(x) & np.isfinite(y)
                if fin.sum() >= 3:
                    lo, hi = np.nanpercentile(np.concatenate([x[fin], y[fin]]), [1, 99])
                    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.6)
                    xx = np.array([lo, hi]); ax.plot(xx, m['slope'] * xx + m['off'], 'r-', lw=0.8)
                    ax.text(0.04, 0.96, f"r={m['r']:.2f} ρ={m['rho']:.2f}\n"
                            f"s={m['slope']:.2f} b={m['off']:+.3f}\nrms={m['rms']:.3f}",
                            transform=ax.transAxes, va='top', fontsize=6)
                ax.set_title(f'Z{j}', fontsize=8); ax.tick_params(labelsize=6)
            for ax in axes.ravel()[nZk:]:
                ax.axis('off')
            _dl = f' [{defocus_of.get(det, "")}]' if defocus_of.get(det) else ''
            fig.suptitle(f'{det}{_dl}: CWFS median (y) vs FAM-interp (x) [{coord}, µm] — {label}', fontsize=11)
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
            _dl = f' [{defocus_of.get(det, "")}]' if defocus_of.get(det) else ''
            fig.suptitle(f'{det}{_dl}: time history vs image ordinal [{coord}, µm] — {label}', fontsize=11)
            pdf.savefig(fig); plt.close(fig)

        # ---- summary: metric vs Zj, all corners ----
        fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
        panels = [('r', 'Pearson r'), ('rho', 'Spearman ρ'), ('slope', 'Huber slope'),
                  ('off', 'offset [µm]'), ('rms', 'robust RMS [µm]')]
        for ax, (key, lab) in zip(axes.ravel(), panels):
            for c, det in enumerate(CORNERS):
                ax.plot(noll, [metrics[(det, j)][key] for j in noll], '-o', ms=4, label=det)
            ax.set_xlabel('Noll j'); ax.set_ylabel(lab); ax.grid(alpha=0.3)
            if key == 'slope':
                ax.axhline(1, color='k', lw=0.5)
            if key in ('off',):
                ax.axhline(0, color='k', lw=0.5)
        axes.ravel()[len(panels)].axis('off')       # 2x3 grid, 5 panels
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle(f'CWFS vs FAM-interp summary — {label} ({coord})', fontsize=12)
        pdf.savefig(fig); plt.close(fig)

        # ---- validation: Z5-Z8 vs azimuth, every val_stride-th triplet ----
        for key in val_keys:
            t = triplets.index(key)
            faz, fzk, caz, czk = val[key]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            agrid = np.arange(0, 360.5, 2.0)
            for ax, j in zip(axes.ravel(), vjs):
                jp = noll.index(j)
                ax.scatter(faz, fzk[:, jp], s=8, alpha=0.25, color='steelblue', label='FAM donuts')
                ax.scatter(caz, czk[:, jp], s=18, alpha=0.6, color='crimson', marker='s', label='CWFS donuts')
                _, curve = cc.azimuth_interp(faz.values, fzk[:, jp], [0.0], **ikw)
                if curve is not None:
                    ax.plot(agrid, np.asarray(curve(agrid), float).ravel(), '-', color='navy', lw=1,
                            alpha=0.7, label=f'FAM {s["interp"]} fit')
                ax.scatter(AZC[t], FAMI[t, :, jp], s=80, color='navy', marker='P',
                           edgecolors='w', zorder=5, label='FAM interp')
                ax.scatter(AZC[t], CWM[t, :, jp], s=80, color='darkred', marker='X',
                           edgecolors='w', zorder=5, label='CWFS median')
                ax.set_title(f'Z{j} {NOLL_NAMES.get(j, "")}', fontsize=9)
                ax.set_xlabel(f'focal-plane azimuth [deg, {coord}]'); ax.set_ylabel(f'Z{j} [µm]')
                ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90)); ax.grid(alpha=0.3)
            axes.ravel()[0].legend(fontsize=7, loc='best')
            fig.suptitle(f'Validation triplet #{t} day {key[0]} fam_seq {key[1]} '
                         f'(annulus {s["r_min_deg"]}-{s["r_max_deg"]}°) — {label}', fontsize=11)
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
