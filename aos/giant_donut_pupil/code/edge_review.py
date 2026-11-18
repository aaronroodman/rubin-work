"""Review of the rim (edge) finder and the spider-bin cut on one giant-donut side.

Four pages per side, for one or more sides of a night2_pipeline fit directory:

1. The data stamp at full resolution with the data edges (white) and the fit c model edges
   (coloured by defining mask element), the spider-flagged bins shaded.
2. The stamp unwrapped about the fit-a center (CCS azimuth against radius), divided by
   each bin's flat top, with the 50 % contour, the data and model edges and the vane
   shadow of the fit c model.
3. Radial profiles in six bins: data and model, flat top, 50 % level, the crossings found;
   each panel runs 15 pixels outside the edge and 25 pixels into the flat top.
4. Spider cut per edge: bins flagged now (any vane shadow anywhere in the ring, ±2 bins)
   against bins where the shadow comes within `PAD_PIX` of that edge.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/edge_review.py --run 20261118_night2_20261107 \\
        --fit-dir v402s --key 388_S4848323907389759872:extra \\
        --key 366_S4848323907389759872:intra
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import psi_edges_report as per  # noqa: E402
import psi_gallery as pg  # noqa: E402
import psi_labels as pl  # noqa: E402
import robust_edges as rob  # noqa: E402

PAD_PIX = 6.0
SHADOW_FRAC = 0.8
FWHM_ARCSEC = 1.0
PROFILE_BINS_DEG = (1.0, 41.0, 91.0, 181.0, 251.0, 331.0)


def unwrap(img, cx, cy, r_max, dr=0.5, daz=0.5):
    """Bilinear (azimuth, radius) resampling about (cx, cy); CCS azimuth in deg."""
    az = np.arange(0.0, 360.0, daz) + daz / 2
    r = np.arange(0.0, r_max, dr) + dr / 2
    A, R = np.meshgrid(np.radians(az), r, indexing='ij')
    x, y = cx + R * np.cos(A), cy + R * np.sin(A)
    return az, r, map_coordinates(img, [y, x], order=1, mode='constant', cval=np.nan)


def shadow_map(row, params, n):
    """Lit pixels the vanes darken by more than 1 - `SHADOW_FRAC`, fit c model."""
    without = {k: v for k, v in params.items() if not k.startswith('Spider')}
    w = pl.fit_model(row, params, FWHM_ARCSEC)
    wo = pl.fit_model(row, without, FWHM_ARCSEC)
    lit = wo > 0.5 * np.median(wo[wo > 0])
    return lit & (w < SHADOW_FRAC * wo), w, wo


def per_edge_flags(shadow, cx, cy, edge_r):
    """Bins where a shadow pixel lies within `PAD_PIX` of the edge radius of that bin."""
    n = shadow.shape[0]
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - cx, yy - cy)
    phi = np.mod(np.degrees(np.arctan2(yy - cy, xx - cx)), 360.0)
    b = np.minimum((phi / (360 / rob.NBIN)).astype(int), rob.NBIN - 1)
    near = shadow & np.isfinite(edge_r[b]) & (np.abs(r - edge_r[b]) <= PAD_PIX)
    hit = np.zeros(rob.NBIN, bool)
    hit[np.unique(b[near])] = True
    return hit


def review(pdf, run, fit_dir, key):
    c, b, mask_run = pl._load(run, fit_dir)
    pl._init(*map(str, mask_run))
    row = c[c.key == key].iloc[0]
    g = b[b.key == key].sort_values('bin').reset_index(drop=True)
    img, msk = pg.stamp(pl.STUDY_OUT / run, key)
    params = pl.mask_for(row)
    shadow, model, _ = shadow_map(row, params, img.shape[0])
    cx, cy = row.cx_a, row.cy_a
    noise = float(row.noise_e)
    valid = ~msk
    prof, rad, top, top_snr, top_n, bad, cnt = rob.profiles(img, valid, cx, cy, row.r_out_pix,
                                                           noise)
    mprof = rob.profiles(model, valid, cx, cy, row.r_out_pix, noise)
    az_bin = g.az_ccs_deg.to_numpy()
    lab = None
    for p in sorted(pl.STUDY_OUT.glob(f'*_psi_labels_{run}_{fit_dir}')):
        lab = pd.read_parquet(p / 'labels.parquet')
    if lab is not None:
        lab = lab[lab.key == key].set_index('bin')
    fit_in = np.where(g.r_inner_c_pix.notna(), g.r_inner_c_pix, np.nan)
    fit_out = np.where(g.r_outer_c_pix.notna(), g.r_outer_c_pix, np.nan)
    # expected edges where the finder rejected the bin: the model's harmonic fit
    m_e = rob.robust_edges(model, valid, cx, cy, row.r_out_pix, noise,
                           np.zeros(rob.NBIN, bool), np.zeros(rob.NBIN, bool))
    new_in = per_edge_flags(shadow, cx, cy, m_e['fit_inner'])
    new_out = per_edge_flags(shadow, cx, cy, m_e['fit_outer'])
    now = g.spider.to_numpy(bool)
    now = now | np.roll(now, 1) | np.roll(now, -1)
    title = (f'{run} {fit_dir} {key}: field {row.field_r_deg:.3f} deg, OCS az '
             f'{row.field_az_ocs_deg:.0f} deg, rtp {row.rtp_deg:.2f} deg')
    # 1. full-resolution stamp
    fig, ax = plt.subplots(figsize=(11, 11))
    vmax = np.nanpercentile(img[valid], 99.0)
    ax.imshow(np.where(msk, np.nan, img), origin='lower', cmap='gray', vmin=0, vmax=vmax,
              interpolation='none')
    a = np.radians(az_bin)
    for e in ('inner', 'outer'):
        cols = ['0.5'] * len(g)
        if lab is not None:
            cols = [per.COLOURS.get(lab[f'element_{e}'].get(k, ''), 'k') for k in g.bin]
        rr = g[f'r_{e}_c_pix'].to_numpy()
        ax.scatter(cx + rr * np.cos(a), cy + rr * np.sin(a), c=cols, s=4, lw=0)
        rr = g[f'r_{e}_data_pix'].to_numpy()
        ax.plot(cx + rr * np.cos(a), cy + rr * np.sin(a), '.', color='w', ms=1.5)
    for k in np.nonzero(now)[0]:
        t = np.radians([az_bin[k] - 1, az_bin[k] + 1])
        ax.fill([cx, cx + 380 * np.cos(t[0]), cx + 380 * np.cos(t[1])],
                [cy, cy + 380 * np.sin(t[0]), cy + 380 * np.sin(t[1])],
                color='yellow', alpha=0.12, lw=0)
    ax.set_title(title + '\nwhite: data edges; coloured: fit c model edges by element; '
                 'yellow: bins cut as spider now', fontsize=9)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)

    # 2. unwrapped, normalised by each bin's flat top
    azu, ru, U = unwrap(np.where(msk, np.nan, img), cx, cy, 1.15 * row.r_out_pix)
    kb = np.minimum((azu / 2.0).astype(int), rob.NBIN - 1)
    Un = U / top[kb][:, None]
    _, _, S = unwrap(shadow.astype(float), cx, cy, 1.15 * row.r_out_pix)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for ax, (lo, hi) in zip(axes, ((0.45, 0.80), (0.85, 1.12))):
        sel = (ru > lo * row.r_out_pix) & (ru < hi * row.r_out_pix)
        ax.imshow(Un[:, sel].T, origin='lower', aspect='auto', cmap='gray', vmin=0, vmax=1.2,
                  extent=(0, 360, ru[sel][0], ru[sel][-1]), interpolation='none')
        ax.contour(azu, ru[sel], Un[:, sel].T, levels=[0.5], colors='tab:orange',
                   linewidths=0.6)
        ax.contour(azu, ru[sel], S[:, sel].T, levels=[0.5], colors='magenta', linewidths=0.6)
        for e, col in (('inner', 'tab:cyan'), ('outer', 'tab:cyan')):
            ax.plot(az_bin, g[f'r_{e}_c_pix'], '.', color=col, ms=3)
            ax.plot(az_bin, g[f'r_{e}_data_pix'], '.', color='w', ms=2)
        ax.set_ylabel('radius from fit-a center (pixels)')
    axes[1].set_xlabel('CCS azimuth about the donut center (deg)')
    axes[0].set_title(title + '\ndata / bin flat top (0.70-0.90 r_out median); orange: 50 % '
                      'contour; white: data edges; cyan: model edges; magenta: vane shadow '
                      f'(model, vanes darken > {100 * (1 - SHADOW_FRAC):.0f} %)', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # 3. radial profiles
    fig, axes = plt.subplots(len(PROFILE_BINS_DEG), 2, figsize=(12, 3 * len(PROFILE_BINS_DEG)))
    for i, azd in enumerate(PROFILE_BINS_DEG):
        k = int(azd / 2.0)
        for j, e in enumerate(('inner', 'outer')):
            ax = axes[i, j]
            rd = g.r_inner_data_pix[k] if e == 'inner' else g.r_outer_data_pix[k]
            rm = g.r_inner_c_pix[k] if e == 'inner' else g.r_outer_c_pix[k]
            ctr = m_e[f'fit_{e}'][k]
            # 25 pixels on the flat-top side, 15 outside the donut
            lo, hi = (ctr - 15, ctr + 25) if e == 'inner' else (ctr - 25, ctr + 15)
            w = (rad > lo) & (rad < hi)
            ax.plot(rad[w], prof[k][w], '.', color='k', ms=3, label='data, 0.5-pixel steps')
            ax.plot(rad[w], rob._smooth(prof[k])[w], '-', color='k', lw=0.8,
                    label='data, smoothed 1.5 pixels')
            ax.plot(rad[w], rob._smooth(mprof[0][k])[w], '-', color='tab:red', lw=0.8,
                    label='fit c model, smoothed')
            ax.axhline(top[k], color='0.5', lw=0.6, ls=':', label='data flat top')
            ax.axhline(0.5 * top[k], color='tab:orange', lw=0.6, label='50 % of flat top')
            for v, col in ((rd, 'k'), (rm, 'tab:red')):
                if np.isfinite(v):
                    ax.axvline(v, color=col, lw=0.6, ls='--')
            ax.set_title(f'bin {k} (CCS az {az_bin[k]:.0f} deg), {e} edge: data '
                         f'{rd:.2f}, model {rm:.2f} pixels; spider cut now: {bool(now[k])}',
                         fontsize=8)
            ax.set_xlabel('radius (pixels)')
            ax.set_ylabel('electrons per pixel')
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.tight_layout()
    pdf.savefig(fig, dpi=120)
    plt.close(fig)

    # 4. spider cut per edge
    fig, ax = plt.subplots(figsize=(14, 4))
    for y, f, lab_ in ((3, now, 'cut now (shadow anywhere in the ring, ±2 bins)'),
                       (2, new_in, f'inner: shadow within {PAD_PIX:.0f} pixels of the edge'),
                       (1, new_out, f'outer: shadow within {PAD_PIX:.0f} pixels of the edge')):
        ax.scatter(az_bin[f], np.full(f.sum(), y), marker='|', s=200, label=f'{lab_}, '
                   f'{int(f.sum())} of {rob.NBIN} bins')
    ax.set_yticks([1, 2, 3], ['outer, per edge', 'inner, per edge', 'now'])
    ax.set_xlabel('CCS azimuth about the donut center (deg)')
    ax.set_xlim(0, 360)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(title + '\nspider-cut bins', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    summary = dict(key=key, bins_cut_now=int(now.sum()), inner_cut_per_edge=int(new_in.sum()),
                   outer_cut_per_edge=int(new_out.sum()))
    print(summary, flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run', required=True)
    p.add_argument('--fit-dir', required=True)
    p.add_argument('--key', action='append', required=True, help='repeatable')
    a = p.parse_args()
    from rubinwork.common import study_run
    out = study_run('aos/giant_donut_pupil', 'edge_review').path
    star = a.key[0].split(':')[0].split('_', 1)[-1]
    name = f'edge_review_{star}.pdf' if all(star in k for k in a.key) else \
        f'edge_review_{a.key[0].replace(":", "_")}.pdf'
    with PdfPages(out / name) as pdf:
        for k in a.key:
            review(pdf, a.run, a.fit_dir, k)
    print('wrote', out / name, flush=True)


if __name__ == '__main__':
    main()
