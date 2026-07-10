"""Validation of the empirical per-cell scatter used as the fit weights.

For each in-focus visit, bin the PSF-star moments with the SAME per-detector
sub-CCD binning as the fit (cell_deg / min_n from config.yaml) and plot:
  1. focal-plane maps (DVCS: x_field vertical, y_field horizontal) of the
     per-cell scatter  err = 1.253 * std / sqrt(n)  for each fit moment, plus
     the star-count-per-cell map;
  2. histograms of that per-cell scatter for each fit moment, plus the
     star-count histogram.

Reads only the psfmoments parquet (no Butler, no fit result).  Use it to judge
whether the per-cell counts are high enough or cell_deg should be widened.

Usage: python plot_bin_scatter.py [seqs=25,28] [day=20260513] [cell=0.10]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from config import load_config
from jax_optatmo import MOMENT_LABELS
import data_fit

FP_R = 1.75
DAY = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('day=')), 20260513)


def visit_of(seq):
    return int(f'{DAY}{seq:05d}')


def _dvcs(ax, lim, title):
    ax.add_patch(Circle((0, 0), FP_R, fill=False, ls='--', color='k', lw=0.5, alpha=0.5))
    ax.set_aspect('equal'); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=6)
    ax.set_xlabel('y_field [deg]', fontsize=7); ax.set_ylabel('x_field [deg]', fontsize=7)


def run(seq, cfg, moments, cell_deg, min_n, out_dir):
    prep = data_fit.load_and_prep(f'data/psfmoments_{visit_of(seq)}.parquet',
                                  sign=1, rot_deg=0.0)
    binned = data_fit.bin_grid(prep, cell_deg=cell_deg, min_n=min_n)
    thx, thy, err, n = binned['thx'], binned['thy'], binned['err'], binned['n']
    lim = 1.9
    idx = [MOMENT_LABELS.index(m) for m in moments]
    print(f'  seq {seq}: {len(n)} cells, stars/cell median {np.median(n):.0f} '
          f'(min {n.min()}, max {n.max()}); '
          f'median per-cell scatter '
          f'{ {m: round(float(np.median(err[:, i])), 4) for m, i in zip(moments, idx)} }')

    panels = moments + ['n_stars']
    cols = 3
    rows = int(np.ceil(len(panels) / cols))

    # ---- 1. FoV maps of the per-cell scatter ----
    fig, axs = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.0 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a, p in zip(axs, panels):
        if p == 'n_stars':
            sc = a.scatter(thy, thx, c=n, s=14, marker='s', cmap='viridis')
            _dvcs(a, lim, 'stars per cell'); fig.colorbar(sc, ax=a, shrink=0.8)
        else:
            v = err[:, MOMENT_LABELS.index(p)]
            vhi = np.percentile(v[np.isfinite(v)], 98)
            sc = a.scatter(thy, thx, c=v, s=14, marker='s', cmap='magma',
                           vmin=0, vmax=vhi)
            _dvcs(a, lim, f'σ({p}) per cell'); fig.colorbar(sc, ax=a, shrink=0.8)
    for a in axs[len(panels):]:
        a.axis('off')
    fig.suptitle(f'20260513 seq={seq}: per-cell scatter over FoV '
                 f'(cell={cell_deg}°, min_n={min_n})  (DVCS)')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/binscatter_map_{seq}.png', dpi=115, bbox_inches='tight')
    plt.close(fig)

    # ---- 2. histograms of the per-cell scatter ----
    fig, axs = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a, p in zip(axs, panels):
        if p == 'n_stars':
            a.hist(n, bins=np.arange(n.min() - 0.5, n.max() + 1.5), color='C0')
            a.axvline(np.median(n), color='r', lw=1, ls='--',
                      label=f'median {np.median(n):.0f}')
            a.set_title('stars per cell', fontsize=9); a.legend(fontsize=7)
        else:
            v = err[:, MOMENT_LABELS.index(p)]
            v = v[np.isfinite(v)]
            a.hist(v, bins=40, range=(0, np.percentile(v, 99)), color='C3')
            a.axvline(np.median(v), color='k', lw=1, ls='--',
                      label=f'median {np.median(v):.4f}')
            a.set_title(f'σ({p})', fontsize=9); a.legend(fontsize=7)
        a.tick_params(labelsize=6)
    for a in axs[len(panels):]:
        a.axis('off')
    fig.suptitle(f'20260513 seq={seq}: per-cell scatter histograms '
                 f'(cell={cell_deg}°, min_n={min_n})')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/binscatter_hist_{seq}.png', dpi=115, bbox_inches='tight')
    plt.close(fig)
    print(f'  seq {seq}: wrote binscatter_map_{seq}.png, binscatter_hist_{seq}.png')


def main():
    cfg = load_config('config.yaml')
    moments = cfg['fit']['moments']
    seqs = next((a.split('=')[1] for a in sys.argv if a.startswith('seqs=')), None)
    seqs = [int(s) for s in seqs.split(',')] if seqs else [25, 28]
    cell = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('cell=')),
                cfg['fit'].get('cell_deg', 0.10))
    min_n = cfg['fit'].get('min_n', 3)
    out_dir = next((a.split('=')[1] for a in sys.argv if a.startswith('out=')), 'output')
    for seq in seqs:
        run(seq, cfg, moments, cell, min_n, out_dir)


if __name__ == '__main__':
    main()
