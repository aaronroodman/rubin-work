#!/usr/bin/env python3
"""Per-param_set visit summary: elevation vs rotator angle, one panel per filter.

Reads ``output/<param_set>/visits.parquet`` and writes
``output/<param_set>/plots/visits_elev_rot_by_band.{pdf,png}`` — a fixed ugrizy
grid (empty bands shown empty so the layout is uniform), each panel a 2-D
histogram of visit counts in fixed-width elevation (deg, from ``alt``) ×
rotator-angle (deg) bins (``--elev-bin`` / ``--rot-bin``, default 2.5°), with
the count printed only on non-zero bins.

    python code/plot_visits_summary.py --param-set all
    python code/plot_visits_summary.py --param-set fam_danish_1_0_wep17_3_0_bin2x

Run on the RSP. Over SSH the repo-relative ``output`` symlink does not resolve,
so pass the absolute data path, e.g.
    --output-root /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.utils import text_hist2d, centered_edges  # noqa: E402

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def plot_param_set(ps, visits_path, plots_dir, bands=BANDS,
                   elev_bin=10.0, rot_bin=15.0):
    df = pq.read_table(visits_path, columns=['alt', 'rotator_angle', 'band']).to_pandas()
    elev = np.degrees(df['alt'].to_numpy(dtype=float))
    rot = df['rotator_angle'].to_numpy(dtype=float)
    band = df['band'].astype(str).to_numpy()

    # Bins centered on the setpoints (rotator multiples of rot_bin, elevation
    # multiples of elev_bin), shared across all panels so they line up.
    rot_edges = centered_edges(np.nanmin(rot), np.nanmax(rot), rot_bin)
    elev_edges = centered_edges(np.nanmin(elev), np.nanmax(elev), elev_bin)
    nr, ne = len(rot_edges) - 1, len(elev_edges) - 1

    ncol = 3
    nrow = int(np.ceil(len(bands) / ncol))
    # Size each panel so every bin gets enough room for a 3-digit number
    # (~0.22 in/bin) — no crowding regardless of bin width.
    per_bin = 0.22
    pw = max(3.5, nr * per_bin)
    ph = max(2.8, ne * per_bin)
    fig, axes = plt.subplots(nrow, ncol, figsize=(pw * ncol, ph * nrow),
                             sharex=True, sharey=True, layout='constrained')
    axes = np.atleast_1d(axes).ravel()
    counts = {}
    for ax, b in zip(axes, bands):
        m = band == b
        counts[b] = int(m.sum())
        # ROOT TEXT style: numbers at non-empty bin centers, white bg, dotted grid.
        text_hist2d(rot[m], elev[m], ax=ax, xbins=rot_edges, ybins=elev_edges,
                    fontsize=7)
        ax.set_title(f'{b}-band  (n={counts[b]})', fontsize=11)
    for ax in axes[len(bands):]:
        ax.set_visible(False)
    for ax in axes[::ncol]:
        ax.set_ylabel('Elevation [deg]')
    for ax in axes[len(bands) - ncol:len(bands)]:
        ax.set_xlabel('Rotator angle [deg]')

    fig.suptitle(f'{ps} — visit counts by elevation & rotator, per filter  '
                 f'(N={len(df)}, {elev_bin:g}°×{rot_bin:g}° bins)', fontsize=13)

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    base = plots_dir / 'visits_elev_rot_by_band'
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    fig.savefig(f'{base}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'{ps}: N={len(df)}  bands={counts}  -> {base}.pdf/.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True,
                    help="param_set name, or 'all' (from snake_config.yaml)")
    ap.add_argument('--output-root', default='output',
                    help='Root of the output/<param_set> tree (default: %(default)s)')
    ap.add_argument('--config', default=None,
                    help='snake_config.yaml path (default: ../snake_config.yaml)')
    ap.add_argument('--elev-bin', type=float, default=10.0,
                    help='Elevation bin width in deg; bins centered on multiples '
                         '(default: %(default)s -> centers 30,40,...,70)')
    ap.add_argument('--rot-bin', type=float, default=15.0,
                    help='Rotator bin width in deg; bins centered on multiples '
                         '(default: %(default)s -> centers -60,-45,...,60)')
    args = ap.parse_args()

    aos_dir = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config) if args.config else aos_dir / 'snake_config.yaml'
    names = (list(yaml.safe_load(open(cfg_path))['param_sets'])
             if args.param_set == 'all' else [args.param_set])

    for ps in names:
        visits = Path(args.output_root) / ps / 'visits.parquet'
        if not visits.exists():
            print(f'{ps}: SKIP — no visits.parquet at {visits}')
            continue
        plot_param_set(ps, visits, Path(args.output_root) / ps / 'plots',
                       elev_bin=args.elev_bin, rot_bin=args.rot_bin)


if __name__ == '__main__':
    main()
