#!/usr/bin/env python3
"""Per-param_set visit summary: elevation vs rotator angle, one panel per filter.

Reads ``output/<param_set>/visits.parquet`` and writes
``output/<param_set>/plots/visits_elev_rot_by_band.{pdf,png}`` — a fixed ugrizy
grid (empty bands shown empty so the layout is uniform), each panel a boxed 2-D
histogram of visit counts binned to the elevation (deg, from ``alt``, rounded to
``--elev-round``) and rotator-angle (deg, rounded to ``--rot-round``) setpoints,
with the count written in each non-empty box.

    python code/plot_visits_summary.py --param-set all
    python code/plot_visits_summary.py --param-set fam_danish_1_0_wep17_3_0_bin2x

Run on the RSP. Over SSH the repo-relative ``output`` symlink does not resolve,
so pass the absolute data path, e.g.
    --output-root /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output
"""
import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def plot_param_set(ps, visits_path, plots_dir, bands=BANDS,
                   elev_round=5.0, rot_round=15.0):
    df = pq.read_table(visits_path, columns=['alt', 'rotator_angle', 'band']).to_pandas()
    elev = np.degrees(df['alt'].to_numpy(dtype=float))
    rot = df['rotator_angle'].to_numpy(dtype=float)
    band = df['band'].astype(str).to_numpy()

    # Snap to setpoints; build a categorical grid shared across all panels so
    # the boxes line up band-to-band.
    elev_s = (np.round(elev / elev_round) * elev_round).astype(int)
    rot_s = (np.round(rot / rot_round) * rot_round).astype(int)
    elev_cats = np.array(sorted(np.unique(elev_s)))          # rows (y)
    rot_cats = np.array(sorted(np.unique(rot_s)))            # cols (x)
    ei = {v: i for i, v in enumerate(elev_cats)}
    ri = {v: i for i, v in enumerate(rot_cats)}
    ne, nr = len(elev_cats), len(rot_cats)
    xe, ye = np.arange(nr + 1), np.arange(ne + 1)            # cell edges

    ncol = 3
    nrow = int(np.ceil(len(bands) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.8 * nrow),
                             layout='constrained')
    axes = np.atleast_1d(axes).ravel()
    counts = {}
    for ax, b in zip(axes, bands):
        m = band == b
        counts[b] = int(m.sum())
        M = np.zeros((ne, nr))
        for e, r in zip(elev_s[m], rot_s[m]):
            M[ei[e], ri[r]] += 1
        vmax = max(M.max(), 1)
        ax.pcolormesh(xe, ye, M, cmap='viridis', vmin=0, vmax=vmax,
                      edgecolors='white', linewidth=0.5)
        for i in range(ne):
            for j in range(nr):
                if M[i, j] > 0:
                    ax.text(j + 0.5, i + 0.5, f'{int(M[i, j])}',
                            ha='center', va='center', fontsize=7, fontweight='bold',
                            color='black' if M[i, j] > 0.6 * vmax else 'white')
        ax.set_xticks(np.arange(nr) + 0.5)
        ax.set_xticklabels(rot_cats, fontsize=7)
        ax.set_yticks(np.arange(ne) + 0.5)
        ax.set_yticklabels(elev_cats, fontsize=8)
        ax.set_xlim(0, nr)
        ax.set_ylim(0, ne)
        ax.set_aspect('auto')
        ax.set_title(f'{b}-band  (n={counts[b]})', fontsize=11)
    for ax in axes[len(bands):]:
        ax.set_visible(False)
    for ax in axes[::ncol]:
        ax.set_ylabel('Elevation [deg]')
    for ax in axes[len(bands) - ncol:len(bands)]:
        ax.set_xlabel('Rotator angle [deg]')

    fig.suptitle(f'{ps} — visit counts by elevation & rotator, per filter  '
                 f'(N={len(df)})', fontsize=13)

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
    ap.add_argument('--elev-round', type=float, default=5.0,
                    help='Elevation bin / setpoint rounding in deg (default: %(default)s)')
    ap.add_argument('--rot-round', type=float, default=15.0,
                    help='Rotator bin / setpoint rounding in deg (default: %(default)s)')
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
                       elev_round=args.elev_round, rot_round=args.rot_round)


if __name__ == '__main__':
    main()
