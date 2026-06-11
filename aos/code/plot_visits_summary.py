#!/usr/bin/env python3
"""Per-param_set visit summary: elevation vs rotator angle, one panel per filter.

Reads ``output/<param_set>/visits.parquet`` and writes
``output/<param_set>/plots/visits_elev_rot_by_band.{pdf,png}`` — a fixed
ugrizy grid (empty bands shown empty so the layout is uniform), each panel a
scatter of visit elevation (deg, from ``alt``) vs rotator angle (deg).

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
BAND_COLORS = {'u': 'tab:purple', 'g': 'tab:green', 'r': 'tab:red',
               'i': 'tab:orange', 'z': 'tab:brown', 'y': 'tab:olive'}


def plot_param_set(ps, visits_path, plots_dir, bands=BANDS):
    df = pq.read_table(visits_path, columns=['alt', 'rotator_angle', 'band']).to_pandas()
    elev = np.degrees(df['alt'].to_numpy(dtype=float))
    rot = df['rotator_angle'].to_numpy(dtype=float)
    band = df['band'].astype(str).to_numpy()

    # Shared axis ranges across panels for direct comparison.
    rlo, rhi = np.nanmin(rot), np.nanmax(rot)
    elo, ehi = np.nanmin(elev), np.nanmax(elev)
    rpad = max(5.0, 0.05 * (rhi - rlo))
    epad = max(2.0, 0.05 * (ehi - elo))

    ncol = 3
    nrow = int(np.ceil(len(bands) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True, layout='constrained')
    axes = np.atleast_1d(axes).ravel()
    counts = {}
    for ax, b in zip(axes, bands):
        m = band == b
        counts[b] = int(m.sum())
        ax.scatter(rot[m], elev[m], s=18, alpha=0.5,
                   color=BAND_COLORS.get(b, 'k'), edgecolors='none')
        ax.set_title(f'{b}-band  (n={counts[b]})', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(rlo - rpad, rhi + rpad)
        ax.set_ylim(elo - epad, ehi + epad)
    for ax in axes[len(bands):]:
        ax.set_visible(False)
    for ax in axes[::ncol]:
        ax.set_ylabel('Elevation [deg]')
    for ax in axes[len(bands) - ncol:len(bands)]:
        ax.set_xlabel('Rotator angle [deg]')

    fig.suptitle(f'{ps} — visit elevation vs rotator by filter  '
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
        plot_param_set(ps, visits, Path(args.output_root) / ps / 'plots')


if __name__ == '__main__':
    main()
