"""Validation plots of the official MIW maps exported by export_official_miw.py.

  1. Z4 (defocus): OCS map | CCS map.  The CCS panel is drawn per-detector
     footprint, so the per-CCD focal-plane-height Z4 steps are visible (the
     focus diversity that breaks the Z4<->seeing degeneracy).
  2. OCS maps of Z5..Z11 (astig, coma, trefoil, spherical).

Writes output/miw_z4_ocs_ccs.png and output/miw_ocs_z5_z11.png.
Usage: python plot_miw_maps.py [--ocs data/...ocs.parquet] [--ccs ...ccs.parquet]
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

FP_R = 1.75


def _vlim(v, pct=98.0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return max(float(np.percentile(np.abs(v), pct)), 1e-6) if v.size else 1.0


def _map(ax, x, y, v, title, R, s=6):
    vl = _vlim(v)
    fin = np.isfinite(v)
    sc = ax.scatter(x[fin], y[fin], c=v[fin], s=s, cmap='RdBu_r',
                    vmin=-vl, vmax=vl, linewidths=0)
    ax.add_patch(Circle((0, 0), FP_R, fill=False, ls='--', color='k',
                        lw=0.5, alpha=0.4))
    ax.set_aspect('equal'); ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=6)
    ax.set_xlabel('x [deg]', fontsize=7); ax.set_ylabel('y [deg]', fontsize=7)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ocs', default='data/intrinsic_official_ocs.parquet')
    ap.add_argument('--ccs', default='data/intrinsic_official_ccs.parquet')
    ap.add_argument('--out-dir', default='output')
    args = ap.parse_args()

    ocs = pd.read_parquet(args.ocs)
    ccs = pd.read_parquet(args.ccs)
    ox, oy = ocs['thx_deg'].to_numpy(), ocs['thy_deg'].to_numpy()
    cx, cy = ccs['thx_deg'].to_numpy(), ccs['thy_deg'].to_numpy()
    Rocs = float(np.hypot(ox, oy).max()) * 1.03
    Rccs = float(np.hypot(cx, cy).max()) * 1.03

    # ---- 1. Z4 OCS vs CCS ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    s0 = _map(ax[0], ox, oy, ocs['Z4_OCS'].to_numpy(), 'Z4 OCS [µm]', Rocs)
    fig.colorbar(s0, ax=ax[0], shrink=0.8)
    s1 = _map(ax[1], cx, cy, ccs['Z4_CCS'].to_numpy(),
              'Z4 CCS [µm] (per-CCD, incl. focal-plane height)', Rccs, s=9)
    fig.colorbar(s1, ax=ax[1], shrink=0.8)
    fig.suptitle('Official MIW — Z4 (defocus): OCS vs CCS')
    fig.tight_layout()
    fig.savefig(f'{args.out_dir}/miw_z4_ocs_ccs.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- 2. OCS maps Z5..Z11 ----
    js = [j for j in range(5, 12) if f'Z{j}_OCS' in ocs.columns]
    cols = 4
    rows = int(np.ceil(len(js) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.7 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a, j in zip(axs, js):
        sc = _map(a, ox, oy, ocs[f'Z{j}_OCS'].to_numpy(), f'Z{j} OCS [µm]', Rocs)
        fig.colorbar(sc, ax=a, shrink=0.8)
    for a in axs[len(js):]:
        a.axis('off')
    fig.suptitle('Official MIW — OCS maps Z5–Z11')
    fig.tight_layout()
    fig.savefig(f'{args.out_dir}/miw_ocs_z5_z11.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out_dir}/miw_z4_ocs_ccs.png and '
          f'{args.out_dir}/miw_ocs_z5_z11.png')


if __name__ == '__main__':
    main()
