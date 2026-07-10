"""Validation plots of the official MIW maps exported by export_official_miw.py.

  1. Z4 (defocus): OCS map | CCS map, both on a dense focal-plane grid.
     - OCS: global linear interp on a 71x71 grid over [-lim, lim].
     - CCS: 151x151 grid; each grid point is assigned to the detector of its
       nearest CCS sample and interpolated within that detector, so the per-CCD
       focal-plane-height Z4 steps stay crisp (a global interp would smear them).
  2. OCS maps of Z5..Z11 (astig, coma, trefoil, spherical) on the 71x71 grid.

Writes output/miw_z4_ocs_ccs.png and output/miw_ocs_z5_z11.png.
Usage:
    python plot_miw_maps.py [--ocs data/...ocs.parquet] [--ccs ...ccs.parquet]
        [--ocs-n 71] [--ccs-n 151] [--lim 1.5] [--out-dir output]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

FP_R = 1.75


def _vlim(v, pct=99.0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return max(float(np.percentile(np.abs(v), pct)), 1e-6) if v.size else 1.0


def _grid(n, lim):
    ax = np.linspace(-lim, lim, n)
    gx, gy = np.meshgrid(ax, ax)
    return gx, gy


def ocs_dense(ocs, col, n, lim):
    """Global linear interp of an OCS map column onto an n x n grid."""
    pts = ocs[['thx_deg', 'thy_deg']].to_numpy(float)
    v = ocs[col].to_numpy(float)
    m = np.isfinite(v)
    gx, gy = _grid(n, lim)
    Z = LinearNDInterpolator(pts[m], v[m])(np.column_stack([gx.ravel(), gy.ravel()]))
    return Z.reshape(gx.shape)


def ccs_dense(ccs, col, n, lim, max_dist=0.2):
    """Per-detector dense CCS map: assign each grid point to the detector of its
    nearest sample, interpolate within that detector (nearest-sample fallback
    outside the footprint hull); blank points farther than `max_dist` from any
    sample (beyond focal-plane coverage)."""
    pts = ccs[['thx_deg', 'thy_deg']].to_numpy(float)
    det = ccs['detector'].to_numpy()
    v = ccs[col].to_numpy(float)
    gx, gy = _grid(n, lim)
    gg = np.column_stack([gx.ravel(), gy.ravel()])
    dist, idx = cKDTree(pts).query(gg)
    gdet = det[idx]
    Z = np.full(len(gg), np.nan)
    for d in np.unique(gdet):
        gm = gdet == d
        sm = det == d
        sp, sv = pts[sm], v[sm]
        vals = LinearNDInterpolator(sp, sv)(gg[gm])
        bad = ~np.isfinite(vals)
        if bad.any():                              # outside this CCD's hull
            _, di = cKDTree(sp).query(gg[gm][bad])
            vals[bad] = sv[di]
        Z[gm] = vals
    Z[dist > max_dist] = np.nan                    # beyond coverage
    return Z.reshape(gx.shape)


def _imshow(ax, Z, lim, title):
    vl = _vlim(Z)
    im = ax.imshow(Z, origin='lower', extent=[-lim, lim, -lim, lim],
                   cmap='RdBu_r', vmin=-vl, vmax=vl, interpolation='nearest')
    ax.add_patch(Circle((0, 0), FP_R, fill=False, ls='--', color='k',
                        lw=0.5, alpha=0.4))
    ax.set_aspect('equal'); ax.set_title(title, fontsize=9)
    ax.set_xlabel('x [deg]', fontsize=7); ax.set_ylabel('y [deg]', fontsize=7)
    ax.tick_params(labelsize=6)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ocs', default='data/intrinsic_official_ocs.parquet')
    ap.add_argument('--ccs', default='data/intrinsic_official_ccs.parquet')
    ap.add_argument('--ocs-n', type=int, default=71)
    ap.add_argument('--ccs-n', type=int, default=151)
    ap.add_argument('--lim', type=float, default=1.75)
    ap.add_argument('--out-dir', default='output')
    args = ap.parse_args()

    ocs = pd.read_parquet(args.ocs)
    ccs = pd.read_parquet(args.ccs)

    # ---- 1. Z4 OCS vs CCS (dense) ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    i0 = _imshow(ax[0], ocs_dense(ocs, 'Z4_OCS', args.ocs_n, args.lim),
                 args.lim, f'Z4 OCS [µm] ({args.ocs_n}x{args.ocs_n})')
    fig.colorbar(i0, ax=ax[0], shrink=0.8)
    i1 = _imshow(ax[1], ccs_dense(ccs, 'Z4_CCS', args.ccs_n, args.lim),
                 args.lim, f'Z4 CCS [µm] ({args.ccs_n}x{args.ccs_n}, '
                 f'per-CCD, incl. focal-plane height)')
    fig.colorbar(i1, ax=ax[1], shrink=0.8)
    fig.suptitle('Official MIW — Z4 (defocus): OCS vs CCS')
    fig.tight_layout()
    fig.savefig(f'{args.out_dir}/miw_z4_ocs_ccs.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- 2. OCS maps Z5..Z11 (dense) ----
    js = [j for j in range(5, 12) if f'Z{j}_OCS' in ocs.columns]
    cols = 4
    rows = int(np.ceil(len(js) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.7 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a, j in zip(axs, js):
        im = _imshow(a, ocs_dense(ocs, f'Z{j}_OCS', args.ocs_n, args.lim),
                     args.lim, f'Z{j} OCS [µm]')
        fig.colorbar(im, ax=a, shrink=0.8)
    for a in axs[len(js):]:
        a.axis('off')
    fig.suptitle(f'Official MIW — OCS maps Z5–Z11 ({args.ocs_n}x{args.ocs_n})')
    fig.tight_layout()
    fig.savefig(f'{args.out_dir}/miw_ocs_z5_z11.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out_dir}/miw_z4_ocs_ccs.png and '
          f'{args.out_dir}/miw_ocs_z5_z11.png')


if __name__ == '__main__':
    main()
