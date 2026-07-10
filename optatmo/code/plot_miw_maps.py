"""Validation plots of the official MIW calibration, using the calib object's
OWN interpolators and field tables (RUN ON USDF -- reads the Butler).

The ip_isr ``IntrinsicZernikes`` calib exposes:
  * ``interpolator_ocs`` / ``interpolator`` -- scipy LinearNDInterpolator over
    the OCS / (per-detector) CCS sample points; return (N, n_noll) in microns,
    NaN outside the sample convex hull (no binning / no extension -- the reach
    is set by how far the calib's own field_x/field_y samples go).
  * ``field_x_ocs, field_y_ocs, values_ocs`` (OCS, shared across detectors),
    ``field_x, field_y, values`` (per-detector CCS), ``noll_indices``.

Plots:
  1. Z4 OCS (calib.interpolator_ocs) | Z4 CCS (per-detector calib.interpolator,
     assembled over the focal plane; per-CCD height steps preserved).
  2. OCS maps Z5..Z11 (calib.interpolator_ocs).
Sample points (field_x/field_y from the tables) are overplotted faintly.

Usage:
    python plot_miw_maps.py [--repo /repo/main] [--collection ...] [--filter i_39]
        [--ocs-n 71] [--ccs-n 151] [--lim 1.75] [--out-dir output]
"""
import argparse

import numpy as np
from lsst.daf.butler import Butler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree

FP_R = 1.75


def _vlim(v, pct=99.0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return max(float(np.percentile(np.abs(v), pct)), 1e-6) if v.size else 1.0


def _grid(n, lim):
    ax = np.linspace(-lim, lim, n)
    gx, gy = np.meshgrid(ax, ax)
    return gx, gy, np.column_stack([gx.ravel(), gy.ravel()])


def _imshow(ax, Z, lim, title, sample_xy=None):
    vl = _vlim(Z)
    im = ax.imshow(Z, origin='lower', extent=[-lim, lim, -lim, lim],
                   cmap='RdBu_r', vmin=-vl, vmax=vl, interpolation='nearest')
    if sample_xy is not None:
        ax.plot(sample_xy[0], sample_xy[1], '.', ms=0.5, color='k', alpha=0.15)
    ax.add_patch(Circle((0, 0), FP_R, fill=False, ls='--', color='k',
                        lw=0.5, alpha=0.5))
    ax.set_aspect('equal'); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=6)
    ax.set_xlabel('x [deg]', fontsize=7); ax.set_ylabel('y [deg]', fontsize=7)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/repo/main')
    ap.add_argument('--collection',
                    default='u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
    ap.add_argument('--filter', default='i_39')
    ap.add_argument('--instrument', default='LSSTCam')
    ap.add_argument('--ocs-n', type=int, default=71)
    ap.add_argument('--ccs-n', type=int, default=151)
    ap.add_argument('--lim', type=float, default=1.75)
    ap.add_argument('--max-dist', type=float, default=0.2,
                    help='blank CCS grid points farther than this (deg) from any '
                         'sample (beyond focal-plane coverage)')
    ap.add_argument('--out-dir', default='output')
    args = ap.parse_args()

    b = Butler(args.repo)
    refs = list(b.registry.queryDatasets(
        'intrinsicZernikes', collections=args.collection,
        where=f"instrument='{args.instrument}' and physical_filter='{args.filter}'",
        findFirst=True))
    dets = sorted({r.dataId['detector'] for r in refs})
    if not dets:
        raise SystemExit(f'no intrinsicZernikes in {args.collection} / {args.filter}')
    print(f'loading {len(dets)} detector calibs ...')
    calibs = {d: b.get('intrinsicZernikes', collections=args.collection,
                       instrument=args.instrument, detector=d,
                       physical_filter=args.filter) for d in dets}
    cal0 = calibs[dets[0]]
    noll = list(np.asarray(cal0.noll_indices))
    k = {j: i for i, j in enumerate(noll)}
    print(f'Noll indices: {noll}')

    gx, gy, gg = _grid(args.ocs_n, args.lim)

    def ocs_map(j):                     # via the calib's OWN OCS interpolator
        return np.asarray(cal0.interpolator_ocs(gg))[:, k[j]].reshape(gx.shape)

    ocs_xy = (np.asarray(cal0.field_x_ocs), np.asarray(cal0.field_y_ocs))
    print(f'OCS samples: {ocs_xy[0].size} pts, r_max='
          f'{np.hypot(*ocs_xy).max():.3f} deg')

    # ---- CCS Z4 assembled per-detector via each calib's OWN interpolator ----
    cgx, cgy, cgg = _grid(args.ccs_n, args.lim)
    sx = np.concatenate([np.asarray(calibs[d].field_x) for d in dets])
    sy = np.concatenate([np.asarray(calibs[d].field_y) for d in dets])
    sdet = np.concatenate([np.full(np.asarray(calibs[d].field_x).size, d)
                           for d in dets])
    dist, idx = cKDTree(np.column_stack([sx, sy])).query(cgg)
    gdet = sdet[idx]
    Zccs = np.full(len(cgg), np.nan)
    for d in np.unique(gdet):
        gm = gdet == d
        vals = np.asarray(calibs[d].interpolator(cgg[gm]))[:, k[4]]
        bad = ~np.isfinite(vals)        # outside this CCD's footprint hull
        if bad.any():
            sm = sdet == d
            _, di = cKDTree(np.column_stack([sx[sm], sy[sm]])).query(cgg[gm][bad])
            vals[bad] = np.asarray(calibs[d].values)[:, k[4]][di]
        Zccs[gm] = vals
    Zccs[dist > args.max_dist] = np.nan
    Zccs = Zccs.reshape(cgx.shape)
    print(f'CCS samples: {sx.size} pts across {len(dets)} detectors, '
          f'r_max={np.hypot(sx, sy).max():.3f} deg')

    # ---- 1. Z4 OCS vs CCS ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    i0 = _imshow(ax[0], ocs_map(4), args.lim,
                 f'Z4 OCS [µm] ({args.ocs_n}x{args.ocs_n}, calib.interpolator_ocs)',
                 sample_xy=ocs_xy)
    fig.colorbar(i0, ax=ax[0], shrink=0.8)
    i1 = _imshow(ax[1], Zccs, args.lim,
                 f'Z4 CCS [µm] ({args.ccs_n}x{args.ccs_n}, per-CCD '
                 f'calib.interpolator, incl. height)', sample_xy=(sx, sy))
    fig.colorbar(i1, ax=ax[1], shrink=0.8)
    fig.suptitle('Official MIW calib — Z4 (defocus): OCS vs CCS')
    fig.tight_layout()
    fig.savefig(f'{args.out_dir}/miw_z4_ocs_ccs.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- 2. OCS maps Z5..Z11 ----
    js = [j for j in range(5, 12) if j in k]
    cols = 4
    rows = int(np.ceil(len(js) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.7 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a, j in zip(axs, js):
        im = _imshow(a, ocs_map(j), args.lim, f'Z{j} OCS [µm]', sample_xy=ocs_xy)
        fig.colorbar(im, ax=a, shrink=0.8)
    for a in axs[len(js):]:
        a.axis('off')
    fig.suptitle(f'Official MIW calib — OCS maps Z5–Z11 '
                 f'({args.ocs_n}x{args.ocs_n}, calib.interpolator_ocs)')
    fig.tight_layout()
    fig.savefig(f'{args.out_dir}/miw_ocs_z5_z11.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out_dir}/miw_z4_ocs_ccs.png and '
          f'{args.out_dir}/miw_ocs_z5_z11.png')


if __name__ == '__main__':
    main()
