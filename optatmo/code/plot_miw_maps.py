"""Validation plots of the official MIW calibration, using the calib object's
OWN interpolators and field tables (RUN ON USDF -- reads the Butler).

The ip_isr ``IntrinsicZernikes`` calib exposes:
  * ``interpolator_ocs`` / ``interpolator`` -- scipy LinearNDInterpolator over
    the OCS / (per-detector) CCS sample points; return (N, n_noll) in microns,
    NaN outside the sample convex hull (no binning / no extension -- the reach
    is set by how far the calib's own field_x/field_y samples go).
  * ``field_x_ocs, field_y_ocs, values_ocs`` (OCS, shared across detectors),
    ``field_x, field_y, values`` (per-detector CCS), ``noll_indices``.

All panels use DVCS orientation (x_field on the vertical axis, y_field on the
horizontal axis).

Plots:
  1. Z4 OCS (calib.interpolator_ocs on a global grid) | Z4 CCS (rendered
     PER CCD: an n_sub x n_sub grid inside each detector's own footprint hull,
     evaluated with that detector's calib.interpolator, then imaged together;
     the per-CCD height steps are preserved and there is no cross-CCD blending).
  2. OCS maps Z5..Zmax -- every Noll index present (--zmax to cap)
     (calib.interpolator_ocs).

The ~16 field-edge CCDs stored height-only on the WHOLE focal-plane grid (not a
compact footprint) are skipped in the CCS panel -- their per-CCD footprint is
not recoverable from the calib samples alone.

Orientation is selectable: --orient dvcs (y_field horizontal, default) or
--orient aos (y_field vertical, to match AOS/Danish/Batoid).

Usage:
    python plot_miw_maps.py [--repo /repo/main] [--collection ...] [--filter i_39]
        [--ocs-n 71] [--ccs-sub 10] [--lim 1.75] [--orient dvcs|aos] [--zmax N]
        [--out-dir output]
"""
import argparse

import numpy as np
from lsst.daf.butler import Butler
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
    return gx, gy, np.column_stack([gx.ravel(), gy.ravel()])


def _axes(ax, lim, title, orient):
    """Field-map axes. orient='dvcs' -> x_field vertical, y_field horizontal;
    orient='aos' -> y_field vertical, x_field horizontal (AOS/Danish/Batoid)."""
    ax.add_patch(Circle((0, 0), FP_R, fill=False, ls='--', color='k',
                        lw=0.5, alpha=0.5))
    ax.set_aspect('equal'); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=6)
    if orient == 'dvcs':
        ax.set_xlabel('y_field [deg]', fontsize=7)
        ax.set_ylabel('x_field [deg]', fontsize=7)
    else:  # aos
        ax.set_xlabel('x_field [deg]', fontsize=7)
        ax.set_ylabel('y_field [deg]', fontsize=7)


# back-compat alias (older call sites)
def _dvcs_axes(ax, lim, title, orient='dvcs'):
    _axes(ax, lim, title, orient)


def _imshow(ax, Z, lim, title, orient='dvcs', sample_xy=None, vlim=None):
    """Z is indexed [iy, ix] (iy over y_field, ix over x_field).
    dvcs: transpose so x_field is vertical; aos: no transpose (y_field vertical).
    vlim: shared symmetric color limit (default: per-panel 99th percentile)."""
    vl = vlim if vlim is not None else _vlim(Z)
    Zp = Z.T if orient == 'dvcs' else Z
    im = ax.imshow(Zp, origin='lower', extent=[-lim, lim, -lim, lim],
                   cmap='RdBu_r', vmin=-vl, vmax=vl, interpolation='nearest')
    if sample_xy is not None:
        fx, fy = sample_xy                    # (field_x, field_y)
        if orient == 'dvcs':
            ax.plot(fy, fx, '.', ms=0.5, color='k', alpha=0.15)
        else:
            ax.plot(fx, fy, '.', ms=0.5, color='k', alpha=0.15)
    _axes(ax, lim, title, orient)
    return im


def ccs_ccd_patches(calibs, dets, kidx, n_sub, ocs_n):
    """One (ex, ey, Z) patch per footprint CCD: an n_sub x n_sub grid of cell
    CENTERS inside the CCD's own sample bounding box (= its interpolator's hull
    bbox), evaluated with that CCD's interpolator (NaN outside the footprint
    hull), plus the matching n_sub+1 cell EDGES so pcolormesh(shading='flat')
    draws the tile spanning EXACTLY the sample bbox -- no half-cell overhang into
    the inter-CCD gaps.  Field-edge CCDs stored height-only on the WHOLE
    focal-plane grid (n_samples >= 0.5*ocs_n) are skipped.  Returns
    (patches, n_skipped); each patch = (ex, ey, Z) with Z indexed [iy, ix]."""
    patches, n_skip = [], 0
    for d in dets:
        c = calibs[d]
        fx, fy = np.asarray(c.field_x), np.asarray(c.field_y)
        if fx.size >= 0.5 * ocs_n:            # whole-FP height-only fallback CCD
            n_skip += 1
            continue
        ex = np.linspace(fx.min(), fx.max(), n_sub + 1)      # field_x cell edges
        ey = np.linspace(fy.min(), fy.max(), n_sub + 1)      # field_y cell edges
        cx = 0.5 * (ex[:-1] + ex[1:])
        cy = 0.5 * (ey[:-1] + ey[1:])
        gx, gy = np.meshgrid(cx, cy)                          # gx[iy,ix], gy[iy,ix]
        z = np.asarray(c.interpolator(np.column_stack([gx.ravel(), gy.ravel()])))
        patches.append((ex, ey, z[:, kidx].reshape(gx.shape)))
    return patches, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/repo/main')
    ap.add_argument('--collection',
                    default='u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
    ap.add_argument('--filter', default='i_39')
    ap.add_argument('--instrument', default='LSSTCam')
    ap.add_argument('--ocs-n', type=int, default=71)
    ap.add_argument('--ccs-sub', type=int, default=10,
                    help='per-CCD grid (n_sub x n_sub) for the CCS panel')
    ap.add_argument('--lim', type=float, default=1.75)
    ap.add_argument('--orient', choices=['dvcs', 'aos'], default='dvcs',
                    help="dvcs: y_field horizontal (x_field vertical); "
                         "aos: y_field vertical (x_field horizontal, AOS/Danish/Batoid)")
    ap.add_argument('--zmax', type=int, default=None,
                    help='highest Noll index for the OCS map grid (default: all present)')
    ap.add_argument('--batoid', action='store_true',
                    help='also compute the batoid design intrinsic on the SAME grid '
                         'and write a per-Noll MIW-vs-batoid comparison PDF (RSP: needs '
                         'batoid_rubin; per-point, so use a modest --batoid-n)')
    ap.add_argument('--batoid-band', default=None,
                    help="band for the batoid design intrinsic (default: first char of --filter)")
    ap.add_argument('--batoid-model', default='Rubin_v3.14',
                    help="batoid yaml stem: Rubin_v3.14 (as-built) or LSST (nominal)")
    ap.add_argument('--batoid-n', type=int, default=41,
                    help='grid size for the batoid comparison (per-point eval; keep modest)')
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

    # ---- CCS Z4: one n_sub x n_sub patch per CCD, each via its OWN interp ----
    patches, n_skip = ccs_ccd_patches(calibs, dets, k[4], args.ccs_sub, ocs_xy[0].size)
    vl_ccs = _vlim(np.concatenate([p[2].ravel() for p in patches]))
    fills = [np.isfinite(p[2]).mean() for p in patches]      # in-hull fraction
    wid = np.median([np.ptp(p[0]) for p in patches])         # median CCD bbox width
    print(f'CCS: {len(patches)} footprint CCDs rendered '
          f'({args.ccs_sub}x{args.ccs_sub} each), {n_skip} height-only CCDs '
          f'skipped; median footprint width {wid:.3f} deg, '
          f'median in-hull cell fraction {np.median(fills):.2f}')

    # ---- 1. Z4 OCS vs CCS (DVCS: x_field vertical, y_field horizontal) ----
    orient = args.orient
    otag = orient.upper()
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    i0 = _imshow(ax[0], ocs_map(4), args.lim,
                 f'Z4 OCS [µm] ({args.ocs_n}x{args.ocs_n}, calib.interpolator_ocs)',
                 orient=orient, sample_xy=ocs_xy)
    fig.colorbar(i0, ax=ax[0], shrink=0.8)
    axc = ax[1]
    im = None
    for ex, ey, z in patches:
        if orient == 'dvcs':      # horizontal = field_y (ey), vertical = field_x (ex)
            im = axc.pcolormesh(ey, ex, np.ma.masked_invalid(z.T), cmap='RdBu_r',
                                vmin=-vl_ccs, vmax=vl_ccs, shading='flat')
        else:                     # aos: horizontal = field_x (ex), vertical = field_y (ey)
            im = axc.pcolormesh(ex, ey, np.ma.masked_invalid(z), cmap='RdBu_r',
                                vmin=-vl_ccs, vmax=vl_ccs, shading='flat')
    _axes(axc, args.lim, f'Z4 CCS [µm] (per-CCD {args.ccs_sub}x'
          f'{args.ccs_sub} interpolation, incl. height)', orient)
    if im is not None:
        fig.colorbar(im, ax=axc, shrink=0.8)
    fig.suptitle(f'Official MIW calib — Z4 (defocus): OCS vs CCS  ({otag})')
    fig.tight_layout()
    fig.savefig(f'{args.out_dir}/miw_z4_ocs_ccs.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- 2. OCS maps: every Noll >= 5 (up to --zmax) ----
    zmax = args.zmax if args.zmax is not None else max(noll)
    js = [j for j in noll if 5 <= j <= zmax]
    cols = 5
    rows = int(np.ceil(len(js) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a, j in zip(axs, js):
        im = _imshow(a, ocs_map(j), args.lim, f'Z{j} OCS [µm]',
                     orient=orient, sample_xy=ocs_xy)
        fig.colorbar(im, ax=a, shrink=0.8)
    for a in axs[len(js):]:
        a.axis('off')
    fig.suptitle(f'Official MIW calib — OCS maps Z5–Z{zmax} '
                 f'({args.ocs_n}x{args.ocs_n}, calib.interpolator_ocs)  ({otag})')
    fig.tight_layout()
    fig.savefig(f'{args.out_dir}/miw_ocs_maps.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out_dir}/miw_z4_ocs_ccs.png and '
          f'{args.out_dir}/miw_ocs_maps.png')

    # ---- 3. optional: batoid design intrinsic vs MIW, same grid & bins ----
    if args.batoid:
        from matplotlib.backends.backend_pdf import PdfPages
        from lsst.ts.intrinsic.wavefront.batoid_intrinsic import batoid_ocs_zernikes
        band = args.batoid_band or args.filter[0].lower()
        bgx, bgy, bgg = _grid(args.batoid_n, args.lim)
        ins = np.hypot(bgg[:, 0], bgg[:, 1]) <= args.lim     # skip corners (vignetted)
        print(f'batoid design intrinsic: model {args.batoid_model}, band {band}, '
              f'{int(ins.sum())} pts on {args.batoid_n}x{args.batoid_n} grid '
              f'(per-point eval, may take a few min)...')
        bat = batoid_ocs_zernikes(bgg[ins, 0], bgg[ins, 1], noll_list=noll,
                                  band=band, optical_model=args.batoid_model)
        miw_same = np.asarray(cal0.interpolator_ocs(bgg))    # MIW on the SAME grid

        def bmap(j):
            full = np.full(bgx.size, np.nan); full[ins] = bat[j]
            return full.reshape(bgx.shape)

        def mmap(j):
            return miw_same[:, k[j]].reshape(bgx.shape)

        js = [j for j in noll if 5 <= j <= (args.zmax if args.zmax is not None else max(noll))]
        outpdf = f'{args.out_dir}/miw_vs_batoid_{orient}.pdf'
        per = 3
        with PdfPages(outpdf) as pdf:
            for s in range(0, len(js), per):
                chunk = js[s:s + per]
                fig, axs = plt.subplots(len(chunk), 3, figsize=(12, 3.7 * len(chunk)),
                                        squeeze=False)
                for row, j in enumerate(chunk):
                    M, B = mmap(j), bmap(j)
                    vl = _vlim(np.concatenate([M.ravel(), B.ravel()]))
                    for a, (Z, ttl, vv) in zip(
                            axs[row],
                            [(M, f'Z{j} MIW', vl), (B, f'Z{j} batoid', vl),
                             (M - B, f'Z{j} MIW-batoid', None)]):
                        im = _imshow(a, Z, args.lim, f'{ttl} [µm]', orient=orient, vlim=vv)
                        fig.colorbar(im, ax=a, shrink=0.8)
                fig.suptitle(f'MIW vs batoid design intrinsic ({args.batoid_model}, '
                             f'{band}-band, OCS)  ({otag})', fontsize=11)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        print(f'wrote {outpdf}')


if __name__ == '__main__':
    main()
