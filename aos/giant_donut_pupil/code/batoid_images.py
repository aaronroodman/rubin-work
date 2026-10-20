"""Batoid ray-traced v4.02 donut image for each giant-donut side (step 2 of the pupil pass).

Per donut side:

1. Camera piston: the extra camera z shift that makes batoid's Z4 at the donut's
   field (v4.02 without spider, the side's offsets) equal the Z4 that
   `z4_fits.py` fitted with v4.02.
2. Reduced model: one `traceFull` of the full v4.02 model (spider rotated by
   rtp, the side's offsets plus the piston) on a pupil grid; the reduced model
   keeps the optical surfaces and every obscuration that is the first to
   vignette at least one ray. Its transmission is checked against the full
   model on a second, offset grid; below `MIN_AGREEMENT` the full model is used.
3. Image: a stratified-random pupil grid dense enough for about `RAYS_PER_PIX`
   rays per illuminated 10 um pixel, traced at 620 nm, binned onto the stamp
   pixel grid, convolved with the danish atmosphere (Kolmogorov, 1.0 arcsec
   FWHM, galsim, as `danish.DonutModel._atm_kernel`).
4. Alignment: the Batoid centroid is put on the centroid of the step 1 v4.02
   model image, then refined by an upsampled cross-correlation with the data
   (unmasked pixels inside 1.1 r_out); the flux is scaled to the data's over
   the same pixels.

The spider is rotated as in the danish fits: danish ``spider_angle = rtp``
equals a batoid rotation of the telescope (everything but the camera) by
``-rtp`` about the optical axis. Batoid detector x, y map to stamp column, row.

Run, from ``code/``::

    python batoid_images.py --run ../output/20261018_pupil --workers 32

The v1000 control (no spider planes in that model; its own Z4 fit sets the
piston and centre), written to another run::

    python batoid_images.py --run ../output/20261018_pupil --model v1000 \
        --out ../output/20261019_pupil2
"""
import argparse
import json
import pathlib
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fit_donuts as fd  # noqa: E402  (puts the blitz checkout first on sys.path)
import obscuration_v402 as ob  # noqa: E402
import z4_fits  # noqa: E402

WAVELENGTH_M = 620e-9
PIXEL_M = 10e-6
FWHM_ARCSEC = z4_fits.FWHM_ARCSEC
RAYS_PER_PIX = 100
PUPIL_INNER_M, PUPIL_OUTER_M = 2.3, 4.25
CLIP_GRID = 500
CHECK_GRID = 420
MIN_AGREEMENT = 0.9999
OPTICAL = ('M1', 'M2', 'M3', 'L1', 'L2', 'Filter', 'L3', 'Detector')
CHUNK = 1_000_000
TRACEFULL_CHUNK = 20_000
PISTON_STEP_M = 20e-6
BATOID_TAG = {'v4.02': 'batoid', 'v1000': 'batoid_v1000'}


def _rebuild(optic, items):
    keep = ('name', 'inMedium', 'outMedium', 'coordSys', 'skip', 'backDist', 'pupilSize',
            'pupilObscuration', 'stopSurface', 'sphereRadius')
    import batoid
    return batoid.CompoundOptic(items, **{k: optic.__dict__[k] for k in keep})


def configure(optic, rtp_deg, offsets, piston_m=0.0):
    """Rotate the telescope (not the camera) by -rtp, then shift detector, camera, M2."""
    import batoid
    rot = batoid.RotZ(np.deg2rad(-rtp_deg))
    optic = _rebuild(optic, [it if it.name == 'LSSTCamera' else
                             it.withGlobalRotation(rot, rotCenter=[0, 0, 0],
                                                   coordSys=batoid.CoordSys())
                             for it in optic.items])
    det, cam, m2 = offsets
    for name, dz in (('Detector', det), ('LSSTCamera', cam + piston_m), ('M2', m2)):
        if dz:
            optic = optic.withGloballyShiftedOptic(name, [0.0, 0.0, dz])
    return optic


def pupil_grid(spacing, rng=None, inner=PUPIL_INNER_M, outer=PUPIL_OUTER_M, offset=0.5):
    """Square grid on the stop plane, cut to an annulus; stratified-random if `rng`."""
    g = (np.arange(int(np.ceil(2 * outer / spacing))) + offset) * spacing - outer
    x, y = np.meshgrid(g, g)
    x, y = x.ravel(), y.ravel()
    if rng is not None:
        x = x + rng.uniform(-0.5, 0.5, x.size) * spacing
        y = y + rng.uniform(-0.5, 0.5, y.size) * spacing
    r = np.hypot(x, y)
    keep = (r >= inner) & (r <= outer)
    return x[keep], y[keep]


def _rays(optic, x, y, thx, thy):
    import batoid
    return batoid.RayVector.fromStop(x, y, optic=optic, wavelength=WAVELENGTH_M,
                                     theta_x=thx, theta_y=thy)


def first_clip(optic, x, y, thx, thy):
    """Index of the first surface that vignettes each ray (-1 if none), and the names.

    Also returns the detector-plane positions (m) of every ray; batoid keeps
    tracing vignetted rays, so blocked rays have positions too.
    """
    first = np.full(len(x), -1, np.int32)
    det_x, det_y = np.empty(len(x)), np.empty(len(x))
    for i in range(0, len(x), TRACEFULL_CHUNK):  # traceFull holds every surface's rays
        s = slice(i, i + TRACEFULL_CHUNK)
        tf = optic.traceFull(_rays(optic, x[s], y[s], thx, thy))
        names = list(tf)
        f = first[s]
        prev = np.zeros(len(f), bool)
        for k, key in enumerate(names):
            out = tf[key]['out']
            stop = np.asarray(out.vignetted) | np.asarray(out.failed)
            f[stop & ~prev & (f < 0)] = k
            prev = stop
        det_x[s], det_y[s] = tf[names[-1]]['out'].x, tf[names[-1]]['out'].y
    return first, names, det_x, det_y


def transmitted(optic, x, y, thx, thy):
    out = optic.trace(_rays(optic, x, y, thx, thy))
    return ~np.asarray(out.vignetted) & ~np.asarray(out.failed), np.asarray(out.x), np.asarray(out.y)


def camera_piston(optic, thx, thy, offsets, r_outer, r_inner, z4_target_m):
    """Camera z shift (m) giving batoid Z4 = `z4_target_m`; two secant steps."""
    dz = 0.0
    z_a = z4_fits.zk_intrinsic(optic, thx, thy, offsets, r_outer, r_inner)[4]
    z_b = z4_fits.zk_intrinsic(optic, thx, thy, offsets, r_outer, r_inner, PISTON_STEP_M)[4]
    slope = (z_b - z_a) / PISTON_STEP_M
    for _ in range(2):
        dz += (z4_target_m - z_a) / slope
        z_a = z4_fits.zk_intrinsic(optic, thx, thy, offsets, r_outer, r_inner, dz)[4]
    return dz, z_a


def atm_kernel(npix, fwhm_arcsec, sky_scale):
    """danish's atmosphere: galsim Kolmogorov drawn at the pixel scale, ``npix//2`` wide."""
    import galsim
    no2 = (npix - 1) // 2
    return galsim.Kolmogorov(fwhm=fwhm_arcsec).drawImage(nx=no2, ny=no2, scale=sky_scale).array


def render(px, py, npix, kernel):
    """Bin ray pixel positions (stamp column, row) and convolve with `kernel`."""
    from scipy.signal import fftconvolve
    edges = np.arange(npix + 1) - 0.5
    hist, _, _ = np.histogram2d(py, px, bins=[edges, edges])
    return hist, fftconvolve(hist, kernel, mode='same')


def side_image(task):
    """Steps 1-4 for one donut side; writes ``images/<cand>_<side>_batoid.npz``."""
    import batoid  # noqa: F401
    from skimage.registration import phase_cross_correlation
    key, row, mask_v402, run_dir, model = task
    t0 = time.time()
    run_dir = pathlib.Path(run_dir)
    out_dir = pathlib.Path(row.pop('out_dir', run_dir))
    thx, thy = np.radians(row['field_x_ccs_deg']), np.radians(row['field_y_ccs_deg'])
    offsets = (row['det_offset_mm'] * 1e-3, row['cam_offset_mm'] * 1e-3, row['m2_offset_mm'] * 1e-3)
    rtp = row['rtp_deg']
    params, r_outer, r_inner, nospider = fd.pupil(model, mask_v402)
    piston, z4_got = camera_piston(nospider, thx, thy, offsets, r_outer, r_inner,
                                   row['z4_um'] * 1e-6)

    full = configure(ob.v402_optic() if model == 'v4.02' else nospider, rtp, offsets, piston)
    cx, cy = pupil_grid(2 * PUPIL_OUTER_M / CLIP_GRID)
    first, names, tf_x, tf_y = first_clip(full, cx, cy, thx, thy)
    clippers = sorted({names[k] for k in np.unique(first[first >= 0])})
    leaf = tuple(n.split('.')[-1] for n in clippers)
    reduced = (configure(ob.reduced_v402(keep=tuple(set(leaf) | set(OPTICAL))), rtp, offsets, piston)
               if model == 'v4.02' else full)
    kx, ky = pupil_grid(2 * PUPIL_OUTER_M / CHECK_GRID, offset=0.21)
    t_full, _, _ = transmitted(full, kx, ky, thx, thy)
    t_red, _, _ = transmitted(reduced, kx, ky, thx, thy)
    agreement = float(np.mean(t_full == t_red))
    use_full = agreement < MIN_AGREEMENT
    optic = full if use_full else reduced

    stamp = np.load(run_dir.parent / '20261018_fits' / 'images' / f'{key[0]}_{key[1]}_stamp.npz')
    img, msk = stamp['stamp'].astype(float), stamp['masked'].astype(bool)
    npix = img.shape[0]
    spacing = r_outer / (row['r_out_pix'] * np.sqrt(RAYS_PER_PIX)) / 1.05
    rng = np.random.default_rng(zlib.crc32(f'{key[0]}_{key[1]}'.encode()))
    gx, gy = pupil_grid(spacing, rng=rng)
    xs, ys = [], []
    for i in range(0, len(gx), CHUNK):
        keep, x, y = transmitted(optic, gx[i:i + CHUNK], gy[i:i + CHUNK], thx, thy)
        xs.append(x[keep])
        ys.append(y[keep])
    x, y = np.concatenate(xs), np.concatenate(ys)
    x0, y0 = x.mean(), y.mean()
    u, v = (x - x0) / PIXEL_M, (y - y0) / PIXEL_M

    model_img = np.load(run_dir / 'images' / f'{key[0]}_{key[1]}_{model}_z4.npz')['model']
    yy, xx = np.mgrid[:npix, :npix]
    mc = (np.sum(model_img * xx) / model_img.sum(), np.sum(model_img * yy) / model_img.sum())
    sky_scale = 3600 * np.rad2deg(PIXEL_M / 10.312)
    kernel = atm_kernel(npix, FWHM_ARCSEC, sky_scale)
    n = npix // 2
    inside = (np.hypot(xx - n - row['dx_pix'], yy - n - row['dy_pix']) < 1.1 * row['r_out_pix']) & ~msk
    shift = np.zeros(2)
    for _ in range(2):
        hist, bat = render(u + mc[0] + shift[0], v + mc[1] + shift[1], npix, kernel)
        ref = np.where(inside, img, 0.0)
        mov = np.where(inside, bat, 0.0)
        s, _, _ = phase_cross_correlation(ref, mov, upsample_factor=20, normalization=None)
        shift += (s[1], s[0])
    hist, bat = render(u + mc[0] + shift[0], v + mc[1] + shift[1], npix, kernel)
    scale = img[inside].sum() / bat[inside].sum()
    bat *= scale
    lit = hist[hist > 0.5 * np.percentile(hist[hist > 0], 90)]
    resid = (img - bat)[inside]
    np.savez_compressed(
        out_dir / 'images' / f'{key[0]}_{key[1]}_{BATOID_TAG[model]}.npz', batoid=bat.astype(np.float32),
        tf_px=((tf_x - x0) / PIXEL_M + mc[0] + shift[0]).astype(np.float32),
        tf_py=((tf_y - y0) / PIXEL_M + mc[1] + shift[1]).astype(np.float32),
        tf_first=first.astype(np.int16), tf_names=np.array(names))
    out = dict(cand_id=key[0], side=key[1], model=model, piston_um=piston * 1e6, z4_target_um=row['z4_um'],
               z4_batoid_um=z4_got * 1e6, n_clippers=len(clippers), clippers=';'.join(leaf),
               agreement=agreement, xor_over_T=float(np.sum(t_full != t_red) / t_full.sum()),
               used_full=bool(use_full), n_rays=int(len(gx)), n_transmitted=int(len(x)),
               rays_per_pix_median=float(np.median(lit)),
               xcorr_dx_pix=float(shift[0]), xcorr_dy_pix=float(shift[1]),
               centre_x_pix=float(mc[0] + shift[0]), centre_y_pix=float(mc[1] + shift[1]),
               flux_scale=float(scale), resid_rms_e=float(np.sqrt(np.mean(resid ** 2))),
               seconds=time.time() - t0)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()
                      if k != 'clippers'}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True, help='the pupil run directory (z4_fits.py)')
    ap.add_argument('--mask-v402', default='../output/20261018_obscuration_v402/'
                    'RubinObsc_v4.02_r_rtpp0_pp0d0.yaml')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--only', default=None, help='one donut side, e.g. G0515_extra')
    ap.add_argument('--model', default='v4.02', choices=tuple(BATOID_TAG),
                    help='Batoid model; its Z4 fit (z4_fits.py) sets the piston and centre')
    ap.add_argument('--out', default=None, help='output run directory (default --run)')
    args = ap.parse_args()
    import multiprocessing as mp
    run_dir = pathlib.Path(args.run).resolve()
    z4 = pd.read_parquet(run_dir / 'z4_fits.parquet')
    z4 = z4[z4.pupil == args.model]
    if args.only:
        z4 = z4[z4.cand_id + '_' + z4.side == args.only]
    out_dir = pathlib.Path(args.out).resolve() if args.out else run_dir
    (out_dir / 'images').mkdir(exist_ok=True)
    fd.pupil(args.model, args.mask_v402)
    tasks = [((r.cand_id, r.side), dict(r._asdict(), out_dir=str(out_dir)), args.mask_v402,
              str(run_dir), args.model) for r in z4.itertuples(index=False)]
    with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
        rows = pool.map(side_image, tasks, chunksize=1)
    df = pd.DataFrame(rows)
    name = ('batoid_images' + ('' if args.model == 'v4.02' else '_v1000')
            + (f'_{args.only}' if args.only else ''))
    df.to_parquet(out_dir / f'{name}.parquet')
    df.to_csv(out_dir / f'{name}.csv', index=False, float_format='%.6g')
    print('done', out_dir)


if __name__ == '__main__':
    main()
