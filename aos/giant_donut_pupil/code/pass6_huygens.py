"""Pupil pass 6 follow-up: Huygens (wave-optics) giant donuts, with and without a vane phase strip.

Tests two causes of the data − model mismatch that geometric optics (danish and the step 3
Batoid sims) leaves out: Fresnel diffraction at the vanes and pupil edges, and refraction by
warm air along the vanes (a phase strip).

Per sim side of step 3 (`pass6_sims.choose`, 26 sides) and wavelength (``WAVELENGTHS_M``):

1. a regular stop-plane grid at ``STOP_SPACING_M`` (0.5 mm: the grid's ghost images sit
   lambda F / s = 12-14 mm away, outside the stamp), traced through Batoid v4.02 (the reduced
   model of the step 3 draw, or the full model when the draw used it), design state, the
   side's field, giant offsets and physical_rotator_angle;
2. the Huygens sum U(x) = sum_j exp(i [k_j . (x - x_j) + k0 (t_j - t_ref)]) over the
   transmitted rays (batoid ``RayVector.amplitude`` convention, unit weights: Rubin obeys
   the sine condition to 4e-6, `optics/fresnel_donuts`), evaluated on a lattice of 4 x 4
   points per 10 um pixel with a type-1 NUFFT (FINUFFT, as `optics/code/fresnel`), summed
   over ray chunks; |U|^2 binned to the stamp pixels, then summed over wavelengths;
3. variants: the plain sum (``diffraction``), and the same rays with an extra phase
   2 pi OPD / lambda * exp(-d^2 / 2 sigma^2), d the stop-plane distance to the nearest
   projected vane centerline (danish ``_project_spider_vane`` of every Spider_3D piece at
   the side's field and rtp), optionally offset along stop-plane y (``PHASE_VARIANTS``);
   a geometric image of the same rays (``geometric``) is the control;
4. blurred with the danish Kolmogorov kernel at the step 3 sim blur, scaled to the data
   flux, the data sky added, Poisson noise, then `pass6_pipeline.measure` (v4.02s mask, no
   delta grid, shadow profiles kept).

Outputs in ``<run>/huygens/``: ``stamps/<sample>_<side>.fits`` (one HDU per variant: the
noiseless unblurred image, then the noisy stamp), ``draws.parquet``, per variant
``<variant>/`` with the pass 5 table shapes, ``translation_fit``, ``centre_test`` and
``spider_profiles.npz``, and ``huygens_summary`` (parquet and csv).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass6_huygens.py --workers 110
    python aos/giant_donut_pupil/code/pass6_huygens.py --test   # 1 side, 1 wavelength, coarse
"""
import argparse
import json
import os
import pathlib
import sys
import time
import zlib

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import finufft  # noqa: E402,F401  (before batoid; see optics/code/fresnel/fresnel_donut.py)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass6_common as p6  # noqa: E402
import pass6_pipeline as pp  # noqa: E402
import pass6_sims as psim  # noqa: E402
import pass5_analysis as pa  # noqa: E402
import batoid_images as bi  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import fit_donuts as fd  # noqa: E402

OUT = p6.RUN / 'huygens'
WAVELENGTHS_M = (570e-9, 595e-9, 620e-9, 645e-9, 670e-9)
STOP_SPACING_M = 0.5e-3
OVERSAMPLE = 2
N_BLOCKS = 8
CHUNK_ROWS = 128
NUFFT_EPS = 1e-6
FIT_MASK = 'v4.02s'
# name: (OPD um, Gaussian sigma m, stop-plane y offset m)
PHASE_VARIANTS = {'plume_m0p1': (-0.10, 0.060, 0.0), 'plume_p0p1': (+0.10, 0.060, 0.0),
                  'plume_m0p2': (-0.20, 0.045, 0.0), 'plume_m0p1_yp40': (-0.10, 0.060, +0.040),
                  'plume_m0p1_ym40': (-0.10, 0.060, -0.040)}
VARIANTS = ('geometric', 'diffraction') + tuple(PHASE_VARIANTS)


def vane_lines(row, rtp_deg):
    """Projected centerline segments (p, unit direction, half length) of every vane piece."""
    from danish.factory import _project_spider_vane
    params = p6.mask_params(FIT_MASK, fd.offsets_m(row.side, row.sensor_class))
    thx, thy = np.radians(row.field_x_ccs_deg), np.radians(row.field_y_ccs_deg)
    out = []
    for v in params['Spider_3D']:
        p1x, p1y, s1, c1, p2x, p2y, s2, c2 = _project_spider_vane(
            v['r0'], v['v0'], v['width'], v['length'], v['angle'] + rtp_deg, thx, thy)
        d = np.array([c1 + c2, s1 + s2])
        out.append((np.array([(p1x + p2x) / 2, (p1y + p2y) / 2]), d / np.hypot(*d), v['length'] / 2))
    return out


def vane_distance(x, y, lines):
    """Stop-plane distance (m) from each point to the nearest vane centerline segment."""
    best = np.full(x.shape, np.inf)
    for p, d, h in lines:
        qx, qy = x - p[0], y - p[1]
        a = np.clip(qx * d[0] + qy * d[1], -h, h)
        best = np.minimum(best, np.hypot(qx - a * d[0], qy - a * d[1]))
    return best


def optic_for(row, rtp_deg):
    dr = pd.read_parquet(psim.SIM_DIR / 'draws.parquet').set_index(['key', 'variant'])
    d = dr.loc[(row.key, 'nominal')]
    offsets = fd.offsets_m(row.side, row.sensor_class)
    if d.used_full:
        return psim.configure(ob.v402_optic(), rtp_deg, offsets, 0.0), d
    keep = tuple(set(d.clippers.split(';')) | set(bi.OPTICAL))
    return psim.configure(ob.reduced_v402(keep=keep), rtp_deg, offsets, 0.0), d


def lattice(row, d, n, xc, yc):
    """Lattice origin (m) and spacing so OVERSAMPLE^2 points fall in each stamp pixel.

    Stamp pixel i is centered on the ray centroid (xc, yc) + (i - centroid_pix) * pixel, as in
    the step 3 draw, so the Huygens and geometric images share the draw's stamp grid.
    """
    dx = bi.PIXEL_M / OVERSAMPLE
    x0 = xc - d.centroid_x_pix * bi.PIXEL_M - bi.PIXEL_M / 2 + dx / 2
    y0 = yc - d.centroid_y_pix * bi.PIXEL_M - bi.PIXEL_M / 2 + dx / 2
    return x0, y0, dx, n * OVERSAMPLE


def draw_task(task):
    """U per variant on the lattice for one block of stop-grid rows, and its geometric image.

    Blocks of one (side, wavelength) share the lattice, the centroid and the time reference
    (from the same coarse trace), so their U add coherently in `main`.
    """
    import batoid
    key, wl, spacing, block, n_blocks = task
    t0 = time.time()
    C = pd.read_parquet(psim.SIM_DIR / 'sim_sides.parquet').set_index('key')
    row = C.loc[key].copy()
    row['key'] = key
    rtp = float(row.rtp_deg)
    optic, d = optic_for(row, rtp)
    thx, thy = np.radians(row.field_x_ccs_deg), np.radians(row.field_y_ccs_deg)
    n = np.load(p6.RUN / 'isr_stamps' / f'{row.sample_id}_{row.side}.npz')['img'].shape[0]
    lines = vane_lines(row, rtp)
    # centroid and time reference from a coarse trace
    cx_, cy_ = bi.pupil_grid(0.02)
    rc = optic.trace(batoid.RayVector.fromStop(cx_, cy_, optic=optic, wavelength=wl, theta_x=thx,
                                               theta_y=thy))
    ok = ~rc.vignetted & ~rc.failed
    xc, yc = float(np.mean(rc.x[ok])), float(np.mean(rc.y[ok]))
    t_ref = float(np.median(rc.t[ok]))
    x0, y0, dx, N = lattice(row, d, n, xc, yc)
    xm, ym = x0 + (N // 2) * dx, y0 + (N // 2) * dx   # NUFFT mode 0 (modeord 0: -N/2..N/2-1)
    k0 = 2 * np.pi / wl
    names = ('diffraction',) + tuple(PHASE_VARIANTS)
    U = np.zeros((len(names), N, N), np.complex128)
    geo = np.zeros((n, n))
    g = (np.arange(int(np.ceil(2 * bi.PUPIL_OUTER_M / spacing))) + 0.5) * spacing - bi.PUPIL_OUTER_M
    plan = finufft.Plan(1, (N, N), n_trans=len(names), eps=NUFFT_EPS, isign=1, nthreads=1,
                        maxbatchsize=1, upsampfac=1.25)
    n_rays = n_tr = 0
    rows = np.array_split(np.arange(len(g)), n_blocks)[block]
    for i in range(rows[0], rows[-1] + 1, CHUNK_ROWS):
        X, Y = np.meshgrid(g, g[i:min(i + CHUNK_ROWS, rows[-1] + 1)])
        X, Y = X.ravel(), Y.ravel()
        r = np.hypot(X, Y)
        keep = (r >= bi.PUPIL_INNER_M) & (r <= bi.PUPIL_OUTER_M)
        X, Y = X[keep], Y[keep]
        if not len(X):
            continue
        n_rays += len(X)
        rv = optic.trace(batoid.RayVector.fromStop(X, Y, optic=optic, wavelength=wl, theta_x=thx,
                                                   theta_y=thy))
        good = ~np.asarray(rv.vignetted) & ~np.asarray(rv.failed)
        if not good.any():
            continue
        n_tr += int(good.sum())
        vx, vy, vz = (np.asarray(a)[good] for a in (rv.vx, rv.vy, rv.vz))
        vsq = vx ** 2 + vy ** 2 + vz ** 2
        kx, ky, kz = k0 * vx / vsq, k0 * vy / vsq, k0 * vz / vsq
        px, py, pz = (np.asarray(a)[good] for a in (rv.x, rv.y, rv.z))
        tt = np.asarray(rv.t)[good]
        phase = kx * (xm - px) + ky * (ym - py) - kz * pz + k0 * (tt - t_ref)
        dist = vane_distance(X[good], Y[good], lines)
        c = np.empty((len(names), len(phase)), np.complex128)
        c[0] = np.exp(1j * phase)
        for j, (opd_um, sig, yoff) in enumerate(PHASE_VARIANTS.values(), start=1):
            dd = dist if yoff == 0 else vane_distance(X[good], Y[good] - yoff, lines)
            c[j] = np.exp(1j * (phase + 2 * np.pi * opd_um * 1e-6 / wl * np.exp(-0.5 * (dd / sig) ** 2)))
        fx = np.mod(kx * dx + np.pi, 2 * np.pi) - np.pi
        fy = np.mod(ky * dx + np.pi, 2 * np.pi) - np.pi
        plan.setpts(fx, fy)
        U += plan.execute(c)
        u = (px - xc) / bi.PIXEL_M + d.centroid_x_pix
        w = (py - yc) / bi.PIXEL_M + d.centroid_y_pix
        geo += np.histogram2d(w, u, bins=[np.arange(n + 1) - 0.5] * 2)[0]
    out = dict(key=key, wavelength_nm=wl * 1e9, block=block, n_rays=n_rays, n_transmitted=n_tr,
               seconds=time.time() - t0)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}),
          flush=True)
    return key, wl, names, U.astype(np.complex64), geo, out


def to_pixels(U, n):
    """|U|^2 binned to the stamp pixels; finufft output [k1, k2] is (x, y), stamps [row y, col x]."""
    I = (np.abs(U.astype(np.complex128)) ** 2).transpose(0, 2, 1)
    return I.reshape(len(I), n, OVERSAMPLE, n, OVERSAMPLE).sum(axis=(2, 4))


def finish_side(key, imgs, seed_tag=''):
    """Blur, scale, sky, Poisson; writes the FITS stamps; returns the noisy stamps and mask."""
    from astropy.io import fits
    C = pd.read_parquet(psim.SIM_DIR / 'sim_sides.parquet').set_index('key')
    row = C.loc[key]
    F = pd.read_parquet(p6.RUN / psim.DATA_CONFIG / 'fits.parquet').set_index('key')
    sky = float(pd.read_parquet(p6.RUN / 'isr_stamps.parquet').set_index('key').sky_e[key])
    d = np.load(p6.RUN / 'isr_stamps' / f'{row.sample_id}_{row.side}.npz')
    n = d['img'].shape[0]
    sky_scale = 3600 * np.rad2deg(bi.PIXEL_M / 10.312)
    from scipy.signal import fftconvolve
    kern = bi.atm_kernel(n, p6.SIM_FWHM_ARCSEC[row.side], sky_scale)
    flux = float(F.loc[key].flux_e)
    hdus = [fits.PrimaryHDU()]
    hdus[0].header.update(SIDEKEY=key, LABEL=row.label, SIDE=row.side, DETECTOR=row.detector,
                          FIELDR=row.field_r_deg, RTP=float(row.rtp_deg), SKY_E=sky,
                          FWHM=p6.SIM_FWHM_ARCSEC[row.side], FRAME='CCS stamp (col x, row y)')
    noisy = {}
    for v, im in imgs.items():
        img = np.clip(fftconvolve(im, kern, mode='same'), 0, None)
        img *= flux / img.sum()
        rng = np.random.default_rng(zlib.crc32(f'{key}_{v}_huygens{seed_tag}'.encode()))
        noisy[v] = rng.poisson(img + sky).astype(np.float32)
        hdus.append(fits.ImageHDU((im / im.sum()).astype(np.float32), name=f'{v}_truth'))
        hdus.append(fits.ImageHDU(noisy[v], name=f'{v}_stamp'))
    (OUT / 'stamps').mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(OUT / 'stamps' / f'{row.sample_id}_{row.side}.fits', overwrite=True)
    return noisy, d['masked'].astype(bool)


def fit_task(task):
    key, variant, img, msk = task
    C = pd.read_parquet(psim.SIM_DIR / 'sim_sides.parquet').set_index('key')
    row = C.loc[key].copy()
    row['key'] = key
    im, _ = psim.subtract_sky(img.astype(float), msk, row.r_out_pix)
    try:
        out = pp.measure(row, im, msk, p6.SIM_FWHM_ARCSEC[row.side], mask=FIT_MASK, delta=False,
                         profiles=True)
    except Exception as err:
        print('error', variant, key, repr(err), flush=True)
        return variant, None
    print('fit', variant, key, f'{out["fit"]["seconds"]:.0f} s', 'found', out['fit']['n_found'],
          f'chi2/dof {out["fit"]["chi2_dof"]:.3f}', flush=True)
    return variant, out


def shadow_shape(t):
    """Median shadow depth, FWHM and equivalent width (depth x FWHM) per side of focus."""
    a = t[(t.centre == 'fit_a') & (t.missing == '')].copy()
    a['ew_pix'] = a.depth_data * a.fwhm_pix_data
    return a.groupby('side').agg(depth=('depth_data', 'median'), fwhm_pix=('fwhm_pix_data', 'median'),
                                 ew_pix=('ew_pix', 'median'))


def summarize(res, C):
    rows = []
    for v, r in res.items():
        if not r:
            continue
        tabs = pp.tables(r, C)
        od = OUT / v
        pp.save(tabs, od)
        prof = {f'{o["fit"]["key"]}:{k}:{nm}_{q}': arr for o in r
                for (k, nm), p in o.get('profiles', {}).items() for q, arr in zip(('off', 'flux'), p)}
        np.savez_compressed(od / 'spider_profiles.npz', **prof)
        tf = pa.translation_fit(tabs['spiders'])
        pa._save(tf, od, 'translation_fit')
        try:
            ct, _ = pa.centre_test(tabs['spiders'], tabs['centres'], tabs['sides'])
            pa._save(ct, od, 'centre_test')
        except Exception as err:  # no shadows found: an empty table, not a crashed run
            print('centre_test', v, repr(err), flush=True)
            ct = pd.DataFrame(columns=['centre', 'side', 'm1_amp_deg', 'inner_edge_dy_ocs_pix',
                                       'outer_edge_dy_ocs_pix'])
        sh = shadow_shape(tabs['spiders'])
        f = tabs['fits']
        for side in ('extra', 'intra'):
            c = ct[(ct.centre == 'fit_a') & (ct.side == side)]
            g = tf[tf.side == side]
            h = f[f.side == side]
            rows.append(dict(variant=v, side=side, n_sides=len(h), chi2_dof=h.chi2_dof.median(),
                             dx_pix=g.dx_pix.median(), dy_pix=g.dy_pix.median(),
                             rot_deg=g.rot_deg.median(),
                             m1_amp_deg=float(c.m1_amp_deg.iloc[0]) if len(c) else np.nan,
                             inner_dy_pix=float(c.inner_edge_dy_ocs_pix.iloc[0]) if len(c) else np.nan,
                             outer_dy_pix=float(c.outer_edge_dy_ocs_pix.iloc[0]) if len(c) else np.nan,
                             edge_inner_pix=h.edge_median_inner_pix.median(),
                             edge_outer_pix=h.edge_median_outer_pix.median(),
                             ratio_excess=h.ratio_excess.median(),
                             depth=sh.depth.get(side, np.nan), fwhm_pix=sh.fwhm_pix.get(side, np.nan),
                             ew_pix=sh.ew_pix.get(side, np.nan)))
    s = pd.DataFrame(rows)
    pa._save(s, OUT, 'huygens_summary')
    return s


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=110)
    ap.add_argument('--test', action='store_true',
                    help='1 side, 1 wavelength, 3 mm stop grid (ghosts in the stamp): code check only')
    ap.add_argument('--validate', action='store_true',
                    help='1 science and 1 corner side, 620 nm, full grid, to huygens_validate/')
    ap.add_argument('--blocks', type=int, default=N_BLOCKS)
    args = ap.parse_args()
    global OUT
    C = pd.read_parquet(psim.SIM_DIR / 'sim_sides.parquet')
    wls, spacing = WAVELENGTHS_M, STOP_SPACING_M
    if args.test:
        OUT = p6.RUN / 'huygens_test'
        C = C.head(1)
        wls, spacing = (620e-9,), 3e-3
    if args.validate:
        OUT = p6.RUN / 'huygens_validate'
        C = pd.concat([C[C.slot == 'science'].head(1), C[C.slot == 'corner'].tail(1)])
        wls = (620e-9,)
    OUT.mkdir(parents=True, exist_ok=True)
    nb = 1 if args.test else args.blocks
    tasks = [(k, wl, spacing, b, nb) for k in C.key for wl in wls for b in range(nb)]
    npix = {k: np.load(p6.RUN / 'isr_stamps' / f'{r.sample_id}_{r.side}.npz')['img'].shape[0]
            for k, r in C.set_index('key').iterrows()}
    t0 = time.time()
    acc, geo, draws, Uacc, nseen = {}, {}, [], {}, {}
    with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
        for key, wl, names, U, g, info in pool.imap_unordered(draw_task, tasks):
            k = (key, wl)
            Uacc[k] = U if k not in Uacc else Uacc[k] + U
            nseen[k] = nseen.get(k, 0) + 1
            geo[key] = geo.get(key, 0) + g
            draws.append(info)
            if nseen[k] == nb:
                a = acc.setdefault(key, {})
                for v, im in zip(names, to_pixels(Uacc.pop(k), npix[key])):
                    a[v] = a.get(v, 0) + im
    pd.DataFrame(draws).to_parquet(OUT / 'draws.parquet')
    print(f'draw wall {time.time() - t0:.0f} s', flush=True)
    fit_tasks = []
    for key in C.key:
        imgs = {'geometric': geo[key], **acc[key]}
        noisy, msk = finish_side(key, imgs)
        fit_tasks += [(key, v, noisy[v], msk) for v in VARIANTS]
    t1 = time.time()
    res = {v: [] for v in VARIANTS}
    with mp.get_context('fork').Pool(min(args.workers, len(fit_tasks))) as pool:
        for v, out in pool.imap_unordered(fit_task, fit_tasks):
            if out is not None:
                res[v].append(out)
    print(f'fit wall {time.time() - t1:.0f} s', flush=True)
    s = summarize(res, C)
    pd.set_option('display.width', 250)
    print(s.round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
