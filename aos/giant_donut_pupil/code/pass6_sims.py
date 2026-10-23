"""Pupil pass 6, step 3: simulated giant donuts through the data pipeline.

(a) Sides (`choose`): 12 science donuts with both sides fitted and all 8 shadows found in
    pass 5, one per OCS field-azimuth quadrant in the central band (nearest 0.5 deg) and
    two per quadrant in the outer band (nearest 1.50 and 1.68 deg), brighter than r = 14;
    plus 2 corner sides (the brightest SW1 extra and SW0 intra with 8 shadows).
(b) Image (`draw`), per side and variant:
    - Batoid v4.02 (full model, then the reduced model of the surfaces that clip a ray,
      as `batoid_images.side_image`), design state, the side's field, giant offsets and
      rotator angle (physical_rotator_angle): the telescope, not the camera, rotated by
      −rtp;
    - a stratified-random stop-plane grid of about ``N_RAYS`` rays, 620 nm;
    - binned on the data stamp's pixel grid (CCS, Batoid detector x, y = column, row),
      the ray centroid on the data fit c model's centroid;
    - convolved with the danish Kolmogorov kernel at ``SIM_FWHM_ARCSEC`` (1.00 arcsec
      extra, 1.60 arcsec intra);
    - scaled to the data's fit c flux, the data's sky level added as a constant, Poisson
      noise.
    Variants: ``nominal``; ``rot20``, drawn at rtp + 0.20 deg; ``tea_y10``, the TEA
    (``Spider*``, ``TopEnd*``, ``TEACentralShade``) shifted +10 mm in telescope y before
    the rotation, as `tea_shift_check.py` in rubin-optical-model.
(c) Fit (`fit`): the sky constant re-estimated and subtracted as `fit_donuts.cut` does
    (sigma-clipped, outside 1.1 r_out, data neighbour mask), then `pass6_pipeline.measure`
    at rtp = physical_rotator_angle, blur fixed at the simulation's, v4.02g (and v4.02s
    for ``nominal``).

Outputs in ``<run>/sims/``: ``sim_sides.parquet``, ``draws.parquet``,
``stamps/<key>_<variant>.npz``, and per fit configuration ``<variant>_<mask>/`` with the
pass 5 table shapes, read by `pass5_analysis.run_all`.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass6_sims.py --steps draw,fit --workers 80
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass6_common as p6  # noqa: E402
import pass6_pipeline as pp  # noqa: E402
import pass5_analysis as pa  # noqa: E402
import batoid_images as bi  # noqa: E402
import fit_donuts as fd  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import pupil_pass3 as p3  # noqa: E402

N_RAYS = 30_000_000
WAVELENGTH_M = 620e-9
VARIANTS = {'nominal': dict(d_rtp=0.0, tea_y=0.0), 'rot20': dict(d_rtp=0.20, tea_y=0.0),
            'tea_y10': dict(d_rtp=0.0, tea_y=0.010)}
FITS = (('nominal', 'v4.02g'), ('nominal', 'v4.02s'), ('rot20', 'v4.02g'),
        ('tea_y10', 'v4.02g'))
SIM_DIR = p6.RUN / 'sims'
DATA_CONFIG = 'isr_phys'


def choose(S, t5):
    """The 12 donuts (24 sides) and 2 corner sides."""
    nf = t5[t5.centre == 'fit_a'].groupby('key').missing.apply(lambda m: int((m == '').sum()))
    S = S.copy()
    S['n_found5'] = S.key.map(nf)
    sc = S[(S.sensor_class == 'science') & (S.mag < 14.0)]
    good = sc.groupby('sample_id').filter(
        lambda g: set(g.side) == {'extra', 'intra'} and g.fit_ok.all() and (g.n_found5 == 8).all())
    d = good.drop_duplicates('sample_id').copy()
    d['quad'] = (d.field_az_ocs_deg // 90).astype(int)
    pick = []
    for q in range(4):
        for target in (0.5, 1.50, 1.68):
            g = d[(d.quad == q) & ~d.sample_id.isin(pick)]
            g = g[(g.field_r_deg <= 0.8) if target < 1 else (g.field_r_deg >= 1.4)]
            if len(g):
                pick.append(g.iloc[np.argmin(np.abs(g.field_r_deg - target))].sample_id)
    out = S[S.sample_id.isin(pick)].copy()
    out['slot'] = 'science'
    cor = S[(S.sensor_class != 'science') & S.fit_ok & (S.n_found5 == 8)]
    for cls, side in (('SW1', 'extra'), ('SW0', 'intra')):
        g = cor[(cor.sensor_class == cls) & (cor.side == side)].sort_values('mag')
        if len(g):
            out = pd.concat([out, g.head(1).assign(slot='corner')])
    return out.reset_index(drop=True)


def tea_items(optic):
    return [it.name for it in optic.items
            if it.name.startswith(('Spider', 'TopEnd')) or it.name == 'TEACentralShade']


def configure(optic, rtp_deg, offsets, tea_y):
    if tea_y:
        for name in tea_items(optic):
            optic = optic.withGloballyShiftedOptic(name, [0.0, tea_y, 0.0])
    return bi.configure(optic, rtp_deg, offsets)


def data_model_centroid(row, fit):
    """Centroid (column, row) of the data's fit c model, and the stamp size."""
    d = np.load(p6.RUN / 'isr_stamps' / f'{row.sample_id}_{row.side}.npz')
    n = d['img'].shape[0]
    M = pp.Model(row, 1.0)
    m0 = M.single(n, p3.FITS['c'])
    z = np.array([fit[f'z{j}_ccs_um'] for j in p3.FITS['c']]) * 1e-6
    x = np.concatenate([[fit['flux_e'], fit['dx_pix'] * m0.sky_scale, fit['dy_pix'] * m0.sky_scale], z])
    img = M.render(n, x, p3.FITS['c'])
    yy, xx = np.mgrid[:n, :n]
    return (float((img * xx).sum() / img.sum()), float((img * yy).sum() / img.sum())), n, d


def draw(task):
    """One simulated stamp; writes ``stamps/<key>_<variant>.npz``."""
    row, variant, fit, sky_e = task
    t0 = time.time()
    v = VARIANTS[variant]
    rtp = float(row.rtp_deg) + v['d_rtp']
    thx, thy = np.radians(row.field_x_ccs_deg), np.radians(row.field_y_ccs_deg)
    offsets = fd.offsets_m(row.side, row.sensor_class)
    full = configure(ob.v402_optic(), rtp, offsets, v['tea_y'])
    cx, cy = bi.pupil_grid(2 * bi.PUPIL_OUTER_M / bi.CLIP_GRID)
    first, names, _, _ = bi.first_clip(full, cx, cy, thx, thy)
    leaf = tuple(n.split('.')[-1] for n in sorted({names[k] for k in np.unique(first[first >= 0])}))
    reduced = configure(ob.reduced_v402(keep=tuple(set(leaf) | set(bi.OPTICAL))), rtp, offsets,
                        v['tea_y'])
    kx, ky = bi.pupil_grid(2 * bi.PUPIL_OUTER_M / bi.CHECK_GRID, offset=0.21)
    agreement = float(np.mean(bi.transmitted(full, kx, ky, thx, thy)[0]
                              == bi.transmitted(reduced, kx, ky, thx, thy)[0]))
    optic = full if agreement < bi.MIN_AGREEMENT else reduced
    area = np.pi * (bi.PUPIL_OUTER_M ** 2 - bi.PUPIL_INNER_M ** 2)
    spacing = np.sqrt(area / N_RAYS)
    rng = np.random.default_rng(zlib.crc32(f'{row.key}_{variant}'.encode()))
    gx, gy = bi.pupil_grid(spacing, rng=rng)
    xs, ys = [], []
    for i in range(0, len(gx), bi.CHUNK):
        k, x, y = bi.transmitted(optic, gx[i:i + bi.CHUNK], gy[i:i + bi.CHUNK], thx, thy)
        xs.append(x[k])
        ys.append(y[k])
    x, y = np.concatenate(xs), np.concatenate(ys)
    mc, n, d = data_model_centroid(row, fit)
    u = (x - x.mean()) / bi.PIXEL_M + mc[0]
    w = (y - y.mean()) / bi.PIXEL_M + mc[1]
    sky_scale = 3600 * np.rad2deg(bi.PIXEL_M / 10.312)
    fwhm = p6.SIM_FWHM_ARCSEC[row.side]
    _, img = bi.render(u, w, n, bi.atm_kernel(n, fwhm, sky_scale))
    img = np.clip(img, 0, None) * (fit['flux_e'] / img.sum())
    counts = rng.poisson(img + sky_e).astype(np.float32)
    out = SIM_DIR / 'stamps' / f'{row.sample_id}_{row.side}_{variant}.npz'
    np.savez_compressed(out, img=counts, truth=img.astype(np.float32), masked=d['masked'],
                        sky_e=sky_e)
    r = dict(key=row.key, variant=variant, rtp_draw_deg=rtp, tea_y_m=v['tea_y'], fwhm_arcsec=fwhm,
             n_rays=int(len(gx)), n_transmitted=int(len(x)), agreement=agreement,
             used_full=bool(agreement < bi.MIN_AGREEMENT), clippers=';'.join(leaf),
             centroid_x_pix=mc[0], centroid_y_pix=mc[1], flux_e=float(fit['flux_e']),
             sky_e=float(sky_e), seconds=time.time() - t0)
    print(json.dumps({k: (round(q, 4) if isinstance(q, float) else q) for k, q in r.items()
                      if k != 'clippers'}), flush=True)
    return r


def subtract_sky(img, msk, r_out):
    from astropy.stats import sigma_clipped_stats
    n = img.shape[0]
    yy, xx = np.mgrid[:n, :n] - n // 2
    r = np.hypot(xx, yy)
    pix = img[(r > 1.1 * r_out) & ~msk]
    _, sky, _ = sigma_clipped_stats(pix if pix.size > 1000 else img[r > 1.1 * r_out],
                                    sigma=3.0, maxiters=5)
    return img - sky, float(sky)


def fit(task):
    row, variant, mask = task
    d = np.load(SIM_DIR / 'stamps' / f'{row.sample_id}_{row.side}_{variant}.npz')
    msk = d['masked'].astype(bool)
    img, sky = subtract_sky(d['img'].astype(float), msk, row.r_out_pix)
    try:
        out = pp.measure(row, img, msk, p6.SIM_FWHM_ARCSEC[row.side], mask=mask,
                         keep_images=(variant == 'nominal'), profiles=(variant == 'nominal'))
    except Exception as err:
        print('error', variant, mask, row.key, repr(err), flush=True)
        return variant, mask, None
    out['fit']['sky_fit_e'] = sky
    out['fit']['sky_true_e'] = float(d['sky_e'])
    if 'images' in out:
        im = out.pop('images')
        np.savez_compressed(SIM_DIR / 'stamps' / f'{row.sample_id}_{row.side}_{variant}_{mask}_model.npz',
                            img=img.astype(np.float32), model=im['model'], center=np.array(im['center']),
                            r_edge=im['r_edge'],
                            **{f'line_{nm}_{k}': np.array([*t['point'], *(t['direction'] if np.isfinite(
                                t['impact_pix']) else (np.nan, np.nan)), rh], np.float32)
                               for nm, g in (('data', im['geom']), ('model', im['geom_m']))
                               for k, (t, rh) in g.items()})
    print(variant, mask, row.key, f'{out["fit"]["seconds"]:.0f} s', 'found', out['fit']['n_found'],
          flush=True)
    return variant, mask, out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--steps', default='draw,fit')
    ap.add_argument('--workers', type=int, default=80)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    steps = args.steps.split(',')
    (SIM_DIR / 'stamps').mkdir(parents=True, exist_ok=True)
    S = p6.with_rtp(p6.sides(), 'phys')
    t5 = pd.read_parquet(p6.PASS5_RUN / 'spiders.parquet')
    if (SIM_DIR / 'sim_sides.parquet').exists():
        C = pd.read_parquet(SIM_DIR / 'sim_sides.parquet')
    else:
        C = choose(S, t5)
        C.to_parquet(SIM_DIR / 'sim_sides.parquet')
        C.to_csv(SIM_DIR / 'sim_sides.csv', index=False, float_format='%.6g')
    if args.limit:
        C = C.head(args.limit)
    print('sim sides', len(C), C.sample_id.nunique(), 'donuts', flush=True)
    variants = args.variants.split(',')
    if 'draw' in steps:
        F = pd.read_parquet(p6.RUN / DATA_CONFIG / 'fits.parquet').set_index('key')
        sky = pd.read_parquet(p6.RUN / 'isr_stamps.parquet').set_index('key').sky_e
        tasks = [(r, v, F.loc[r.key].to_dict(), float(sky[r.key])) for _, r in C.iterrows()
                 for v in variants]
        t0 = time.time()
        with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
            rows = pool.map(draw, tasks, chunksize=1)
        dr = pd.DataFrame(rows)
        if (SIM_DIR / 'draws.parquet').exists():
            old = pd.read_parquet(SIM_DIR / 'draws.parquet')
            dr = pd.concat([old[~old.set_index(['key', 'variant']).index.isin(
                dr.set_index(['key', 'variant']).index)], dr], ignore_index=True)
        dr.to_parquet(SIM_DIR / 'draws.parquet')
        dr.to_csv(SIM_DIR / 'draws.csv', index=False, float_format='%.6g')
        print(f'draw wall {time.time() - t0:.0f} s', flush=True)
    if 'fit' in steps:
        cfgs = [(v, m) for v, m in FITS if v in variants]
        tasks = [(r, v, m) for v, m in cfgs for _, r in C.iterrows()]
        t0 = time.time()
        res = {c: [] for c in cfgs}
        with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
            for v, m, out in pool.imap_unordered(fit, tasks):
                if out is not None:
                    res[(v, m)].append(out)
        for (v, m), r in res.items():
            out_dir = SIM_DIR / f'{v}_{m.replace(".", "")}'
            tabs = pp.tables(r, C)
            pp.save(tabs, out_dir)
            if v == 'nominal':
                prof = {f'{o["fit"]["key"]}:{k}:{nm}_{q}': arr for o in r
                        for (k, nm), p in o.get('profiles', {}).items()
                        for q, arr in zip(('off', 'flux'), p)}
                np.savez_compressed(out_dir / 'spider_profiles.npz', **prof)
            a = pa.run_all(out_dir)
            print(v, m, 'sides', len(tabs['fits']))
            print(a['ts'].round(3).to_string(index=False))
            print(a['rs'].round(3).to_string(index=False))
        print(f'fit wall {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
