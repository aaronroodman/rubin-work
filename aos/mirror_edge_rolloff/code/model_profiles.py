"""Batoid model rim profiles of each giant-donut side on a grid of M1 roll-off, M3 turn-up and blur.

Per side of `giant_donuts` (science sides at 1.6-1.725 deg by default):

1. **Coarse trace, full v4.02 with vanes** (`CLIP_GRID` stop points per side length): the
   ray centroid sets the image center; the first clipping surface of the rays landing
   just beyond each edge labels the edge's defining aperture per 2 deg bin; vane-blocked
   rays landing within `PAD_PIX` of an edge cut that bin for that edge.
2. **Dense trace, reduced v4.02 without vanes** (the surfaces that clip, plus the optics):
   stop points at `DENSITY_RAYS` per full annulus, kept in `ZONES_M` (the stop radii that
   land within about 30 pixels of the inner and outer rims). One `traceFull` gives each
   ray's M1 and M3 hit points (mirror frame) and detector position.
3. **Slope response:** two more traces each with a small Zernike tip and tilt on M1 (zone 1
   rays) and on M3 (zone 2 rays) give each ray's detector shift per unit surface slope, a
   2 x 2 matrix J. A surface perturbation dz moves a ray by J ∇dz at its hit point.
4. **Grids:** M1 roll-off dz = −A exp(−(r − 2.558 m)/w) (SUMMARY.md form) over `M1_GRID`;
   M3 turn-up dz = +B exp(−(R3 − r)/u), R3 the M3 outer clear radius, over `M3_GRID`. Each
   model's rays are binned on the stamp grid and convolved with the danish Kolmogorov
   kernel at each `FWHM_GRID` value; `rubinwork.giant_donuts.edge_profiles.profiles` gives
   the per-bin radial profile about the image center (electrons per pixel, scaled to the
   data flux), kept in a 50-pixel window around each bin's nominal edge (inner rim for the
   M1 grid, outer rim for the M3 grid).

Writes ``profiles/<key>.npz`` into the run, one per side, and ``sides.csv``.

Usage, from the repo root::

    python aos/mirror_edge_rolloff/code/model_profiles.py --workers 100
    python aos/mirror_edge_rolloff/code/model_profiles.py --test   # one side, few rays
"""
import argparse
import json
import pathlib
import time
import zlib

import numpy as np

from rubinwork.giant_donuts import batoid_model as bm
from rubinwork.giant_donuts import edge_profiles as ep

STUDY = 'aos/mirror_edge_rolloff'
VARIANT = 'v402s_fitc'
DENSITY_RAYS = 30_000_000
ZONES_M = ((2.30, 3.10), (3.60, 4.25))
CLIP_GRID = 1000
PAD_PIX = 6.0
WINDOW_PIX = 25.0
TILT_M = 1e-7
R0_M1_M = 2.558
M1_GRID = {'A_nm': [0, 25, 50, 75, 100, 150, 200, 300], 'w_mm': [8, 12, 18, 27, 40, 60]}
M3_GRID = {'B_nm': [0, 40, 80, 160, 320], 'u_mm': [10, 25, 50]}
FWHM_GRID = [0.8, 1.0, 1.2, 1.4, 1.7, 2.0]
SAMPLE = dict(sensor_class='science', field_r_deg=(1.6, 1.725))


def sample(tables):
    s = tables['sides'].merge(tables['fits'][['key', 'fit_ok', 'noise_e', 'cx_a', 'cy_a',
                                              'flux_e']], on='key')
    s = s[(s.sensor_class == SAMPLE['sensor_class']) & s.field_r_deg.between(*SAMPLE['field_r_deg'])
          & s.fit_ok]
    return s.reset_index(drop=True)


def _surface_key(names, leaf):
    hits = [n for n in names if n.split('.')[-1] == leaf]
    if len(hits) != 1:
        raise KeyError(f'{leaf}: {hits}')
    return hits[0]


def _trace_full(optic, x, y, thx, thy, chunk=200_000):
    """Detector position, transmission and M1 / M3 hit points (local frames), m."""
    out = {k: np.empty(len(x)) for k in ('dx', 'dy', 'm1x', 'm1y', 'm3x', 'm3y')}
    ok = np.empty(len(x), bool)
    for i in range(0, len(x), chunk):
        s = slice(i, i + chunk)
        tf = optic.traceFull(bm.rays(optic, x[s], y[s], thx, thy))
        names = list(tf)
        k1, k3 = _surface_key(names, 'M1'), _surface_key(names, 'M3')
        last = tf[names[-1]]['out']
        out['dx'][s], out['dy'][s] = last.x, last.y
        ok[s] = ~np.asarray(last.vignetted) & ~np.asarray(last.failed)
        out['m1x'][s], out['m1y'][s] = tf[k1]['out'].x, tf[k1]['out'].y
        out['m3x'][s], out['m3y'][s] = tf[k3]['out'].x, tf[k3]['out'].y
    return out, ok


def _response(optic, mirror, r_outer, x, y, thx, thy, base_x, base_y):
    """Detector shift per unit surface slope (2 x 2 per ray), from Zernike tip and tilt."""
    import batoid
    J = np.empty((len(x), 2, 2))
    for col, coef in ((0, [0, 0, TILT_M]), (1, [0, 0, 0, TILT_M])):
        o = optic.withPerturbedSurface(mirror, batoid.Zernike(coef, R_outer=r_outer))
        _, px, py = bm.transmitted(o, x, y, thx, thy)
        slope = 2 * TILT_M / r_outer
        J[:, 0, col] = (px - base_x) / slope
        J[:, 1, col] = (py - base_y) / slope
    return J


def _m3_outer_radius():
    d = bm.v402_dict()
    for it in d['opticalSystem']['items']:
        if it['name'] == 'M3':
            return float(it['obscuration']['outer'])
    raise KeyError('M3 outer radius')


def side_task(task):
    row, out_dir, density, grids = task
    import batoid  # noqa: F401
    t0 = time.time()
    path = pathlib.Path(out_dir) / 'profiles' / f'{row["key"]}.npz'
    if path.exists():
        return row['key'], 'exists'
    thx, thy = np.radians(row['field_x_ccs_deg']), np.radians(row['field_y_ccs_deg'])
    offsets = bm.offsets_m(row['side'], row['sensor_class'], row['step_mm'])
    rtp = float(row['rtp_deg'])
    n = int(row['npix'])
    r_out = float(row['r_out_pix'])
    noise = float(row['noise_e'])

    # 1. coarse trace with vanes: center, edge labels, spider cut
    full = bm.configure(bm.v402_optic(spider=True), rtp, offsets)
    cx, cy = bm.pupil_grid(2 * bm.PUPIL_OUTER_M / CLIP_GRID)
    first, names, fx, fy = bm.first_clip(full, cx, cy, thx, thy)
    leaf = np.array([nm.split('.')[-1] for nm in names] + [''])
    first_leaf = leaf[first]
    spider = np.array([str(v).startswith(bm.SPIDER_PREFIXES) for v in first_leaf])
    lit = (first < 0) | spider
    x0, y0 = fx[lit].mean(), fy[lit].mean()
    cpx, cpy = (fx - x0) / bm.PIXEL_M + n // 2, (fy - y0) / bm.PIXEL_M + n // 2
    clip_names = sorted({first_leaf[k] for k in np.nonzero(first >= 0)[0]})

    # 2. dense trace, reduced optic without vanes
    keep = tuple(set(clip_names) - {nm for nm in clip_names if nm.startswith(bm.SPIDER_PREFIXES)}
                 | set(bm.OPTICAL))
    red = bm.configure(bm.reduced_v402(keep=keep), rtp, offsets)
    area = np.pi * (bm.PUPIL_OUTER_M ** 2 - bm.PUPIL_INNER_M ** 2)
    rng = np.random.default_rng(zlib.crc32(row['key'].encode()))
    gx, gy = bm.pupil_grid(np.sqrt(area / density), rng=rng)
    # electrons per dense ray: the data flux over the transmitted share of the full grid
    e_per_ray = float(row['flux_e']) / (len(gx) * lit.mean())
    gr = np.hypot(gx, gy)
    zone = np.full(len(gx), -1)
    for k, (lo, hi) in enumerate(ZONES_M):
        zone[(gr >= lo) & (gr <= hi)] = k
    gx, gy, zone = gx[zone >= 0], gy[zone >= 0], zone[zone >= 0]
    tr, ok = _trace_full(red, gx, gy, thx, thy)
    sel = ok
    gx, gy, zone = gx[sel], gy[sel], zone[sel]
    tr = {k: v[sel] for k, v in tr.items()}

    # 3. slope responses
    z1, z3 = zone == 0, zone == 1
    J1 = _response(red, 'M1', 4.18, gx[z1], gy[z1], thx, thy, tr['dx'][z1], tr['dy'][z1])
    r3_outer = _m3_outer_radius()
    J3 = _response(red, 'M3', r3_outer, gx[z3], gy[z3], thx, thy, tr['dx'][z3], tr['dy'][z3])

    # 4. grids
    px = (tr['dx'] - x0) / bm.PIXEL_M + n // 2
    py = (tr['dy'] - y0) / bm.PIXEL_M + n // 2
    edges = np.arange(n + 1) - 0.5
    sky_scale = 3600 * np.rad2deg(bm.PIXEL_M / bm.FOCAL_LENGTH_M)
    kernels = [bm.atm_kernel(n, f, sky_scale) for f in FWHM_GRID]
    from scipy.signal import fftconvolve

    def hist(ux, uy, weight=e_per_ray):
        h, _, _ = np.histogram2d(uy, ux, bins=[edges, edges])
        return h * weight

    h_fixed = {0: hist(px[z3], py[z3]), 1: hist(px[z1], py[z1])}
    valid = np.ones((n, n), bool)
    c = n // 2

    # Nominal edges (no roll-off, 1.0 arcsec) set the windows and the labels. From the
    # coarse full-pupil trace: the dense zones leave the flat-top window empty.
    t = first < 0
    nominal = fftconvolve(hist(cpx[t], cpy[t], float(row['flux_e']) / t.sum()),
                          kernels[FWHM_GRID.index(1.0)], mode='same')
    nom = ep.robust_edges(nominal, valid, c, c, r_out, noise, np.zeros(ep.NBIN, bool),
                          np.zeros(ep.NBIN, bool), pad_bins=0)
    e_in, e_out = nom['fit_inner'], nom['fit_outer']
    rad = (np.arange(int(1.3 * r_out / ep.DR_PIX) + 1) + 0.5) * ep.DR_PIX
    nwin = int(2 * WINDOW_PIX / ep.DR_PIX)
    start_in = np.clip(np.round((e_in - WINDOW_PIX) / ep.DR_PIX).astype(int), 0, len(rad) - nwin)
    start_out = np.clip(np.round((e_out - WINDOW_PIX) / ep.DR_PIX).astype(int), 0, len(rad) - nwin)
    take = np.arange(nwin)

    def windows(img, start):
        prof = ep.profiles(img, valid, c, c, r_out, noise)[0]
        return prof[np.arange(ep.NBIN)[:, None], start[:, None] + take[None, :]].astype(np.float32)

    r1 = np.hypot(tr['m1x'][z1], tr['m1y'][z1])
    u1 = np.stack([tr['m1x'][z1], tr['m1y'][z1]], 1) / r1[:, None]
    r3 = np.hypot(tr['m3x'][z3], tr['m3y'][z3])
    u3 = np.stack([tr['m3x'][z3], tr['m3y'][z3]], 1) / r3[:, None]
    A, W = np.meshgrid(grids['M1']['A_nm'], grids['M1']['w_mm'], indexing='ij')
    prof_m1 = np.empty((A.size, len(FWHM_GRID), ep.NBIN, nwin), np.float32)
    for k, (a, w) in enumerate(zip(A.ravel() * 1e-9, W.ravel() * 1e-3)):
        g = (a / w) * np.exp(-(r1 - R0_M1_M) / w)  # d(dz)/dr for dz = -a exp(-(r-R0)/w)
        d = np.einsum('nij,nj->ni', J1, g[:, None] * u1) / bm.PIXEL_M
        h = h_fixed[0] + hist(px[z1] + d[:, 0], py[z1] + d[:, 1])
        for f, ker in enumerate(kernels):
            prof_m1[k, f] = windows(fftconvolve(h, ker, mode='same'), start_in)
    B, U = np.meshgrid(grids['M3']['B_nm'], grids['M3']['u_mm'], indexing='ij')
    prof_m3 = np.empty((B.size, len(FWHM_GRID), ep.NBIN, nwin), np.float32)
    for k, (b, u) in enumerate(zip(B.ravel() * 1e-9, U.ravel() * 1e-3)):
        g = (b / u) * np.exp(-(r3_outer - r3) / u)  # d(dz)/dr for dz = +b exp(-(R3-r)/u)
        d = np.einsum('nij,nj->ni', J3, g[:, None] * u3) / bm.PIXEL_M
        h = h_fixed[1] + hist(px[z3] + d[:, 0], py[z3] + d[:, 1])
        for f, ker in enumerate(kernels):
            prof_m3[k, f] = windows(fftconvolve(h, ker, mode='same'), start_out)

    # labels and spider cut per bin and edge, from the coarse trace
    rr = np.hypot(cpx - c, cpy - c)
    phi = np.mod(np.degrees(np.arctan2(cpy - c, cpx - c)), 360.0)
    kb = np.minimum((phi / (360 / ep.NBIN)).astype(int), ep.NBIN - 1)
    lab_in = np.full(ep.NBIN, '', object)
    lab_out = np.full(ep.NBIN, '', object)
    cut_in = np.zeros(ep.NBIN, bool)
    cut_out = np.zeros(ep.NBIN, bool)
    blocked = (first >= 0) & ~spider
    for b in range(ep.NBIN):
        m = kb == b
        if np.isfinite(e_in[b]):
            just = m & blocked & (rr > e_in[b] - 8) & (rr < e_in[b])
            if just.any():
                v, cnt = np.unique(first_leaf[just], return_counts=True)
                lab_in[b] = v[np.argmax(cnt)]
            cut_in[b] = (m & spider & (np.abs(rr - e_in[b]) <= PAD_PIX)).any()
        if np.isfinite(e_out[b]):
            just = m & blocked & (rr > e_out[b]) & (rr < e_out[b] + 8)
            if just.any():
                v, cnt = np.unique(first_leaf[just], return_counts=True)
                lab_out[b] = v[np.argmax(cnt)]
            cut_out[b] = (m & spider & (np.abs(rr - e_out[b]) <= PAD_PIX)).any()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, prof_m1=prof_m1, prof_m3=prof_m3, start_in=start_in,
                        start_out=start_out, e_in=e_in, e_out=e_out,
                        A_nm=A.ravel(), w_mm=W.ravel(), B_nm=B.ravel(), u_mm=U.ravel(),
                        fwhm_arcsec=np.array(FWHM_GRID), label_inner=lab_in.astype(str),
                        label_outer=lab_out.astype(str), cut_inner=cut_in, cut_outer=cut_out,
                        r3_outer_m=r3_outer, n_rays=len(gx))
    tmp.rename(path)
    info = dict(key=row['key'], n_rays=int(len(gx)), seconds=round(time.time() - t0))
    print(json.dumps(info), flush=True)
    return row['key'], 'done'


def main():
    import multiprocessing as mp
    from rubinwork.common import study_run
    from rubinwork.products import giant_donuts
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--test', action='store_true', help='one side, 1 M rays, a small grid')
    p.add_argument('--out', default=None, help='output directory; a new study run by default')
    a = p.parse_args()
    S = sample(giant_donuts.load(VARIANT))
    density, grids = DENSITY_RAYS, {'M1': M1_GRID, 'M3': M3_GRID}
    if a.test:
        S = S.head(1)
        density = 1_000_000
        grids = {'M1': {'A_nm': [0, 76.4], 'w_mm': [27.1]}, 'M3': {'B_nm': [0, 80], 'u_mm': [25]}}
    out = pathlib.Path(a.out) if a.out else study_run(STUDY, 'model_profiles').path
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / 'sides.csv', index=False, float_format='%.6g')
    tasks = [(r, str(out), density, grids) for r in S.to_dict('records')]
    if a.workers > 1:
        with mp.get_context('spawn').Pool(min(a.workers, len(tasks)), maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(side_task, tasks):
                pass
    else:
        for t in tasks:
            side_task(t)
    print('done', out, len(S), 'sides', flush=True)


if __name__ == '__main__':
    main()
