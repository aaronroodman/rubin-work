"""Pupil pass 5, step 2: which v4.02 surface sets the inner and outer donut edge, per psi.

One field, 1.7 deg at focal-plane azimuth 45 deg, science sensor, at the two giant
configurations (detector, camera, M2) = (0, -4, -4) mm intra and (0, +4, +4) mm extra.
Rotator angle 0, so OCS = CCS.

1. A polar grid on the stop plane (``NRAD`` x ``NAZ``, 1.5-4.6 m, wider than the
   annulus) is traced through the full batoid v4.02 (spider on) with ``traceFull``; per
   ray, the first surface that vignettes it (`batoid_images.first_clip`) and its
   detector position (batoid traces blocked rays on, so a blocked ray has the position
   it would have reached). Surfaces are grouped by element (``TMASection003_top`` ->
   ``TMASection``). Rays first blocked by a spider vane are left out of the attribution;
   a psi bin where they are over ``SPIDER_FRAC`` of the rays reaching the ring, and its
   two neighbours, is a ``spider`` bin, as the data's spider bins.
2. Edges per 1 deg psi bin, about the centroid of the transmitted rays: the smallest
   and largest transmitted radius (pixels of 10 um).
3. The surface setting an edge in a bin is the commonest first clipper among blocked
   rays landing within ``BAND_PIX`` outside it (inside the inner edge, outside the outer
   edge).
4. The danish v4.02g mask at the same field, on the same stop-plane rays and Batoid's
   ray positions, gives the mask edges, so the mask-against-Batoid gap is in pixels.

psi = (az - (phi + 180)) mod 360, az the image azimuth about the centroid and phi = 45
deg, as pass 4.

Outputs in the run: ``vignetting_bins.parquet`` (per side and psi bin: edges and
surface), ``vignetting_ranges.parquet``/``.csv`` (psi ranges per surface, edge, side, after a
7 deg running mode that absorbs the 1-3 deg fragments next to the spider shadows),
``vignetting_images.npz`` (Batoid image per side).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass5_vignetting.py --workers 64
"""
import argparse
import os
import pathlib
import re
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass5_common as pc  # noqa: E402
import batoid_images as bi  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import obscuration_v402g as og  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

FIELD_R_DEG = 1.7
FIELD_AZ_DEG = 45.0
CONFIGS = {'intra': (0.0, -4.0, -4.0), 'extra': (0.0, 4.0, 4.0)}
NRAD, NAZ = 700, 1800
R_IN_M, R_OUT_M = 1.5, 4.6
BAND_PIX = 3.0
SPIDER_FRAC = 0.05
MODE_HALF_DEG = 3
NPSI = 360
PIXEL_M = 10e-6
CHUNK = 20_000
SPIDER = ('SpiderFlat', 'SpiderInclined')
_OPT = {}


def group_of(name):
    leaf = name.split('.')[-1]
    return re.sub(r'_z\d.*$', '', re.sub(r'\d{3}_(top|bottom)$', '', leaf))


def optic_for(side):
    if side not in _OPT:
        _OPT[side] = og.shift(ob.v402_optic(spider=True), CONFIGS[side])
    return _OPT[side]


def stop_grid():
    r = np.linspace(R_IN_M, R_OUT_M, NRAD)
    a = (np.arange(NAZ) + 0.5) * 2 * np.pi / NAZ
    rr, aa = np.meshgrid(r, a)
    return (rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel()


def field():
    th = np.deg2rad(FIELD_R_DEG)
    return th * np.cos(np.deg2rad(FIELD_AZ_DEG)), th * np.sin(np.deg2rad(FIELD_AZ_DEG))


def _chunk(task):
    side, i0, i1 = task
    x, y = stop_grid()
    thx, thy = field()
    optic = optic_for(side)
    rays = bi._rays(optic, x[i0:i1], y[i0:i1], thx, thy)
    tf = optic.traceFull(rays)
    names = list(tf)
    first = np.full(i1 - i0, -1, np.int32)
    failed = np.zeros(i1 - i0, bool)
    for k, key in enumerate(names):
        out = tf[key]['out']
        vig = np.asarray(out.vignetted)
        f = np.asarray(out.failed)
        first[vig & (first < 0)] = k
        failed |= f
    det = tf[names[-1]]['out']
    u, v = ob._stop_xy(optic, bi._rays(optic, x[i0:i1], y[i0:i1], thx, thy))
    groups = np.array([group_of(n) for n in names] + ['none'])
    return (i0, groups[first], failed, np.asarray(det.x), np.asarray(det.y), u, v)


def side_bins(side, parts):
    order = np.argsort([p[0] for p in parts])
    grp = np.concatenate([parts[i][1] for i in order])
    failed = np.concatenate([parts[i][2] for i in order])
    dx = np.concatenate([parts[i][3] for i in order])
    dy = np.concatenate([parts[i][4] for i in order])
    u = np.concatenate([parts[i][5] for i in order])
    v = np.concatenate([parts[i][6] for i in order])
    lit = (grp == 'none') & ~failed
    vane = np.isin(grp, SPIDER) & ~failed
    cx, cy = dx[lit | vane].mean(), dy[lit | vane].mean()  # centroid of the vane-free pupil
    px, py = (dx - cx) / PIXEL_M, (dy - cy) / PIXEL_M
    r = np.hypot(px, py)
    az = np.mod(np.degrees(np.arctan2(py, px)), 360.0)
    psi = np.mod(az - (FIELD_AZ_DEG + 180.0), 360.0)
    k = np.minimum(psi.astype(int), NPSI - 1)
    # danish v4.02g mask on the same rays, no vanes
    import yaml
    params = yaml.safe_load(og.mask_path(pc.p3.MASK_RUN, tuple(c * 1e-3 for c in CONFIGS[side]))
                            .read_text())
    r_o, r_i = ob._annulus(params)
    thx, thy = field()
    dk = ob.danish_keep(params, u, v, thx, thy, spider=False, r_outer=r_o, r_inner=r_i) & ~failed
    blocked = ~lit & ~failed & ~np.isin(grp, SPIDER)
    n_lit = np.bincount(k[lit | vane], minlength=NPSI)
    spider = np.bincount(k[vane], minlength=NPSI) > SPIDER_FRAC * np.maximum(n_lit, 1)
    spider = spider | np.roll(spider, 1) | np.roll(spider, -1)
    rows = []
    for b in range(NPSI):
        s = k == b
        row = dict(side=side, psi_bin=b, psi_deg=b + 0.5, spider=bool(spider[b]))
        for name, sel in (('batoid', lit), ('danish', dk)):
            rb = r[s & sel]
            row[f'r_inner_{name}_pix'] = float(rb.min()) if len(rb) else np.nan
            row[f'r_outer_{name}_pix'] = float(rb.max()) if len(rb) else np.nan
        for edge, lo, hi in (('inner', row['r_inner_batoid_pix'] - BAND_PIX, row['r_inner_batoid_pix']),
                             ('outer', row['r_outer_batoid_pix'], row['r_outer_batoid_pix'] + BAND_PIX)):
            g = grp[s & blocked & (r >= lo) & (r <= hi)]
            if spider[b]:
                row[f'surface_{edge}'] = 'spider'
                row[f'surface_{edge}_frac'] = np.nan
                row[f'n_{edge}'] = int(len(g))
            elif len(g):
                names, counts = np.unique(g, return_counts=True)
                j = int(np.argmax(counts))
                row[f'surface_{edge}'] = names[j]
                row[f'surface_{edge}_frac'] = float(counts[j] / len(g))
                row[f'n_{edge}'] = int(len(g))
            else:
                row[f'surface_{edge}'] = 'none'
                row[f'surface_{edge}_frac'] = np.nan
                row[f'n_{edge}'] = 0
        rows.append(row)
    img, _, _ = np.histogram2d(py[lit], px[lit], bins=800, range=[[-400, 400], [-400, 400]])
    return pd.DataFrame(rows), img.astype(np.float32), (cx, cy)


def _mode_filter(s, half=MODE_HALF_DEG):
    """Commonest surface in a +-`half` deg circular window; removes 1-3 deg fragments."""
    out = s.copy()
    n = len(s)
    for i in range(n):
        w = s[np.arange(i - half, i + half + 1) % n]
        w = w[~np.isin(w, ('none', 'spider'))] if (~np.isin(w, ('none', 'spider'))).any() else w
        names, counts = np.unique(w, return_counts=True)
        out[i] = s[i] if s[i] == 'spider' else (names[np.argmax(counts)] if counts.max() > 1 else s[i])
    return out


def ranges(bins):
    """Contiguous psi ranges (deg) per side, edge and surface, after `_mode_filter`."""
    rows = []
    for side, g in bins.groupby('side'):
        g = g.sort_values('psi_bin')
        for edge in ('inner', 'outer'):
            s = _mode_filter(g[f'surface_{edge}'].to_numpy())
            starts = [0] + [i for i in range(1, NPSI) if s[i] != s[i - 1]]
            segs = [(starts[i], (starts[i + 1] if i + 1 < len(starts) else NPSI)) for i in
                    range(len(starts))]
            if len(segs) > 1 and s[0] == s[-1]:  # join across 0 deg
                segs[0] = (segs[-1][0] - NPSI, segs[0][1])
                segs = segs[:-1]
            for a, b in segs:
                rows.append(dict(side=side, edge=edge, surface=s[a % NPSI], psi_from_deg=a % 360,
                                 psi_to_deg=b % 360 if b % 360 else 360, width_deg=b - a))
    return pd.DataFrame(rows)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=64)
    args = ap.parse_args()
    run = study_run(pc.STUDY, pc.RUN_LABEL, date=pc.RUN_DATE).path
    n = NRAD * NAZ
    for side in CONFIGS:
        optic_for(side)
    tasks = [(side, i, min(i + CHUNK, n)) for side in CONFIGS for i in range(0, n, CHUNK)]
    print('rays per side', n, 'tasks', len(tasks), flush=True)
    t0 = time.time()
    with mp.get_context('fork').Pool(args.workers) as pool:
        out = pool.map(_chunk, tasks, chunksize=1)
    print(f'trace {time.time() - t0:.0f} s', flush=True)
    bins, imgs = [], {}
    for side in CONFIGS:
        parts = [o for t, o in zip(tasks, out) if t[0] == side]
        b, img, c = side_bins(side, parts)
        bins.append(b)
        imgs[side] = img
        imgs[f'{side}_centroid_m'] = np.array(c)
    bins = pd.concat(bins, ignore_index=True)
    bins.to_parquet(run / 'vignetting_bins.parquet')
    bins.to_csv(run / 'vignetting_bins.csv', index=False, float_format='%.5g')
    rg = ranges(bins)
    rg.to_parquet(run / 'vignetting_ranges.parquet')
    rg.to_csv(run / 'vignetting_ranges.csv', index=False)
    np.savez_compressed(run / 'vignetting_images.npz', **imgs)
    print(rg.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
