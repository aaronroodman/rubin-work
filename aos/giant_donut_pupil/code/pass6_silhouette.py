"""Pupil pass 6, step 4a: the camera-silhouette circle refitted to the Batoid inner edge (v4.02s).

The v4.02g silhouette circle (`obscuration_v402g.fit_config`) is fitted to the whole
silhouette shadow, traced alone. Pass 5 step 2 found that where the silhouette sets the
donut's inner edge (psi 330–30 deg intra, 160–200 deg extra at 1.7 deg) that circle sits
3–5 pixels inside the Batoid edge. Here, per giant configuration (`og.CONFIGS_MM`) and
field radius ``FIELD_DEG`` (1.30–1.75 deg, along +x, rotator 0):

1. a polar grid of stop-plane rays (``NRAD`` x ``NAZ``, ``R_IN_M``–``R_OUT_M``, the
   inner-edge region) is traced through v4.02 without the vanes, with and without the
   CameraSilhouette planes. A ray blocked only with them is silhouette-blocked;
2. per 1 deg psi bin of the image (about the transmitted centroid, psi = az − (phi +
   180), phi = 0), the Batoid inner edge is the smallest transmitted radius (pixels).
   A bin is silhouette-set when over ``SIL_FRAC`` of the rays within ``BAND_PIX``
   inside that edge are silhouette-blocked;
3. the boundary points: per silhouette-set bin, the stop-plane midpoint of the innermost
   transmitted ray and the outermost silhouette-blocked ray;
4. a circle with its center on the field axis is fitted to them (center and radius, m),
   then a cubic in field radius over ``FIELD_DEG``, as the other mask edges;
5. danish − Batoid inner edge (pixels) in the silhouette-set bins, with the v4.02g and
   the v4.02s masks on the same rays.

v4.02s = v4.02g with ``CameraSilhouette`` split in two edges: ``outer`` (the v4.02g
circle) for 0–1.4 deg and ``outer_s`` (this fit) for 1.4–2.0 deg. The silhouette first
sets the inner edge at 1.40–1.45 deg; below that the cubic is an extrapolation, and at
1.30 deg it would cut up to 4 pixels into bins the silhouette does not set.

`data_fit` (step 4b) turns a measured data − model inner-edge miss per psi bin into a
change of the silhouette radius and center (mm, stop plane), from the edge response to
each parameter at one field.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass6_silhouette.py --workers 24
"""
import argparse
import copy
import os
import pathlib
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass6_common as p6  # noqa: E402
import batoid_images as bi  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import obscuration_v402g as og  # noqa: E402

FIELD_DEG = tuple(np.round(np.arange(1.30, 1.7501, 0.05), 2))
THETA_SPLIT_DEG = 1.4
NRAD, NAZ = 300, 1080
R_IN_M, R_OUT_M = 1.8, 3.2
NPSI = 360
BAND_PIX = 3.0
SIL_FRAC = 0.5
PIXEL_M = 10e-6
_OPT = {}


def optics(cfg):
    if cfg not in _OPT:
        d = ob._v402_dict()
        sysd = d['opticalSystem']
        sysd['items'] = [it for it in sysd['items'] if not it['name'].startswith(ob.SPIDER_PREFIXES)]
        full = ob._optic(copy.deepcopy(d))
        sysd['items'] = [it for it in sysd['items'] if not it['name'].startswith('CameraSilhouette')]
        nosil = ob._optic(d)
        _OPT[cfg] = (og.shift(full, cfg), og.shift(nosil, cfg))
    return _OPT[cfg]


def grid():
    r = np.linspace(R_IN_M, R_OUT_M, NRAD)
    a = (np.arange(NAZ) + 0.5) * 2 * np.pi / NAZ
    rr, aa = np.meshgrid(r, a)
    return (rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel()


def trace(task):
    cfg, th_deg = task
    full, nosil = optics(cfg)
    x, y = grid()
    thx = np.deg2rad(th_deg)
    k_full, px, py = bi.transmitted(full, x, y, thx, 0.0)
    k_nosil, _, _ = bi.transmitted(nosil, x, y, thx, 0.0)
    u, v = ob._stop_xy(full, bi._rays(full, x, y, thx, 0.0))
    return cfg, th_deg, dict(k=k_full, sil=k_nosil & ~k_full, px=px, py=py, u=u, v=v)


def bins_of(t, th_deg):
    """Per psi bin: Batoid inner edge (pixels), silhouette-set flag, boundary point (m)."""
    k = t['k']
    cx, cy = t['px'][k].mean(), t['py'][k].mean()
    qx, qy = (t['px'] - cx) / PIXEL_M, (t['py'] - cy) / PIXEL_M
    r = np.hypot(qx, qy)
    psi = np.mod(np.degrees(np.arctan2(qy, qx)) - 180.0, 360.0)
    b = np.minimum(psi.astype(int), NPSI - 1)
    rows = []
    for j in range(NPSI):
        s = b == j
        rk = r[s & k]
        if not len(rk):
            continue
        i_in = np.flatnonzero(s & k)[np.argmin(rk)]
        r_edge = r[i_in]
        near = s & ~k & (r < r_edge) & (r >= r_edge - BAND_PIX)
        frac = float(t['sil'][near].mean()) if near.any() else 0.0
        row = dict(psi_bin=j, psi_deg=j + 0.5, r_inner_batoid_pix=float(r_edge), sil_frac=frac,
                   sil_set=frac > SIL_FRAC)
        sb = s & t['sil']
        if row['sil_set'] and sb.any():
            i_b = np.flatnonzero(sb)[np.argmax(r[sb])]
            row['u_m'] = 0.5 * (t['u'][i_in] + t['u'][i_b])
            row['v_m'] = 0.5 * (t['v'][i_in] + t['v'][i_b])
        rows.append(row)
    return pd.DataFrame(rows), (cx, cy, qx, qy, r, b)


def danish_edge(params, t, geom, th_deg):
    """danish inner edge per psi bin (pixels) from its mask on the same rays."""
    r_o, r_i = ob._annulus(params)
    _, _, _, _, r, b = geom
    dk = ob.danish_keep(params, t['u'], t['v'], np.deg2rad(th_deg), 0.0, spider=False,
                        r_outer=r_o, r_inner=r_i)
    out = np.full(NPSI, np.nan)
    np.fmin.at(out, b[dk], r[dk])
    return out


def circle_x(u, v):
    """Circle with its center on the x (field) axis: center, radius, rms (m)."""
    from scipy.optimize import least_squares
    res = least_squares(lambda p: np.hypot(u - p[0], v) - p[1], (np.mean(u) + 2.0, 2.0))
    return float(res.x[0]), float(res.x[1]), float(np.sqrt(np.mean(res.fun ** 2)))


def v402s(params_g, centre_poly, radius_poly):
    p = copy.deepcopy(params_g)
    old = p['CameraSilhouette']['outer']
    p['CameraSilhouette'] = {
        'outer': dict(old, thetaMax=THETA_SPLIT_DEG),
        'outer_s': dict(clear=False, thetaMin=THETA_SPLIT_DEG, thetaMax=2.0,
                        center=[float(c) for c in centre_poly],
                        radius=[float(c) for c in radius_poly])}
    return p


def data_fit(miss, side, field_deg=1.65, d_m=0.01):
    """Silhouette radius and center change (mm, stop plane) that removes a data − model miss.

    Parameters
    ----------
    miss : `pandas.DataFrame`
        ``psi_deg`` and ``inner_dmm`` (data − model inner edge, pixels), the stacked bins.
    side : `str`
        ``'intra'`` or ``'extra'``.

    Returns
    -------
    fit : `dict`
        ``d_radius_mm``, ``d_center_mm`` (stop plane, center along the field direction),
        their errors, the bins used and the rms before and after (pixels).
    """
    cfg = (0.0, 4.0, 4.0) if side == 'extra' else (0.0, -4.0, -4.0)
    _, _, t = trace((cfg, field_deg))
    bins, geom = bins_of(t, field_deg)
    p0 = p6.mask_params('v4.02s', tuple(c * 1e-3 for c in cfg))
    e0 = danish_edge(p0, t, geom, field_deg)
    resp = []
    for which in ('radius', 'center'):
        p = copy.deepcopy(p0)
        p['CameraSilhouette']['outer_s'][which][-1] += d_m
        resp.append((danish_edge(p, t, geom, field_deg) - e0) / (d_m * 1e3))
    sel = bins[bins.sil_set].psi_bin.to_numpy()
    m = miss.copy()
    m['psi_bin'] = np.minimum(m.psi_deg.astype(int), NPSI - 1)
    m = m[m.psi_bin.isin(sel) & np.isfinite(m.inner_dmm)]
    A = np.column_stack([resp[0][m.psi_bin], resp[1][m.psi_bin]])
    ok = np.isfinite(A).all(axis=1)
    import statsmodels.api as sm
    r = sm.RLM(m.inner_dmm.to_numpy()[ok], A[ok], M=sm.robust.norms.HuberT()).fit()
    return dict(side=side, n_bins=int(ok.sum()), d_radius_mm=float(r.params[0]),
                d_center_mm=float(r.params[1]), d_radius_err_mm=float(r.bse[0]),
                d_center_err_mm=float(r.bse[1]),
                miss_rms_before_pix=float(np.sqrt(np.mean(m.inner_dmm.to_numpy()[ok] ** 2))),
                miss_rms_after_pix=float(np.sqrt(np.mean(r.resid ** 2))),
                resp_radius_pix_per_mm=float(np.nanmedian(resp[0][sel])),
                resp_center_pix_per_mm=float(np.nanmedian(resp[1][sel])))


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()
    run = p6.run_dir()
    (run / 'masks').mkdir(exist_ok=True)
    tasks = [(cfg, th) for cfg in og.CONFIGS_MM for th in FIELD_DEG]
    t0 = time.time()
    with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
        traced = pool.map(trace, tasks, chunksize=1)
    print(f'traced {len(traced)} in {time.time() - t0:.0f} s', flush=True)
    circ, allbins, cache = [], [], {}
    for cfg, th, t in traced:
        b, geom = bins_of(t, th)
        cache[(cfg, th)] = (t, geom, b)
        s = b[b.sil_set & b.u_m.notna()] if 'u_m' in b else b.iloc[:0]
        row = dict(det_mm=cfg[0], cam_mm=cfg[1], m2_mm=cfg[2], field_deg=th, n_bins=len(s))
        if len(s) >= 5:
            c, r, rms = circle_x(s.u_m.to_numpy(), s.v_m.to_numpy())
            row.update(center_m=c, radius_m=r, rms_m=rms)
        circ.append(row)
    circ = pd.DataFrame(circ)
    pd.set_option('display.width', 200)
    print(circ.round(5).to_string(index=False), flush=True)
    for cfg in og.CONFIGS_MM:
        g = circ[(circ.det_mm == cfg[0]) & (circ.cam_mm == cfg[1]) & circ.center_m.notna()]
        params_g = yaml.safe_load(og.mask_path(p6.pc.p3.MASK_RUN,
                                               tuple(c * 1e-3 for c in cfg)).read_text())
        deg = 3 if len(g) >= 6 else max(len(g) - 1, 0)
        cp = np.polyfit(g.field_deg, g.center_m, deg)
        rp = np.polyfit(g.field_deg, g.radius_m, deg)
        cp = np.concatenate([np.zeros(3 - deg), cp])
        rp = np.concatenate([np.zeros(3 - deg), rp])
        params_s = v402s(params_g, cp, rp)
        (run / 'masks' / p6.mask_name_s(cfg)).write_text(
            yaml.safe_dump(params_s, sort_keys=False, default_flow_style=None))
        for th in FIELD_DEG:
            t, geom, b = cache[(cfg, th)]
            for name, p in (('v4.02g', params_g), ('v4.02s', params_s)):
                e = danish_edge(p, t, geom, th)
                b[f'r_inner_{name}_pix'] = e[b.psi_bin]
                b[f'dmb_{name}_pix'] = b[f'r_inner_{name}_pix'] - b.r_inner_batoid_pix
            b = b.assign(det_mm=cfg[0], cam_mm=cfg[1], m2_mm=cfg[2], field_deg=th)
            allbins.append(b)
    bins = pd.concat(allbins, ignore_index=True)
    summ = []
    for k, g in bins.groupby(['det_mm', 'cam_mm', 'field_deg']):
        s = g[g.sil_set]
        o = g[~g.sil_set]
        row = dict(det_mm=k[0], cam_mm=k[1], field_deg=k[2], n_sil_bins=len(s),
                   psi_sil_min_deg=float(s.psi_deg.min()) if len(s) else np.nan,
                   psi_sil_max_deg=float(s.psi_deg.max()) if len(s) else np.nan)
        for name in ('v4.02g', 'v4.02s'):
            row[f'sil_median_{name}_pix'] = float(s[f'dmb_{name}_pix'].median())
            row[f'sil_min_{name}_pix'] = float(s[f'dmb_{name}_pix'].min())
            row[f'other_median_{name}_pix'] = float(o[f'dmb_{name}_pix'].median())
            row[f'other_absmax_{name}_pix'] = float(o[f'dmb_{name}_pix'].abs().max())
        summ.append(row)
    summ = pd.DataFrame(summ)
    for name, df in (('silhouette_circles', circ), ('silhouette_bins', bins),
                     ('silhouette_summary', summ)):
        df.to_parquet(run / f'{name}.parquet')
        df.to_csv(run / f'{name}.csv', index=False, float_format='%.6g')
    print(summ.round(2).to_string(index=False))
    print(f'wall {time.time() - t0:.0f} s', run, flush=True)


if __name__ == '__main__':
    main()
