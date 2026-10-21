"""Pupil pass 5, step 6 tables: spider rotation, translation and centre test.

Reads ``spiders.parquet``, ``spider_delta.parquet`` and ``centres.parquet`` (from
`pass5_spiders.py`) in the pass 5 run. All azimuths are OCS image azimuth at r_half
(deg); data − model differences are wrapped to ±180 deg. Only found shadows of sides with a
good fit c enter.

(a) ``rotation_fit``: per side, the spider-angle offset delta (deg) that minimises the sum
    over its found shadows of (data − model(rtp + delta))^2, on the 0.05 deg grid then a
    parabola through the three grid points about the minimum. Its error is from the
    curvature, with the per-shadow scatter about the best delta as the noise.
(b) ``translation_fit``: per side, data − model azimuth (rad) = w + (-dx sin(az) + dy
    cos(az)) / r_half, a rotation w plus a translation (dx, dy) of the shadow pattern, in
    OCS image pixels, Huber fit over the found shadows (needs 5).
(c) ``centre_test``: per centre definition (fit a, own outer-edge circle, own inner-edge
    circle), the common term and m = 1 amplitude and phase of data − model against az
    (Huber), and the inner- and outer-edge circle-centre offset, data − model, relative to
    that centre (OCS image pixels).

Written as parquet and csv: ``rotation_fit``, ``rotation_summary``, ``translation_fit``,
``translation_summary``, ``m1_summary``, ``vane_residuals``, ``centre_test``.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass5_common as pc  # noqa: E402

from scipy.stats import pearsonr, spearmanr  # noqa: E402
from rubinwork.common.utils import nmad  # noqa: E402

MIN_SHADOWS = 5


def _save(df, run, name):
    df.to_parquet(run / f'{name}.parquet')
    df.to_csv(run / f'{name}.csv', index=False, float_format='%.5g')


def found(t, centre='fit_a'):
    return t[(t.centre == centre) & (t.missing == '') & t.fit_ok].copy()


def rotation_fit(t, dt):
    a = found(t)
    m = dt.merge(a[['key', 'model_idx', 'az_half_ocs_deg_data']], on=['key', 'model_idx'])
    m['res'] = pc.wrap(m.az_half_ocs_deg_data - m.az_half_ocs_deg_model)
    rows = []
    for key, g in m.groupby('key'):
        n = g.groupby('delta_deg').res.count()
        if n.max() < MIN_SHADOWS:
            continue
        nmax = n.max()
        ok = n[n == nmax].index
        g = g[g.delta_deg.isin(ok)]
        ss = g.groupby('delta_deg').res.apply(lambda r: float(np.sum(r ** 2))).sort_index()
        d = ss.index.to_numpy()
        i = int(np.argmin(ss.to_numpy()))
        best, err = d[i], np.nan
        if 0 < i < len(d) - 1 and np.allclose(np.diff(d[i - 1:i + 2]), 0.05):
            y0, y1, y2 = ss.iloc[i - 1:i + 2].to_numpy()
            h = 0.05
            curv = (y0 - 2 * y1 + y2) / h ** 2
            if curv > 0:
                best = d[i] + h * (y0 - y2) / (2 * (y0 - 2 * y1 + y2))
                s2 = ss.iloc[i] / max(nmax - 1, 1)
                err = float(np.sqrt(2 * s2 / curv))
        # model azimuth response to delta (deg per deg), a check of the grid
        slope = g.groupby('model_idx').apply(
            lambda x: np.polyfit(x.delta_deg, pc.wrap(x.az_half_ocs_deg_model
                                                      - x.az_half_ocs_deg_model.iloc[0]), 1)[0]
            if x.az_half_ocs_deg_model.notna().sum() > 5 else np.nan, include_groups=False)
        r0 = g[g.delta_deg == 0].res
        rb = g[np.isclose(g.delta_deg, d[i])].res
        meta = a[a.key == key].iloc[0]
        rows.append(dict(key=key, label=meta.label, side=meta.side, sensor_class=meta.sensor_class,
                         field_r_deg=meta.field_r_deg, field_az_ocs_deg=meta.field_az_ocs_deg,
                         n_shadows=int(nmax), delta_best_deg=float(best), delta_err_deg=err,
                         at_grid_edge=bool(i in (0, len(d) - 1)),
                         rms_at_0_deg=float(np.sqrt(np.mean(r0 ** 2))),
                         rms_at_best_deg=float(np.sqrt(np.mean(rb ** 2))),
                         dmodel_ddelta=float(np.nanmedian(slope))))
    return pd.DataFrame(rows)


def rotation_summary(rf):
    rows = []
    for name, g in [('all', rf)] + list(rf.groupby('side')):
        x = g.delta_best_deg.to_numpy()
        row = dict(group=name, n_sides=len(g), delta_median_deg=float(np.median(x)),
                   delta_nmad_deg=nmad(x), delta_median_err_deg=1.2533 * nmad(x) / np.sqrt(len(x)),
                   delta_err_median_deg=float(np.nanmedian(g.delta_err_deg)),
                   n_at_grid_edge=int(g.at_grid_edge.sum()),
                   dmodel_ddelta_median=float(g.dmodel_ddelta.median()))
        for q in ('field_r_deg', 'field_az_ocs_deg'):
            v = g[q].to_numpy()
            if q == 'field_az_ocs_deg':  # circular: correlate against cos and sin
                for f, fn in (('cos', np.cos), ('sin', np.sin)):
                    u = fn(np.deg2rad(v))
                    row[f'pearson_r_vs_{f}_field_az'] = pearsonr(u, x)[0]
                    row[f'spearman_rho_vs_{f}_field_az'] = spearmanr(u, x)[0]
            else:
                row['pearson_r_vs_field_r'] = pearsonr(v, x)[0]
                row['spearman_rho_vs_field_r'] = spearmanr(v, x)[0]
        rows.append(row)
    return pd.DataFrame(rows)


def per_side_delta(t, dt, rf):
    """Found shadows with data − model at delta 0 and at each side's best delta (interpolated)."""
    a = found(t)
    best = rf.set_index('key').delta_best_deg
    a = a[a.key.isin(best.index)]
    out = []
    for key, g in dt[dt.key.isin(best.index)].groupby(['key', 'model_idx']):
        g = g.sort_values('delta_deg')
        y = g.az_half_ocs_deg_model.to_numpy()
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        y = y[0] + pc.wrap(y - y[0])
        out.append(dict(key=key[0], model_idx=key[1], az_model_best=float(
            np.interp(best[key[0]], g.delta_deg.to_numpy()[ok], y[ok]))))
    b = pd.DataFrame(out)
    a = a.merge(b, on=['key', 'model_idx'], how='left')
    a['dmm_best'] = pc.wrap(a.az_half_ocs_deg_data - a.az_model_best)
    return a


def m1_summary(a, cols):
    rows = []
    for side, g in a.groupby('side'):
        for c in cols:
            f = pc.m1_fit(g[c], g.az_half_ocs_deg_model)
            rows.append(dict(side=side, quantity=c, **f))
    return pd.DataFrame(rows)


def translation_fit(t, centre='fit_a'):
    import statsmodels.api as sm
    a = found(t, centre)
    rows = []
    for key, g in a.groupby('key'):
        if len(g) < MIN_SHADOWS:
            continue
        az = np.deg2rad(g.az_half_ocs_deg_model.to_numpy())
        rh = g.r_half_pix_model.to_numpy()
        y = np.deg2rad(g.az_half_ocs_deg_dmm.to_numpy())
        X = np.column_stack([np.ones_like(az), -np.sin(az) / rh, np.cos(az) / rh])
        r = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        m = g.iloc[0]
        rows.append(dict(key=key, centre=centre, label=m.label, side=m.side,
                         sensor_class=m.sensor_class, field_r_deg=m.field_r_deg,
                         field_az_ocs_deg=m.field_az_ocs_deg, n_shadows=len(g),
                         rot_deg=float(np.degrees(r.params[0])), dx_pix=float(r.params[1]),
                         dy_pix=float(r.params[2]), rot_err_deg=float(np.degrees(r.bse[0])),
                         dx_err_pix=float(r.bse[1]), dy_err_pix=float(r.bse[2]),
                         resid_rms_deg=float(np.degrees(np.sqrt(np.mean(r.resid ** 2))))))
    return pd.DataFrame(rows)


def summarize(df, cols, by=('centre', 'side')):
    rows = []
    for k, g in df.groupby(list(by)):
        row = dict(zip(by, k if isinstance(k, tuple) else (k,)), n_sides=len(g))
        for c in cols:
            x = g[c].dropna().to_numpy()
            row[f'{c}_median'] = float(np.median(x))
            row[f'{c}_nmad'] = nmad(x)
            row[f'{c}_median_err'] = 1.2533 * nmad(x) / np.sqrt(len(x))
        rows.append(row)
    return pd.DataFrame(rows)


def centre_test(t, cen, s):
    """m = 1 of the spider data − model and the edge-centre offsets, per centre definition."""
    rot = np.deg2rad(s.set_index('key').rtp_deg)
    c = cen.set_index('key')

    def ocs(dx, dy, keys):
        r = rot.reindex(keys).to_numpy()
        return dx * np.cos(r) - dy * np.sin(r), dx * np.sin(r) + dy * np.cos(r)

    rows = []
    ref = {'fit_a': (lambda e: (c.cx_fit_a, c.cy_fit_a)),
           'outer': (lambda e: (c[f'cx_outer_{e}'], c[f'cy_outer_{e}'])),
           'inner': (lambda e: (c[f'cx_inner_{e}'], c[f'cy_inner_{e}']))}
    edge_off = []
    for cname, fn in ref.items():
        for edge in ('inner', 'outer'):
            off = {}
            for e in ('data', 'model'):
                rx, ry = fn(e)
                off[e] = (c[f'cx_{edge}_{e}'] - rx, c[f'cy_{edge}_{e}'] - ry)
            dx, dy = ocs(off['data'][0] - off['model'][0], off['data'][1] - off['model'][1], c.index)
            edge_off.append(pd.DataFrame(dict(key=c.index.to_numpy(), centre=cname, edge=edge,
                                              dx_ocs_pix=np.asarray(dx), dy_ocs_pix=np.asarray(dy))))
    eo = pd.concat(edge_off).merge(s[['key', 'side', 'fit_ok']], on='key')
    eo = eo[eo.fit_ok]
    for cname in ref:
        a = found(t, cname)
        for side, g in a.groupby('side'):
            f = pc.m1_fit(g.az_half_ocs_deg_dmm, g.az_half_ocs_deg_model)
            row = dict(centre=cname, side=side, n_shadows=f['n'], common_deg=f['common'],
                       m1_amp_deg=f['amp'], m1_phase_deg=f['phase_deg'],
                       resid_nmad_deg=f['resid_nmad'])
            for edge in ('inner', 'outer'):
                e = eo[(eo.centre == cname) & (eo.side == side) & (eo.edge == edge)]
                row[f'{edge}_edge_dx_ocs_pix'] = float(e.dx_ocs_pix.median())
                row[f'{edge}_edge_dy_ocs_pix'] = float(e.dy_ocs_pix.median())
                row[f'{edge}_edge_dy_ocs_nmad_pix'] = nmad(e.dy_ocs_pix.dropna().to_numpy())
            rows.append(row)
    return pd.DataFrame(rows), eo


def vane_residuals(a):
    rows = []
    for (side, v), g in a[a.vane_id >= 0].groupby(['side', 'vane_id']):
        rows.append(dict(side=side, vane_id=int(v), n=len(g),
                         pupil_az_median_deg=float(g.pupil_az_deg.median()),
                         dmm_rtp_median_deg=float(g.az_half_ocs_deg_dmm.median()),
                         dmm_rtp_nmad_deg=nmad(g.az_half_ocs_deg_dmm.dropna().to_numpy()),
                         dmm_best_median_deg=float(g.dmm_best.median()),
                         dmm_best_nmad_deg=nmad(g.dmm_best.dropna().to_numpy()),
                         dmm_trans_median_deg=float(g.dmm_trans.median()),
                         dmm_trans_nmad_deg=nmad(g.dmm_trans.dropna().to_numpy())))
    return pd.DataFrame(rows)


def run_all(run):
    run = pathlib.Path(run)
    t = pd.read_parquet(run / 'spiders.parquet')
    dt = pd.read_parquet(run / 'spider_delta.parquet')
    cen = pd.read_parquet(run / 'centres.parquet')
    s = pd.read_parquet(run / 'sides.parquet')
    rf = rotation_fit(t, dt)
    rs = rotation_summary(rf)
    a = per_side_delta(t, dt, rf)
    tf = pd.concat([translation_fit(t, c) for c in ('fit_a', 'outer', 'inner')], ignore_index=True)
    ts = summarize(tf, ('rot_deg', 'dx_pix', 'dy_pix', 'resid_rms_deg'))
    # residual after the per-side translation fit (fit-a centre)
    tp = tf[tf.centre == 'fit_a'].set_index('key')
    az = np.deg2rad(a.az_half_ocs_deg_model)
    p = tp.reindex(a.key)
    a['dmm_trans'] = a.az_half_ocs_deg_dmm.to_numpy() - np.degrees(
        p.rot_deg.to_numpy() * np.pi / 180 + (-p.dx_pix.to_numpy() * np.sin(az)
                                              + p.dy_pix.to_numpy() * np.cos(az))
        / a.r_half_pix_model.to_numpy())
    # a common rotation per side of focus, the same delta for every side
    common = rs.set_index('group').delta_median_deg
    a['dmm_common'] = a.az_half_ocs_deg_dmm - a.side.map(common)
    m1 = m1_summary(a, ('az_half_ocs_deg_dmm', 'dmm_best', 'dmm_common', 'dmm_trans'))
    vr = vane_residuals(a)
    ct, eo = centre_test(t, cen, s)
    for name, df in (('rotation_fit', rf), ('rotation_summary', rs), ('translation_fit', tf),
                     ('translation_summary', ts), ('m1_summary', m1), ('vane_residuals', vr),
                     ('centre_test', ct), ('centre_edge_offsets', eo)):
        _save(df, run, name)
    return dict(rf=rf, rs=rs, a=a, tf=tf, ts=ts, m1=m1, vr=vr, ct=ct, eo=eo)


if __name__ == '__main__':
    out = run_all(pc.p3.STUDY_DIR / 'output' / f'{pc.RUN_DATE}_{pc.RUN_LABEL}')
    pd.set_option('display.width', 250)
    for k in ('rs', 'ts', 'm1', 'vr', 'ct'):
        print(k)
        print(out[k].round(3).to_string(index=False))
