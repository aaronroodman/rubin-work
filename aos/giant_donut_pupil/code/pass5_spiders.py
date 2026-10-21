"""Pupil pass 5, steps 4-6: spider shadows with the guided tracer, the spider-angle grid
and the centre test.

Per distinct side (`pass5_common.sides`, 188):

1. fit c is re-rendered from the fit table (with and without the vanes), the data and
   model robust edges are taken about the fit-a centre, as pass 4;
2. shadows: `pass5_common.guided_shadows` on the data and the fit c model, about the fit-a
   centre (step 4, 5 and the step 6 baseline);
3. spider-angle grid (step 6a): the fit c model re-rendered at danish ``spider_angle =
   rtp + delta`` for delta -1.0 to +1.0 deg in 0.05 deg steps, and its shadow azimuths
   at r_half traced at each, near the delta = 0 shadows turned by delta;
4. centre test (step 6c): data and model each traced about their own outer-edge centre
   and their own inner-edge centre (a circle fitted to the robust edge, two iterations
   from the fit-a centre), with the edge-circle centres recorded.

Outputs in the run: ``spiders.parquet`` (one row per side, shadow and centre definition),
``spider_delta.parquet`` (model azimuth per side, shadow and delta), ``centres.parquet``
(per side: the fit-a, outer- and inner-edge centres of data and model, pixels),
``spider_profiles.npz`` (cross-vane profiles, fit-a centre) and ``spider_geometry.npz``
(traced centres and chords, for the gallery).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass5_spiders.py --workers 80
"""
import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass5_common as pc  # noqa: E402
import geom_edges as ge  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

DELTAS = np.round(np.arange(-1.0, 1.0001, 0.05), 3)
CENTRES = ('fit_a', 'outer', 'inner')
AZ = (np.arange(pc.p3.pe.NBIN) + 0.5) * 360 / pc.p3.pe.NBIN


def edge_centre(image, valid, c, row, noise, flags, edge, n_iter=2):
    """Centre (pixels) of the circle fitted to the robust `edge` of `image`."""
    for _ in range(n_iter):
        e = pc.edges(image, valid, c[0], c[1], row, noise, flags)
        ox, oy, _ = ge.circle_centre(e[f'r_{edge}'], AZ)
        c = (c[0] + ox, c[1] + oy)
    return c


def _rows(row, centre_def, sd, sm, reasons, sign):
    out = []
    for k in range(pc.N_SHADOW):
        r = dict(key=row.key, sample_id=row.sample_id, pass3_id=row.pass3_id, label=row.label,
                 side=row.side, band=row.band, sensor_class=row.sensor_class,
                 detector=row.detector, mag=row.mag, field_r_deg=row.field_r_deg,
                 field_az_ocs_deg=row.field_az_ocs_deg, rtp_deg=row.rtp_deg,
                 fit_ok=bool(row.fit_ok), in_sample=bool(row.in_sample),
                 in_bands=bool(row.in_bands), centre=centre_def, model_idx=k,
                 missing=reasons[k])
        for name, s in (('data', sd[k]), ('model', sm[k])):
            for q in ('n_points', 'rms_pix', 'b_pix', 'phi0_ocs_deg', 'az_half_ocs_deg',
                      'r_half_pix', 'depth', 'fwhm_pix'):
                r[f'{q}_{name}'] = s[q]
            t = s['traj']
            r[f'dir_x_{name}'], r[f'dir_y_{name}'] = (t['direction'] if np.isfinite(
                t['impact_pix']) else (np.nan, np.nan))
        r['pupil_az_deg'] = (sm[k]['az_half_ocs_deg'] + sign) % 360
        out.append(r)
    return out


def one_side(row):
    t0 = time.time()
    d = pc.load_side(row)
    R = pc.Renderer(row, d['zk_ta'])
    n = d['img'].shape[0]
    msk = d['masked']
    valid = ~msk
    model0 = R.render(n)
    without0 = R.render(n, spider=False)
    qa = float(np.nanmax(np.abs(model0 - d['model_c'])))
    data = np.where(msk, np.nan, d['img'])
    model = np.where(msk, np.nan, model0)
    without = np.where(msk, 0.0, without0)
    ca = (d['cx'], d['cy'])
    e = pc.edges(np.nan_to_num(data), valid, ca[0], ca[1], row, d['noise'], d['flags'])
    r_edge = float(np.nanmedian(e['r_outer']))
    fit_in = np.where(np.isfinite(e['fit_inner']), e['fit_inner'], np.nanmedian(e['r_inner']))
    fit_out = np.where(np.isfinite(e['fit_outer']), e['fit_outer'], r_edge)
    ef = (fit_in, fit_out)
    sign = 180.0 if row.side == 'extra' else 0.0

    # Step 4/5/6 baseline: fit-a centre, delta 0
    sd, sm, reasons, aux = pc.guided_shadows(data, model, without, ca, r_edge, ef, row.rtp_deg)
    rows = _rows(row, 'fit_a', sd, sm, reasons, sign)
    prof = {}
    geom = {}
    for k in range(pc.N_SHADOW):
        for name, s in (('data', sd[k]), ('model', sm[k])):
            if s['profile'] is not None:
                prof[f'{row.key}:{k}:{name}_off'], prof[f'{row.key}:{k}:{name}_flux'] = s['profile']
            geom[f'{row.key}:{k}:{name}_centres'] = s['centres'].astype(np.float32)
            t = s['traj']
            geom[f'{row.key}:{k}:{name}_line'] = np.array(
                [*t['point'], *(t['direction'] if np.isfinite(t['impact_pix']) else (np.nan, np.nan)),
                 s['r_half_pix'], s['az_half_ocs_deg']], np.float32)
    geom[f'{row.key}:meta'] = np.array([ca[0], ca[1], r_edge, n], np.float32)

    # Step 6a: model shadows at rtp + delta
    drow = []
    for delta in DELTAS:
        if delta == 0:
            az_md = aux['az_m']
        else:
            m_d = np.where(msk, np.nan, R.render(n, delta_deg=float(delta)))
            az_md = pc.trace_near(m_d, ca, r_edge, aux['az_m'] + delta, pc.MODEL_WIN_DEG)
        s_d = pc.chords(az_md, None, ca, r_edge, ef, row.rtp_deg, profiles=False)
        for k, s in enumerate(s_d):
            drow.append(dict(key=row.key, model_idx=k, delta_deg=float(delta),
                             az_half_ocs_deg_model=s['az_half_ocs_deg']))

    # Step 6c: own outer- and inner-edge centres
    crow = dict(key=row.key, cx_fit_a=ca[0], cy_fit_a=ca[1])
    m_img = np.nan_to_num(model)
    for edge in ('outer', 'inner'):
        cd_ = edge_centre(np.nan_to_num(data), valid, ca, row, d['noise'], d['flags'], edge)
        cm_ = edge_centre(m_img, valid, ca, row, d['noise'], d['flags'], edge)
        crow.update({f'cx_{edge}_data': cd_[0], f'cy_{edge}_data': cd_[1],
                     f'cx_{edge}_model': cm_[0], f'cy_{edge}_model': cm_[1]})
        g = pc.guide_azimuths(model, without, cm_, r_edge)
        az_m = pc.trace_near(model, cm_, r_edge, g, pc.MODEL_WIN_DEG)
        az_d = pc.trace_near(data, cd_, r_edge, az_m, pc.DATA_WIN_DEG)
        smc = pc.chords(az_m, model, cm_, r_edge, ef, row.rtp_deg, profiles=False)
        sdc = pc.chords(az_d, data, cd_, r_edge, ef, row.rtp_deg, profiles=False)
        rs = ['' if np.isfinite(a['az_half_ocs_deg']) and np.isfinite(b['az_half_ocs_deg'])
              and a['rms_pix'] <= pc.sp4.MAX_RMS_PIX else 'not traced' for a, b in zip(sdc, smc)]
        rows += _rows(row, edge, sdc, smc, rs, sign)
    crow['render_qa_max_e'] = qa
    crow['seconds'] = time.time() - t0
    print(row.key, row.label, row.side, f'{crow["seconds"]:.0f} s', 'found',
          sum(r == '' for r in reasons), flush=True)
    return rows, drow, crow, prof, geom


def number(t):
    """psi_num and vane_id per side and centre definition (as `spider_pass4.number`)."""
    t = t.copy()
    az = t.az_half_ocs_deg_data.where(t.missing == '', t.az_half_ocs_deg_model)
    t['psi_deg'] = np.mod(az - (t.field_az_ocs_deg + 180.0), 360.0)
    t['psi_num'] = -1
    t['vane_id'] = -1
    for _, g in t.groupby(['centre', 'key']):
        g = g[np.isfinite(g.psi_deg)]
        if len(g):
            start = g.psi_deg.apply(lambda p: min(p, 360 - p)).idxmin()
            order = np.mod(g.psi_deg - g.psi_deg[start], 360.0).sort_values().index
            t.loc[order, 'psi_num'] = np.arange(len(order))
        gv = g[np.isfinite(g.pupil_az_deg)]
        t.loc[gv.pupil_az_deg.sort_values().index, 'vane_id'] = np.arange(len(gv))
    return t


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=80)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    run = study_run(pc.STUDY, pc.RUN_LABEL, date=pc.RUN_DATE).path
    S = pc.sides()
    S = S[S.error == ''].reset_index(drop=True)
    if args.limit:
        S = S.head(args.limit)
    S.to_parquet(run / 'sides.parquet')
    S.to_csv(run / 'sides.csv', index=False, float_format='%.6g')
    print('sides', len(S), flush=True)
    t0 = time.time()
    rows, drows, crows, prof, geom = [], [], [], {}, {}
    tasks = [r for _, r in S.iterrows()]
    with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
        for r, dr, cr, p, g in pool.imap_unordered(one_side, tasks):
            rows += r
            drows += dr
            crows.append(cr)
            prof.update(p)
            geom.update(g)
    t = pd.DataFrame(rows)
    t = number(t)
    for q in ('az_half_ocs_deg', 'phi0_ocs_deg'):
        t[f'{q}_dmm'] = pc.wrap(t[f'{q}_data'] - t[f'{q}_model'])
    for q in ('b_pix', 'depth', 'fwhm_pix'):
        t[f'{q}_dmm'] = t[f'{q}_data'] - t[f'{q}_model']
    t.loc[t.missing != '', [c for c in t if c.endswith('_dmm')]] = np.nan
    t.to_parquet(run / 'spiders.parquet')
    t.to_csv(run / 'spiders.csv', index=False, float_format='%.5g')
    dt = pd.DataFrame(drows)
    dt.to_parquet(run / 'spider_delta.parquet')
    pd.DataFrame(crows).to_parquet(run / 'centres.parquet')
    np.savez_compressed(run / 'spider_profiles.npz', **prof)
    np.savez_compressed(run / 'spider_geometry.npz', **geom)
    print(f'wall {time.time() - t0:.0f} s', run, flush=True)


if __name__ == '__main__':
    main()
