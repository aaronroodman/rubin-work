"""Pupil pass 4, step 6: spider shadows in the data against the fit c model.

Per donut side, the eight spider shadows are traced in the data stamp and in the fit c
model image with `spider_trajectory` (``wfs/code/giant_donuts``): shadow centres at seven
narrow radii (`trace_shadow_centres`), a straight chord through them
(`fit_trajectories`), the flux profile across the chord (`trajectory_profile`) and its
depth and width (`vane_depth`). Both images are traced about the same point, the fit-a
centre the robust edges use, and at the same radii (the data's median robust r_outer).

A traced chord whose points scatter by more than ``MAX_RMS_PIX`` about the line has its
worst point dropped and is refitted, down to ``MIN_POINTS``; a shadow still off, or with
no model shadow within ``MATCH_DEG``, is recorded as missing with a reason.

Per shadow: phi_0 (deg, OCS; the azimuth of the chord's outward direction, so the shadow
sits at phi(r) = phi_0 + arcsin(b / r)), b (pixels, signed as `fit_trajectories`), the
azimuth at r_half = (r_inner + r_outer) / 2 from the data's harmonic robust edges at that
azimuth (deg, OCS), and the depth (dimensionless) and FWHM (pixels).

Numbering:
- ``psi_num`` 0-7: by relative azimuth psi = (az_ocs - (phi + 180)) mod 360 of the data
  (else model) shadow at r_half, starting from the shadow nearest psi = 0 and going
  anticlockwise;
- ``vane_id`` 0-7: the pupil-frame shadow. Pupil azimuth is the model's OCS azimuth at
  r_half on intra-focal sides and that plus 180 deg on extra-focal sides; the ids count
  anticlockwise from 0 deg, so id 0 is the first shadow past 0 deg (the shadows sit at
  45 ± 6 deg + k 90 deg, well clear of 0).

Sets: ``sample`` (the 49 pass 3 sample sides, fit c from run 20261020_pupil3) and
``bands`` (the 180 pass 4 central- and outer-band sides, run 20261021_pupil4). Outputs
in the pass 4 run: ``spiders.parquet``/``.csv`` (one row per side and shadow),
``spider_profiles.npz`` (sample sides) and ``pupil_pass4_spiders.pdf``.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/spider_pass4.py
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import candidates as cd  # noqa: E402  (blitz path first)
import pupil_pass3 as p3  # noqa: E402

sys.path.insert(0, str(cd._GIANT_CODE))
import spider_trajectory as stj  # noqa: E402

PASS3_RUN = p3.STUDY_DIR / 'output' / '20261020_pupil3'
PASS4_RUN = p3.STUDY_DIR / 'output' / '20261021_pupil4'
N_SHADOW = stj.N_SPIDER_SHADOWS
MAX_RMS_PIX = 2.0
MIN_POINTS = 4
MATCH_DEG = 5.0


def _wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def _robust_trajectories(centres, center):
    """`fit_trajectories`, dropping the worst traced point while the rms is over MAX_RMS_PIX."""
    centres = centres.copy()
    tr = stj.fit_trajectories(centres, center)
    for k in range(len(tr)):
        while tr[k]['rms_pix'] > MAX_RMS_PIX and tr[k]['n_points'] > MIN_POINTS:
            pts = centres[k]
            d = np.array(tr[k]['direction'])
            nrm = np.array([-d[1], d[0]])
            off = np.abs((pts - np.array(tr[k]['point'])) @ nrm)
            off[~np.isfinite(pts).all(axis=1)] = -1
            centres[k, int(np.argmax(off))] = np.nan
            tr[k] = stj.fit_trajectories(centres[k:k + 1], center)[0]
    return tr


def _az_at(t, center, r):
    """Azimuth (deg, pixel frame) where chord `t` crosses radius `r`, on its own half."""
    p = np.array(t['point']) - np.array(center)
    d = np.array(t['direction'])
    if np.dot(p, d) < 0:
        d = -d
    # |p + s d| = r, the root nearest the traced centroid
    pd_ = p @ d
    disc = pd_ ** 2 - (p @ p - r ** 2)
    if not np.isfinite(disc) or disc < 0:
        return np.nan, d
    s = min((-pd_ + np.sqrt(disc), -pd_ - np.sqrt(disc)), key=abs)
    q = p + s * d
    return float(np.degrees(np.arctan2(q[1], q[0])) % 360.0), d


def trace(image, center, r_edge, edges_fit, rtp_deg):
    """Shadow rows of one image: chords, r_half azimuths, profiles, depth and width."""
    centres, _, _ = stj.trace_shadow_centres(image, center, r_edge)
    tr = _robust_trajectories(centres, center)
    az_bin = (np.arange(len(edges_fit[0])) + 0.5) * 360 / len(edges_fit[0])
    out = []
    for t in tr:
        row = dict(n_points=t['n_points'], rms_pix=t['rms_pix'], b_pix=t['impact_pix'],
                   phi0_ocs_deg=np.nan, az_half_ocs_deg=np.nan, r_half_pix=np.nan,
                   depth=np.nan, fwhm_pix=np.nan, profile=None)
        if np.isfinite(t['impact_pix']):
            az0 = t['azimuth_deg']
            k = int(np.argmin(np.abs(_wrap(az_bin - az0))))
            r_half = 0.5 * (edges_fit[0][k] + edges_fit[1][k])
            az_h, d = _az_at(t, center, r_half)
            off, flux, err, npx = stj.trajectory_profile(image, t, center, r_edge)
            depth, fwhm, _ = stj.vane_depth(off, flux)
            row.update(phi0_ocs_deg=float((np.degrees(np.arctan2(d[1], d[0])) + rtp_deg) % 360),
                       az_half_ocs_deg=(az_h + rtp_deg) % 360 if np.isfinite(az_h) else np.nan,
                       r_half_pix=r_half, depth=depth, fwhm_pix=fwhm,
                       profile=(off.astype(np.float32), flux.astype(np.float32)))
        out.append(row)
    return out


def one_side(task):
    """Data and model shadows of one side, matched; rows per shadow."""
    meta, img_path, keep_profiles = task
    d = np.load(img_path)
    msk = d['masked'].astype(bool)
    data = np.where(msk, np.nan, d['img'].astype(float))
    model = np.where(msk, np.nan, d['model_c'].astype(float))
    center = (float(d['cx']), float(d['cy']))
    flags = (d['spider'], d['dark'], d['m_in'], d['m_out']) if 'spider' in d else meta['flags']
    noise = float(d['noise']) if 'noise' in d else meta['noise_e']
    e = p3.edges_of(np.nan_to_num(data), ~msk, center[0], center[1], meta['r_out_pix'], noise,
                    flags)
    r_edge = float(np.nanmedian(e['r_outer']))
    fit_in = np.where(np.isfinite(e['fit_inner']), e['fit_inner'], np.nanmedian(e['r_inner']))
    fit_out = np.where(np.isfinite(e['fit_outer']), e['fit_outer'], r_edge)
    rtp = meta['rtp_deg']
    sd = trace(data, center, r_edge, (fit_in, fit_out), rtp)
    sm = trace(model, center, r_edge, (fit_in, fit_out), rtp)
    sign = 180.0 if meta['side'] == 'extra' else 0.0
    rows, profiles = [], {}
    used = set()
    for j, m in enumerate(sm):
        row = dict(meta_row(meta), model_idx=j)
        for k, v in m.items():
            if k != 'profile':
                row[f'{k}_model'] = v
        # the data shadow nearest this model shadow, each used once
        best, dbest = None, MATCH_DEG
        for i, s in enumerate(sd):
            if i in used or not np.isfinite(s['az_half_ocs_deg']) \
                    or not np.isfinite(m['az_half_ocs_deg']):
                continue
            dd = abs(_wrap(s['az_half_ocs_deg'] - m['az_half_ocs_deg']))
            if dd < dbest:
                best, dbest = i, dd
        reason = ''
        if not np.isfinite(m['az_half_ocs_deg']):
            reason = 'model shadow not traced'
        elif best is None:
            reason = 'no data shadow within 5 deg'
        else:
            s = sd[best]
            used.add(best)
            if s['rms_pix'] > MAX_RMS_PIX:
                reason = f'data chord rms over {MAX_RMS_PIX} pixels'
            for k, v in s.items():
                if k != 'profile':
                    row[f'{k}_data'] = v
            if keep_profiles:
                profiles[j] = (s['profile'], m['profile'])
        row['missing'] = reason
        row['pupil_az_deg'] = (m['az_half_ocs_deg'] + sign) % 360 \
            if np.isfinite(m['az_half_ocs_deg']) else np.nan
        rows.append(row)
    if keep_profiles:
        for j, m in enumerate(sm):
            profiles.setdefault(j, (None, m['profile']))
    return rows, (meta['key'], profiles) if keep_profiles else None


def meta_row(meta):
    return {k: meta[k] for k in ('set', 'sample_id', 'pass3_id', 'side', 'band', 'sensor_class',
                                  'detector', 'mag', 'field_r_deg', 'field_az_ocs_deg', 'rtp_deg')}


def number(t):
    """psi_num and vane_id per side (see the module docstring)."""
    t = t.copy()
    az = t.az_half_ocs_deg_data.where(t.missing == '', t.az_half_ocs_deg_model)
    t['psi_deg'] = np.mod(az - (t.field_az_ocs_deg + 180.0), 360.0)
    t['psi_num'] = -1
    t['vane_id'] = -1
    for _, g in t.groupby(['set', 'sample_id', 'side']):
        g = g[np.isfinite(g.psi_deg)]
        start = g.psi_deg.apply(lambda p: min(p, 360 - p)).idxmin()
        order = np.mod(g.psi_deg - g.psi_deg[start], 360.0).sort_values().index
        t.loc[order, 'psi_num'] = np.arange(len(order))
        gv = g[np.isfinite(g.pupil_az_deg)]
        t.loc[gv.pupil_az_deg.sort_values().index, 'vane_id'] = np.arange(len(gv))
    return t


def tasks_for(set_name, run, fits, rows):
    out = []
    for r in rows.itertuples():
        path = run / 'images' / f'{r.sample_id}_{r.side}.npz'
        if not path.exists():
            continue
        f = fits[(fits.sample_id == r.sample_id) & (fits.side == r.side)].iloc[0]
        meta = dict(set=set_name, sample_id=r.sample_id,
                    pass3_id=getattr(r, 'pass3_id', r.sample_id) or '', side=r.side,
                    band=getattr(r, 'band', ''), sensor_class=r.sensor_class, detector=r.detector,
                    mag=r.mag, field_r_deg=r.field_r_deg, field_az_ocs_deg=r.field_az_ocs_deg,
                    rtp_deg=r.rtp_deg, r_out_pix=r.r_out_pix, noise_e=float(f.noise_e),
                    key=f'{set_name}:{r.sample_id}:{r.side}')
        if set_name == 'sample':
            b = p3_bins[(p3_bins.sample_id == r.sample_id) & (p3_bins.side == r.side)] \
                .sort_values('bin')
            meta['flags'] = tuple(b[c].to_numpy() for c in
                                  ('spider', 'vignetted', 'masked_inner', 'masked_outer'))
        out.append((meta, path, set_name == 'sample'))
    return out


p3_bins = None


def main():
    global p3_bins
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--steps', default='trace,report')
    args = ap.parse_args()
    if 'trace' in args.steps:
        p3_bins = pd.read_parquet(PASS3_RUN / 'edges_bins.parquet')
        s3 = pd.read_parquet(PASS3_RUN / 'sample.parquet')
        f3 = pd.read_parquet(PASS3_RUN / 'fits.parquet')
        s4 = pd.read_parquet(PASS4_RUN / 'sample.parquet')
        f4 = pd.read_parquet(PASS4_RUN / 'fits.parquet')
        tasks = tasks_for('sample', PASS3_RUN, f3, s3) + tasks_for('bands', PASS4_RUN, f4, s4)
        print('sides', len(tasks), flush=True)
        rows, profiles = [], {}
        with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
            for r, p in pool.imap_unordered(one_side, tasks):
                rows += r
                if p:
                    profiles[p[0]] = p[1]
        t = number(pd.DataFrame(rows))
        for q in ('az_half_ocs_deg', 'phi0_ocs_deg'):
            t[f'{q}_dmm'] = _wrap(t[f'{q}_data'] - t[f'{q}_model'])
        for q in ('b_pix', 'depth', 'fwhm_pix'):
            t[f'{q}_dmm'] = t[f'{q}_data'] - t[f'{q}_model']
        t.loc[t.missing != '', [c for c in t if c.endswith('_dmm')]] = np.nan
        t.to_parquet(PASS4_RUN / 'spiders.parquet')
        t.to_csv(PASS4_RUN / 'spiders.csv', index=False, float_format='%.5g')
        flat = {}
        for key, p in profiles.items():
            for j, (pd_, pm) in p.items():
                if pd_ is not None:
                    flat[f'{key}:{j}:data_off'], flat[f'{key}:{j}:data_flux'] = pd_
                if pm is not None:
                    flat[f'{key}:{j}:model_off'], flat[f'{key}:{j}:model_flux'] = pm
        np.savez_compressed(PASS4_RUN / 'spider_profiles.npz', **flat)
    if 'report' in args.steps:
        import spider_pass4_report as rep
        rep.report(PASS4_RUN)


if __name__ == '__main__':
    main()
