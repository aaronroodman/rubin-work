"""Pupil pass 4: fit (c) and a central-Zernike fit (c0) on the central and outer bands.

Setup as pass 3 (`pupil_pass3.py`, ``method.md`` "Pupil fit setup"): v4.02, v4.02g masks,
intrinsicTA reference, blur fixed at 1.0 arcsec FWHM, spider on. Per selected, not-bad
donut side on 337 and 340 in the central band (0-0.8 deg field radius) or the outer band
(1.4-1.75 deg):

1. ``fit``: ISR and stamp as pass 3; fit a (Z4, only for the centre the edges are
   measured about) and fit c (Z4-Z11, Z14, Z15, Z20-Z22); robust edges of the data and
   the fit c model. Every stamp is cached in ``images/<id>_<side>.npz``.
2. ``medians``: the median fit c deviation per term except Z4, OCS, over the central-band
   science sides, both sides of focus combined (``central_medians``).
3. ``fit0``: fit c0 on every side, from the cached stamps: the reference is intrinsicTA
   plus the central medians (rotated to the side's CCS), and only flux, centre and Z4
   float. Robust edges of the c0 model.
4. ``report``: `pupil_pass4_report.report`.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pupil_pass4.py --steps fit,medians,fit0,report
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pupil_pass3 as p3  # noqa: E402  (blitz path first)
import fit_donuts as fd  # noqa: E402
import fit_giant_donut  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import obscuration_v402g as og  # noqa: E402
import pupil_edges as pe  # noqa: E402

from rubinwork.common import study_run  # noqa: E402
from rubinwork.common.utils import nmad  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
RUN_LABEL = 'pupil4'
RUN_DATE = '20261021'
PASS3_RUN = p3.STUDY_DIR / 'output' / '20261020_pupil3'
CENTRAL_BAND = (0.0, 0.8)
OUTER_BAND = (1.4, 1.75)
FIT_C = p3.FITS['c']
MEDIAN_TERMS = tuple(j for j in FIT_C if j != 4)


def build_sample():
    """Every selected, not-bad side in the two bands, with the pass 3 sample id."""
    t = p3.all_candidates()
    r = t.field_r_deg
    t['band'] = np.where(r <= CENTRAL_BAND[1], 'central',
                         np.where(r.between(*OUTER_BAND), 'outer', ''))
    t = t[t.band != ''].copy()
    s3 = pd.read_parquet(PASS3_RUN / 'sample.parquet')
    s3 = s3[['visit', 'detector', 'peak_id', 'sample_id']]
    t = t.merge(s3.rename(columns={'sample_id': 'pass3_id'}), on=['visit', 'detector', 'peak_id'],
                how='left')
    t['pass3_id'] = t.pass3_id.fillna('')
    return t.reset_index(drop=True)


def _flags(d):
    return d['spider'], d['dark'], d['m_in'], d['m_out']


def _bins(row, flags):
    spider, dark, m_in, m_out = flags
    az_ccs = (np.arange(pe.NBIN) + 0.5) * 360 / pe.NBIN
    return pd.DataFrame(dict(sample_id=row.sample_id, side=row.side, bin=np.arange(pe.NBIN),
                             az_ccs_deg=az_ccs, az_ocs_deg=np.mod(az_ccs + row.rtp_deg, 360.0),
                             spider=spider, vignetted=dark, masked_inner=m_in,
                             masked_outer=m_out))


def _fit_row(base, k, r, sky_scale, var, n_good, rtp, noll):
    blitz_utils, _ = fit_giant_donut._import_blitz()
    dev = np.full(len(p3.NOLL_MODEL), np.nan)
    for j, v in zip(noll, r['x'][3:]):
        dev[j] = v * 1e6
    dev_ocs = blitz_utils._rotate_zk(dev[None, :], -rtp)[0]
    out = dict(base, fit=k, n_free=3 + len(noll), nfev=r['nfev'], success=r['success'],
               fit_seconds=r['seconds'], flux_e=float(r['x'][0]),
               dx_pix=float(r['x'][1] / sky_scale), dy_pix=float(r['x'][2] / sky_scale))
    chi = r['chi'](var)
    out['dof'] = n_good - 3 - len(noll)
    out['chi2_dof'] = float(np.sum(chi ** 2) / max(out['dof'], 1))
    for j in p3.NOLL_REPORT:
        out[f'z{j}_ccs_um'] = dev[j]
        out[f'z{j}_ocs_um'] = dev_ocs[j]
    return out


def _edge_stats(rows, bins):
    qd = (bins.r_inner_data_pix / bins.r_outer_data_pix).dropna()
    for r in rows:
        k = r['fit']
        if f'r_inner_{k}_pix' not in bins:
            continue
        for edge in ('inner', 'outer'):
            d = (bins[f'r_{edge}_data_pix'] - bins[f'r_{edge}_{k}_pix']).dropna()
            r[f'edge_rms_{edge}_pix'] = float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan
            r[f'edge_median_{edge}_pix'] = float(np.median(d)) if len(d) else np.nan
            r[f'edge_n_{edge}'] = len(d)
        q = (bins[f'r_inner_{k}_pix'] / bins[f'r_outer_{k}_pix']).dropna()
        r['ratio_median'] = float(q.median()) if len(q) else np.nan
        r['ratio_median_data'] = float(qd.median()) if len(qd) else np.nan


def _base(row, offsets, noise, n, msk, img):
    return dict(sample_id=row.sample_id, pass3_id=row.pass3_id, band=row.band, side=row.side,
                visit=int(row.visit), detector=row.detector, sensor_class=row.sensor_class,
                mag=row.mag, field_r_deg=row.field_r_deg, field_az_ocs_deg=row.field_az_ocs_deg,
                field_x_ccs_deg=row.field_x_ccs_deg, field_y_ccs_deg=row.field_y_ccs_deg,
                rtp_deg=row.rtp_deg, det_offset_mm=offsets[0] * 1e3, r_out_pix=row.r_out_pix,
                noise_e=float(noise), stamp_npix=n, masked_frac=float(msk.mean()),
                stamp_sum_e=float(img[~msk].sum()))


def side_c(row, stamp, masked, out_dir):
    """Fits a and c and the edges of one side; caches the stamp."""
    import yaml
    from scipy.stats import median_abs_deviation
    t0 = time.time()
    offsets = fd.offsets_m(row.side, row.sensor_class)
    params = yaml.safe_load(og.mask_path(p3.MASK_RUN, offsets).read_text())
    r_outer, r_inner = ob._annulus(params)
    thx, thy = np.radians(row.field_x_ccs_deg), np.radians(row.field_y_ccs_deg)
    zk_ta, _ = p3.reference_zk(row.field_x_ccs_deg, row.field_y_ccs_deg, offsets, row.side,
                               r_inner / r_outer)
    factory = p3._factory(params, row.rtp_deg)
    img = np.asarray(stamp, float)
    msk = masked.astype(bool)
    n = img.shape[0]
    diff = (img[1:] - img[:-1])[~(msk[1:] | msk[:-1])]
    noise = median_abs_deviation(diff, scale='normal') / np.sqrt(2.0)
    var = np.where(msk, fd.MASKED_VAR, noise ** 2)
    n_good = int((~msk).sum())
    res = {'a': p3.fit(img, var, factory, zk_ta, thx, thy, p3.FITS['a'])}
    sky_scale = res['a']['danish'].sky_scale
    cx = n // 2 + res['a']['x'][1] / sky_scale
    cy = n // 2 + res['a']['x'][2] / sky_scale
    spider, dark = p3.spider_and_vignetting(params, row.rtp_deg, zk_ta, thx, thy, n, cx, cy,
                                            row.r_out_pix)
    _, _, m_in, m_out, _ = pe.edges(img, ~msk, cx, cy, row.r_out_pix)
    flags = (spider, dark, m_in, m_out)
    res['c'] = p3.fit(img, var, factory, zk_ta, thx, thy, FIT_C, x0=res['a']['x'])
    bins = _bins(row, flags)
    for name, im in (('data', img), ('c', res['c']['model'])):
        e = p3.edges_of(im, ~msk, cx, cy, row.r_out_pix, noise, flags)
        for edge in ('inner', 'outer'):
            bins[f'r_{edge}_{name}_pix'] = e[f'r_{edge}']
            bins[f'q_{edge}_{name}'] = e[f'q_{edge}']
    base = _base(row, offsets, noise, n, msk, img)
    rtp = np.radians(row.rtp_deg)
    rows = [_fit_row(base, k, res[k], sky_scale, var, n_good, rtp, p3.FITS[k]) for k in res]
    _edge_stats(rows, bins)
    for r in rows:
        r['side_seconds'] = time.time() - t0
    np.savez_compressed(out_dir / 'images' / f'{row.sample_id}_{row.side}.npz',
                        img=img.astype(np.float32), masked=msk, cx=cx, cy=cy, noise=noise,
                        spider=spider, dark=dark, m_in=m_in, m_out=m_out, x_a=res['a']['x'],
                        zk_ta=zk_ta, sky_scale=sky_scale,
                        model_c=res['c']['model'].astype(np.float32))
    return rows, bins


def chunk_c(task):
    """ISR one (visit, detector), then fits a and c on every side on it."""
    (visit, det_name), sides, out_dir = task
    import lsst.daf.butler as dafButler
    import candidates as cd
    sys.path.insert(0, str(cd._GIANT_CODE))
    import donut_images as di
    if 'butler' not in p3._BUTLER:
        p3._BUTLER['butler'] = dafButler.Butler('/repo/main')
    camera = p3._CTX['camera']
    out_dir = pathlib.Path(out_dir)
    image = di.run_isr(p3._BUTLER['butler'], visit, det_name, camera, p3.RAW_COLLECTION)
    neigh = p3.neighbours(visit, det_name, camera, p3._CTX['cat'][p3._CTX['cat'].visit == visit],
                          p3._CTX['orphans'])
    n_quarter = camera[det_name].getOrientation().getNQuarter()
    rows, bins = [], []
    for row in sides.itertuples():
        try:
            stamp, masked, info = fd.cut(image, pd.Series(dict(
                r_out_pix=row.r_out_pix, x_pix=row.x_pix, y_pix=row.y_pix, mag_r=row.mag)),
                neigh, n_quarter)
            r, b = side_c(row, stamp, masked, out_dir)
            for x in r:
                x.update(info, error='')
            rows += r
            bins.append(b)
            print(json.dumps(dict(id=row.sample_id, side=row.side, s=round(r[0]['side_seconds']),
                                  chi2_dof={x['fit']: round(x['chi2_dof'], 3) for x in r})),
                  flush=True)
        except Exception as err:  # a failed side is a row, not a crashed run
            rows.append(dict(sample_id=row.sample_id, side=row.side, visit=visit,
                             detector=det_name, error=repr(err)))
            print('error', row.sample_id, row.side, repr(err), flush=True)
    return rows, bins


def step_fit(run, cand, workers):
    import multiprocessing as mp
    from lsst.obs.lsst import LsstCam
    (run / 'images').mkdir(exist_ok=True)
    p3._CTX['camera'] = LsstCam.getCamera()
    p3._CTX['cat'] = pd.read_parquet(p3.METRICS_RUN / 'refcat_fp.parquet')
    pk = pd.read_parquet(p3.REFINE_RUN / 'peak_metrics.parquet')
    p3._CTX['orphans'] = pk[pk.match_refcat_id < 0][['visit', 'detector', 'x_pix', 'y_pix']]
    p3.optic_v402(0.62)  # build before forking
    tasks = [(key, g, str(run)) for key, g in cand.groupby(['visit', 'detector'])]
    tasks.sort(key=lambda t: -len(t[1]))
    print(f'fit: sides {len(cand)}, chunks {len(tasks)}, workers {workers}', flush=True)
    t0 = time.time()
    rows, bins = [], []
    with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
        for r, b in pool.imap_unordered(chunk_c, tasks):
            rows += r
            bins += b
    fits = pd.DataFrame(rows)
    fits['error'] = fits.error.fillna('')
    fits.to_parquet(run / 'fits_c.parquet')
    pd.concat(bins, ignore_index=True).to_parquet(run / 'edges_bins_c.parquet')
    print(f'fit: wall {time.time() - t0:.0f} s', flush=True)


def central_medians(fits):
    """Median and nMAD of fit c deviations (OCS, µm) over central-band science sides."""
    import pupil_pass3_report as rep3
    fits = rep3.flag_failed(fits.copy())
    c = fits[(fits.fit == 'c') & fits.fit_ok & (fits.band == 'central')
             & (fits.sensor_class == 'science')]
    rows = []
    for j in MEDIAN_TERMS:
        col = f'z{j}_ocs_um'
        row = dict(noll=j, n_both=int(c[col].notna().sum()), median_both_um=c[col].median(),
                   nmad_both_um=nmad(c[col].dropna().to_numpy()))
        for side in ('extra', 'intra'):
            s = c.loc[c.side == side, col].dropna()
            row[f'n_{side}'] = len(s)
            row[f'median_{side}_um'] = s.median()
            row[f'nmad_{side}_um'] = nmad(s.to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def side_c0(task):
    """Fit c0 of one side from its cached stamp: Z4, flux and centre free."""
    import yaml
    row, out_dir, dev_ocs_um = task
    blitz_utils, _ = fit_giant_donut._import_blitz()
    out_dir = pathlib.Path(out_dir)
    try:
        d = np.load(out_dir / 'images' / f'{row.sample_id}_{row.side}.npz')
        t0 = time.time()
        offsets = fd.offsets_m(row.side, row.sensor_class)
        params = yaml.safe_load(og.mask_path(p3.MASK_RUN, offsets).read_text())
        factory = p3._factory(params, row.rtp_deg)
        thx, thy = np.radians(row.field_x_ccs_deg), np.radians(row.field_y_ccs_deg)
        rtp = np.radians(row.rtp_deg)
        dev_ccs = blitz_utils._rotate_zk(dev_ocs_um[None, :], rtp)[0]
        zk_ref = d['zk_ta'] + dev_ccs * 1e-6
        img = d['img'].astype(float)
        msk = d['masked'].astype(bool)
        noise = float(d['noise'])
        var = np.where(msk, fd.MASKED_VAR, noise ** 2)
        r = p3.fit(img, var, factory, zk_ref, thx, thy, (4,), x0=d['x_a'])
        flags = _flags(d)
        cx, cy = float(d['cx']), float(d['cy'])
        bins = _bins(row, flags)
        for name, im in (('data', img), ('c0', r['model'])):
            e = p3.edges_of(im, ~msk, cx, cy, row.r_out_pix, noise, flags)
            for edge in ('inner', 'outer'):
                bins[f'r_{edge}_{name}_pix'] = e[f'r_{edge}']
        bins = bins.drop(columns=['r_inner_data_pix', 'r_outer_data_pix'])
        out = _fit_row(_base(row, offsets, noise, img.shape[0], msk, img), 'c0', r,
                       float(d['sky_scale']), var, int((~msk).sum()), rtp, (4,))
        # The fixed central medians are part of c0's wavefront; report them with Z4.
        for j in MEDIAN_TERMS:
            out[f'z{j}_ocs_um'] = dev_ocs_um[j]
            out[f'z{j}_ccs_um'] = dev_ccs[j]
        out['side_seconds'] = time.time() - t0
        out['error'] = ''
        np.savez_compressed(out_dir / 'images' / f'{row.sample_id}_{row.side}_c0.npz',
                            model_c0=r['model'].astype(np.float32))
        print(json.dumps(dict(id=row.sample_id, side=row.side, s=round(out['side_seconds']),
                              chi2_dof=round(out['chi2_dof'], 3))), flush=True)
        return [out], [bins]
    except Exception as err:
        print('error', row.sample_id, row.side, repr(err), flush=True)
        return [dict(sample_id=row.sample_id, side=row.side, fit='c0', error=repr(err))], []


def step_fit0(run, cand, workers):
    import multiprocessing as mp
    med = pd.read_csv(run / 'central_medians.csv')
    dev = np.zeros(len(p3.NOLL_MODEL))
    dev[med.noll.to_numpy()] = med.median_both_um.to_numpy()
    tasks = [(row, str(run), dev) for _, row in cand.iterrows()]
    print(f'fit0: sides {len(tasks)}, workers {workers}', flush=True)
    t0 = time.time()
    rows, bins = [], []
    with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
        for r, b in pool.imap_unordered(side_c0, tasks):
            rows += r
            bins += b
    fits = pd.DataFrame(rows)
    fits.to_parquet(run / 'fits_c0.parquet')
    pd.concat(bins, ignore_index=True).to_parquet(run / 'edges_bins_c0.parquet')
    print(f'fit0: wall {time.time() - t0:.0f} s', flush=True)


def merge(run):
    """One fits table (fits a, c, c0) and one bins table with the c0 edges joined."""
    c = pd.read_parquet(run / 'fits_c.parquet')
    c0 = pd.read_parquet(run / 'fits_c0.parquet')
    fits = pd.concat([c, c0], ignore_index=True)
    fits['error'] = fits.error.fillna('')
    b = pd.read_parquet(run / 'edges_bins_c.parquet')
    b0 = pd.read_parquet(run / 'edges_bins_c0.parquet')
    keep = ['sample_id', 'side', 'bin', 'r_inner_c0_pix', 'r_outer_c0_pix']
    bins = b.merge(b0[keep], on=['sample_id', 'side', 'bin'], how='left')
    for (sid, side), g in bins.groupby(['sample_id', 'side']):
        rows = fits[(fits.sample_id == sid) & (fits.side == side) & (fits.fit == 'c0')]
        if len(rows):
            recs = [rows.iloc[0].to_dict()]
            _edge_stats(recs, g)
            for k in ('edge_rms_inner_pix', 'edge_rms_outer_pix', 'edge_median_inner_pix',
                      'edge_median_outer_pix', 'edge_n_inner', 'edge_n_outer', 'ratio_median',
                      'ratio_median_data'):
                fits.loc[rows.index, k] = recs[0].get(k, np.nan)
    import pupil_pass3_report as rep3
    fits = rep3.flag_failed(fits)
    fits.to_parquet(run / 'fits.parquet')
    fits.to_csv(run / 'fits.csv', index=False, float_format='%.5g')
    bins.to_parquet(run / 'edges_bins.parquet')
    return fits, bins


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--steps', default='fit,medians,fit0,report')
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--limit', type=int, default=None, help='first N sides per band only (a test)')
    args = ap.parse_args()
    run = study_run(STUDY, RUN_LABEL, date=RUN_DATE).path
    cand = build_sample()
    if args.limit:
        cand = cand.groupby("band").head(args.limit).reset_index(drop=True)
    cand.to_parquet(run / 'sample.parquet')
    cand.to_csv(run / 'sample.csv', index=False, float_format='%.6g')
    steps = args.steps.split(',')
    if 'fit' in steps:
        step_fit(run, cand, args.workers)
    if 'medians' in steps:
        med = central_medians(pd.read_parquet(run / 'fits_c.parquet'))
        med.to_csv(run / 'central_medians.csv', index=False, float_format='%.5f')
        med.to_parquet(run / 'central_medians.parquet')
        print(med.round(3).to_string(index=False), flush=True)
    if 'fit0' in steps:
        step_fit0(run, cand, args.workers)
    if 'report' in steps:
        import pupil_pass4_report as rep
        fits, bins = merge(run)
        rep.report(run, cand, fits, bins)
    print('done', run, flush=True)


if __name__ == '__main__':
    main()
