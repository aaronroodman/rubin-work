"""Night 2, part B: the pass 6 pupil analysis on giant donuts of other nights.

Per night (`NIGHTS`), into run ``<date>_night2_<day_obs>``, with the pass 6 conventions
(`method.md` "Conventions in force"): v4.02, intrinsicTA, fit c, rtp = ConsDB
physical_rotator_angle, full ISR (flat, CTI), guided spider tracer, robust edges. The
camera and M2 z offsets of each exposure (4 or 8 mm) set the mask files and the reference.

Steps (``--steps``, in order):

- ``meta`` (needs ConsDB and value_added; run interactively): ``exposures.parquet`` (band,
  exposure time, physical_rotator_angle, pointing, the Trim of every DOF against the in-focus
  reference), ``thermal.parquet`` (camera, M1M3 and dome temperatures, value_added
  ``visit_telemetry`` and ``m1m3_thermal_r2``) and ``reference.parquet`` (the reference's PSF
  FWHM and optical state);
- ``masks``: the v4.02g and v4.02s masks the exposures need in ``masks/``: the 4 mm ones
  copied from runs ``20261019_pupil2`` and ``20261023_pupil6``, the 8 mm ones fitted here
  (`obscuration_v402g.fit_config`, then `pass6_silhouette.refit`);
- ``cache``: the 4×4-binned minimal ISR of every science and corner detector
  (``isr_bin4/<visit>/``, as `isr_cache.py`);
- ``select``: the final selection rule (`method.md` section 9, no hand labels) per exposure:
  ``candidates.parquet``, ``tally.csv``, ``candidates_<visit>_<side>.pdf`` with the ^ and v
  markers;
- ``sample``: the selected donuts within 1.75 deg in the central (0–0.8 deg) and outer
  (1.4–1.75 deg) bands, ``sides.parquet``;
- ``stamps``: full ISR (`pass6_isr.isr_config`) of every detector with a side, stamps cut as
  passes 3–6, one FITS file per side (``stamps/<key>.fits``: IMAGE and MASK HDUs, CCS);
- ``fit``: `pass6_pipeline.measure` with the v4.02g masks (with the δ grid, as the night 1
  baseline) and the v4.02s masks (without), into ``v402g/`` and ``v402s/``; then the pattern fits
  (`pass5_analysis`);
- ``vanes``: on the central-band science sides, fit c with the blur free (0.5–2.5 arcsec),
  then one free vane width (`pass6_vanes.one`, v4.02g);
- ``report``: `night2_report.report`.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/night2_pipeline.py --night 20260127 --steps meta
    python aos/giant_donut_pupil/code/night2_pipeline.py --night 20260127 \\
        --steps masks,cache,select,sample,stamps,fit,vanes,report --workers 110
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import candidates as cd  # noqa: E402  (blitz path first)

STUDY = 'aos/giant_donut_pupil'
STUDY_DIR = CODE.parents[0]
OUTPUT = STUDY_DIR / 'output'
PASS2_MASKS = OUTPUT / '20261019_pupil2' / 'masks'
PASS6_RUN = OUTPUT / '20261023_pupil6'
RUN_DATE = '20261023'
NIGHTS = {
    # T680, the field at RA 185 deg (galactic latitude +61.8 deg), three rotator groups; and
    # the three 8 mm exposures of the same block (RA 146 deg, +37.7 deg).
    20260127: dict(band='r', visits=[2026012700000 + s for s in list(range(393, 409))
                                     + [220, 221, 222]]),
    # T621 z band, the one 8 mm exposure (M2 dx −1000 µm), galactic latitude −67 deg.
    20250920: dict(band='z', visits=[2025092000274]),
    # T621 i band, the M2 dx steps (0, −1000, +2000, +1000, −2000, 0 µm).
    20251027: dict(band='i', visits=[2025102700000 + s for s in range(206, 212)]),
}
CENTRAL_BAND = (0.0, 0.8)
OUTER_BAND = (1.4, 1.75)
FWHM_ARCSEC = 1.0
FREE_FWHM_BOUNDS = (0.5, 2.5)
# v4.02g carries the δ grid, as the night 1 baseline (pass 6 ``isr_phys``) does; v4.02s is
# for the silhouette check.
CONFIGS = {'v402g': dict(mask='v4.02g', delta=True), 'v402s': dict(mask='v4.02s', delta=False)}
CORNER_SCALE = 1.19
R_GUESS_PIX = 344.0


def run_path(night):
    return OUTPUT / f'{RUN_DATE}_night2_{night}'


def make_run(night):
    from rubinwork.common import study_run
    return study_run(STUDY, f'night2_{night}', date=RUN_DATE).path


def offsets_mm(step_mm, sensor_class='science'):
    det = {'science': 0.0, 'SW0': 1.5, 'SW1': -1.5}[sensor_class]
    s = float(np.sign(step_mm))
    return (det, s * abs(step_mm), s * abs(step_mm))


# ---------------------------------------------------------------- worker set-up

def _init(run, band):
    """Point the pass 3–6 modules at this run's masks and band (pool initializer)."""
    import pupil_pass3 as p3
    import pass6_common as p6
    p3.MASK_RUN = pathlib.Path(run)
    p3.REF_BAND = band
    p6.RUN = pathlib.Path(run)


def _set_step(step_mm):
    import fit_donuts as fd
    fd.CAMERA_OFFSET_M = abs(step_mm) * 1e-3


def _pool(workers, n, run, band):
    import multiprocessing as mp
    # spawn: a fork after the parent opened Butler or database clients can deadlock.
    # maxtasksperchild: the danish factories and mask caches grow per side.
    return mp.get_context('spawn').Pool(max(1, min(workers, n)), initializer=_init,
                                        initargs=(str(run), band), maxtasksperchild=8)


# ---------------------------------------------------------------- meta (interactive)

def step_meta(run, night):
    import night2_review as nr
    from rubinwork.common.telemetry_clients import make_consdb_client
    visits = NIGHTS[night]['visits']
    _, e = nr.sets_table()
    e = e[e.visit_id.isin(visits)].copy()
    d = nr.trim_deltas(e)
    e = e.join(d)
    cdb = make_consdb_client()
    ids = ','.join(map(str, visits))
    q = cdb.query('select visit_id, physical_rotator_angle, donut_blur_fwhm from '
                  f'cdb_lsstcam.visit1_quicklook where visit_id in ({ids})').to_pandas()
    x = cdb.query('select exposure_id as visit_id, altitude, azimuth, airmass, air_temp, '
                  f'dimm_seeing from cdb_lsstcam.exposure where exposure_id in ({ids})').to_pandas()
    e = e.merge(q, on='visit_id', how='left').merge(x, on='visit_id', how='left')
    e['rtp_deg'] = pd.to_numeric(e.physical_rotator_angle)
    e['ref_visit'] = (e.day_obs.astype(str) + e.ref_seq.map(lambda s: f'{int(s):05d}')).astype(int)
    keep = ['visit_id', 'day_obs', 'seq_num', 'ref_seq', 'ref_visit', 'band', 'exp_time', 'side',
            'step_mm', 'science_program', 'observation_reason', 's_ra', 's_dec', 'rtp_deg',
            'altitude', 'azimuth', 'airmass', 'air_temp', 'dimm_seeing', 'perturbed', 'stepped',
            'dz_mismatch_um'] + [f'd_dof{i}' for i in range(50)]
    e = e[keep].sort_values('seq_num').reset_index(drop=True)
    e.to_parquet(run / 'exposures.parquet')
    e.to_csv(run / 'exposures.csv', index=False, float_format='%.5g')
    thermal(run, e)
    reference(run, e)
    print(e[['seq_num', 'band', 'exp_time', 'step_mm', 'rtp_deg', 'stepped']].to_string(index=False))


def thermal(run, e):
    """Camera, M1M3 and dome temperatures (°C) of the exposures and their references."""
    sys.path.insert(0, str(STUDY_DIR.parents[1] / 'value_added' / 'code'))
    import efd_db
    con = efd_db.connect()
    ids = sorted(set(e.visit_id) | set(e.ref_visit))
    cols = ['cam_AverageTemp', 'cam_CamBodyXPlusTemp', 'cam_CamBodyYPlusTemp',
            'cam_CamBodyYMinusTemp', 'cam_L1XMinusTemp', 'cam_L1YMinusTemp', 'cam_AmbAirtemp',
            'cam_DomeYMinusTemp', 'm1m3_x_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
            'm1m3_z_gradient_c_per_m', 'm1m3_radial_gradient_c_per_m', 'wind_speed_ms']
    t = con.execute(f'select visit_id, day_obs, seq_num, {", ".join(cols)} from visit_telemetry '
                    f'where visit_id in ({",".join(map(str, ids))})').fetchdf()
    r2 = con.execute('select visit_id, m1m3_mean_temp_c, m1_mean_temp_c, m3_mean_temp_c from '
                     f'm1m3_thermal_r2 where visit_id in ({",".join(map(str, ids))})').fetchdf()
    t = t.merge(r2, on='visit_id', how='left')
    t['cam_body_minus_amb_c'] = t[['cam_CamBodyXPlusTemp', 'cam_CamBodyYPlusTemp',
                                   'cam_CamBodyYMinusTemp']].mean(axis=1) - t.cam_AmbAirtemp
    t['cam_l1_minus_amb_c'] = t[['cam_L1XMinusTemp', 'cam_L1YMinusTemp']].mean(axis=1) - t.cam_AmbAirtemp
    t['m1m3_minus_amb_c'] = t.m1m3_mean_temp_c - t.cam_AmbAirtemp
    t['is_reference'] = t.visit_id.isin(e.ref_visit) & ~t.visit_id.isin(e.visit_id)
    t.to_parquet(run / 'thermal.parquet')
    t.to_csv(run / 'thermal.csv', index=False, float_format='%.4g')


def reference(run, e):
    """The in-focus references: ConsDB PSF FWHM (arcsec) and value_added optical state."""
    import night2_review as nr
    from rubinwork.common.telemetry_clients import make_consdb_client
    refs = sorted(e.ref_visit.unique())
    q = make_consdb_client().query(
        'select visit_id, psf_sigma_median, physical_rotator_angle from '
        f'cdb_lsstcam.visit1_quicklook where visit_id in ({",".join(map(str, refs))})').to_pandas()
    q['psf_fwhm_arcsec'] = 2.3548 * pd.to_numeric(q.psf_sigma_median) * nr.PIXEL_ARCSEC
    st = nr.state_of(refs).reset_index()
    q = q.merge(st, on='visit_id', how='left')
    q.to_parquet(run / 'reference.parquet')
    q.to_csv(run / 'reference.csv', index=False, float_format='%.4g')


def exposures(run):
    return pd.read_parquet(run / 'exposures.parquet')


# ---------------------------------------------------------------- masks

def step_masks(run, night, workers):
    import yaml
    import obscuration_v402g as og
    import pass6_common as p6
    import pass6_silhouette as psil
    _init(run, NIGHTS[night]['band'])
    m = run / 'masks'
    m.mkdir(exist_ok=True)
    steps = sorted(exposures(run).step_mm.unique())
    for st in steps:
        if abs(st) == 4.0:
            for cls in ('science', 'SW0', 'SW1'):
                off = offsets_mm(st, cls)
                shutil.copy(PASS2_MASKS / og.mask_name(off), m / og.mask_name(off))
                shutil.copy(PASS6_RUN / 'masks' / p6.mask_name_s(off), m / p6.mask_name_s(off))
    new = [offsets_mm(st) for st in steps if abs(st) != 4.0]
    new = [off for off in new if not (m / p6.mask_name_s(off)).exists()]
    for off in new:
        t0 = time.time()
        params, table = og.fit_config(off)
        (m / og.mask_name(off)).write_text(yaml.safe_dump(params, sort_keys=False,
                                                          default_flow_style=None))
        pd.DataFrame([dict(det_mm=off[0], cam_mm=off[1], m2_mm=off[2], **t) for t in table]) \
            .to_csv(run / f'v402g_silhouette_{off[1]:+.0f}mm.csv', index=False, float_format='%.6g')
        print('v4.02g', off, f'{time.time() - t0:.0f} s', flush=True)
    if new:
        psil.refit(run, new, run, workers)


# ---------------------------------------------------------------- cache

def cache_task(task):
    visit, names, run = task
    import lsst.daf.butler as dafButler
    from lsst.obs.lsst import LsstCam
    import isr_cache
    sys.path.insert(0, str(cd._GIANT_CODE))
    import donut_images
    butler = dafButler.Butler('/repo/main')
    camera = LsstCam.getCamera()
    out = pathlib.Path(run) / 'isr_bin4' / str(visit)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for name in names:
        path = out / f'{name}.npy'
        hdr = out / f'{name}.wcs.json'
        if path.exists() and hdr.exists():
            done.append(name)
            continue
        try:
            image = donut_images.run_isr(butler, visit, name, camera, isr_cache.RAW_COLLECTION)
            wcs = butler.get('raw.wcs', collections=isr_cache.RAW_COLLECTION, instrument='LSSTCam',
                             exposure=visit, detector=camera[name].getId())
        except Exception as exc:
            print(visit, name, 'FAILED', repr(exc)[:200], flush=True)
            continue
        np.save(path, isr_cache.bin_mean(image, isr_cache.CACHE_BIN).astype(np.float32))
        hdr.write_text(json.dumps(wcs.getFitsMetadata().toDict()))
        done.append(name)
    return visit, done


def step_cache(run, night, workers):
    from lsst.obs.lsst import LsstCam
    from lsst.afw.cameraGeom import DetectorType
    camera = LsstCam.getCamera()
    names = [d.getName() for d in camera
             if d.getType() in (DetectorType.SCIENCE, DetectorType.WAVEFRONT)]
    tasks = [(int(v), names[i:i + 10], str(run)) for v in exposures(run).visit_id
             for i in range(0, len(names), 10)]
    t0 = time.time()
    with _pool(workers, len(tasks), run, NIGHTS[night]['band']) as pool:
        for visit, done in pool.imap_unordered(cache_task, tasks):
            pass
    for v in exposures(run).visit_id:
        d = run / 'isr_bin4' / str(int(v))
        headers = {p.name[:-len('.wcs.json')]: json.loads(p.read_text())
                   for p in sorted(d.glob('*.wcs.json'))}
        (d / 'wcs.json').write_text(json.dumps(headers))
        print(v, 'detectors cached', len(headers), flush=True)
    print(f'cache wall {time.time() - t0:.0f} s', flush=True)


# ---------------------------------------------------------------- select

def select_task(task):
    visit, side, step_mm, ra, dec, run = task
    import lsst.daf.butler as dafButler
    from lsst.obs.lsst import LsstCam
    import select_candidates as sc
    import selection_v2 as sv
    import peak_quality as pq
    import peak_refine as pr
    import score_selection as ss
    import final_candidates as fc
    import night2_review as nr
    t0 = time.time()
    run = pathlib.Path(run)
    camera = LsstCam.getCamera()
    images, wcss = sc.load_cache(run / 'isr_bin4' / str(visit), camera)
    refcat = cd.load_refcat_bands(dafButler.Butler('/repo/main'), ra, dec)
    refcat['mag_best'], refcat['mag_source'], _ = cd.fill_r_magnitude(refcat)
    bt, cc, tc = cd.import_blitz()
    bkg = {n: cd.background_map(im, block=128, percentile=10.0) for n, im in images.items()}
    radii = nr.measure_radii(images, bkg, camera, bt, r_guess_pix=R_GUESS_PIX * abs(step_mm) / 4.0)
    for cls in ('SW0', 'SW1'):
        if cls in radii and not np.isfinite(radii[cls][0]):
            radii[cls] = tuple(CORNER_SCALE * v for v in radii['science'])
    sv.VISITS[visit] = side
    orig = sc.measure_radii
    sc.measure_radii = lambda *a, **k: radii  # noqa: E731  (measured above)
    try:
        stars, peaks, gaps, _, diag = sv.build_visit(visit, images, wcss, camera, refcat, (cc, tc),
                                                     bt)
    finally:
        sc.measure_radii = orig
    peaks = peaks[peaks.pass_rel].copy()
    cat = pq.refcat_positions(visit, wcss, camera, refcat, diag['affine'])
    rad = {k: tuple(v) for k, v in diag['radii_pix'].items()}
    pt = pq.peak_rows(visit, images, camera, peaks, cat, rad)
    pt, _ = pq.ring_level_magnitudes(visit, images, camera, pt)
    pt = pq.add_ring_metrics(visit, images, camera, pt, cat)
    pt['saturated'] = pt.ring_p99_e > sc.SATURATION_E
    pt['good_peak'] = pq.good_peak(pt)
    pt = pr.refine_visit(visit, images, camera, pt, cat)
    stars = stars.merge(pt[['visit', 'peak_id', 'overlap_w', 'overlap_w_sep', 'annulus_ratio_sky',
                            'ring_snr']], on=['visit', 'peak_id'], how='left')
    m = ss.marked_donuts(stars, pt, ss.star_rule(stars, 'final'), ss.peak_rule(pt, 'final'))
    lab = pd.DataFrame(columns=['visit', 'detector', 'x_pix', 'y_pix', 'fp_x_mm', 'fp_y_mm',
                                'sensor_class', 'label', 'mark'])
    t = fc.candidate_table(stars, pt, m, lab, camera)
    t['side'] = side
    tally = fc_tally(t, visit, side)
    fc.write_pdf(run / f'candidates_{visit}_{side}.pdf', visit, side, t, images, camera,
                 tally.to_string(index=False),
                 'No hand labels on this night: the final rule (method.md section 9) alone.')
    d = run / 'select' / str(visit)
    d.mkdir(parents=True, exist_ok=True)
    t.to_parquet(d / 'candidates.parquet')
    pt.to_parquet(d / 'peak_metrics.parquet')
    cat.to_parquet(d / 'refcat_fp.parquet')
    stars.to_parquet(d / 'stars.parquet')
    diag['seconds'] = round(time.time() - t0)
    (d / 'diagnostics.json').write_text(json.dumps(diag, default=float))
    print(visit, 'selected', int(t.selected.sum()), f'{time.time() - t0:.0f} s', flush=True)
    return visit


def fc_tally(t, visit, side):
    rows = []
    for cls, g in t.groupby('sensor_class'):
        s = g[g.selected]
        rows.append(dict(visit=visit, side=side, sensor_class=cls, selected=len(s),
                         both=int((s.marker == 'both').sum()), star_only=int((s.marker == '^').sum()),
                         peak_only=int((s.marker == 'v').sum()),
                         within_1p75_deg=int((s.field_r_deg <= 1.75).sum())))
    return pd.DataFrame(rows)


def step_select(run, night, workers):
    e = exposures(run)
    tasks = [(int(r.visit_id), r.side, float(r.step_mm), float(r.s_ra), float(r.s_dec), str(run))
             for r in e.itertuples()]
    with _pool(workers, len(tasks), run, NIGHTS[night]['band']) as pool:
        list(pool.imap_unordered(select_task, tasks))
    cands = pd.concat([pd.read_parquet(run / 'select' / str(int(v)) / 'candidates.parquet')
                       for v in e.visit_id], ignore_index=True)
    cands.to_parquet(run / 'candidates.parquet')
    cands.to_csv(run / 'candidates.csv', index=False, float_format='%.4f')
    tally = pd.concat([fc_tally(cands[cands.visit == v], v, cands[cands.visit == v].side.iloc[0])
                       for v in cands.visit.unique()], ignore_index=True)
    tally.to_csv(run / 'tally.csv', index=False)
    print(tally.to_string(index=False), flush=True)


# ---------------------------------------------------------------- sample

def step_sample(run):
    e = exposures(run).set_index('visit_id')
    c = pd.read_parquet(run / 'candidates.parquet')
    c = c[c.selected & (c.field_r_deg <= OUTER_BAND[1])].copy()
    r = c.field_r_deg
    c['band'] = np.where(r <= CENTRAL_BAND[1], 'central', np.where(r.between(*OUTER_BAND), 'outer', ''))
    c = c[c.band != ''].copy()
    c['seq_num'] = c.visit.map(e.seq_num)
    c['step_mm'] = c.visit.map(e.step_mm)
    c['rtp_deg'] = c.visit.map(e.rtp_deg)
    c['sample_id'] = [f'{int(q):03d}_' + (f'S{int(a)}' if a >= 0 else f'K{int(b)}')
                      for q, a, b in zip(c.seq_num, c.refcat_id, c.peak_id)]
    c['field_x_ccs_deg'], c['field_y_ccs_deg'] = c.field_y_deg, c.field_x_deg
    rt = np.radians(c.rtp_deg.to_numpy())
    x, y = c.field_x_ccs_deg.to_numpy(), c.field_y_ccs_deg.to_numpy()
    c['field_az_ocs_deg'] = np.mod(np.degrees(np.arctan2(np.sin(rt) * x + np.cos(rt) * y,
                                                         np.cos(rt) * x - np.sin(rt) * y)), 360.0)
    # The 8 mm donut is twice the size: the corner sensors cannot hold one whole.
    c = c[(np.abs(c.step_mm) == 4.0) | (c.sensor_class == 'science')]
    c['label'] = c.sample_id
    c['pass3_id'] = ''
    c['in_sample'] = False
    c['in_bands'] = True
    c['fit_ok'] = True
    c['key'] = c.sample_id + ':' + c.side
    c['error'] = ''
    s = c[['key', 'sample_id', 'label', 'pass3_id', 'visit', 'seq_num', 'step_mm', 'side', 'band',
           'detector', 'sensor_class', 'mag', 'field_r_deg', 'field_az_ocs_deg', 'field_x_ccs_deg',
           'field_y_ccs_deg', 'rtp_deg', 'r_out_pix', 'x_pix', 'y_pix', 'fit_ok', 'in_sample',
           'in_bands', 'error']].sort_values(['visit', 'detector', 'key']).reset_index(drop=True)
    s.to_parquet(run / 'sides.parquet')
    s.to_csv(run / 'sides.csv', index=False, float_format='%.6g')
    print(s.groupby(['step_mm', 'band', 'sensor_class']).size().to_string(), flush=True)


def sides(run):
    return pd.read_parquet(run / 'sides.parquet')


# ---------------------------------------------------------------- stamps

def write_stamp(path, img, masked, row, info):
    from astropy.io import fits
    h = fits.Header()
    for k in ('key', 'visit', 'detector', 'side', 'sensor_class'):
        h[k[:8].upper()] = str(getattr(row, k))
    for k, name in (('field_x_ccs_deg', 'FLDX_CCS'), ('field_y_ccs_deg', 'FLDY_CCS'),
                    ('rtp_deg', 'RTP_DEG'), ('step_mm', 'STEP_MM'), ('r_out_pix', 'R_OUT')):
        h[name] = float(getattr(row, k))
    h['SKY_E'] = info['sky_e']
    h['COMMENT'] = 'CCS orientation (fit_donuts.cut), electrons, one sky constant subtracted'
    hdus = [fits.PrimaryHDU(header=h),
            fits.CompImageHDU(img.astype(np.float32), name='IMAGE', compression_type='GZIP_2',
                              quantize_level=0.0),
            fits.CompImageHDU(masked.astype(np.uint8), name='MASK', compression_type='GZIP_2')]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def read_stamp(path):
    from astropy.io import fits
    with fits.open(path) as f:
        return np.asarray(f['IMAGE'].data, float), np.asarray(f['MASK'].data, bool)


def stamp_task(task):
    (visit, det_name), grp, run = task
    import lsst.daf.butler as dafButler
    from lsst.ip.isr import IsrTaskLSST
    from lsst.obs.lsst import LsstCam
    import fit_donuts as fd
    import pass6_isr as pi
    import pupil_pass3 as p3
    run = pathlib.Path(run)
    t0 = time.time()
    camera = LsstCam.getCamera()
    b = dafButler.Butler('/repo/main', collections=['LSSTCam/raw/all', 'LSSTCam/calib'])
    did = dict(instrument='LSSTCam', exposure=int(visit), detector=camera[det_name].getId())
    raw_ref = b.find_dataset('raw', did, collections='LSSTCam/raw/all')
    raw = b.get(raw_ref)
    did = b.registry.expandDataId(raw_ref.dataId)
    span = did.records['exposure'].timespan
    inputs, runs = {}, dict(visit=int(visit), detector=det_name)
    for arg, dt in pi.CALIBS:
        ref = b.find_dataset(dt, did, collections='LSSTCam/calib', timespan=span)
        inputs[arg] = b.get(dt, did)
        runs[dt] = ref.run if ref is not None else ''
    image = np.asarray(IsrTaskLSST(config=pi.isr_config()).run(raw, camera=camera, **inputs)
                       .exposure.image.array, float)
    d = run / 'select' / str(int(visit))
    cat = pd.read_parquet(d / 'refcat_fp.parquet')
    pk = pd.read_parquet(d / 'peak_metrics.parquet')
    orphans = pk[pk.match_refcat_id < 0][['visit', 'detector', 'x_pix', 'y_pix']]
    neigh = p3.neighbours(visit, det_name, camera, cat, orphans)
    nq = camera[det_name].getOrientation().getNQuarter()
    rows = []
    for row in grp.itertuples():
        stamp, masked, info = fd.cut(image, pd.Series(dict(r_out_pix=row.r_out_pix, x_pix=row.x_pix,
                                                           y_pix=row.y_pix, mag_r=row.mag)), neigh, nq)
        write_stamp(run / 'stamps' / f'{row.key.replace(":", "_")}.fits', stamp, masked, row, info)
        rows.append(dict(key=row.key, visit=int(visit), detector=det_name, **info))
    print(det_name, visit, len(grp), f'{time.time() - t0:.0f} s', flush=True)
    return rows, runs


def step_stamps(run, night, workers):
    import pass6_isr as pi
    (run / 'stamps').mkdir(exist_ok=True)
    (run / 'isr_config.py').write_text(pi.isr_config().saveToString())
    S = sides(run)
    tasks = [(k, g, str(run)) for k, g in S.groupby(['visit', 'detector'])]
    rows, runs = [], []
    t0 = time.time()
    with _pool(workers, len(tasks), run, NIGHTS[night]['band']) as pool:
        for r, c in pool.imap_unordered(stamp_task, tasks):
            rows += r
            runs.append(c)
    pd.DataFrame(rows).to_parquet(run / 'stamps.parquet')
    pd.DataFrame(runs).to_csv(run / 'isr_calibs.csv', index=False)
    print(f'stamps {len(rows)} wall {time.time() - t0:.0f} s', flush=True)


# ---------------------------------------------------------------- fit

def _part(run, name, key):
    return pathlib.Path(run) / 'fit_parts' / name / f'{key.replace(":", "_")}.pkl'


def fit_task(task):
    """One side and configuration; the result is kept in ``fit_parts/`` so a rerun resumes."""
    import pickle
    name, row, run = task
    import pass6_pipeline as pp
    part = _part(run, name, row.key)
    if part.exists():
        return name, row.key
    _set_step(row.step_mm)
    cfg = CONFIGS[name]
    img, msk = read_stamp(pathlib.Path(run) / 'stamps' / f'{row.key.replace(":", "_")}.fits')
    try:
        out = pp.measure(row, img, msk, FWHM_ARCSEC, mask=cfg['mask'], delta=cfg['delta'])
    except Exception as err:  # a failed side is a row, not a crashed run
        print('error', name, row.key, repr(err)[:200], flush=True)
        return name, None
    out['fit']['step_mm'] = row.step_mm
    out['fit']['visit'] = int(row.visit)
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix('.tmp')
    tmp.write_bytes(pickle.dumps(out))
    tmp.rename(part)
    print(name, row.key, f'{out["fit"]["seconds"]:.0f} s', 'found', out['fit']['n_found'], flush=True)
    return name, row.key


def step_fit(run, night, workers, configs=('v402g', 'v402s')):
    import pass6_pipeline as pp
    import pass5_analysis as pa
    _init(run, NIGHTS[night]['band'])
    S = sides(run)
    S = S[[(run / 'stamps' / f'{k.replace(":", "_")}.fits').exists() for k in S.key]]
    tasks = [(n, r, str(run)) for n in configs for _, r in S.iterrows()]
    # largest stamps (8 mm) and the δ-grid configuration first
    tasks.sort(key=lambda t: (-abs(t[1].step_mm), not CONFIGS[t[0]]['delta']))
    import pickle
    t0 = time.time()
    with _pool(workers, len(tasks), run, NIGHTS[night]['band']) as pool:
        for _ in pool.imap_unordered(fit_task, tasks):
            pass
    res = {n: [pickle.loads(p.read_bytes()) for p in sorted((run / 'fit_parts' / n).glob('*.pkl'))]
           for n in configs}
    for name in configs:
        tabs = pp.tables(res[name], S)
        tabs['fits']['step_mm'] = tabs['fits'].key.map(S.set_index('key').step_mm)
        d = run / name
        pp.save(tabs, d)
        if CONFIGS[name]['delta']:
            pa.run_all(d)
        else:
            tf = pa.translation_fit(tabs['spiders'])
            pa._save(tf, d, 'translation_fit')
            ct, eo = pa.centre_test(tabs['spiders'], tabs['centres'], tabs['sides'])
            pa._save(ct, d, 'centre_test')
            pa._save(eo, d, 'centre_edge_offsets')
        print(name, 'sides', len(tabs['fits']), flush=True)
    print(f'fit wall {time.time() - t0:.0f} s', flush=True)


# ---------------------------------------------------------------- vanes

def free_blur(row, img, msk, fitc):
    """fit c from its solution with the Kolmogorov FWHM free (arcsec), v4.02g."""
    from scipy.optimize import least_squares
    import fit_donuts as fd
    import pass6_pipeline as pp
    import pupil_pass3 as p3
    noise = pp.noise_of(img, msk)
    var = np.where(msk, fd.MASKED_VAR, noise ** 2)
    M = pp.Model(row, FWHM_ARCSEC)
    m = M.single(img.shape[0], p3.FITS['c'])
    z = [fitc[f'z{j}_ccs_um'] * 1e-6 for j in p3.FITS['c']]
    p0 = np.array([fitc['flux_e'], fitc['dx_pix'] * m.sky_scale, fitc['dy_pix'] * m.sky_scale,
                   FWHM_ARCSEC] + z)
    lo = [0.0, -np.inf, -np.inf, FREE_FWHM_BOUNDS[0]] + [-np.inf] * len(z)
    hi = [np.inf, np.inf, np.inf, FREE_FWHM_BOUNDS[1]] + [np.inf] * len(z)
    r = least_squares(lambda p: m.chi(p, img, var), x0=p0, jac=lambda p: m.jac(p, img, var),
                      bounds=(lo, hi), xtol=fd.TOL, ftol=fd.TOL, gtol=fd.TOL, x_scale='jac',
                      tr_solver='lsmr', max_nfev=fd.MAX_NFEV)
    return float(r.x[3])


def vane_task(task):
    row, fitc, run = task
    import pass6_vanes as pv
    _set_step(row.step_mm)
    img, msk = read_stamp(pathlib.Path(run) / 'stamps' / f'{row.key.replace(":", "_")}.fits')
    try:
        fwhm = free_blur(row, img, msk, fitc)
        out, chi, sh = pv.one((row, img, msk, fwhm, fitc, 'data'))
    except Exception as err:
        print('error vanes', row.key, repr(err)[:200], flush=True)
        return None
    out['step_mm'] = row.step_mm
    out['visit'] = int(row.visit)
    return out, chi, sh


def step_vanes(run, night, workers):
    S = sides(run).set_index('key')
    F = pd.read_parquet(run / 'v402g' / 'fits.parquet')
    F = F[F.fit_ok & F.key.isin(S.index)]
    F = F[(S.loc[F.key].band == 'central').to_numpy() & (S.loc[F.key].sensor_class == 'science').to_numpy()]
    tasks = []
    for f in F.itertuples():
        row = S.loc[f.key].copy()
        row['key'] = f.key
        tasks.append((row, f._asdict(), str(run)))
    out, chi, sh = [], [], []
    t0 = time.time()
    with _pool(workers, len(tasks), run, NIGHTS[night]['band']) as pool:
        for r in pool.imap_unordered(vane_task, tasks):
            if r is not None:
                out.append(r[0])
                chi += r[1]
                sh += r[2]
    for name, rows in (('vane_width_data', out), ('vane_width_chi2', chi),
                       ('vane_width_spiders', sh)):
        df = pd.DataFrame(rows)
        df.to_parquet(run / f'{name}.parquet')
        df.to_csv(run / f'{name}.csv', index=False, float_format='%.5g')
    print(f'vanes {len(out)} wall {time.time() - t0:.0f} s', flush=True)


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--night', type=int, required=True, choices=sorted(NIGHTS))
    ap.add_argument('--steps', default='masks,cache,select,sample,stamps,fit,vanes,report')
    ap.add_argument('--workers', type=int, default=110)
    args = ap.parse_args()
    run = make_run(args.night)
    t0 = time.time()
    for step in args.steps.split(','):
        print('step', step, f'{time.time() - t0:.0f} s', flush=True)
        if step == 'meta':
            step_meta(run, args.night)
        elif step == 'masks':
            step_masks(run, args.night, args.workers)
        elif step == 'cache':
            step_cache(run, args.night, args.workers)
        elif step == 'select':
            step_select(run, args.night, args.workers)
        elif step == 'sample':
            step_sample(run)
        elif step == 'stamps':
            step_stamps(run, args.night, args.workers)
        elif step == 'fit':
            step_fit(run, args.night, args.workers)
        elif step == 'vanes':
            step_vanes(run, args.night, args.workers)
        elif step == 'report':
            import night2_report
            night2_report.report(run, args.night)
        else:
            raise ValueError(f'unknown step {step}')
    print('done', run, f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
