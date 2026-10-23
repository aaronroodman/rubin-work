"""Pupil pass 3: four danish fits and the robust edges per giant-donut side, v4.02 only.

One driver for any candidate table. Per donut side:

1. the stamp: overscan-and-gain ISR (`donut_images.run_isr`), cut and rotated to
   CCS with neighbours masked (`fit_donuts.cut`; neighbours are the_monster stars
   brighter than the target plus 2.5 mag, and orphan peaks);
2. the reference wavefront, "intrinsicTA": ts_wep
   ``Instrument.getOffAxisCoeff`` plus ``Instrument.getIntrinsicZernikes`` on the
   batoid ``LSST_r_v4.02`` at the defocused configuration (see `reference_zk`);
3. four fits, blur fixed at 1.0 arcsec FWHM, spider on at the rotator angle, every
   other Zernike fixed at the reference (a deviation of 0):
   ``a`` Z4; ``b`` Z4, Z11, Z22; ``c`` Z4-Z11, Z14, Z15, Z20, Z21, Z22; ``d`` Z4,
   Z7, Z8, Z11, Z22 fitted only to the pixels within ``ANNULUS_PIX`` of the data's
   robust inner and outer edges, spider bins excluded;
4. robust edges (`robust_edges.robust_edges`) of the data and the four models,
   about the fit-a centre, with the data's noise.

The danish mask is the v4.02g file for the side's (detector, camera, M2) offsets.
Deviations are the fitted perturbations: the reference minus the in-focus
intrinsic is held fixed, so a deviation is the Zernike minus the in-focus
intrinsic, as ts_wep reports with ``returnWfDev``. Zernikes are µm of wavefront,
CCS and OCS.

The sides run in parallel chunks of one (visit, detector) each, so the ISR runs
once per detector. Steps: ``fit`` (writes ``fits.parquet``, ``edges_bins.parquet``
and, for the sides that get pages, ``images/<id>_<side>.npz``) and ``report``
(the PDF). Run each as one background task.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pupil_pass3.py --run 20261020_pupil3 --steps fit,report
    python aos/giant_donut_pupil/code/pupil_pass3.py --run 20261020_pupil3 \\
        --candidates all --pages none --steps fit,report
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
import candidates as cd  # noqa: E402  (blitz path first)
import fit_donuts as fd  # noqa: E402
import fit_giant_donut  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import obscuration_v402g as og  # noqa: E402
import pupil_edges as pe  # noqa: E402
import pupil_sample as ps  # noqa: E402
import robust_edges as rob  # noqa: E402

STUDY_DIR = pathlib.Path(__file__).resolve().parents[1]
MASK_RUN = STUDY_DIR / 'output' / '20261019_pupil2'
METRICS_RUN = STUDY_DIR / 'output' / '20261020_peak_metrics'
REFINE_RUN = STUDY_DIR / 'output' / '20261020_peak_refine'
CANDIDATES_RUN = STUDY_DIR / 'output' / '20261020_candidates_v2'
RAW_COLLECTION = 'LSSTCam/raw/all'
FWHM_ARCSEC = 1.0
NOLL_MODEL = np.arange(0, 67)
NOLL_INTR = np.arange(4, 67)
FITS = {'a': (4,), 'b': (4, 11, 22), 'c': (4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 20, 21, 22),
        'd': (4, 7, 8, 11, 22)}
FIT_LABELS = {'a': 'a: Z4', 'b': 'b: Z4, Z11, Z22', 'c': 'c: Z4-Z11, Z14, Z15, Z20-Z22',
              'd': 'd: annulus, Z4, Z7, Z8, Z11, Z22'}
ANNULUS_PIX = 8.0
NOLL_REPORT = (4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 20, 21, 22)
VIGNETTED_FRAC = 0.3
# Filter band of the reference wavefront; the night 2 drivers set it per night.
REF_BAND = 'r'

_OPTIC = {}
_BUTLER = {}


def optic_v402(eps):
    """batoid v4.02 without its spider planes; ``pupilObscuration`` set to danish's."""
    if 'optic' not in _OPTIC:
        _OPTIC['optic'] = ob.v402_optic(spider=False)
    optic = _OPTIC['optic']
    optic.pupilObscuration = eps
    return optic


def reference_zk(thx_deg, thy_deg, offsets_m, side, eps):
    """intrinsicTA and the in-focus intrinsic, m of wavefront, Noll 0-66.

    ts_wep's danish path (`lsst.ts.wep.estimation.danish.DanishAlgorithm._prepDanish`
    with ``startWithIntrinsic``) builds the reference as::

        off = instrument.getOffAxisCoeff(thx, thy, defocalType, band,
                                         nollIndicesModel=NOLL_MODEL,
                                         nollIndicesIntr=NOLL_INTR)
        zk_ref = off;  zk_ref[NOLL_INTR] += instrument.getIntrinsicZernikes(
            thx, thy, band=band, nollIndices=NOLL_INTR)

    which is ``batoid.zernikeTA`` (jmax 66, nrad 10, naz 62) of the model shifted by
    ``Instrument._applyBatoidOffsets``. The in-focus intrinsic is
    ``getIntrinsicZernikes(thx, thy, defocalType=None)``: ``batoid.zernikeGQ``, rings
    12, unshifted. The Instrument is ``policy:instruments/LsstCam.yaml`` with
    ``batoidOffsetOptic = ['Detector', 'LSSTCamera', 'M2']`` and
    ``batoidOffsetValue = [sign * det, 4 mm, 4 mm]``, sign +1 extra-focal, so the
    detector offset of a corner sensor keeps its sign on both sides; the batoid
    model is v4.02 (`optic_v402`), with ``pupilObscuration`` set to danish's
    ``R_inner / R_outer`` so the reference is on danish's annulus.

    Parameters
    ----------
    thx_deg, thy_deg : `float`
        Field angle, CCS, deg.
    offsets_m : `tuple` [`float`]
        Signed (detector, camera, M2) z offsets, m.
    side : `str`
        ``'extra'`` or ``'intra'``.
    eps : `float`
        Annulus obscuration ratio (dimensionless).

    Returns
    -------
    zk_ta, zk_focus : `numpy.ndarray`
        Noll 0-66, m of wavefront.
    """
    from lsst.ts.wep.instrument import Instrument
    from lsst.ts.wep.utils import BandLabel, DefocalType
    sign = 1.0 if side == 'extra' else -1.0
    det, cam, m2 = offsets_m
    inst = Instrument(configFile='policy:instruments/LsstCam.yaml',
                      batoidOffsetOptic=['Detector', 'LSSTCamera', 'M2'],
                      batoidOffsetValue=[sign * det, sign * cam, sign * m2])
    optic = optic_v402(eps)
    inst.getBatoidModel = lambda band=BandLabel.REF: optic
    dtype = DefocalType.Extra if side == 'extra' else DefocalType.Intra
    band = BandLabel(REF_BAND)
    off = inst.getOffAxisCoeff(thx_deg, thy_deg, dtype, band, nollIndicesModel=NOLL_MODEL,
                               nollIndicesIntr=NOLL_INTR)
    focus = np.zeros(len(NOLL_MODEL))
    focus[NOLL_INTR] = inst.getIntrinsicZernikes(thx_deg, thy_deg, None, band,
                                                 nollIndices=NOLL_INTR)
    zk_ta = off.copy()
    zk_ta[NOLL_INTR] += focus[NOLL_INTR]
    return zk_ta, focus


def _factory(params, rtp_deg, spider=True):
    import danish
    blitz_utils, _ = fit_giant_donut._import_blitz()
    inst = blitz_utils._INSTRUMENT
    r_outer, r_inner = ob._annulus(params)
    if not spider:
        params = {k: v for k, v in params.items() if k != 'Spider_3D'}
    return danish.DonutFactory(R_outer=r_outer, R_inner=r_inner, mask_params=params,
                               focal_length=inst.focalLength, pixel_scale=inst.pixelSize,
                               spider_angle=rtp_deg)


def fit(img, var, factory, zk_ref, thx, thy, noll, x0=None):
    """danish fit of flux, centre and the `noll` deviations, blur fixed.

    Returns
    -------
    res : `dict`
        ``x`` (flux e, dx, dy arcsec, deviations m), ``model``, ``cost``, ``nfev``,
        ``success``, ``seconds`` and the danish model object.
    """
    import danish
    from scipy.optimize import least_squares
    model = danish.SingleDonutModel(factory, z_ref=zk_ref, z_terms=tuple(noll), thx=thx, thy=thy,
                                    npix=img.shape[0], bkg_order=-1)

    def full(x):
        return np.insert(x, 3, FWHM_ARCSEC)

    def chi(x):
        return model.chi(full(x), img, var)

    def jac(x):
        return np.delete(model.jac(full(x), img, var), 3, axis=1)

    if x0 is None:
        good = var < fd.MASKED_VAR
        x0 = np.array([float(np.clip(np.sum(img[good]), 1e3, 1e10)), 0.0, 0.0])
    x0 = np.concatenate([x0[:3], np.zeros(len(noll))])
    bounds = ([0.0] + [-np.inf] * (len(x0) - 1), [np.inf] * len(x0))
    t0 = time.time()
    r = least_squares(chi, x0=x0, jac=jac, bounds=bounds, xtol=fd.TOL, ftol=fd.TOL, gtol=fd.TOL,
                      x_scale='jac', tr_solver='lsmr', max_nfev=fd.MAX_NFEV)
    p = model.unpack_params(full(r.x))
    img_model = np.asarray(model.model(p['flux'], p['dx'], p['dy'], FWHM_ARCSEC, p['z_fit']), float)
    return dict(x=r.x, model=img_model, cost=float(r.cost), nfev=int(r.nfev),
                success=bool(r.success), seconds=time.time() - t0, danish=model,
                chi=lambda v: model.chi(full(r.x), img, v))


def spider_and_vignetting(params, rtp_deg, zk_ref, thx, thy, npix, cx, cy, r_out):
    """Spider and vignetted azimuth bins (CCS, about the donut centre).

    From two danish images at the reference, no blur, with and without the vanes
    (as `pupil_edges.spider_bins`): a bin is spider where the vanes remove more
    than 20 % of the light from a ring pixel, plus the neighbouring bins. A bin is
    vignetted where the no-vane image's flat top is under 0.3 of its maximum.
    """
    imgs = [np.asarray(_factory(params, rtp_deg, spider=s).image(
        aberrations=zk_ref, thx=thx, thy=thy, npix=npix)) for s in (True, False)]
    with_v, without = imgs
    yy, xx = np.mgrid[:npix, :npix] - npix // 2
    lit = without > 0.5 * np.median(without[without > 0])
    shadow = lit & (with_v < 0.8 * without)
    phi = np.mod(np.degrees(np.arctan2(yy[shadow], xx[shadow])), 360.0)
    hit = np.zeros(pe.NBIN, bool)
    hit[np.minimum((phi / (360 / pe.NBIN)).astype(int), pe.NBIN - 1)] = True
    spider = hit | np.roll(hit, 1) | np.roll(hit, -1)
    tops = pe.edges(without, np.ones_like(without, bool), cx, cy, r_out)[4]
    dark = ~(tops > VIGNETTED_FRAC * np.nanmax(tops))
    return spider, dark | np.roll(dark, 1) | np.roll(dark, -1)


def annulus_region(n, cx, cy, fit_in, fit_out, spider, pad=ANNULUS_PIX):
    """Pixels within `pad` of the data's robust inner or outer edge, spider bins out."""
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - cx, yy - cy)
    phi = np.mod(np.degrees(np.arctan2(yy - cy, xx - cx)), 360.0)
    b = np.minimum((phi / (360 / pe.NBIN)).astype(int), pe.NBIN - 1)
    near = (np.abs(r - fit_in[b]) <= pad) | (np.abs(r - fit_out[b]) <= pad)
    return near & ~spider[b] & np.isfinite(fit_in[b]) & np.isfinite(fit_out[b])


def edges_of(image, valid, cx, cy, r_out, noise, flags):
    spider, dark, m_in, m_out = flags
    return rob.robust_edges(image, valid, cx, cy, r_out, noise, spider | dark | m_in,
                            spider | dark | m_out)


def one_side(row, stamp, masked, out_dir, save_images):
    """Fits a-d and edges of one side; returns (fit rows, bins frame)."""
    import yaml
    from scipy.stats import median_abs_deviation
    blitz_utils, _ = fit_giant_donut._import_blitz()
    t0 = time.time()
    offsets = fd.offsets_m(row.side, row.sensor_class)
    params = yaml.safe_load(og.mask_path(MASK_RUN, offsets).read_text())
    r_outer, r_inner = ob._annulus(params)
    thx, thy = np.radians(row.field_x_ccs_deg), np.radians(row.field_y_ccs_deg)
    zk_ta, zk_focus = reference_zk(row.field_x_ccs_deg, row.field_y_ccs_deg, offsets, row.side,
                                   r_inner / r_outer)
    factory = _factory(params, row.rtp_deg)
    img = np.asarray(stamp, float)
    msk = masked.astype(bool)
    n = img.shape[0]
    diff = (img[1:] - img[:-1])[~(msk[1:] | msk[:-1])]
    noise = median_abs_deviation(diff, scale='normal') / np.sqrt(2.0)
    var = np.where(msk, fd.MASKED_VAR, noise ** 2)
    n_good = int((~msk).sum())

    res = {'a': fit(img, var, factory, zk_ta, thx, thy, FITS['a'])}
    sky_scale = res['a']['danish'].sky_scale
    cx = n // 2 + res['a']['x'][1] / sky_scale
    cy = n // 2 + res['a']['x'][2] / sky_scale
    spider, dark = spider_and_vignetting(params, row.rtp_deg, zk_ta, thx, thy, n, cx, cy,
                                         row.r_out_pix)
    _, _, m_in, m_out, _ = pe.edges(img, ~msk, cx, cy, row.r_out_pix)
    flags = (spider, dark, m_in, m_out)
    e_data = edges_of(img, ~msk, cx, cy, row.r_out_pix, noise, flags)
    for k in ('b', 'c'):
        res[k] = fit(img, var, factory, zk_ta, thx, thy, FITS[k], x0=res['a']['x'])
    region = annulus_region(n, cx, cy, e_data['fit_inner'], e_data['fit_outer'], spider) & ~msk
    var_d = np.where(region, noise ** 2, fd.MASKED_VAR)
    res['d'] = fit(img, var_d, factory, zk_ta, thx, thy, FITS['d'], x0=res['a']['x'])
    seconds_fit = time.time() - t0

    rtp = np.radians(row.rtp_deg)
    base = dict(sample_id=row.sample_id, side=row.side, visit=int(row.visit),
                detector=row.detector, sensor_class=row.sensor_class, mag=row.mag,
                field_r_deg=row.field_r_deg, field_az_ocs_deg=row.field_az_ocs_deg,
                field_x_ccs_deg=row.field_x_ccs_deg, field_y_ccs_deg=row.field_y_ccs_deg,
                rtp_deg=row.rtp_deg, det_offset_mm=offsets[0] * 1e3,
                cam_offset_mm=offsets[1] * 1e3, m2_offset_mm=offsets[2] * 1e3,
                r_out_pix=row.r_out_pix, noise_e=float(noise), stamp_npix=n,
                masked_frac=float(msk.mean()), stamp_sum_e=float(img[~msk].sum()))
    for j in range(4, 12):
        base[f'z{j}_ta_um'] = zk_ta[j] * 1e6
        base[f'z{j}_focus_um'] = zk_focus[j] * 1e6
    e_models = {}
    rows = []
    for k, r in res.items():
        nz = len(FITS[k])
        dev = np.full(len(NOLL_MODEL), np.nan)
        for j, v in zip(FITS[k], r['x'][3:]):
            dev[j] = v * 1e6
        dev_ocs = blitz_utils._rotate_zk(dev[None, :], -rtp)[0]
        out = dict(base, fit=k, fit_label=FIT_LABELS[k], n_free=3 + nz, nfev=r['nfev'],
                   success=r['success'], fit_seconds=r['seconds'],
                   flux_e=float(r['x'][0]), dx_pix=float(r['x'][1] / sky_scale),
                   dy_pix=float(r['x'][2] / sky_scale))
        chi_all = r['chi'](var)
        out['chi2_dof'] = float(np.sum(chi_all ** 2) / max(n_good - 3 - nz, 1))
        out['dof'] = n_good - 3 - nz
        if k == 'd':
            n_reg = int(region.sum())
            out['chi2_dof_fitted'] = float(2 * r['cost'] / max(n_reg - 3 - nz, 1))
            out['dof_fitted'] = n_reg - 3 - nz
        for j in NOLL_REPORT:
            out[f'z{j}_ccs_um'] = dev[j]
            out[f'z{j}_ocs_um'] = dev_ocs[j]
        e_models[k] = edges_of(r['model'], ~msk, cx, cy, row.r_out_pix, noise, flags)
        rows.append(out)

    az_ccs = (np.arange(pe.NBIN) + 0.5) * 360 / pe.NBIN
    bins = pd.DataFrame(dict(sample_id=row.sample_id, side=row.side, bin=np.arange(pe.NBIN),
                             az_ccs_deg=az_ccs, az_ocs_deg=np.mod(az_ccs + row.rtp_deg, 360.0),
                             spider=spider, vignetted=dark, masked_inner=m_in,
                             masked_outer=m_out, field_r_deg=row.field_r_deg))
    for name, e in [('data', e_data)] + list(e_models.items()):
        for edge in ('inner', 'outer'):
            bins[f'r_{edge}_{name}_pix'] = e[f'r_{edge}']
            bins[f'fit_{edge}_{name}_pix'] = e[f'fit_{edge}']
            bins[f'q_{edge}_{name}'] = e[f'q_{edge}']
    for r in rows:
        k = r['fit']
        for edge in ('inner', 'outer'):
            d = (bins[f'r_{edge}_data_pix'] - bins[f'r_{edge}_{k}_pix']).dropna()
            r[f'edge_rms_{edge}_pix'] = float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan
            r[f'edge_median_{edge}_pix'] = float(np.median(d)) if len(d) else np.nan
            r[f'edge_n_{edge}'] = len(d)
        q = (bins[f'r_inner_{k}_pix'] / bins[f'r_outer_{k}_pix']).dropna()
        r['ratio_median'] = float(q.median()) if len(q) else np.nan
    qd = (bins.r_inner_data_pix / bins.r_outer_data_pix).dropna()
    for r in rows:
        r['ratio_median_data'] = float(qd.median()) if len(qd) else np.nan
        r['ratio_n_data'] = len(qd)
        r['side_seconds'] = time.time() - t0
        r['fit_side_seconds'] = seconds_fit
    if save_images:
        np.savez_compressed(out_dir / 'images' / f'{row.sample_id}_{row.side}.npz',
                            img=img.astype(np.float32), masked=msk, region=region,
                            cx=cx, cy=cy, **{f'model_{k}': r['model'].astype(np.float32)
                                             for k, r in res.items()})
    return rows, bins


def neighbours(visit, det_name, camera, cat, orphans):
    """Neighbour centres (pixels) and magnitudes on one detector, for `fit_donuts.cut`."""
    det = camera[det_name]
    c = det.getCenter(cd.FOCAL_PLANE)
    near = cat[np.hypot(cat.x_mm - c.getX(), cat.y_mm - c.getY()) < 45.0]
    px, py = cd.focal_to_pixels(det, near.x_mm.to_numpy(), near.y_mm.to_numpy())
    o = orphans[(orphans.visit == visit) & (orphans.detector == det_name)]
    xy = np.vstack([np.column_stack([px, py]), o[['x_pix', 'y_pix']].to_numpy()])
    mag = np.concatenate([near.mag_best.to_numpy(), np.full(len(o), np.nan)])
    return xy, mag


_CTX = {}


def chunk(task):
    """ISR one (visit, detector), then every side on it."""
    (visit, det_name), sides, out_dir, page_ids = task
    import lsst.daf.butler as dafButler
    sys.path.insert(0, str(cd._GIANT_CODE))
    import donut_images as di
    if 'butler' not in _BUTLER:
        _BUTLER['butler'] = dafButler.Butler('/repo/main')
    camera = _CTX['camera']
    out_dir = pathlib.Path(out_dir)
    t0 = time.time()
    image = di.run_isr(_BUTLER['butler'], visit, det_name, camera, RAW_COLLECTION)
    isr_s = time.time() - t0
    neigh = neighbours(visit, det_name, camera, _CTX['cat'][_CTX['cat'].visit == visit],
                       _CTX['orphans'])
    n_quarter = camera[det_name].getOrientation().getNQuarter()
    rows, bins = [], []
    for row in sides.itertuples():
        try:
            stamp, masked, info = fd.cut(image, pd.Series(dict(
                r_out_pix=row.r_out_pix, x_pix=row.x_pix, y_pix=row.y_pix, mag_r=row.mag)),
                neigh, n_quarter)
            r, b = one_side(row, stamp, masked, out_dir, row.sample_id in page_ids)
            for x in r:
                x.update(info, isr_seconds=isr_s / len(sides), error='')
            rows += r
            bins.append(b)
            print(json.dumps(dict(id=row.sample_id, side=row.side,
                                  s=round(r[0]['side_seconds']),
                                  chi2_dof={x['fit']: round(x['chi2_dof'], 3) for x in r})),
                  flush=True)
        except Exception as err:  # a failed side is a row, not a crashed run
            rows.append(dict(sample_id=row.sample_id, side=row.side, visit=visit,
                             detector=det_name, error=repr(err)))
            print('error', row.sample_id, row.side, repr(err), flush=True)
    return rows, bins


def all_candidates():
    """Every selected, not-bad donut side on 337 and 340 within 1.75 deg."""
    cand = pd.read_parquet(CANDIDATES_RUN / 'candidates_v2.parquet')
    t = cand[cand.selected & cand.visit.isin(ps.VISITS.values())
             & (cand.field_r_deg <= ps.FIELD_MAX_DEG) & (cand.hand_label != 'bad')].copy()
    t = ps.add_ocs(t)
    t['side'] = t.visit.map({v: k for k, v in ps.VISITS.items()})
    t['sample_id'] = np.where(t.refcat_id >= 0, 'S' + t.refcat_id.astype(str),
                              'K' + t.peak_id.astype(str))
    return t.reset_index(drop=True)


def step_fit(run, cand, pages, workers):
    import multiprocessing as mp
    from lsst.obs.lsst import LsstCam
    (run / 'images').mkdir(exist_ok=True)
    _CTX['camera'] = LsstCam.getCamera()
    _CTX['cat'] = pd.read_parquet(METRICS_RUN / 'refcat_fp.parquet')
    pk = pd.read_parquet(REFINE_RUN / 'peak_metrics.parquet')
    _CTX['orphans'] = pk[pk.match_refcat_id < 0][['visit', 'detector', 'x_pix', 'y_pix']]
    optic_v402(0.62)  # build before forking
    if pages == 'all':
        page_ids = set(cand.sample_id)
    elif pages == 'sample':
        in_sample = cand.refcat_id.isin(ps.SCIENCE) | (cand.sensor_class != 'science')
        page_ids = set(cand.sample_id[in_sample])
    else:
        page_ids = set()
    tasks = [(key, g, str(run), page_ids) for key, g in cand.groupby(['visit', 'detector'])]
    tasks.sort(key=lambda t: -len(t[1]))
    print(f'sides {len(cand)}, chunks {len(tasks)}, workers {workers}', flush=True)
    t0 = time.time()
    rows, bins = [], []
    with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
        for r, b in pool.imap_unordered(chunk, tasks):
            rows += r
            bins += b
    wall = time.time() - t0
    fits = pd.DataFrame(rows)
    fits.to_parquet(run / 'fits.parquet')
    fits.to_csv(run / 'fits.csv', index=False, float_format='%.5g')
    pd.concat(bins, ignore_index=True).to_parquet(run / 'edges_bins.parquet')
    ok = fits[fits.error == '']
    per = ok.drop_duplicates(['sample_id', 'side'])
    timing = dict(sides=int(len(per)), failed=int(fits[fits.error != ''].drop_duplicates(
        ['sample_id', 'side']).shape[0]), chunks=len(tasks), workers=workers, wall_s=wall,
        side_s_median=float(per.side_seconds.median()), side_s_max=float(per.side_seconds.max()),
        fit_s_median=float(per.fit_side_seconds.median()),
        isr_s_per_detector_median=float(per.groupby(['visit', 'detector']).isr_seconds.sum()
                                        .median()))
    (run / 'timing.json').write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing), flush=True)
    return fits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True, help='pupil3 run under output/ (holds sample.parquet)')
    ap.add_argument('--candidates', default='sample',
                    help="'sample' (the run's sample.parquet), 'all', or a parquet path")
    ap.add_argument('--pages', choices=('none', 'sample', 'all'), default='all')
    ap.add_argument('--steps', default='fit,report')
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--limit', type=int, default=None, help='first N sides only (a timing test)')
    ap.add_argument('--out-label', default=None,
                    help='write fits into this subdirectory of the run (default: the run)')
    args = ap.parse_args()
    run = STUDY_DIR / 'output' / args.run
    if args.candidates == 'sample':
        cand = pd.read_parquet(run / 'sample.parquet')
    elif args.candidates == 'all':
        cand = all_candidates()
    else:
        cand = pd.read_parquet(args.candidates)
    if args.limit:
        cand = cand.iloc[:args.limit]
    out = run / args.out_label if args.out_label else run
    out.mkdir(exist_ok=True)
    steps = args.steps.split(',')
    if 'fit' in steps:
        step_fit(out, cand, args.pages, args.workers)
    if 'report' in steps:
        import pupil_pass3_report as rep
        rep.report(out, cand)
    print('done', out)


if __name__ == '__main__':
    main()
