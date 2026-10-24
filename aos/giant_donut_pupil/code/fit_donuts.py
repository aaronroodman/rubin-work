"""Danish fits of the 20 picked giant donuts: 4 fits per donut and side.

{Z4 + Z11 + Z22, blitz set Noll 4-19 and 22-26} x {v1000, v4.02}, single-sided,
spider on, blur FWHM bounded 0.1-1.5 arcsec as in `fit_giant_donut.fit_stamp`.

Conventions follow blitz (`cutDonutStampsTask`): the stamp is rotated into CCS
with ``np.rot90(stamp, k=-n_quarter).T``, the CCS field angle is
``(FIELD_ANGLE[1], FIELD_ANGLE[0])``, and the spider angle is the rotator
angle ``rtp``. OCS values rotate by ``rtp`` as blitz's catalog does.

Each pupil swaps both representations together: the danish mask (R_outer,
R_inner and ``mask_params``) and the batoid model that supplies the reference
wavefront and the defocus.

- v1000: danish ``RubinObsc_v1000_r_rtpp0_azp45_pp0d0.yaml``, ``Rubin_v1000_r.yaml``.
- v4.02: the file `obscuration_v402` writes, ``LSST_r_v4.02.yaml`` without its
  spider planes (the spider is in the mask; in the batoid model it would only
  punch holes in the reference-wavefront ray grid).

Neighbouring donuts are masked with the selection's own neighbour list: refcat
stars brighter than the target's magnitude + 2.5, through the raw WCS and the
per-visit affine correction, plus blind detections no star accounts for. Pixels
within 1.05 r_out of a neighbour get a variance of 1e20 electrons^2, so they
drop out of the fit.

Run::

    python aos/giant_donut_pupil/code/fit_donuts.py \\
        --mask-v402 <obscuration run>/RubinObsc_v4.02_r_rtpp0_pp0d0.yaml --workers 12
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import candidates as cd  # noqa: E402  (must precede the lsst imports: blitz path)
import obscuration_v402 as ob  # noqa: E402
from rubinwork.giant_donuts import fit_giant_donut  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
VISITS = {'extra': 2025102300337, 'intra': 2025102300340}
POINTING_DEG = (34.032, -28.371)
CANDIDATES = 'output/20261018_candidates'
ISR_CACHE = 'output/20261018_isr_bin4'
RAW_COLLECTION = 'LSSTCam/raw/all'

ZSETS = {'z4_11_22': [4, 11, 22], 'blitz': list(range(4, 20)) + list(range(22, 27))}
PUPILS = ('v1000', 'v4.02')
FIELD_RADIUS_LIMIT_DEG = 1.725
CAMERA_OFFSET_M = 4.0e-3
CORNER_DET_OFFSET_M = 1.5e-3
NEIGHBOUR_MAG_DIFF = 2.5
MASK_RADIUS_FACTOR = 1.05
MASKED_VAR = 1e20
FWHM_BOUNDS_ARCSEC = (0.1, 1.5)
BINNING = 1
TOL = 1e-3
MAX_NFEV = 200
NOLL_MAX = 26


def flags(rows):
    """Add ``beyond_fov``, ``corner_not_isolated`` and ``clean``."""
    rows = rows.copy()
    rows['beyond_fov'] = rows.field_r_deg > FIELD_RADIUS_LIMIT_DEG
    corner = rows.sensor_class != 'science'
    rows['corner_not_isolated'] = corner & ((rows.iso_class != 'isolated')
                                            | (rows.overlap_frac > 0.02)
                                            | (rows.annulus_ratio > 0.05))
    rows['clean'] = ~rows.beyond_fov & ~rows.corner_not_isolated
    return rows


def offsets_m(side, sensor_class):
    """Signed (detector, camera, M2) offsets, m, positive extra-focal."""
    sign = 1.0 if side == 'extra' else -1.0
    det = {'science': 0.0, 'SW0': CORNER_DET_OFFSET_M, 'SW1': -CORNER_DET_OFFSET_M}[sensor_class]
    return (det, sign * CAMERA_OFFSET_M, sign * CAMERA_OFFSET_M)


def rtp_rad(visit_info):
    """Rotator angle as blitz computes it, rad."""
    return ((visit_info.boresightParAngle.asRadians()
             - visit_info.boresightRotAngle.asRadians() - np.pi / 2 + np.pi)
            % (2 * np.pi) - np.pi)


def neighbours(study_dir, butler, camera, rows):
    """Neighbour centres per (visit, detector), full-resolution pixels.

    Returns
    -------
    out : `dict`
        ``(visit, detector) -> (xy, mag)``; blind detections carry mag NaN.
    """
    import lsst.daf.base as dafBase
    from lsst.afw.geom import makeSkyWcs
    from lsst.afw.cameraGeom import FOCAL_PLANE
    diag = json.loads((study_dir / CANDIDATES / 'diagnostics.json').read_text())
    _, ra, dec, mag = cd.load_refcat(butler, *POINTING_DEG, mag_max=14.5 + NEIGHBOUR_MAG_DIFF)
    build_template = cd.import_blitz()[0]
    out = {}
    for (side, visit, name), grp in rows.groupby(['side', 'visit', 'detector']):
        cache = study_dir / ISR_CACHE / str(visit)
        header = json.loads((cache / 'wcs.json').read_text())[name]
        metadata = dafBase.PropertyList()
        for key, value in header.items():
            metadata.set(key, value)
        det = camera[name]
        idx, x, y = cd.project_refcat(makeSkyWcs(metadata), det, ra, dec, margin_pix=1500)
        fx, fy = cd.pixels_to(det, x, y, FOCAL_PLANE)
        fx, fy = cd.apply_affine(np.array(diag[side]['affine']), fx, fy)
        sx, sy = cd.focal_to_pixels(det, fx, fy)
        stars = np.column_stack([sx, sy])
        r_out = float(grp.r_out_pix.iloc[0])
        image = np.load(cache / f'{name}.npy').astype(float)
        bkg = cd.background_map(image, block=128, percentile=10.0)
        blind = cd.blind_detect(image, bkg, r_out / cd.CACHE_BIN,
                                0.62, build_template)
        blind = (blind + 0.5) * cd.CACHE_BIN - 0.5
        if len(blind) and len(stars):
            dist = np.min(np.hypot(*(blind[:, None, :] - stars[None, :, :]).transpose(2, 0, 1)),
                          axis=1)
            blind = blind[dist > 0.25 * r_out]
        out[(visit, name)] = (np.vstack([stars, blind]),
                              np.concatenate([mag[idx], np.full(len(blind), np.nan)]))
    return out


def cut(image, row, neigh, n_quarter):
    """CCS stamp, its neighbour mask and the subtracted sky.

    Returns
    -------
    stamp, masked : `numpy.ndarray`
        Sky-subtracted stamp (electrons) and the neighbour mask, CCS.
    info : `dict`
        ``stamp_half_pix``, ``sky_e``, ``mask_applied``, ``masked_ring_frac``
        (fraction of the target's ring pixels masked, dimensionless), ``n_neighbours``.
    """
    from astropy.stats import sigma_clipped_stats
    r_out = row.r_out_pix
    cx, cy = int(round(row.x_pix)), int(round(row.y_pix))
    ny, nx = image.shape
    half = int(min(1.27 * r_out, cx, cy, nx - 1 - cx, ny - 1 - cy))
    stamp = image[cy - half:cy + half + 1, cx - half:cx + half + 1].copy()
    yy, xx = np.mgrid[cy - half:cy + half + 1, cx - half:cx + half + 1]
    xy, mag = neigh
    near = np.hypot(xy[:, 0] - row.x_pix, xy[:, 1] - row.y_pix)
    use = (near > 0.25 * r_out) & (near < 2 * MASK_RADIUS_FACTOR * r_out + 1.5 * half)
    use &= ~(mag > row.mag_r + NEIGHBOUR_MAG_DIFF)
    masked = np.zeros(stamp.shape, bool)
    for x0, y0 in xy[use]:
        masked |= np.hypot(xx - x0, yy - y0) < MASK_RADIUS_FACTOR * r_out
    r = np.hypot(xx - row.x_pix, yy - row.y_pix)
    ring = (r < r_out) & (r > 0.62 * r_out)
    sky_pix = stamp[(r > 1.1 * r_out) & ~masked]
    _, sky, _ = sigma_clipped_stats(sky_pix if sky_pix.size > 1000 else stamp[r > 1.1 * r_out],
                                    sigma=3.0, maxiters=5)
    stamp -= sky
    info = dict(stamp_half_pix=half, sky_e=float(sky), mask_applied=bool(masked.any()),
                masked_ring_frac=float(masked[ring].mean()), n_neighbours=int(use.sum()))
    to_ccs = lambda a: np.rot90(a, k=-n_quarter).T  # noqa: E731
    return to_ccs(stamp), to_ccs(masked), info


_PUPIL_CACHE = {}


def pupil(name, mask_v402):
    """(mask_params, R_outer m, R_inner m, batoid optic) for one pupil."""
    if name not in _PUPIL_CACHE:
        import batoid
        import yaml
        if name == 'v1000':
            params = ob.load_danish()
            optic = batoid.Optic.fromYaml('Rubin_v1000_r.yaml')
        else:
            params = yaml.safe_load(open(mask_v402))
            optic = ob.v402_optic(spider=False)
        r_outer, r_inner = ob._annulus(params)
        _PUPIL_CACHE[name] = (params, r_outer, r_inner, optic)
    return _PUPIL_CACHE[name]


def fit_one(stamp, masked, thx, thy, offsets, pupil_name, mask_v402, noll, rtp_deg,
            r_out_pix):
    """One danish fit, as `fit_giant_donut.fit_stamp` with a pixel mask and a set pupil.

    Returns
    -------
    result : `dict`
        Fit summary; ``img``, ``model_img`` and ``masked`` on the fit grid.
    """
    import batoid
    import danish
    from scipy.optimize import least_squares
    from scipy.stats import median_abs_deviation
    blitz_utils, wf_task = fit_giant_donut._import_blitz()
    instrument = blitz_utils._INSTRUMENT
    params, r_outer, r_inner, optic = pupil(pupil_name, mask_v402)

    img = wf_task._bin_stamp_odd(np.asarray(stamp, float), BINNING)
    msk = wf_task._bin_stamp_odd(masked.astype(float), BINNING) > 0
    diff = (img[1:] - img[:-1])[~(msk[1:] | msk[:-1])]
    bkg_std = median_abs_deviation(diff, scale='normal') / np.sqrt(2.0)
    var = np.where(msk, MASKED_VAR, bkg_std ** 2)

    wavelength = {b.value: w for b, w in instrument.wavelength.items()}['r']
    shifted = optic
    for name, dz in zip(('Detector', 'LSSTCamera', 'M2'), offsets):
        if dz:
            shifted = shifted.withGloballyShiftedOptic(name, [0.0, 0.0, dz])
    eps = r_inner / r_outer
    nrad = 10
    zk_ref = batoid.zernikeTA(shifted, thx, thy, wavelength, jmax=blitz_utils._ZK_JMAX,
                              eps=eps, focal_length=instrument.focalLength, nrad=nrad,
                              naz=int(2 * np.pi * nrad / (1 - eps))) * wavelength
    factory = danish.DonutFactory(R_outer=r_outer, R_inner=r_inner, mask_params=params,
                                  focal_length=instrument.focalLength,
                                  pixel_scale=instrument.pixelSize * BINNING,
                                  spider_angle=rtp_deg)
    dz_terms = [(1, j) for j in noll]
    model = danish.DZMultiDonutModel(
        factory, z_refs=[zk_ref], dz_terms=dz_terms,
        field_radius=wf_task._DANISH_FIELD_RADIUS_RAD, thxs=[thx], thys=[thy],
        npix=img.shape[0], bkg_order=0)
    flux0 = float(np.clip(np.sum(img[~msk]), 1e3, 1e9))
    x0 = model.pack_params(fluxes=[flux0], dxs=[0.0], dys=[0.0], fwhm=1.0,
                           bkgs=[[0.0] * model.nbkg], wavefront_params=[0.0] * len(noll))
    bounds = model.pack_params(
        fluxes=[[0.0, np.inf]], dxs=[[-np.inf, np.inf]], dys=[[-np.inf, np.inf]],
        fwhm=list(FWHM_BOUNDS_ARCSEC), bkgs=[[[-np.inf, np.inf]] * model.nbkg],
        wavefront_params=[[-np.inf, np.inf]] * len(noll))
    bounds = [list(b) for b in zip(*bounds)]
    x0 = np.clip(x0, bounds[0], bounds[1])
    t0 = time.time()
    fit = least_squares(model.chi, x0=x0, jac=model.jac, bounds=bounds, args=([img], [var]),
                        xtol=TOL, ftol=TOL, gtol=TOL, x_scale='jac', tr_solver='lsmr',
                        max_nfev=MAX_NFEV)
    p = model.unpack_params(fit.x)
    fwhm = float(np.atleast_1d(p['fwhm'])[0])
    model_img = np.asarray(model.model(p['fluxes'], p['dxs'], p['dys'], fwhm=p['fwhm'],
                                       wavefront_params=p['wavefront_params'],
                                       bkgs=p['bkgs'], sky_levels=[0.0])[0], float)
    n = img.shape[0]
    yy, xx = np.mgrid[:n, :n] - n // 2
    inside = (np.hypot(xx, yy) < 1.1 * r_out_pix / BINNING) & ~msk
    resid = (img - model_img)[inside]
    n_good = int((~msk).sum())
    dev = np.full(NOLL_MAX + 1, np.nan)
    for j, v in zip(noll, p['wavefront_params']):
        dev[j] = v * 1e6
    lo, hi = FWHM_BOUNDS_ARCSEC
    return dict(
        img=img, model_img=model_img, masked=msk,
        zk_dev_um=dev, zk_ref_um=np.asarray(zk_ref)[:NOLL_MAX + 1] * 1e6,
        fwhm_arcsec=fwhm,
        fwhm_at_bound=bool(min(abs(fwhm - lo), abs(fwhm - hi)) < 0.02 * (hi - lo)),
        chi2_per_dof=float(2 * fit.cost / max(n_good - len(x0), 1)),
        dof=int(n_good - len(x0)),
        resid_rms_e=float(np.sqrt(np.mean(resid ** 2))),
        flux_e=float(np.atleast_1d(p['fluxes'])[0]),
        dx_pix=float(np.atleast_1d(p['dxs'])[0]), dy_pix=float(np.atleast_1d(p['dys'])[0]),
        nfev=int(fit.nfev), success=bool(fit.success), fit_seconds=time.time() - t0,
        bkg_std_e=float(bkg_std))


_STAMPS = {}


def _work(task):
    key, pupil_name, zset, mask_v402, out_dir = task
    s = _STAMPS[key]
    try:
        res = fit_one(s['stamp'], s['masked'], s['thx'], s['thy'], s['offsets'],
                      pupil_name, mask_v402, ZSETS[zset], s['rtp_deg'], s['r_out_pix'])
    except Exception as err:  # a failed fit is a row, not a crashed run
        return dict(key=key, pupil=pupil_name, zset=zset, error=repr(err))
    np.savez_compressed(
        pathlib.Path(out_dir) / 'images' / f'{key[0]}_{key[1]}_{pupil_name}_{zset}.npz',
        img=res['img'].astype(np.float32), model=res['model_img'].astype(np.float32),
        masked=res['masked'])
    for k in ('img', 'model_img', 'masked'):
        res.pop(k)
    res.update(key=key, pupil=pupil_name, zset=zset, error='')
    print(json.dumps(dict(key=key, pupil=pupil_name, zset=zset,
                          chi2_per_dof=round(res['chi2_per_dof'], 3),
                          resid_rms_e=round(res['resid_rms_e'], 1),
                          fwhm=round(res['fwhm_arcsec'], 3), nfev=res['nfev'],
                          s=round(res['fit_seconds']))), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mask-v402', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=None, help='first N (donut, side) rows only')
    args = ap.parse_args()

    import multiprocessing as mp
    import lsst.daf.butler as dafButler
    from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
    import lsst.geom
    from lsst.obs.lsst import LsstCam
    from rubinwork.giant_donuts import donut_images as di
    blitz_utils, _ = fit_giant_donut._import_blitz()

    study_dir = pathlib.Path(__file__).resolve().parents[1]
    run = study_run(STUDY, 'fits')
    (run.path / 'images').mkdir(exist_ok=True)
    table = pd.read_parquet(study_dir / CANDIDATES / 'candidates.parquet')
    rows = flags(table[table.proposed]).sort_values(['cand_id', 'side']).reset_index(drop=True)
    if args.limit:
        rows = rows.iloc[:args.limit]
    camera = LsstCam.getCamera()
    butler = dafButler.Butler('/repo/main')
    neigh = neighbours(study_dir, butler, camera, rows)

    meta = []
    for (visit, name), grp in rows.groupby(['visit', 'detector']):
        image = di.run_isr(butler, visit, name, camera, RAW_COLLECTION)
        det = camera[name]
        vi = butler.get('raw.visitInfo', collections=RAW_COLLECTION,
                        dataId={'instrument': 'LSSTCam', 'exposure': visit,
                                'detector': det.getId()})
        rtp = rtp_rad(vi)
        n_quarter = det.getOrientation().getNQuarter()
        for _, row in grp.iterrows():
            stamp, masked, info = cut(image, row, neigh[(visit, name)], n_quarter)
            fa = det.transform([lsst.geom.Point2D(round(row.x_pix), round(row.y_pix))],
                               PIXELS, FIELD_ANGLE)[0]
            thx, thy = fa[1], fa[0]
            key = (row.cand_id, row.side)
            _STAMPS[key] = dict(stamp=stamp, masked=masked, thx=thx, thy=thy,
                                offsets=offsets_m(row.side, row.sensor_class),
                                rtp_deg=float(np.degrees(rtp)), r_out_pix=row.r_out_pix)
            c, s = np.cos(rtp), np.sin(rtp)
            meta.append(dict(
                cand_id=row.cand_id, side=row.side, visit=visit, detector=name,
                sensor_class=row.sensor_class, mag_r=row.mag_r,
                field_x_ccs_deg=float(np.degrees(thx)), field_y_ccs_deg=float(np.degrees(thy)),
                field_x_ocs_deg=float(np.degrees(c * thx - s * thy)),
                field_y_ocs_deg=float(np.degrees(s * thx + c * thy)),
                field_r_deg=row.field_r_deg, rtp_deg=float(np.degrees(rtp)),
                det_offset_mm=_STAMPS[key]['offsets'][0] * 1e3,
                cam_offset_mm=_STAMPS[key]['offsets'][1] * 1e3,
                m2_offset_mm=_STAMPS[key]['offsets'][2] * 1e3,
                iso_class=row.iso_class, overlap_frac=row.overlap_frac,
                annulus_ratio=row.annulus_ratio, beyond_fov=bool(row.beyond_fov),
                corner_not_isolated=bool(row.corner_not_isolated), clean=bool(row.clean),
                **info))
            np.savez_compressed(run.path / 'images' / f'{key[0]}_{key[1]}_stamp.npz',
                                stamp=stamp.astype(np.float32), masked=masked)
        print('stamps', visit, name, len(grp), flush=True)
    meta = pd.DataFrame(meta)
    meta.to_csv(run.path / 'stamps.csv', index=False)

    for name in PUPILS:  # build before forking, so workers inherit them
        pupil(name, args.mask_v402)
    tasks = [(key, p, z, args.mask_v402, str(run.path))
             for key in _STAMPS for p in PUPILS for z in ZSETS]
    print('fits', len(tasks), flush=True)
    with mp.get_context('fork').Pool(args.workers) as pool:
        results = pool.map(_work, tasks, chunksize=1)

    out = []
    for res in results:
        m = meta[(meta.cand_id == res['key'][0]) & (meta.side == res['key'][1])].iloc[0].to_dict()
        row = dict(m, pupil=res['pupil'], zset=res['zset'], error=res['error'])
        if not res['error']:
            rtp = np.radians(m['rtp_deg'])
            dev = res['zk_dev_um']
            dev_ocs = blitz_utils._rotate_zk(dev[None, :], -rtp)[0]
            for j in range(4, NOLL_MAX + 1):
                row[f'z{j}_ccs_um'] = dev[j]
                row[f'z{j}_ocs_um'] = dev_ocs[j]
            ref_ocs = blitz_utils._rotate_zk(res['zk_ref_um'][None, :], -rtp)[0]
            for j in (4, 5, 6, 7, 8, 11, 22):
                row[f'z{j}_ref_ocs_um'] = ref_ocs[j]
            row.update({k: res[k] for k in ('fwhm_arcsec', 'fwhm_at_bound', 'chi2_per_dof',
                                            'dof', 'resid_rms_e', 'flux_e', 'dx_pix',
                                            'dy_pix', 'nfev', 'success', 'fit_seconds',
                                            'bkg_std_e')})
        out.append(row)
    df = pd.DataFrame(out)
    df.to_parquet(run.path / 'fits.parquet')
    df.to_csv(run.path / 'fits.csv', index=False, float_format='%.5g')
    print('failures', int((df.error != '').sum()), 'of', len(df))
    print('done', run.path)


if __name__ == '__main__':
    main()
