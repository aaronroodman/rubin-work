"""Z4-only danish fits of the 38 giant-donut sides, v1000 and v4.02 (step 1 of the pupil pass).

Each fit floats Z4, flux and centre only. Noll 5-28 are fixed at the batoid
intrinsic of that pupil's own batoid model, at the donut's field position with
that side's (detector, camera, M2) offsets; the blur is a Kolmogorov profile of
fixed 1.0 arcsec FWHM; no background term (the stamps are sky-subtracted). The
wavefront then cannot reshape the outline, so the residual measures the pupil
model.

Inputs are the stamps and neighbour masks `fit_donuts.py` saved in its run
(``images/<cand>_<side>_stamp.npz``, full-resolution ISR, CCS) and its
``stamps.csv``, so the donuts, flags and masks are those of that run.

Run, from ``code/``::

    python z4_fits.py --fits-run ../output/20261018_fits \\
        --mask-v402 ../output/20261018_obscuration_v402/RubinObsc_v4.02_r_rtpp0_pp0d0.yaml
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fit_donuts as fd  # noqa: E402  (puts the blitz checkout first on sys.path)
import fit_giant_donut  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
FWHM_ARCSEC = 1.0
NOLL_FIXED_MAX = 28
PUPILS = fd.PUPILS

_STAMPS = {}


def zk_intrinsic(optic, thx, thy, offsets, r_outer, r_inner, extra_camera_dz=0.0):
    """Batoid intrinsic Zernikes, Noll 0..28, m of wavefront.

    Same call as `fit_donuts.fit_one`: `batoid.zernikeTA` on the optic with the
    side's offsets, annulus ``r_inner/r_outer``, at the r-band wavelength.

    Parameters
    ----------
    thx, thy : `float`
        Field angle, CCS, rad.
    offsets : `tuple` [`float`]
        (detector, camera, M2) z offsets, m.
    extra_camera_dz : `float`, optional
        Additional camera piston, m.
    """
    import batoid
    blitz_utils, _ = fit_giant_donut._import_blitz()
    instrument = blitz_utils._INSTRUMENT
    wavelength = {b.value: w for b, w in instrument.wavelength.items()}['r']
    det, cam, m2 = offsets
    shifted = optic
    for name, dz in (('Detector', det), ('LSSTCamera', cam + extra_camera_dz), ('M2', m2)):
        if dz:
            shifted = shifted.withGloballyShiftedOptic(name, [0.0, 0.0, dz])
    eps = r_inner / r_outer
    nrad = 10
    zk = batoid.zernikeTA(shifted, thx, thy, wavelength, jmax=NOLL_FIXED_MAX, eps=eps,
                          focal_length=instrument.focalLength, nrad=nrad,
                          naz=int(2 * np.pi * nrad / (1 - eps))) * wavelength
    return np.asarray(zk)


def fit_z4(stamp, masked, thx, thy, offsets, pupil_name, mask_v402, rtp_deg, r_out_pix,
           noll=(4,), mask_params=None, optic=None):
    """Flux + centre + `noll` danish fit with the other Zernikes and the blur fixed.

    Parameters
    ----------
    noll : `tuple` [`int`], optional
        Noll terms floated about the intrinsic; Z4 only by default.
    mask_params : `dict`, optional
        danish mask replacing the pupil's own (a v4.02g file); the batoid
        optic for the intrinsic stays the pupil's.
    optic : `batoid.Optic`, optional
        Optic for the intrinsic, replacing the pupil's (a perturbed v4.02).

    Returns
    -------
    result : `dict`
        Fit summary, plus ``img``, ``model_img`` and ``masked`` arrays.
        ``z{j}_dev_um`` is each floated term's deviation from the intrinsic,
        µm of wavefront.
    """
    import danish
    from scipy.optimize import least_squares
    from scipy.stats import median_abs_deviation
    blitz_utils, wf_task = fit_giant_donut._import_blitz()
    instrument = blitz_utils._INSTRUMENT
    params, r_outer, r_inner, pupil_optic = fd.pupil(pupil_name, mask_v402)
    optic = pupil_optic if optic is None else optic
    if mask_params is not None:
        params = mask_params
        r_outer, r_inner = fd.ob._annulus(params)

    img = np.asarray(stamp, float)
    msk = masked.astype(bool)
    diff = (img[1:] - img[:-1])[~(msk[1:] | msk[:-1])]
    bkg_std = median_abs_deviation(diff, scale='normal') / np.sqrt(2.0)
    var = np.where(msk, fd.MASKED_VAR, bkg_std ** 2)

    zk_ref = zk_intrinsic(optic, thx, thy, offsets, r_outer, r_inner)
    factory = danish.DonutFactory(R_outer=r_outer, R_inner=r_inner, mask_params=params,
                                  focal_length=instrument.focalLength,
                                  pixel_scale=instrument.pixelSize, spider_angle=rtp_deg)
    model = danish.DZMultiDonutModel(
        factory, z_refs=[zk_ref], dz_terms=[(1, j) for j in noll],
        field_radius=wf_task._DANISH_FIELD_RADIUS_RAD, thxs=[thx], thys=[thy],
        npix=img.shape[0], bkg_order=-1)
    # Full danish vector is (flux, dx, dy, fwhm, z...); fwhm is held out of the solve.
    i_fwhm = 3

    def full(x):
        return np.insert(x, i_fwhm, FWHM_ARCSEC)

    def chi(x):
        return model.chi(full(x), [img], [var])

    def jac(x):
        return np.delete(model.jac(full(x), [img], [var]), i_fwhm, axis=1)

    flux0 = float(np.clip(np.sum(img[~msk]), 1e3, 1e9))
    nz = len(noll)
    x0 = np.array([flux0, 0.0, 0.0] + [0.0] * nz)
    bounds = ([0.0] + [-np.inf] * (2 + nz), [np.inf] * (3 + nz))
    t0 = time.time()
    fit = least_squares(chi, x0=x0, jac=jac, bounds=bounds, xtol=fd.TOL, ftol=fd.TOL,
                        gtol=fd.TOL, x_scale='jac', tr_solver='lsmr', max_nfev=fd.MAX_NFEV)
    p = model.unpack_params(full(fit.x))
    model_img = np.asarray(model.model(p['fluxes'], p['dxs'], p['dys'], fwhm=FWHM_ARCSEC,
                                       wavefront_params=p['wavefront_params'],
                                       sky_levels=[0.0])[0], float)
    n = img.shape[0]
    yy, xx = np.mgrid[:n, :n] - n // 2
    inside = (np.hypot(xx, yy) < 1.1 * r_out_pix) & ~msk
    resid = (img - model_img)[inside]
    n_good = int((~msk).sum())
    devs = {f'z{j}_dev_um': float(v) * 1e6 for j, v in zip(noll, p['wavefront_params']) if j != 4}
    dz4 = float(p['wavefront_params'][list(noll).index(4)])
    return dict(**devs, **{f'z{j}_ref_um': float(zk_ref[j]) * 1e6 for j in noll if j != 4},
        img=img, model_img=model_img, masked=msk,
        z4_um=(zk_ref[4] + dz4) * 1e6, z4_ref_um=zk_ref[4] * 1e6, z4_dev_um=dz4 * 1e6,
        chi2_per_dof=float(2 * fit.cost / max(n_good - len(x0), 1)),
        dof=int(n_good - len(x0)),
        resid_rms_e=float(np.sqrt(np.mean(resid ** 2))),
        flux_e=float(p['fluxes'][0]),
        dx_arcsec=float(p['dxs'][0]), dy_arcsec=float(p['dys'][0]),
        dx_pix=float(p['dxs'][0]) / model.sky_scale, dy_pix=float(p['dys'][0]) / model.sky_scale,
        nfev=int(fit.nfev), success=bool(fit.success), fit_seconds=time.time() - t0,
        bkg_std_e=float(bkg_std))


def _work(task):
    key, pupil_name, mask_v402, out_dir = task
    s = _STAMPS[key]
    try:
        res = fit_z4(s['stamp'], s['masked'], s['thx'], s['thy'], s['offsets'], pupil_name,
                     mask_v402, s['rtp_deg'], s['r_out_pix'])
    except Exception as err:  # a failed fit is a row, not a crashed run
        return dict(key=key, pupil=pupil_name, error=repr(err))
    np.savez_compressed(pathlib.Path(out_dir) / 'images' / f'{key[0]}_{key[1]}_{pupil_name}_z4.npz',
                        model=res.pop('model_img').astype(np.float32))
    res.pop('img')
    res.pop('masked')
    res.update(key=key, pupil=pupil_name, error='')
    print(json.dumps(dict(key=key, pupil=pupil_name, z4_um=round(res['z4_um'], 2),
                          resid_rms_e=round(res['resid_rms_e'], 1),
                          s=round(res['fit_seconds']))), flush=True)
    return res


def load_stamps(fits_run):
    """Stamps, masks and per-side metadata of a `fit_donuts.py` run."""
    meta = pd.read_csv(fits_run / 'stamps.csv')
    cand = pd.read_parquet(fits_run.parents[0] / fd.CANDIDATES.split('/')[-1] / 'candidates.parquet')
    meta = meta.merge(cand[['cand_id', 'side', 'r_out_pix', 'x_pix', 'y_pix']],
                      on=['cand_id', 'side'], how='left')
    stamps = {}
    for row in meta.itertuples():
        f = np.load(fits_run / 'images' / f'{row.cand_id}_{row.side}_stamp.npz')
        stamps[(row.cand_id, row.side)] = dict(
            stamp=f['stamp'].astype(float), masked=f['masked'].astype(bool),
            thx=np.radians(row.field_x_ccs_deg), thy=np.radians(row.field_y_ccs_deg),
            offsets=(row.det_offset_mm * 1e-3, row.cam_offset_mm * 1e-3, row.m2_offset_mm * 1e-3),
            rtp_deg=row.rtp_deg, r_out_pix=row.r_out_pix)
    return meta, stamps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fits-run', required=True)
    ap.add_argument('--mask-v402', required=True)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    import multiprocessing as mp

    run = study_run(STUDY, 'pupil')
    (run.path / 'images').mkdir(exist_ok=True)
    fits_run = pathlib.Path(args.fits_run).resolve()
    meta, stamps = load_stamps(fits_run)
    _STAMPS.update(stamps)
    for name in PUPILS:  # build before forking, so workers inherit them
        fd.pupil(name, args.mask_v402)
    tasks = [(key, p, args.mask_v402, str(run.path)) for key in _STAMPS for p in PUPILS]
    print('fits', len(tasks), flush=True)
    with mp.get_context('fork').Pool(args.workers) as pool:
        results = pool.map(_work, tasks, chunksize=1)

    out = []
    for res in results:
        m = meta[(meta.cand_id == res['key'][0]) & (meta.side == res['key'][1])].iloc[0].to_dict()
        row = dict(m, pupil=res['pupil'], error=res['error'])
        row.update({k: v for k, v in res.items() if k not in ('key', 'pupil', 'error')})
        out.append(row)
    df = pd.DataFrame(out)
    df.to_parquet(run.path / 'z4_fits.parquet')
    df.to_csv(run.path / 'z4_fits.csv', index=False, float_format='%.5g')
    print('failures', int((df.error != '').sum()), 'of', len(df))
    print('done', run.path)


if __name__ == '__main__':
    main()
