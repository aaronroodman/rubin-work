"""Pupil pass 5, step 7: fit c with the blur free, then the spider shadow widths and depths.

On the central-band (0-0.8 deg) science sides of pass 4 with a good fit c, fit c is
refitted from its cached stamp with the Kolmogorov FWHM free, bounded to 0.5-2.5 arcsec,
starting from the fit c solution. Then the shadows are traced in the data and in both
models (blur 1.0 arcsec and the fitted blur) with `pass5_common.guided_shadows` about the
fit-a centre, and their depth and FWHM measured as pass 4.

Outputs in the run: ``blur_fits.parquet`` (one row per side) and ``blur_spiders.parquet``
(one row per side, shadow and model).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass5_blur.py --workers 60
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

from rubinwork.common import study_run  # noqa: E402

FWHM_BOUNDS = (0.5, 2.5)


def fit_free_blur(R, img, var, x0):
    """danish fit of flux, centre, FWHM and the fit c terms; x0 = (flux, dx, dy, z...)."""
    from scipy.optimize import least_squares
    m = R.model(img.shape[0])
    p0 = np.concatenate([x0[:3], [pc.p3.FWHM_ARCSEC], x0[3:]])
    lo = [0.0, -np.inf, -np.inf, FWHM_BOUNDS[0]] + [-np.inf] * (len(p0) - 4)
    hi = [np.inf, np.inf, np.inf, FWHM_BOUNDS[1]] + [np.inf] * (len(p0) - 4)
    t0 = time.time()
    r = least_squares(lambda p: m.chi(p, img, var), x0=p0, jac=lambda p: m.jac(p, img, var),
                      bounds=(lo, hi), xtol=pc.fd.TOL, ftol=pc.fd.TOL, gtol=pc.fd.TOL,
                      x_scale='jac', tr_solver='lsmr', max_nfev=pc.fd.MAX_NFEV)
    chi = m.chi(r.x, img, var)
    return r, chi, time.time() - t0, m.sky_scale


def one_side(row):
    t0 = time.time()
    d = pc.load_side(row)
    R = pc.Renderer(row, d['zk_ta'])
    img, msk = d['img'], d['masked']
    n = img.shape[0]
    var = np.where(msk, pc.fd.MASKED_VAR, d['noise'] ** 2)
    n_good = int((~msk).sum())
    m0 = R.model(n)
    z = np.array([row[f'z{j}_ccs_um'] for j in pc.p3.FITS['c']]) * 1e-6
    x0 = np.concatenate([[row.flux_e, row.dx_pix * m0.sky_scale, row.dy_pix * m0.sky_scale], z])
    chi0 = m0.chi(np.concatenate([x0[:3], [pc.p3.FWHM_ARCSEC], x0[3:]]), img, var)
    r, chi, secs, sky = fit_free_blur(R, img, var, x0)
    nz = len(pc.p3.FITS['c'])
    out = dict(key=row.key, sample_id=row.sample_id, label=row.label, side=row.side,
               mag=row.mag, field_r_deg=row.field_r_deg, fwhm_fit_arcsec=float(r.x[3]),
               at_bound=bool(np.isclose(r.x[3], FWHM_BOUNDS).any()), success=bool(r.success),
               nfev=int(r.nfev), fit_seconds=secs,
               chi2_dof_fixed=float(np.sum(chi0 ** 2) / (n_good - 3 - nz)),
               chi2_dof_free=float(np.sum(chi ** 2) / (n_good - 4 - nz)),
               dof_free=n_good - 4 - nz, flux_e=float(r.x[0]), dx_pix=float(r.x[1] / sky),
               dy_pix=float(r.x[2] / sky))
    for j, v in zip(pc.p3.FITS['c'], r.x[4:]):
        out[f'z{j}_ccs_um_free'] = v * 1e6
        out[f'z{j}_ccs_um_fixed'] = row[f'z{j}_ccs_um']
    x_free = (r.x[0], r.x[1], r.x[2], r.x[4:])
    models = {'blur_1.0': R.render(n),
              'blur_free': R.render(n, fwhm=float(r.x[3]), x=x_free)}
    without = R.render(n, spider=False)
    data = np.where(msk, np.nan, img)
    ca = (d['cx'], d['cy'])
    e = pc.edges(np.nan_to_num(data), ~msk, ca[0], ca[1], row, d['noise'], d['flags'])
    r_edge = float(np.nanmedian(e['r_outer']))
    ef = (np.where(np.isfinite(e['fit_inner']), e['fit_inner'], np.nanmedian(e['r_inner'])),
          np.where(np.isfinite(e['fit_outer']), e['fit_outer'], r_edge))
    rows = []
    for name, mod in models.items():
        sd, sm, reasons, _ = pc.guided_shadows(data, np.where(msk, np.nan, mod),
                                               np.where(msk, 0.0, without), ca, r_edge, ef,
                                               row.rtp_deg)
        for k in range(pc.N_SHADOW):
            rows.append(dict(key=row.key, label=row.label, side=row.side, mag=row.mag,
                             model=name, model_idx=k, missing=reasons[k],
                             depth_data=sd[k]['depth'], fwhm_pix_data=sd[k]['fwhm_pix'],
                             depth_model=sm[k]['depth'], fwhm_pix_model=sm[k]['fwhm_pix'],
                             az_half_ocs_deg_data=sd[k]['az_half_ocs_deg'],
                             az_half_ocs_deg_model=sm[k]['az_half_ocs_deg']))
    print(row.key, row.label, row.side, f'fwhm {out["fwhm_fit_arcsec"]:.3f} arcsec',
          f'chi2/dof {out["chi2_dof_fixed"]:.3f} -> {out["chi2_dof_free"]:.3f}',
          f'{time.time() - t0:.0f} s', flush=True)
    return out, rows


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=60)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    run = study_run(pc.STUDY, pc.RUN_LABEL, date=pc.RUN_DATE).path
    S = pc.sides()
    S = S[(S.band == 'central') & (S.sensor_class == 'science') & S.fit_ok & (S.error == '')]
    if args.limit:
        S = S.head(args.limit)
    print('sides', len(S), flush=True)
    fits, rows = [], []
    with mp.get_context('fork').Pool(min(args.workers, len(S))) as pool:
        for f, r in pool.imap_unordered(one_side, [r for _, r in S.iterrows()]):
            fits.append(f)
            rows += r
    pd.DataFrame(fits).to_parquet(run / 'blur_fits.parquet')
    t = pd.DataFrame(rows)
    t.to_parquet(run / 'blur_spiders.parquet')
    print('done', run, flush=True)


if __name__ == '__main__':
    main()
