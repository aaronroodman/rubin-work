"""Pupil pass 6, step 5: the danish Spider_3D vane width fitted per side.

Per side, every Spider_3D vane gets one width w (pupil, m). The reference is
intrinsicTA plus the side's fit c deviations (so only the vanes change), the blur is
fixed, and flux, center and Z4 are refitted at each w on ``WIDTHS_MM``. The best w is the
minimum of a parabola through the three grid points about the chi2 minimum; its error is
from the curvature (Δchi2 = 1, with chi2 scaled to chi2/dof = 1). At the best w the
shadows are traced (`pass5_common.guided_shadows`) and their depth and FWHM measured.

Data: the 53 central-band (0–0.8 deg) science sides of pass 5 step 7, ISR stamps, rtp =
physical_rotator_angle, v4.02g, blur at the pass 5 free-blur fit per side. Control: the
step 3 ``nominal`` simulations at their simulated blur.

Outputs in the run: ``vane_width_data.parquet``, ``vane_width_sims.parquet`` (one row per
side), ``vane_width_chi2.parquet`` (chi2 per side and w) and ``vane_width_spiders.parquet``
(shadows at the best w).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass6_vanes.py --sets data,sims --workers 80
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
import pass6_common as p6  # noqa: E402
import pass6_pipeline as pp  # noqa: E402
import pass6_sims as psim  # noqa: E402
import pass5_common as pc  # noqa: E402
import fit_donuts as fd  # noqa: E402
import pupil_pass3 as p3  # noqa: E402

WIDTHS_MM = (30, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120)


def one(task):
    row, img, msk, fwhm, fitc, set_name = task
    t0 = time.time()
    msk = msk.astype(bool)
    noise = pp.noise_of(img, msk)
    var = np.where(msk, fd.MASKED_VAR, noise ** 2)
    n = img.shape[0]
    base = pp.Model(row, fwhm)
    zk = base.zk_ta.copy()
    for j in p3.FITS['c']:
        if j != 4:
            zk[j] += fitc[f'z{j}_ccs_um'] * 1e-6
    sky_scale = base.single(n, (4,)).sky_scale
    x0 = np.array([fitc['flux_e'], fitc['dx_pix'] * sky_scale, fitc['dy_pix'] * sky_scale])
    chi2, fits = [], {}
    for w in WIDTHS_MM:
        M = pp.Model(row, fwhm, vane_width_m=w * 1e-3, zk_ta=zk)
        r = M.fit(img, var, (4,), x0=x0)
        chi2.append(r['chi2'])
        fits[w] = (M, r)
    chi2 = np.array(chi2)
    dof = int((~msk).sum()) - 4
    i = int(np.argmin(chi2))
    best, err = float(WIDTHS_MM[i]), np.nan
    if 0 < i < len(WIDTHS_MM) - 1:
        x = np.array(WIDTHS_MM[i - 1:i + 2], float)
        c = np.polyfit(x, chi2[i - 1:i + 2] / (chi2[i] / dof), 2)
        if c[0] > 0:
            best = float(-c[1] / (2 * c[0]))
            err = float(1 / np.sqrt(c[0]))
    rows_chi = [dict(key=row.key, set=set_name, width_mm=w, chi2=float(c), chi2_dof=float(c / dof))
                for w, c in zip(WIDTHS_MM, chi2)]
    # shadows at the best width and at the CAD width
    sh = []
    for label, w in (('best', best), ('cad', 50.0)):
        M = pp.Model(row, fwhm, vane_width_m=w * 1e-3, zk_ta=zk)
        r = M.fit(img, var, (4,), x0=x0)
        model = r['model']
        without = M.render(n, r['x'], (4,), spider=False)
        cx, cy = n // 2 + r['x'][1] / r['sky_scale'], n // 2 + r['x'][2] / r['sky_scale']
        spider, dark = p3.spider_and_vignetting(M.params, M.rtp, M.zk_ta, M.thx, M.thy, n, cx, cy,
                                                row.r_out_pix)
        _, _, m_in, m_out, _ = pp.pe.edges(img, ~msk, cx, cy, row.r_out_pix)
        e = p3.edges_of(img, ~msk, cx, cy, row.r_out_pix, noise, (spider, dark, m_in, m_out))
        r_edge = float(np.nanmedian(e['r_outer']))
        ef = (np.where(np.isfinite(e['fit_inner']), e['fit_inner'], np.nanmedian(e['r_inner'])),
              np.where(np.isfinite(e['fit_outer']), e['fit_outer'], r_edge))
        sd, sm, reasons, _ = pc.guided_shadows(np.where(msk, np.nan, img), np.where(msk, np.nan, model),
                                               np.where(msk, 0.0, without), (cx, cy), r_edge, ef,
                                               M.rtp, True)
        for k in range(pc.N_SHADOW):
            if reasons[k]:
                continue
            sh.append(dict(key=row.key, set=set_name, side=row.side, which=label, width_mm=w,
                           model_idx=k, depth_data=sd[k]['depth'], depth_model=sm[k]['depth'],
                           fwhm_data_pix=sd[k]['fwhm_pix'], fwhm_model_pix=sm[k]['fwhm_pix']))
    s = pd.DataFrame(sh)
    out = dict(key=row.key, set=set_name, label=row.label, side=row.side, field_r_deg=row.field_r_deg,
               sensor_class=row.sensor_class, fwhm_arcsec=fwhm, width_best_mm=best, width_err_mm=err,
               at_grid_edge=bool(i in (0, len(WIDTHS_MM) - 1)), chi2_dof_cad=float(chi2[3] / dof),
               chi2_dof_best=float(chi2[i] / dof), dof=dof, seconds=time.time() - t0)
    for label in ('best', 'cad'):
        g = s[s.which == label] if len(s) else s
        for q in ('depth_data', 'depth_model', 'fwhm_data_pix', 'fwhm_model_pix'):
            out[f'{q}_{label}'] = float(g[q].median()) if len(g) else np.nan
    print(set_name, row.key, row.side, f'w {best:.1f} mm', f'{out["seconds"]:.0f} s', flush=True)
    return out, rows_chi, sh


def tasks_data():
    S = p6.with_rtp(p6.sides(), 'phys').set_index('key')
    b = pd.read_parquet(p6.PASS5_RUN / 'blur_fits.parquet')
    F = pd.read_parquet(p6.RUN / 'isr_phys' / 'fits.parquet').set_index('key')
    out = []
    for r in b.itertuples():
        row = S.loc[r.key].copy()
        row['key'] = r.key
        d = np.load(p6.RUN / 'isr_stamps' / f'{row.sample_id}_{row.side}.npz')
        out.append((row, d['img'].astype(float), d['masked'], float(r.fwhm_fit_arcsec),
                    F.loc[r.key].to_dict(), 'data'))
    return out


def tasks_sims():
    C = pd.read_parquet(psim.SIM_DIR / 'sim_sides.parquet')
    F = pd.read_parquet(psim.SIM_DIR / 'nominal_v402g' / 'fits.parquet').set_index('key')
    out = []
    for _, row in C.iterrows():
        d = np.load(psim.SIM_DIR / 'stamps' / f'{row.sample_id}_{row.side}_nominal.npz')
        msk = d['masked'].astype(bool)
        img, _ = psim.subtract_sky(d['img'].astype(float), msk, row.r_out_pix)
        out.append((row, img, msk, p6.SIM_FWHM_ARCSEC[row.side], F.loc[row.key].to_dict(), 'sims'))
    return out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sets', default='data,sims')
    ap.add_argument('--workers', type=int, default=80)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    run = p6.run_dir()
    for set_name in args.sets.split(','):
        tasks = tasks_data() if set_name == 'data' else tasks_sims()
        if args.limit:
            tasks = tasks[:args.limit]
        t0 = time.time()
        with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
            res = pool.map(one, tasks, chunksize=1)
        side = pd.DataFrame([r[0] for r in res])
        chi = pd.DataFrame([c for r in res for c in r[1]])
        sh = pd.DataFrame([s for r in res for s in r[2]])
        for name, df in ((f'vane_width_{set_name}', side), (f'vane_width_chi2_{set_name}', chi),
                         (f'vane_width_spiders_{set_name}', sh)):
            df.to_parquet(run / f'{name}.parquet')
            df.to_csv(run / f'{name}.csv', index=False, float_format='%.5g')
        print(side.groupby('side')[['width_best_mm', 'width_err_mm', 'depth_data_best',
                                     'depth_model_best', 'depth_model_cad']].median().round(3))
        print(f'{set_name} wall {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
