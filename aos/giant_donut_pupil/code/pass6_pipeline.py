"""Pupil pass 6: one side through the pass 5 measurement chain, on any stamp.

The same chain runs on the ISR data stamps (step 1), at either rotator angle (step 2),
on the simulated stamps (step 3) and with the v4.02s masks (step 4):

1. noise from adjacent-row differences, as passes 3–5;
2. fit a (Z4) then fit c (Z4–Z11, Z14, Z15, Z20–Z22) with danish, intrinsicTA reference,
   blur fixed at ``fwhm`` (arcsec), spider at ``spider_angle = rtp``, the mask variant's
   file for the side's offsets;
3. spider and vignetted bins from the reference model (`pupil_pass3.spider_and_vignetting`),
   robust edges of the data and the fit c model about the fit-a center;
4. shadows with the guided tracer (`pass5_common.guided_shadows`), fit-a center;
5. the δ grid: fit c re-rendered at ``rtp + δ``, δ −1.0 to +1.0 deg in 0.05 deg steps,
   shadow azimuths at r_half;
6. edge-circle centers (inner and outer, data and model), two iterations about the fit-a
   center, as pass 5 step 6c.

`measure` returns plain rows; the drivers assemble the tables, and
`pass5_analysis` (rotation, translation, m = 1, centre test) runs on them unchanged.
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass6_common as p6  # noqa: E402
import pass5_common as pc  # noqa: E402
import pass5_spiders as ps5  # noqa: E402
import fit_donuts as fd  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import pupil_edges as pe  # noqa: E402
import pupil_pass3 as p3  # noqa: E402

DELTAS = ps5.DELTAS
AZ = ps5.AZ


def noise_of(img, msk):
    from scipy.stats import median_abs_deviation
    diff = (img[1:] - img[:-1])[~(msk[1:] | msk[:-1])]
    return float(median_abs_deviation(diff, scale='normal') / np.sqrt(2.0))


class Model:
    """danish fit c model of one side at a set blur, mask variant, rtp and vane width."""

    def __init__(self, row, fwhm, mask='v4.02g', vane_width_m=None, zk_ta=None):
        import danish
        self.row = row
        self.offsets = fd.offsets_m(row.side, row.sensor_class)
        self.params = p6.mask_params(mask, self.offsets, vane_width_m)
        r_outer, r_inner = ob._annulus(self.params)
        self.thx, self.thy = np.radians(row.field_x_ccs_deg), np.radians(row.field_y_ccs_deg)
        if zk_ta is None:
            zk_ta, _ = p3.reference_zk(row.field_x_ccs_deg, row.field_y_ccs_deg, self.offsets,
                                       row.side, r_inner / r_outer)
        self.zk_ta = zk_ta
        self.rtp = float(row.rtp_deg)
        self.fwhm = float(fwhm)
        self.fac = {s: p3._factory(self.params, self.rtp, spider=s) for s in (True, False)}
        self.danish = danish

    def single(self, npix, noll, spider=True):
        return self.danish.SingleDonutModel(self.fac[spider], z_ref=self.zk_ta, z_terms=tuple(noll),
                                            thx=self.thx, thy=self.thy, npix=npix, bkg_order=-1)

    def fit(self, img, var, noll, x0=None):
        """As `pupil_pass3.fit`, at this model's blur."""
        from scipy.optimize import least_squares
        m = self.single(img.shape[0], noll)
        f = self.fwhm

        def full(x):
            return np.insert(x, 3, f)
        if x0 is None:
            good = var < fd.MASKED_VAR
            x0 = np.array([float(np.clip(np.sum(img[good]), 1e3, 1e10)), 0.0, 0.0])
        x0 = np.concatenate([x0[:3], np.zeros(len(noll))])
        bounds = ([0.0] + [-np.inf] * (len(x0) - 1), [np.inf] * len(x0))
        r = least_squares(lambda x: m.chi(full(x), img, var), x0=x0,
                          jac=lambda x: np.delete(m.jac(full(x), img, var), 3, axis=1),
                          bounds=bounds, xtol=fd.TOL, ftol=fd.TOL, gtol=fd.TOL, x_scale='jac',
                          tr_solver='lsmr', max_nfev=fd.MAX_NFEV)
        p = m.unpack_params(full(r.x))
        model = np.asarray(m.model(p['flux'], p['dx'], p['dy'], f, p['z_fit']), float)
        chi = m.chi(full(r.x), img, var)
        return dict(x=r.x, model=model, chi2=float(np.sum(chi ** 2)), nfev=int(r.nfev),
                    success=bool(r.success), sky_scale=m.sky_scale)

    def render(self, npix, x, noll, delta_deg=0.0, spider=True):
        m = self.single(npix, noll, spider)
        self.fac[True].spider_angle = self.rtp + delta_deg
        img = np.asarray(m.model(x[0], x[1], x[2], self.fwhm, np.asarray(x[3:])), float)
        self.fac[True].spider_angle = self.rtp
        return img


def measure(row, img, msk, fwhm, mask='v4.02g', vane_width_m=None, zk_ta=None, delta=True,
            centers=True, profiles=False, keep_images=False):
    """Fits a and c, edges, shadows, δ grid and edge centers of one side.

    Returns
    -------
    out : `dict`
        ``fit`` (one row), ``bins`` (edges per azimuth bin), ``shadows`` (rows as
        pass 5 ``spiders.parquet``, fit-a center), ``delta`` (rows as
        ``spider_delta.parquet``), ``centers`` (one row as ``centres.parquet``), and
        ``images`` if `keep_images`.
    """
    t0 = time.time()
    img = np.asarray(img, float)
    msk = np.asarray(msk, bool)
    n = img.shape[0]
    noise = noise_of(img, msk)
    var = np.where(msk, fd.MASKED_VAR, noise ** 2)
    n_good = int((~msk).sum())
    M = Model(row, fwhm, mask, vane_width_m, zk_ta)
    ra = M.fit(img, var, p3.FITS['a'])
    sky_scale = ra['sky_scale']
    cx, cy = n // 2 + ra['x'][1] / sky_scale, n // 2 + ra['x'][2] / sky_scale
    spider, dark = p3.spider_and_vignetting(M.params, M.rtp, M.zk_ta, M.thx, M.thy, n, cx, cy,
                                            row.r_out_pix)
    _, _, m_in, m_out, _ = pe.edges(img, ~msk, cx, cy, row.r_out_pix)
    flags = (spider, dark, m_in, m_out)
    rc = M.fit(img, var, p3.FITS['c'], x0=ra['x'])
    model = rc['model']
    without = M.render(n, rc['x'], p3.FITS['c'], spider=False)
    dof = n_good - 3 - len(p3.FITS['c'])
    fit = dict(key=row.key, side=row.side, label=row.label, mask=mask, fwhm_arcsec=fwhm,
               rtp_deg=M.rtp, vane_width_m=vane_width_m if vane_width_m else p6.CAD_VANE_WIDTH_M,
               noise_e=noise, flux_e=float(rc['x'][0]), dx_pix=float(rc['x'][1] / sky_scale),
               dy_pix=float(rc['x'][2] / sky_scale), chi2_dof=rc['chi2'] / max(dof, 1), dof=dof,
               chi2_dof_a=ra['chi2'] / max(n_good - 4, 1), nfev=rc['nfev'], success=rc['success'],
               stamp_sum_e=float(img[~msk].sum()), cx_a=cx, cy_a=cy)
    for j, v in zip(p3.FITS['c'], rc['x'][3:]):
        fit[f'z{j}_ccs_um'] = v * 1e6

    e_d = p3.edges_of(img, ~msk, cx, cy, row.r_out_pix, noise, flags)
    e_m = p3.edges_of(model, ~msk, cx, cy, row.r_out_pix, noise, flags)
    bins = pd.DataFrame(dict(key=row.key, side=row.side, bin=np.arange(pe.NBIN), az_ccs_deg=AZ,
                             az_ocs_deg=np.mod(AZ + M.rtp, 360.0), spider=spider, vignetted=dark,
                             r_inner_data_pix=e_d['r_inner'], r_outer_data_pix=e_d['r_outer'],
                             r_inner_c_pix=e_m['r_inner'], r_outer_c_pix=e_m['r_outer']))
    for edge in ('inner', 'outer'):
        d = (bins[f'r_{edge}_data_pix'] - bins[f'r_{edge}_c_pix']).dropna()
        fit[f'edge_median_{edge}_pix'] = float(d.median()) if len(d) else np.nan
        fit[f'edge_rms_{edge}_pix'] = float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan
    q = (bins.r_inner_data_pix / bins.r_outer_data_pix - bins.r_inner_c_pix / bins.r_outer_c_pix)
    fit['ratio_excess'] = float(q.median())

    data = np.where(msk, np.nan, img)
    mod = np.where(msk, np.nan, model)
    wo = np.where(msk, 0.0, without)
    ca = (cx, cy)
    r_edge = float(np.nanmedian(e_d['r_outer']))
    fin = np.where(np.isfinite(e_d['fit_inner']), e_d['fit_inner'], np.nanmedian(e_d['r_inner']))
    fout = np.where(np.isfinite(e_d['fit_outer']), e_d['fit_outer'], r_edge)
    ef = (fin, fout)
    sign = 180.0 if row.side == 'extra' else 0.0
    sd, sm, reasons, aux = pc.guided_shadows(data, mod, wo, ca, r_edge, ef, M.rtp, profiles)
    shadows = ps5._rows(row, 'fit_a', sd, sm, reasons, sign)
    for r in shadows:
        r['rtp_deg'] = M.rtp
    out = dict(fit=fit, bins=bins, shadows=shadows)
    if profiles:
        out['profiles'] = {(k, nm): s['profile'] for k in range(pc.N_SHADOW)
                           for nm, s in (('data', sd[k]), ('model', sm[k])) if s['profile'] is not None}
    if delta:
        drow = []
        for dl in DELTAS:
            if dl == 0:
                az_md = aux['az_m']
            else:
                m_d = np.where(msk, np.nan, M.render(n, rc['x'], p3.FITS['c'], delta_deg=float(dl)))
                az_md = pc.trace_near(m_d, ca, r_edge, aux['az_m'] + dl, pc.MODEL_WIN_DEG)
            for k, s in enumerate(pc.chords(az_md, None, ca, r_edge, ef, M.rtp, profiles=False)):
                drow.append(dict(key=row.key, model_idx=k, delta_deg=float(dl),
                                 az_half_ocs_deg_model=s['az_half_ocs_deg']))
        out['delta'] = drow
    if centers:
        import geom_edges as ge
        crow = dict(key=row.key, cx_fit_a=cx, cy_fit_a=cy)
        for edge in ('outer', 'inner'):
            for name, im in (('data', np.nan_to_num(data)), ('model', np.nan_to_num(mod))):
                c = ca
                for _ in range(2):
                    e = p3.edges_of(im, ~msk, c[0], c[1], row.r_out_pix, noise, flags)
                    ox, oy, _ = ge.circle_centre(e[f'r_{edge}'], AZ)
                    c = (c[0] + ox, c[1] + oy)
                crow[f'cx_{edge}_{name}'], crow[f'cy_{edge}_{name}'] = c
        out['centers'] = crow
    if keep_images:
        out['images'] = dict(model=model.astype(np.float32), without=without.astype(np.float32),
                             geom={k: (s['traj'], s['r_half_pix']) for k, s in enumerate(sd)},
                             geom_m={k: (s['traj'], s['r_half_pix']) for k, s in enumerate(sm)},
                             center=ca, r_edge=r_edge)
    fit['seconds'] = time.time() - t0
    fit['n_found'] = int(sum(r == '' for r in reasons))
    return out


def tables(results, sides):
    """Assemble `measure` outputs into the pass 5 table shapes, numbered and differenced."""
    fits = pd.DataFrame([r['fit'] for r in results])
    bins = pd.concat([r['bins'] for r in results], ignore_index=True)
    t = pd.DataFrame([s for r in results for s in r['shadows']])
    # fit_ok: the pass 3 flux-ratio test on this pass's fit
    fits['flux_ratio'] = fits.flux_e / fits.stamp_sum_e
    med = fits.groupby('side').flux_ratio.transform('median')
    q = fits.flux_ratio / med
    import pupil_pass3_report as rep3
    f = rep3.FLUX_RATIO_FACTOR
    fits['fit_ok'] = (q < f) & (q > 1 / f)
    ok = fits.set_index('key').fit_ok
    t['fit_ok'] = t.key.map(ok).fillna(False).astype(bool)
    t = ps5.number(t)
    for c in ('az_half_ocs_deg', 'phi0_ocs_deg'):
        t[f'{c}_dmm'] = pc.wrap(t[f'{c}_data'] - t[f'{c}_model'])
    for c in ('b_pix', 'depth', 'fwhm_pix'):
        t[f'{c}_dmm'] = t[f'{c}_data'] - t[f'{c}_model']
    t.loc[t.missing != '', [c for c in t if c.endswith('_dmm')]] = np.nan
    dt = pd.DataFrame([d for r in results for d in r.get('delta', [])])
    cen = pd.DataFrame([r['centers'] for r in results if 'centers' in r])
    s = sides[sides.key.isin(fits.key)].copy()
    s['fit_ok'] = s.key.map(ok).astype(bool)
    return dict(fits=fits, bins=bins, spiders=t, spider_delta=dt, centres=cen, sides=s)


def save(tabs, out_dir, prefix=''):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tabs.items():
        if df is None or len(df) == 0:
            continue
        df.to_parquet(out_dir / f'{prefix}{name}.parquet')
        if name not in ('bins', 'spider_delta'):
            df.to_csv(out_dir / f'{prefix}{name}.csv', index=False, float_format='%.6g')
