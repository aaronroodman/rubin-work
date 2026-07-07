#!/usr/bin/env python3
"""run_psf_fp_maps — focal-plane PSF maps (FWHM, e1/e2, coma, trefoil, kurtosis)
rendered with GalSim from a wavefront, measured with HSM.  RubinTV psfPlotting style.

Wavefront cases (one page each unless noted):
  miw     Measured Intrinsic Wavefront only (ideal correction of all DOF), rotator=0.
          Z4 = Z4_OCS + Z4_CCS (5rot split) + CCD-height Z4; other Zj = Zj_OCS only.
  fam50   MIW + FAM per-visit residual after 50-DOF / 34-vmode correction  (TODO next increment)
  fam22   MIW + FAM per-visit residual after 22-DOF / 12-vmode correction  (TODO next increment)

PSF model: galsim.OpticalPSF(i-band, diam=8.36 m, obsc=0.612, aberrations=Zj/lambda)
           convolved with galsim.Kolmogorov(fwhm=0.6"), drawn at 0.2"/pix, then HSM.
Moments (matches summit_extras psfPlotting):
  Ixx,Iyy,Ixy from HSM (arcsec^2);  T=Ixx+Iyy;  FWHM=sqrt(T/2*ln256);
  e1=(Ixx-Iyy)/T, e2=2Ixy/T, e=hypot(e1,e2);
  coma1=M30+M12, coma2=M21+M03;  trefoil1=M30-3M12, trefoil2=3M21-M03;
  kurtosis=M40+M04+2M22   (M_pq = standardized higher moments in the HSM-whitened frame).

RSP only: needs galsim, lsst.afw.cameraGeom, lsst.obs.lsst, batoid_rubin (CCD height).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('NUMEXPR_MAX_THREADS', '8')   # silence galsim/numexpr thread warning

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling psf_render
from psf_render import (   # noqa: E402  GalSim render + HSM (shared with run_wfs_dof_compare)
    PIXSCALE, DIAM, OBSC, ATM_FWHM, STAMP, LAM_NM, LN256,
    build_psf_tools, render_measure, optics_fwhm)

FP_RADIUS = 1.75        # deg, field normalization + FoV edge

# fixed plot scales (same across all pages so cases are comparable)
EXTENT = 2.0                              # deg, map axis half-range
SCALE_ELLIP, KEY_ELLIP = 0.5, 0.2         # quiver scale (e per 0.4deg) + reference key
SCALE_COMA,  KEY_COMA = 0.125, 0.05
TREFOIL_AREA, KEY_TREFOIL = 1500.0, 0.1   # marker area per unit amplitude + reference key


# ------------------------------------------------------------------ star sampling
def sample_science_stars(camera, n, r_max_deg, seed):
    """n random positions uniformly over science-CCD pixels, inside r<r_max_deg.
    Returns df(det_name, x_pix, y_pix, thx_deg, thy_deg).  thx/thy are cameraGeom
    FIELD_ANGLE (rad->deg) — the same field frame the build grids use at rotator 0."""
    from lsst.afw import cameraGeom
    from lsst.geom import Point2D
    rng = np.random.default_rng(seed)
    sci = [d for d in camera if d.getType() == cameraGeom.DetectorType.SCIENCE]
    rows = []
    while len(rows) < n:
        det = sci[rng.integers(len(sci))]
        bb = det.getBBox()
        x = rng.uniform(bb.getMinX(), bb.getMaxX())
        y = rng.uniform(bb.getMinY(), bb.getMaxY())
        fa = det.getTransform(cameraGeom.PIXELS, cameraGeom.FIELD_ANGLE).applyForward(Point2D(x, y))
        thx, thy = np.rad2deg(fa.getX()), np.rad2deg(fa.getY())
        if np.hypot(thx, thy) <= r_max_deg:
            rows.append((det.getName(), x, y, thx, thy))
    df = pd.DataFrame(rows, columns=['det_name', 'x_pix', 'y_pix', 'thx_deg', 'thy_deg'])
    print(f'[sample] {len(df)} science-CCD stars; thx {df.thx_deg.min():.2f}..{df.thx_deg.max():.2f} '
          f'thy {df.thy_deg.min():.2f}..{df.thy_deg.max():.2f} deg')
    return df


# ------------------------------------------------------------------ MIW wavefront
def miw_zernikes(stars, maps_path, camera, hmap_dir):
    """Per-star MIW Zernike vector (µm) + noll list, from the 5rot OCS/CCS split maps
    (Z4 = OCS+CCS+CCD-height; other Zj = OCS only)."""
    from scipy.interpolate import LinearNDInterpolator
    from lsst.ts.intrinsic.wavefront import ccd_height as cch
    M = pd.read_parquet(maps_path)
    noll = sorted(int(c[1:-4]) for c in M.columns if c.endswith('_OCS'))
    pts = np.column_stack([M.thx_deg, M.thy_deg]); q = np.column_stack([stars.thx_deg, stars.thy_deg])
    zk = np.zeros((len(stars), len(noll)))
    for ji, j in enumerate(noll):
        zk[:, ji] = LinearNDInterpolator(pts, M[f'Z{j}_OCS'].values)(q)
        if j == 4:
            zk[:, ji] += LinearNDInterpolator(pts, M['Z4_CCS'].values)(q)
    # CCD-height Z4 per star (intra==extra centroid = the star pixel)
    df = pd.DataFrame({'detector': stars.det_name.values,
                       'centroid_x_intra': stars.x_pix, 'centroid_y_intra': stars.y_pix,
                       'centroid_x_extra': stars.x_pix, 'centroid_y_extra': stars.y_pix})
    z4h = np.asarray(cch.compute_ccd_heights(df, camera, source='batoid_rubin',
                                             height_map_dir=hmap_dir)['Z4_height'], float)
    zk[:, noll.index(4)] += np.nan_to_num(z4h)
    print(f'[miw] noll={noll}; median |Z4_height|={np.nanmedian(np.abs(z4h)):.3f} µm')
    return zk, noll


# ------------------------------------------------------------------ render + measure
# render_measure / build_psf_tools / _higher_moments now live in psf_render.py
# (shared with run_wfs_dof_compare); imported at the top.
def measure_zk(zk, noll, lam_nm, atm, aper):
    """Render+measure every star's wavefront row. Returns a DataFrame (with idx)."""
    recs = []
    for i in range(len(zk)):
        r = render_measure(zk[i], noll, lam_nm, atm, aper)
        if r is not None:
            r['idx'] = i; recs.append(r)
    return pd.DataFrame(recs)


# ------------------------------------------------------------------ plotting (psfPlotting style)
def psf_page(stars, meas, title, fwhm_atm, pdf):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    m = meas.set_index('idx')
    x = stars.thx_deg.values[m.index]; y = stars.thy_deg.values[m.index]
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.9], hspace=0.22, wspace=0.25)

    def setup_map(ax, ttl):
        ax.set_xlim(-EXTENT, EXTENT); ax.set_ylim(-EXTENT, EXTENT); ax.set_aspect('equal')
        ax.add_patch(Circle((0, 0), FP_RADIUS, fill=False, ls='--', color='r', lw=0.8))
        ax.set_title(ttl, fontsize=9, pad=3); ax.tick_params(labelsize=7)
        ax.set_xlabel('FP x [deg]', fontsize=7); ax.set_ylabel('FP y [deg]', fontsize=7)

    def whisker(ax, ttl, ang, amp, scale, key):
        Q = ax.quiver(x, y, amp * np.cos(ang), amp * np.sin(ang), angles='xy',
                      scale_units='xy', scale=scale, headlength=0, headaxislength=0,
                      width=0.004, pivot='mid', color='k')
        setup_map(ax, ttl)
        ax.quiverkey(Q, 0.14, 0.95, key, f'{key:g}', labelpos='E', coordinates='axes',
                     fontproperties={'size': 7})

    # row 0: ellipticity whisker | FWHM map
    whisker(fig.add_subplot(gs[0, 0]), 'ellipticity',
            0.5 * np.arctan2(m.e2, m.e1), m.e.values, SCALE_ELLIP, KEY_ELLIP)
    ax = fig.add_subplot(gs[0, 1]); setup_map(ax, 'FWHM [arcsec]')
    vlo, vhi = np.nanpercentile(m.fwhm, [2, 98])
    fig.colorbar(ax.scatter(x, y, c=m.fwhm, s=10, cmap='viridis', vmin=vlo, vmax=vhi),
                 ax=ax, shrink=0.85)
    # row 1: e1 | e2
    for col, key in [(0, 'e1'), (1, 'e2')]:
        ax = fig.add_subplot(gs[1, col]); setup_map(ax, key)
        v = np.nanpercentile(np.abs(m[key]), 98) or 0.01
        fig.colorbar(ax.scatter(x, y, c=m[key], s=10, cmap='RdBu_r', vmin=-v, vmax=v),
                     ax=ax, shrink=0.85)
    # row 2: coma whisker | trefoil markers
    whisker(fig.add_subplot(gs[2, 0]), 'coma',
            np.arctan2(m.coma2, m.coma1), np.hypot(m.coma1, m.coma2), SCALE_COMA, KEY_COMA)
    ax = fig.add_subplot(gs[2, 1]); setup_map(ax, 'trefoil')
    tamp = np.hypot(m.trefoil1, m.trefoil2); tang = np.degrees(np.arctan2(m.trefoil2, m.trefoil1)) / 3
    for xi, yi, ai, si in zip(x, y, tang, tamp * TREFOIL_AREA):
        ax.scatter(xi, yi, marker=(3, 0, 30 + ai), s=si, color='k', lw=0.1)
    ax.scatter(-1.55, 1.6, marker=(3, 0, 30), s=KEY_TREFOIL * TREFOIL_AREA, color='k')
    ax.text(-1.25, 1.6, f'{KEY_TREFOIL:g}', fontsize=7, va='center')

    # histogram column (independent x-axes), quartile lines + values
    for r, (key, lab, col) in enumerate([('fwhm', 'FWHM [arcsec]', 'steelblue'),
                                         ('e', 'e', 'darkorange'), ('kurtosis', 'kurtosis', 'firebrick')]):
        ax = fig.add_subplot(gs[r, 2]); v = m[key].values; v = v[np.isfinite(v)]
        ax.hist(v, bins=40, color=col)
        q = np.nanpercentile(v, [25, 50, 75])
        for qq, lw in zip(q, (1, 2, 1)):
            ax.axvline(qq, color='k', lw=lw)
        ax.text(0.97, 0.96, f'{lab}\n25%: {q[0]:.3f}\n50%: {q[1]:.3f}\n75%: {q[2]:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def fwhm_optics_hist(meas, fwhm_atm, title, pdf):
    """Single histogram of optics FWHM contribution = FWHM_total - FWHM_atm."""
    import matplotlib.pyplot as plt
    d = meas['fwhm'].values - fwhm_atm; d = d[np.isfinite(d)]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.hist(d, bins=40, color='slategray')
    ax.text(0.97, 0.95, f'mean={np.mean(d):.4f}"\nrms={np.std(d):.4f}"',
            transform=ax.transAxes, ha='right', va='top', fontsize=10)
    ax.set_xlabel('optics FWHM contribution = FWHM − FWHM_atm [arcsec]')
    ax.set_title(title, fontsize=10); pdf.savefig(fig); plt.close(fig)


# ------------------------------------------------------------------ FAM residual
# 22-DOF subset: 5 M2 hex (0-4) + 5 Cam hex (5-9) + first 7 M1M3 (10-16) + first 5 M2 (30-34)
DOF22 = list(range(0, 10)) + list(range(10, 17)) + list(range(30, 35))


def build_svd(noll, n_keep, n_dof):
    from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd
    return build_ofc_svd(list(noll), k_min=1, k_max=6, n_keep=n_keep, n_dof=n_dof)


def residual_W(row, prefix, svd):
    """Per-visit uncorrectable DZ residual (1, n_kj) = W - n_keep-mode reconstruction."""
    W = np.array([[float(row.get(f'{prefix}_z{j}_c{k}', np.nan)) for (k, j) in svd.kj_grid]])
    A = svd.project_amplitudes(W)
    return W - A @ svd.U_eff.T


def eval_dz_field(W_resid, svd, noll, stars):
    """Evaluate a DZ-coefficient vector (over svd.kj_grid) to a per-star pupil-Zernike
    matrix (n_star, n_noll) [µm], using the k=1..6 focal-plane Zernike basis."""
    from lsst.ts.intrinsic.wavefront.ofc_svd import focal_zernike_at_points
    rho = np.hypot(stars.thx_deg.values, stars.thy_deg.values) / FP_RADIUS
    theta = np.arctan2(stars.thy_deg.values, stars.thx_deg.values)
    jpos = {j: i for i, j in enumerate(noll)}
    zk = np.zeros((len(stars), len(noll)))
    Wr = W_resid.ravel()
    for ci, (k, j) in enumerate(svd.kj_grid):
        if j in jpos and np.isfinite(Wr[ci]):
            zk[:, jpos[j]] += Wr[ci] * focal_zernike_at_points(k, rho, theta)
    return zk


# WFS-mimic corner wedges (run_wfs_mimic DEFAULT / analysis_config): mid-radius + 4 azimuths
MIMIC_RMID = 0.5 * (1.60 + 1.725)         # deg
MIMIC_OFFSETS = [0.0, 90.0, 180.0, 270.0]


def mimic_corner_matrix(svd, noll, delta):
    """B (4*nj, n_kj): maps a DZ-coeff vector (svd.kj_grid order) to the pupil-Zernike
    vectors at the 4 WFS-mimic corner centers (corner-major), via the focal basis."""
    from lsst.ts.intrinsic.wavefront.ofc_svd import focal_zernike_at_points
    rho = MIMIC_RMID / FP_RADIUS
    jpos = {j: i for i, j in enumerate(noll)}; nj = len(noll)
    B = np.zeros((4 * nj, len(svd.kj_grid)))
    for c, off in enumerate(MIMIC_OFFSETS):
        th = np.deg2rad(delta + off)
        for ci, (k, j) in enumerate(svd.kj_grid):
            if j in jpos:
                B[c * nj + jpos[j], ci] = float(focal_zernike_at_points(k, rho, th))
    return B


def residual_W_mimic(row, prefix, svd, B, z_corners):
    """Residual after correcting with amplitudes estimated ONLY from the 4 WFS-mimic
    corner regions.  Truth W = full FAM DZ fit (this visit); the correction amplitudes
    A = pinv(B·U_eff)·z come from the REAL per-donut wedge-median measurement
    ``z_corners`` (4 x nZk, corner-major) — so both the 4-corner field-sampling
    degeneracy AND the finite-donut measurement noise propagate into the residual.
    W_corr = U_eff·A;  residual = W - W_corr."""
    W = np.nan_to_num(np.array([float(row.get(f'{prefix}_z{j}_c{k}', np.nan))
                                for (k, j) in svd.kj_grid]))
    A = np.linalg.pinv(B @ svd.U_eff) @ np.asarray(z_corners, float).ravel()
    return (W - svd.U_eff @ A)[None, :]


def mimic_measurements(base_ps, fam_mi_dir, coord, noll, sec, visits):
    """Per-visit 4-corner wedge-median of the REAL per-donut MI deviations (the
    run_wfs_mimic measurement).  Returns {(day,seq): (4, nZk) in `noll` order or None}."""
    import pyarrow.parquet as _pq
    from run_wfs_mimic import _wedge_medians
    dd = _pq.read_table(str(base_ps / 'donuts.parquet'),
                        columns=['day_obs', 'seq_num', f'thx_{coord}', f'thy_{coord}',
                                 f'zk_{coord}']).to_pandas()
    sc = _pq.read_table(str(fam_mi_dir / 'zk_intrinsic.parquet'),
                        columns=['zk_intrinsic_MI']).to_pandas()
    if len(sc) != len(dd):
        raise SystemExit(f'sidecar rows ({len(sc)}) != donuts rows ({len(dd)})')
    vt = _pq.read_table(str(base_ps / 'visits.parquet'), columns=['nollIndices']).to_pandas()
    noll_m = [int(x) for x in np.asarray(vt['nollIndices'].iloc[0])]
    md = _pq.read_schema(str(fam_mi_dir / 'zk_intrinsic.parquet')).metadata or {}
    noll_i = (np.frombuffer(md[b'nollIndices'], dtype=int).tolist()
              if b'nollIndices' in md else noll_m)
    im = [noll_m.index(j) for j in noll]; ii = [noll_i.index(j) for j in noll]
    dev = np.stack(dd[f'zk_{coord}'].values).astype(float)[:, im] \
        - np.stack(sc['zk_intrinsic_MI'].values).astype(float)[:, ii]
    day = dd['day_obs'].astype(int).values; seq = dd['seq_num'].astype(int).values
    thx = np.rad2deg(dd[f'thx_{coord}'].astype(float).values)
    thy = np.rad2deg(dd[f'thy_{coord}'].astype(float).values)
    Z = {}
    for r in visits.itertuples():
        d, s = int(r.day_obs), int(r.seq_num); idx = np.where((day == d) & (seq == s))[0]
        Z[(d, s)] = _wedge_medians(dev[idx], thx[idx], thy[idx], sec) if len(idx) else None
    n_ok = sum(v is not None and np.all(np.isfinite(v)) for v in Z.values())
    print(f'[mimic] real-donut wedge medians: {n_ok}/{len(Z)} visits with all 4 wedges populated')
    return Z


def load_fam_visits(fits_path, day_obs, rot_lim, n):
    df = pd.read_parquet(fits_path)
    sel = df[(df['day_obs'].astype(int) == day_obs)
             & (np.abs(df['rotator_angle'].astype(float)) <= rot_lim)]
    if 'visit_quality_pass' in sel.columns:
        sel = sel[sel['visit_quality_pass'].astype(bool)]
    sel = sel.sort_values('mjd' if 'mjd' in sel.columns else 'seq_num').head(n)
    print(f'[fam] {len(sel)} visits (day_obs={day_obs}, |rot|<={rot_lim})')
    return sel.reset_index(drop=True)


def multi_fwhm_hist(visit_arrs, title, pdf, ncol=4):
    """6x4 grid of per-visit optics-FWHM histograms with mean/rms boxes."""
    import matplotlib.pyplot as plt
    n = len(visit_arrs); nrow = int(np.ceil(n / ncol))
    lo = min(np.nanmin(a) for _, a in visit_arrs); hi = max(np.nanmax(a) for _, a in visit_arrs)
    bins = np.linspace(lo, hi, 30)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.4 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax, (lab, a) in zip(axes.ravel(), visit_arrs):
        a = a[np.isfinite(a)]
        ax.hist(a, bins=bins, color='slategray')
        ax.text(0.96, 0.95, f'{lab}\nμ={np.mean(a):.3f}\nrms={np.std(a):.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=6)
        ax.tick_params(labelsize=5)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.suptitle(title + '  — optics FWHM contribution [arcsec]', fontsize=11)
    pdf.savefig(fig); plt.close(fig)


# ------------------------------------------------------------------ closed-loop (P control)
def loop_corner_data(base_ps, fam_mi_dir, coord, noll, sec, visits, intrinsic):
    """Per-visit 4-corner WFS measurement for the closed loop.  Donuts are selected in the
    FIXED CAMERA (CCS) corner wedges; the OCS measurement medians and OCS corner positions
    are returned (so all rotators are handled from the data, no OCS<->CCS rotation assumed).
    The measured quantity is the wavefront the *controller* sees after subtracting its
    assumed intrinsic:  tabulated -> zk_deviation (zk - design);  miw -> zk - zk_intrinsic_MI;
    none -> raw zk.  Returns {(day,seq): (z (4,nZk), pos (4,2) thx/thy_OCS deg)} or None."""
    import pyarrow.parquet as _pq
    base_cols = ['day_obs', 'seq_num', 'thx_CCS', 'thy_CCS', 'thx_OCS', 'thy_OCS']
    val = f'zk_deviation_{coord}' if intrinsic == 'tabulated' else f'zk_{coord}'
    dd = _pq.read_table(str(base_ps / 'donuts.parquet'), columns=base_cols + [val]).to_pandas()
    vt = _pq.read_table(str(base_ps / 'visits.parquet'), columns=['nollIndices']).to_pandas()
    noll_m = [int(x) for x in np.asarray(vt['nollIndices'].iloc[0])]
    im = [noll_m.index(j) for j in noll]
    meas = np.stack(dd[val].values).astype(float)[:, im]
    if intrinsic == 'miw':
        sc = _pq.read_table(str(fam_mi_dir / 'zk_intrinsic.parquet'), columns=['zk_intrinsic_MI']).to_pandas()
        md = _pq.read_schema(str(fam_mi_dir / 'zk_intrinsic.parquet')).metadata or {}
        noll_i = (np.frombuffer(md[b'nollIndices'], dtype=int).tolist() if b'nollIndices' in md else noll_m)
        meas = meas - np.stack(sc['zk_intrinsic_MI'].values).astype(float)[:, [noll_i.index(j) for j in noll]]
    day = dd.day_obs.astype(int).values; seq = dd.seq_num.astype(int).values
    cx = np.rad2deg(dd.thx_CCS.astype(float).values); cy = np.rad2deg(dd.thy_CCS.astype(float).values)
    ox = np.rad2deg(dd.thx_OCS.astype(float).values); oy = np.rad2deg(dd.thy_OCS.astype(float).values)
    rC = np.hypot(cx, cy); azC = np.degrees(np.arctan2(cy, cx)) % 360.0
    half = sec['wfs_azimuth_width_deg'] / 2.0
    rin, rout = sec['wfs_inner_radius_deg'], sec['wfs_outer_radius_deg']
    out = {}
    for r in visits.itertuples():
        d, s = int(r.day_obs), int(r.seq_num); vis = (day == d) & (seq == s)
        z = np.full((4, len(noll)), np.nan); pos = np.full((4, 2), np.nan); ok = True
        for ci, off in enumerate(MIMIC_OFFSETS):
            ctr = (sec['delta_deg'] + off) % 360.0; lo, hi = (ctr - half) % 360.0, (ctr + half) % 360.0
            azin = (azC >= lo) & (azC <= hi) if lo < hi else (azC >= lo) | (azC <= hi)
            w = vis & (rC >= rin) & (rC <= rout) & azin
            if int(w.sum()) < sec['min_donuts_per_wedge']:
                ok = False; break
            z[ci] = np.nanmedian(meas[w], axis=0); pos[ci] = [np.nanmedian(ox[w]), np.nanmedian(oy[w])]
        out[(d, s)] = (z, pos) if ok else None
    print(f'[loop] intrinsic={intrinsic}: {sum(v is not None for v in out.values())}/{len(out)} '
          f'visits with all 4 CCS-corner wedges')
    return out


def corner_matrix_at(svd, noll, pos_deg):
    """B (4*nj, n_kj): focal basis evaluated at 4 explicit corner OCS positions (deg)."""
    from lsst.ts.intrinsic.wavefront.ofc_svd import focal_zernike_at_points
    jpos = {j: i for i, j in enumerate(noll)}; nj = len(noll)
    B = np.zeros((4 * nj, len(svd.kj_grid)))
    for ci, (tx, ty) in enumerate(pos_deg):
        rho = np.hypot(tx, ty) / FP_RADIUS; th = np.arctan2(ty, tx)
        for k_i, (k, j) in enumerate(svd.kj_grid):
            if j in jpos:
                B[ci * nj + jpos[j], k_i] = float(focal_zernike_at_points(k, rho, th))
    return B


def _design_intrinsic_at(stars, band, noll):
    """Design (tabulated GQ) intrinsic Zernikes (µm) at each star field position."""
    from lsst.ts.wep.utils import getTaskInstrument
    inst = getTaskInstrument('LSSTCam', 'R22_S11', None); jmax = max(noll)
    out = np.zeros((len(stars), len(noll)))
    tx = stars.thx_deg.values; ty = stars.thy_deg.values
    for i in range(len(stars)):
        zk = inst._getIntrinsicZernikesCached(float(tx[i]), float(ty[i]), None, band, jmax) * 1e6
        out[i] = [zk[j] for j in noll]
    return out


def run_closed_loop(case, base, args, svd, noll, stars, atm, aper, lam_nm, fwhm_atm, pdf):
    """Design-relative proportional closed loop over a FAM disturbance sequence (all rotators).
    Truth W = deviation-from-design (top-level fits); baseline = design intrinsic per star.
    Each visit i:
      r = W_i - c ;  render PSF from design + r
      (measured images only) y = z_corners_i - B·c ;  A = pinv(B·U_eff)·y
      schedule  c += g·U_eff·A  to take effect at image i+L   (n+2 latency: L=2, every other
      image measured; n+1: L=1, every image).  --intrinsic sets what the controller subtracts."""
    import matplotlib.pyplot as plt
    from run_wfs_mimic import DEFAULT as MD
    sec = {**MD, 'delta_deg': args.mimic_delta}
    prefix, g = args.dz_prefix, args.gain
    fits = pd.read_parquet(base / 'fits.parquet')          # top-level = deviation from design
    fits = (fits.sample(frac=1, random_state=args.seed) if args.order == 'random'
            else fits.sort_values(['day_obs', 'seq_num']))
    if args.max_visits and args.max_visits > 0:
        fits = fits.head(args.max_visits)
    fits = fits.reset_index(drop=True)
    corners = loop_corner_data(base, base / args.fam_mi, args.coord, noll, sec, fits, args.intrinsic)
    design = _design_intrinsic_at(stars, args.band, noll)
    L = 2 if args.latency == 'nplustwo' else 1
    tag = (('50DOF/34vmode' if case.endswith('50') else '22DOF/12vmode')
           + f' g={g} {args.intrinsic} {args.order} {args.latency}')
    c = np.zeros(len(svd.kj_grid)); scheduled = {}; ts = []; optics = []; sample = {}
    step = 0; last_meas = None
    for i, row in fits.iterrows():
        if i in scheduled:
            c = c + scheduled.pop(i)
        W = np.nan_to_num(np.array([float(row.get(f'{prefix}_z{j}_c{k}', np.nan)) for (k, j) in svd.kj_grid]))
        meas = measure_zk(design + eval_dz_field((W - c)[None, :], svd, noll, stars), noll, lam_nm, atm, aper)
        rtp = float(row.get('rotator_angle', np.nan))
        ts.append((step, rtp, float(meas.fwhm.median()), float(meas.e.median())))
        optics.append((step, meas['fwhm'].values - fwhm_atm))
        if i % L == 0:                                     # measured image (stride = latency)
            cm = corners.get((int(row.day_obs), int(row.seq_num)))
            if cm is not None and np.all(np.isfinite(cm[0])):
                z_corners, pos = cm; B = corner_matrix_at(svd, noll, pos)
                A = np.linalg.pinv(B @ svd.U_eff) @ (z_corners.ravel() - B @ c)
                scheduled[i + L] = scheduled.get(i + L, 0.0) + g * (svd.U_eff @ A)
        if step == 0:
            sample[0] = meas
        last_meas = meas; step += 1
        print(f'  [{case}] step {step} {int(row.day_obs)}/{int(row.seq_num)} rtp={rtp:+.0f}: '
              f'medFWHM={meas.fwhm.median():.3f}" e={meas.e.median():.3f}')
    if not ts:
        print(f'[{case}] no usable visits'); return
    sample[ts[-1][0]] = last_meas
    a = np.array([(t[0], t[2], t[3]) for t in ts])
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax[0].plot(a[:, 0], a[:, 1], '-o', ms=3); ax[0].axhline(fwhm_atm, ls=':', color='gray', label='atm')
    ax[0].set_ylabel('median FWHM ["]'); ax[0].legend(fontsize=8); ax[0].set_title(f'Closed loop — {tag}')
    ax[1].plot(a[:, 0], a[:, 2], '-o', ms=3, color='darkorange')
    ax[1].set_ylabel('median e'); ax[1].set_xlabel('visit step')
    pdf.savefig(fig); plt.close(fig)
    burn = args.burn_in
    pool = np.concatenate([o for st, o in optics if st >= burn] or [o for _, o in optics])
    pool = pool[np.isfinite(pool)]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.hist(pool, bins=50, color='slategray')
    ax.text(0.97, 0.95, f'steps≥{burn}\nmean={pool.mean():.3f}"\nrms={pool.std():.3f}"',
            transform=ax.transAxes, ha='right', va='top', fontsize=10)
    ax.set_xlabel('optics FWHM contribution = FWHM − atm [arcsec]')
    ax.set_title(f'Closed-loop steady-state optics FWHM — {tag}')
    pdf.savefig(fig); plt.close(fig)
    for st in sorted(sample):
        psf_page(stars, sample[st], f'Closed loop step {st} — {tag} ({args.band})', fwhm_atm, pdf)


# ------------------------------------------------------------------ formula validation
def _formula_fwhm(zk_um, noll, lam_nm):
    """The convertZernikesToPsfWidth formula under test: per-Zernike arcsec
    contributions (µm in, Noll>=4, Z1-3 excluded), quadrature-summed."""
    from lsst.ts.wep.utils import convertZernikesToPsfWidth
    jmax = max(noll)
    full = np.zeros(jmax - 3)                              # index 0 == Noll 4
    for j, z in zip(noll, zk_um):
        if j >= 4 and np.isfinite(z):
            full[j - 4] = z
    dpsf = np.asarray(convertZernikesToPsfWidth(full), float)
    return float(np.sqrt(np.nansum(dpsf ** 2)))


def validate_formula_page(miw_zk, noll, lam_nm, atm, aper, pdf, n_scatter=300, seed=1):
    """Validate convertZernikesToPsfWidth against GalSim+HSM truth.

    Page 1 — scatter over realistic wavefronts: for up to ``n_scatter`` real
      per-star MIW Zernike vectors, formula FWHM (x) vs GalSim optics FWHM (y,
      atmosphere removed in quadrature).  1:1 line + median ratio.
    Page 2 — per-Noll amplitude sweep: each Noll j alone swept 0..~0.8 µm,
      formula vs GalSim, to expose where the linear/quadrature formula departs
      from the true (nonlinear, cross-term-free here) PSF width."""
    import matplotlib.pyplot as plt
    T_atm = render_measure(None, noll, lam_nm, atm, aper, with_optics=False)['T']

    # ---- Page 1: realistic-wavefront scatter ----
    rng = np.random.default_rng(seed)
    n = min(n_scatter, len(miw_zk))
    sel = rng.choice(len(miw_zk), size=n, replace=False)
    xf = np.array([_formula_fwhm(miw_zk[i], noll, lam_nm) for i in sel])
    yg = np.array([optics_fwhm(miw_zk[i], noll, lam_nm, atm, aper, T_atm) for i in sel])
    ok = np.isfinite(xf) & np.isfinite(yg) & (yg > 0) & (xf > 1e-6)
    xf, yg = xf[ok], yg[ok]
    ratio = np.nanmedian(yg / xf) if xf.size else np.nan
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.plot(xf, yg, '.', ms=4, alpha=0.5)
    lim = max(xf.max(), yg.max()) * 1.05 if xf.size else 1.0
    ax.plot([0, lim], [0, lim], 'k-', lw=1, label='1:1')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
    ax.set_xlabel('convertZernikesToPsfWidth FWHM [arcsec]')
    ax.set_ylabel('GalSim+HSM optics FWHM (atm removed) [arcsec]')
    ax.set_title(f'FWHM formula vs GalSim truth — {xf.size} real MIW wavefronts\n'
                 f'median(GalSim/formula) = {ratio:.3f}')
    ax.legend(); ax.grid(alpha=0.3)
    pdf.savefig(fig); plt.close(fig)
    print(f'[validate] scatter: n={xf.size}  median GalSim/formula = {ratio:.3f}  '
          f'(formula med {np.nanmedian(xf):.3f}″, GalSim med {np.nanmedian(yg):.3f}″)')

    # ---- Page 2: per-Noll amplitude sweep ----
    amps = np.linspace(0.0, 0.8, 9)                        # µm
    ncol = 4; nrow = int(np.ceil(len(noll) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                            constrained_layout=True, squeeze=False)
    for ji, j in enumerate(noll):
        ax = axs[ji // ncol][ji % ncol]
        f_form, f_gal = [], []
        for a in amps:
            zk = np.zeros(len(noll)); zk[ji] = a
            f_form.append(_formula_fwhm(zk, noll, lam_nm))
            f_gal.append(optics_fwhm(zk, noll, lam_nm, atm, aper, T_atm))
        ax.plot(amps, f_form, 'o-', ms=3, label='formula')
        ax.plot(amps, f_gal, 's--', ms=3, label='GalSim')
        ax.set_title(f'Z{j}', fontsize=8); ax.grid(alpha=0.3); ax.tick_params(labelsize=6)
        if ji == 0:
            ax.legend(fontsize=6)
    for k in range(len(noll), nrow * ncol):
        axs[k // ncol][k % ncol].set_visible(False)
    fig.supxlabel('single-Zernike amplitude [µm]'); fig.supylabel('optics FWHM [arcsec]')
    fig.suptitle('Per-Noll amplitude sweep: formula vs GalSim+HSM (single Zernike, isolated)')
    pdf.savefig(fig); plt.close(fig)
    print(f'[validate] per-Noll sweep written ({len(noll)} Zernikes, amps 0-0.8 µm)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ps', default='fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x')
    ap.add_argument('--split-mi', default='pathA_50_34_i_5rot', help='mi config for the MIW OCS/CCS split')
    ap.add_argument('--fam-mi', default='pathA_50_34_i', help='mi config whose per-visit FAM fits to use')
    ap.add_argument('--case', default='miw',
                    choices=['miw', 'fam50', 'fam22', 'all', 'mimic50', 'mimic22', 'mimic',
                             'loop50', 'loop22', 'loop', 'validate'],
                    help="'validate' = check convertZernikesToPsfWidth vs GalSim+HSM")
    ap.add_argument('--gain', type=float, default=0.3, help='proportional control gain (closed loop)')
    ap.add_argument('--burn-in', type=int, default=5, help='loop steps to skip for the steady-state histogram')
    ap.add_argument('--intrinsic', default='tabulated', choices=['tabulated', 'miw', 'none'],
                    help='intrinsic the controller subtracts to estimate DOF (loop): tabulated=design, '
                         'miw=measured-intrinsic, none=raw OPD')
    ap.add_argument('--order', default='ordered', choices=['ordered', 'random'],
                    help='loop visit order: ordered=day_obs,seq_num (groups of same-position visits); '
                         'random=shuffled (random slewing). Reality is in between.')
    ap.add_argument('--latency', default='nplustwo', choices=['nplustwo', 'nplusone'],
                    help='nplustwo: correction from image n applied at n+2, n+1 measurement ignored '
                         '(realistic); nplusone: applied at n+1 (future goal)')
    ap.add_argument('--mimic-delta', type=float, default=0.0,
                    help='azimuth offset of the 4 WFS-mimic corners (deg); matches wfs_mimic delta_deg')
    ap.add_argument('--band', default='i')
    ap.add_argument('--n-stars', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--day-obs', type=int, default=20260315)
    ap.add_argument('--rot-lim', type=float, default=3.0)
    ap.add_argument('--max-visits', type=int, default=24)
    ap.add_argument('--dz-prefix', default='z1toz6')
    ap.add_argument('--coord', default='OCS', choices=['OCS', 'CCS'],
                    help='donut frame for the WFS-mimic per-donut deviations')
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--height-map-dir', default='~/u/LSST/packages/batoid_rubin_data')
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling run_wfs_mimic
    from lsst.obs.lsst import LsstCam
    camera = LsstCam.getCamera()
    base = Path(args.output_root) / args.ps
    hmap = os.path.expanduser(args.height_map_dir); lam_nm = LAM_NM[args.band]
    cases = {'all': ['miw', 'fam50', 'fam22'], 'mimic': ['mimic50', 'mimic22'],
             'loop': ['loop50', 'loop22']}.get(args.case, [args.case])
    # 'validate' checks the convertZernikesToPsfWidth formula vs GalSim+HSM truth.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    # shared: stars, MIW base wavefront, psf tools, atm reference
    stars = sample_science_stars(camera, args.n_stars, FP_RADIUS, args.seed)
    miw_zk, noll = miw_zernikes(stars, base / args.split_mi / 'intrinsic_split_maps.parquet', camera, hmap)
    atm, aper = build_psf_tools(lam_nm)
    fwhm_atm = render_measure(None, noll, lam_nm, atm, aper, with_optics=False)['fwhm']
    print(f'[atm] pure Kolmogorov({ATM_FWHM}") HSM FWHM = {fwhm_atm:.4f}"  (subtracted in step 4)')

    # WFS-mimic real-donut corner measurements (loaded once, shared by mimic50/mimic22)
    Zc = None
    if any(c.startswith('mimic') for c in cases):
        from run_wfs_mimic import DEFAULT as MIMIC_DEFAULT
        sec = {**MIMIC_DEFAULT, 'delta_deg': args.mimic_delta}
        mvis = load_fam_visits(base / args.fam_mi / 'fits.parquet', args.day_obs, args.rot_lim, args.max_visits)
        Zc = mimic_measurements(base, base / args.fam_mi, args.coord, noll, sec, mvis)

    for case in cases:
        suffix = (f'_{args.intrinsic}_{args.order}_{args.latency}_g{args.gain:g}'
                  if case.startswith('loop') else '')
        out = base / 'plots' / f'psf_fp_maps_{case}_{args.band}{suffix}.pdf'
        out.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(str(out)) as pdf:
            if case == 'validate':
                validate_formula_page(miw_zk, noll, lam_nm, atm, aper, pdf,
                                      n_scatter=args.n_stars, seed=args.seed)
            elif case == 'miw':
                meas = measure_zk(miw_zk, noll, lam_nm, atm, aper)
                print(f'[miw] {len(meas)}/{len(stars)} stars; med FWHM={meas.fwhm.median():.3f}" '
                      f'e={meas.e.median():.3f}')
                psf_page(stars, meas, f'MIW (rotator 0, {args.band}) — {args.ps}', fwhm_atm, pdf)
                fwhm_optics_hist(meas, fwhm_atm, f'MIW optics FWHM contribution ({args.band})', pdf)
            elif case in ('loop50', 'loop22'):
                n_keep, n_dof = (34, None) if case.endswith('50') else (12, DOF22)
                run_closed_loop(case, base, args, build_svd(noll, n_keep, n_dof),
                                noll, stars, atm, aper, lam_nm, fwhm_atm, pdf)
            else:
                mimic = case.startswith('mimic')
                n_keep, n_dof = (34, None) if case.endswith('50') else (12, DOF22)
                svd = build_svd(noll, n_keep, n_dof)
                B = mimic_corner_matrix(svd, noll, args.mimic_delta) if mimic else None
                visits = load_fam_visits(base / args.fam_mi / 'fits.parquet',
                                         args.day_obs, args.rot_lim, args.max_visits)
                tag = ('50DOF/34vmode' if case.endswith('50') else '22DOF/12vmode') \
                    + (' WFS-mimic' if mimic else '')
                if mimic:    # nominal MIW page for comparison (per request)
                    psf_page(stars, measure_zk(miw_zk, noll, lam_nm, atm, aper),
                             f'MIW nominal (rotator 0, {args.band}) — {args.ps}', fwhm_atm, pdf)
                optics = []
                for vi, row in visits.iterrows():
                    lab = f"{int(row['day_obs'])}/{int(row['seq_num'])}"
                    if mimic:
                        z = Zc.get((int(row['day_obs']), int(row['seq_num'])))
                        if z is None or not np.all(np.isfinite(z)):
                            print(f'  [{case}] visit {lab}: <4 populated wedges, skip'); continue
                        Wr = residual_W_mimic(row, args.dz_prefix, svd, B, z)
                    else:
                        Wr = residual_W(row, args.dz_prefix, svd)
                    zk = miw_zk + eval_dz_field(Wr, svd, noll, stars)
                    meas = measure_zk(zk, noll, lam_nm, atm, aper)
                    print(f'  [{case}] visit {vi+1}/{len(visits)} {lab}: '
                          f'med FWHM={meas.fwhm.median():.3f}" e={meas.e.median():.3f}')
                    psf_page(stars, meas, f'MIW + FAM {lab} corrected {tag} ({args.band}) — {args.ps}',
                             fwhm_atm, pdf)
                    optics.append((lab, meas['fwhm'].values - fwhm_atm))
                multi_fwhm_hist(optics, f'MIW + FAM corrected {tag} ({args.band})', pdf)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
