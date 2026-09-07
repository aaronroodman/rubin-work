"""Shared machinery for rendering focal-plane PSF maps from an AOS wavefront.

Used by the `psf` study (`run_psf_fp_maps.py`, PSF from a given optical state) and the
`closedloop` study (`run_closed_loop.py`, AOS closed-loop simulation), which need the
same star sampling, wavefront evaluation, GalSim rendering and page layout.

The GalSim render/measure primitives themselves are in `common/psf_render.py` — nothing
in those is AOS-specific. This module holds the parts that are: the MIW wavefront
lookup, the Double Zernike (DZ) residual evaluation, and the corner-recovery matrices.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))          # aos/code
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # repo root -> common/
from common.psf_render import (   # noqa: E402
    PIXSCALE, DIAM, OBSC, ATM_FWHM, STAMP, LAM_NM, LN256,
    build_psf_tools, render_measure, optics_fwhm)

FP_RADIUS = 1.75        # deg, field normalization + FoV edge


# fixed plot scales (same across all pages so cases are comparable)
EXTENT = 2.0                              # deg, map axis half-range
SCALE_ELLIP, KEY_ELLIP = 0.5, 0.2         # quiver scale (e per 0.4deg) + reference key
SCALE_COMA,  KEY_COMA = 0.125, 0.05
TREFOIL_AREA, KEY_TREFOIL = 1500.0, 0.1   # marker area per unit amplitude + reference key



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



def measure_zk(zk, noll, lam_nm, atm, aper):
    """Render+measure every star's wavefront row. Returns a DataFrame (with idx)."""
    recs = []
    for i in range(len(zk)):
        r = render_measure(zk[i], noll, lam_nm, atm, aper)
        if r is not None:
            r['idx'] = i; recs.append(r)
    return pd.DataFrame(recs)



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
        # NaN is truthy, so `x or 0.01` does NOT guard an all-NaN percentile:
        # it would propagate NaN into vmin/vmax and silently break the scale.
        v = np.nanpercentile(np.abs(m[key]), 98)
        v = 0.01 if not np.isfinite(v) or v == 0 else v
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
