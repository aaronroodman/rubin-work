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

PIXSCALE = 0.2          # arcsec / pixel (LSSTCam)
DIAM, OBSC = 8.36, 0.612
FP_RADIUS = 1.75        # deg, field normalization + FoV edge
ATM_FWHM = 0.6          # arcsec, Kolmogorov seeing
STAMP = 64              # pixels
LAM_NM = {'u': 368., 'g': 478., 'r': 622., 'i': 754., 'z': 869., 'y': 971.}
LN256 = np.log(256.0)

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
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import ccd_height as cch
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
def _higher_moments(arr, ix0, iy0, Mxx, Mxy, Myy):
    """Standardized 3rd/4th moments in the adaptive-Gaussian-whitened frame."""
    ny, nx = arr.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = xx - ix0; dy = yy - iy0
    M = np.array([[Mxx, Mxy], [Mxy, Myy]]); Minv = np.linalg.inv(M)
    r2 = Minv[0, 0] * dx**2 + 2 * Minv[0, 1] * dx * dy + Minv[1, 1] * dy**2
    w = np.exp(-0.5 * r2) * arr
    ev, R = np.linalg.eigh(M)
    Mhalfinv = R @ np.diag(1.0 / np.sqrt(ev)) @ R.T          # M^{-1/2}
    xi = Mhalfinv[0, 0] * dx + Mhalfinv[0, 1] * dy
    eta = Mhalfinv[1, 0] * dx + Mhalfinv[1, 1] * dy
    s = w.sum()
    m = lambda p, q: float((w * xi**p * eta**q).sum() / s)
    M30, M21, M12, M03 = m(3, 0), m(2, 1), m(1, 2), m(0, 3)
    M40, M22, M04 = m(4, 0), m(2, 2), m(0, 4)
    return dict(coma1=M30 + M12, coma2=M21 + M03,
                trefoil1=M30 - 3 * M12, trefoil2=3 * M21 - M03,
                kurtosis=M40 + M04 + 2 * M22)


def build_psf_tools(lam_nm):
    """Kolmogorov atmosphere + cached Aperture (pupil reused across all renders)."""
    import galsim
    atm = galsim.Kolmogorov(fwhm=ATM_FWHM)
    aper = galsim.Aperture(diam=DIAM, obscuration=OBSC, lam=lam_nm)
    return atm, aper


def render_measure(zk_um, noll, lam_nm, atm, aper, with_optics=True):
    """Render OpticalPSF(zk)⊗atm, measure with HSM. Returns dict or None on failure."""
    import galsim
    if with_optics:
        ab = np.zeros(max(noll) + 1)
        lam_um = lam_nm / 1000.0
        for j, z in zip(noll, zk_um):
            if np.isfinite(z):
                ab[j] = z / lam_um                      # waves
        psf = galsim.Convolve(galsim.OpticalPSF(lam=lam_nm, aper=aper, aberrations=ab.tolist()), atm)
    else:
        psf = atm
    try:
        img = psf.drawImage(nx=STAMP, ny=STAMP, scale=PIXSCALE)
        res = galsim.hsm.FindAdaptiveMom(img, strict=True)
    except Exception:
        return None
    sig = res.moments_sigma                              # pixels (det radius)
    e1, e2 = res.observed_shape.e1, res.observed_shape.e2
    e = np.hypot(e1, e2)
    T = 2.0 * sig**2 / np.sqrt(max(1.0 - e**2, 1e-6))    # pixels^2 (trace)
    fwhm = np.sqrt(T / 2.0 * LN256) * PIXSCALE           # arcsec
    out = dict(fwhm=fwhm, e1=e1, e2=e2, e=e)
    Mxx = (T / 2) * (1 + e1); Myy = (T / 2) * (1 - e1); Mxy = (T / 2) * e2
    ix0 = res.moments_centroid.x - img.xmin; iy0 = res.moments_centroid.y - img.ymin
    out.update(_higher_moments(img.array, ix0, iy0, Mxx, Mxy, Myy))
    return out


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
    from ofc_svd import build_ofc_svd
    return build_ofc_svd(list(noll), k_min=1, k_max=6, n_keep=n_keep, n_dof=n_dof)


def residual_W(row, prefix, svd):
    """Per-visit uncorrectable DZ residual (1, n_kj) = W - n_keep-mode reconstruction."""
    W = np.array([[float(row.get(f'{prefix}_z{j}_c{k}', np.nan)) for (k, j) in svd.kj_grid]])
    A = svd.project_amplitudes(W)
    return W - A @ svd.U_eff.T


def eval_dz_field(W_resid, svd, noll, stars):
    """Evaluate a DZ-coefficient vector (over svd.kj_grid) to a per-star pupil-Zernike
    matrix (n_star, n_noll) [µm], using the k=1..6 focal-plane Zernike basis."""
    from ofc_svd import focal_zernike_at_points
    rho = np.hypot(stars.thx_deg.values, stars.thy_deg.values) / FP_RADIUS
    theta = np.arctan2(stars.thy_deg.values, stars.thx_deg.values)
    jpos = {j: i for i, j in enumerate(noll)}
    zk = np.zeros((len(stars), len(noll)))
    Wr = W_resid.ravel()
    for ci, (k, j) in enumerate(svd.kj_grid):
        if j in jpos and np.isfinite(Wr[ci]):
            zk[:, jpos[j]] += Wr[ci] * focal_zernike_at_points(k, rho, theta)
    return zk


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ps', default='fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x')
    ap.add_argument('--split-mi', default='pathA_50_34_i_5rot', help='mi config for the MIW OCS/CCS split')
    ap.add_argument('--fam-mi', default='pathA_50_34_i', help='mi config whose per-visit FAM fits to use')
    ap.add_argument('--case', default='miw', choices=['miw', 'fam50', 'fam22', 'all'])
    ap.add_argument('--band', default='i')
    ap.add_argument('--n-stars', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--day-obs', type=int, default=20260315)
    ap.add_argument('--rot-lim', type=float, default=3.0)
    ap.add_argument('--max-visits', type=int, default=24)
    ap.add_argument('--dz-prefix', default='z1toz6')
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--height-map-dir', default='~/u/LSST/packages/batoid_rubin_data')
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # ccd_height, ofc_svd
    from lsst.obs.lsst import LsstCam
    camera = LsstCam.getCamera()
    base = Path(args.output_root) / args.ps
    hmap = os.path.expanduser(args.height_map_dir); lam_nm = LAM_NM[args.band]
    cases = ['miw', 'fam50', 'fam22'] if args.case == 'all' else [args.case]
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    # shared: stars, MIW base wavefront, psf tools, atm reference
    stars = sample_science_stars(camera, args.n_stars, FP_RADIUS, args.seed)
    miw_zk, noll = miw_zernikes(stars, base / args.split_mi / 'intrinsic_split_maps.parquet', camera, hmap)
    atm, aper = build_psf_tools(lam_nm)
    fwhm_atm = render_measure(None, noll, lam_nm, atm, aper, with_optics=False)['fwhm']
    print(f'[atm] pure Kolmogorov({ATM_FWHM}") HSM FWHM = {fwhm_atm:.4f}"  (subtracted in step 4)')

    for case in cases:
        out = base / 'plots' / f'psf_fp_maps_{case}_{args.band}.pdf'
        out.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(str(out)) as pdf:
            if case == 'miw':
                meas = measure_zk(miw_zk, noll, lam_nm, atm, aper)
                print(f'[miw] {len(meas)}/{len(stars)} stars; med FWHM={meas.fwhm.median():.3f}" '
                      f'e={meas.e.median():.3f}')
                psf_page(stars, meas, f'MIW (rotator 0, {args.band}) — {args.ps}', fwhm_atm, pdf)
                fwhm_optics_hist(meas, fwhm_atm, f'MIW optics FWHM contribution ({args.band})', pdf)
            else:
                n_keep, n_dof = (34, None) if case == 'fam50' else (12, DOF22)
                svd = build_svd(noll, n_keep, n_dof)
                visits = load_fam_visits(base / args.fam_mi / 'fits.parquet',
                                         args.day_obs, args.rot_lim, args.max_visits)
                tag = '50DOF/34vmode' if case == 'fam50' else '22DOF/12vmode'
                optics = []
                for vi, row in visits.iterrows():
                    Wr = residual_W(row, args.dz_prefix, svd)
                    zk = miw_zk + eval_dz_field(Wr, svd, noll, stars)
                    meas = measure_zk(zk, noll, lam_nm, atm, aper)
                    lab = f"{int(row['day_obs'])}/{int(row['seq_num'])}"
                    print(f'  [{case}] visit {vi+1}/{len(visits)} {lab}: '
                          f'med FWHM={meas.fwhm.median():.3f}" e={meas.e.median():.3f}')
                    psf_page(stars, meas, f'MIW + FAM {lab} corrected {tag} ({args.band}) — {args.ps}',
                             fwhm_atm, pdf)
                    optics.append((lab, meas['fwhm'].values - fwhm_atm))
                multi_fwhm_hist(optics, f'MIW + FAM corrected {tag} ({args.band})', pdf)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
