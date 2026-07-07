#!/usr/bin/env python3
"""wfs_dof_compare — corner-WFS vs FAM optical-state (v-modes + DoF) per image.

Extract the optical state two ways per FAM triplet and compare:
  FAM  (reference) — project the per-visit full-focal-plane double-Zernike fit
        (<mi>/fits.parquet, MIW-subtracted) onto the OFC sensitivity SVD:
        A_fam = W @ U_eff  ->  v-modes = A/sigma,  DoF = V-recon(A).
  CWFS            — from the 4 corner medians only: subtract the MIW (5rot
        intrinsic_split OCS maps, interpolated to each corner) and the per-(Zj,corner)
        offset, then the corner OFC inverse  A_cwfs = pinv(B·U_eff, rcond)·z_corner_dev
        (B = focal basis at the 4 corner OCS positions).  Same SVD/normalization/
        truncation as FAM, so the two are directly comparable (only the measurement
        differs: 4 corners vs full field).

Both 50-DOF/34-vmode and 22-DOF/12-vmode are produced.  Pages per SVD: for v-modes,
time history (vs image ordinal), CWFS-vs-FAM scatter (per-panel slope/offset/r/robust
RMS/drop) and a per-mode summary of slope/offset/r/robust-RMS; for DoF, the same time
history + scatter, then a 2-page grouped recovery summary (hexapod translations /
rotations / M1M3 / M2 bending, unit-consistent per panel) with each DoF as a point =
fit offset, error bar = robust RMS.  Scatter fits reject points far (>K·nMAD) from the mass.

Finally, AOS-FWHM pages.  The CWFS−FAM v-mode error is the optical-state recovery error;
it maps to a residual double-Zernike wavefront, evaluated as a pupil-Zernike vector at
field points and converted to a PSF FWHM (ts_wep convertZernikesToPsfWidth, Z4+ quadrature).
Per SVD, a page reports the within-scheme recovery FWHM as a focal-plane area-average and
an average at the 4 rotated CWFS corner positions (the ConsDB AOS_FWHM analog).  A final
focal-plane contributions page (both schemes share kj_grid, so reconstructions difference
directly) overlays: the MIW baseline itself (mean over each FAM visit's science-field
donuts), the FAM 50/34 excursion beyond MIW (uncorrected), CWFS-22/12 vs FAM-50/34
(recovery + the truncation error from FAM modes 13-34 the 22/12 CWFS cannot represent),
CWFS-50/34 vs FAM-50/34 (recovery only — tests whether more CWFS modes help), and the
truncation-only term (FAM-22/12 vs FAM-50/34).

Needs ts_ofc (build_ofc_svd) + TS_CONFIG_MTTCS_DIR; runs in the LSST stack env.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CORNERS = ['R00_SW0', 'R04_SW0', 'R40_SW0', 'R44_SW0']
FP_RADIUS = 1.75
GRID_STEP = 0.35          # focal-plane grid step (deg); sets the fp_grid + binning cell
RAFT_CELL = 0.71          # deg: raft pitch — binning half-box for the 1-star-per-raft
                          # GalSim sampling (each donut lands in its raft's cell)
DOF22 = list(range(0, 10)) + list(range(10, 17)) + list(range(30, 35))
# CWFS - FAM offsets (OCS, µm) from the corner-compare study; subtracted from CWFS.
DEFAULT_OFFSETS = {4: {'R00_SW0': -0.11, 'R04_SW0': -0.11, 'R40_SW0': -0.11, 'R44_SW0': -0.19},
                   11: {c: -0.07 for c in CORNERS},
                   14: {c: +0.13 for c in CORNERS}}


def nmad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size >= 3 else np.nan


def corner_matrix_at(svd, noll, pos_deg):
    """B (ncorner*nj, n_kj): focal basis at the corner OCS positions (deg)."""
    from lsst.ts.intrinsic.wavefront.ofc_svd import focal_zernike_at_points
    jpos = {j: i for i, j in enumerate(noll)}; nj = len(noll)
    B = np.zeros((len(pos_deg) * nj, len(svd.kj_grid)))
    for ci, (tx, ty) in enumerate(pos_deg):
        rho = np.hypot(tx, ty) / FP_RADIUS; th = np.arctan2(ty, tx)
        for k_i, (k, j) in enumerate(svd.kj_grid):
            if j in jpos:
                B[ci * nj + jpos[j], k_i] = float(focal_zernike_at_points(k, rho, th))
    return B


def robust_fit(x, y, K):
    """OLS fit + Pearson r + robust RMS after dropping points >K·nMAD from the mass."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 3:                                    # keep-mask is over the finite subset (len == len(x))
        return np.zeros(len(x), bool), dict(n=len(x), ndrop=0, r=np.nan, slope=np.nan, off=np.nan, rms=np.nan)
    keep = ((np.abs(x - np.median(x)) <= K * nmad(x) + 1e-12)
            & (np.abs(y - np.median(y)) <= K * nmad(y) + 1e-12))
    xk, yk = x[keep], y[keep]
    if keep.sum() < 3:
        return keep, dict(n=len(x), ndrop=int((~keep).sum()), r=np.nan, slope=np.nan, off=np.nan, rms=np.nan)
    s, b = np.polyfit(xk, yk, 1); res = yk - (s * xk + b)
    return keep, dict(n=len(x), ndrop=int((~keep).sum()), r=float(np.corrcoef(xk, yk)[0, 1]),
                      slope=float(s), off=float(b), rms=float(nmad(res)))


def th_pages(labels, ordn, fam, cwfs, title, pdf, per=20):
    import matplotlib.pyplot as plt
    n = len(labels)
    for p0 in range(0, n, per):
        idx = range(p0, min(p0 + per, n)); nr = int(np.ceil(len(list(idx)) / 4))
        fig, axes = plt.subplots(nr, 4, figsize=(15, 2.2 * nr), constrained_layout=True, squeeze=False)
        for ax, i in zip(axes.ravel(), idx):
            ax.plot(ordn, fam[:, i], '-', lw=0.6, color='steelblue', label='FAM')
            ax.plot(ordn, cwfs[:, i], '-', lw=0.6, color='crimson', label='CWFS')
            ax.axhline(0, color='k', lw=0.3); ax.set_title(labels[i], fontsize=7); ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(list(idx)):]:
            ax.axis('off')
        axes.ravel()[0].legend(fontsize=6)
        fig.suptitle(f'{title} — time history vs image ordinal  [FAM blue, CWFS red]', fontsize=11)
        pdf.savefig(fig); plt.close(fig)


def scatter_pages(labels, fam, cwfs, title, pdf, K, per=20):
    import matplotlib.pyplot as plt
    n = len(labels)
    for p0 in range(0, n, per):
        idx = list(range(p0, min(p0 + per, n))); nr = int(np.ceil(len(idx) / 4))
        fig, axes = plt.subplots(nr, 4, figsize=(15, 3.0 * nr), constrained_layout=True, squeeze=False)
        for ax, i in zip(axes.ravel(), idx):
            x, y = fam[:, i], cwfs[:, i]
            keep, m = robust_fit(x, y, K)
            fm = np.isfinite(x) & np.isfinite(y)
            xf, yf = x[fm], y[fm]
            ax.scatter(xf[keep], yf[keep], s=6, alpha=0.4)
            if (~keep).any():
                ax.scatter(xf[~keep], yf[~keep], s=6, alpha=0.4, color='lightgray')
            if np.isfinite(m['slope']):
                lo, hi = np.nanpercentile(np.concatenate([xf[keep], yf[keep]]), [1, 99])
                ax.plot([lo, hi], [lo, hi], 'k--', lw=0.6)
                ax.plot([lo, hi], [m['slope'] * lo + m['off'], m['slope'] * hi + m['off']], 'r-', lw=0.8)
                ax.text(0.04, 0.96, f"r={m['r']:.2f} s={m['slope']:.2f}\noff={m['off']:.3f} rms={m['rms']:.3f}\ndrop={m['ndrop']}",
                        transform=ax.transAxes, va='top', fontsize=6)
            ax.set_title(labels[i], fontsize=7); ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(idx):]:
            ax.axis('off')
        fig.suptitle(f'{title} — CWFS (y) vs FAM (x)', fontsize=11)
        pdf.savefig(fig); plt.close(fig)


def summary_page(labels, fam, cwfs, title, pdf, K):
    """One page: slope, offset, Pearson r and robust RMS of the CWFS-vs-FAM fit per mode."""
    import matplotlib.pyplot as plt
    n = len(labels)
    st = [robust_fit(fam[:, i], cwfs[:, i], K)[1] for i in range(n)]
    x = np.arange(n)
    rows = [('slope', [s['slope'] for s in st], 1.0),
            ('offset', [s['off'] for s in st], 0.0),
            ('correlation r', [s['r'] for s in st], 1.0),
            ('robust RMS', [s['rms'] for s in st], 0.0)]
    fig, axes = plt.subplots(4, 1, figsize=(max(8, 0.32 * n), 9), constrained_layout=True,
                             sharex=True, squeeze=False)
    for ax, (lab, vals, ref) in zip(axes[:, 0], rows):
        ax.plot(x, vals, 'o-', ms=3, lw=0.6, color='steelblue')
        if ref is not None:
            ax.axhline(ref, color='k', lw=0.5, ls='--')
        ax.set_ylabel(lab, fontsize=8); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
    axes[-1, 0].set_xticks(x); axes[-1, 0].set_xticklabels(labels, rotation=90, fontsize=6)
    fig.suptitle(f'{title} — CWFS-vs-FAM summary per mode  '
                 f'(slope/r ref=1, offset ref=0; {int(K)}·nMAD outliers dropped)', fontsize=11)
    pdf.savefig(fig); plt.close(fig)


def dof_summary_pages(labels, units, fam, cwfs, title, pdf, K):
    """DoF recovery summary grouped by the standard scheme (hexapod translations /
    rotations / M1M3 / M2 bending), unit-consistent per panel.  Each DoF is one
    point: y = CWFS-vs-FAM fit offset, error bar = robust RMS.  2 pages
    (rigid body, then bending modes)."""
    import matplotlib.pyplot as plt
    n = len(labels)
    st = [robust_fit(fam[:, i], cwfs[:, i], K)[1] for i in range(n)]
    off = np.array([s['off'] for s in st]); rms = np.array([s['rms'] for s in st])

    def grp(pred):
        return [i for i in range(n) if pred(labels[i])]
    pages = [('rigid body', [('Hexapod translations', grp(lambda l: l.endswith(('_dz', '_dx', '_dy')))),
                             ('Hexapod rotations', grp(lambda l: l.endswith(('_rx', '_ry'))))]),
             ('bending modes', [('M1M3 bending modes', grp(lambda l: l.startswith('B1_'))),
                                ('M2 bending modes', grp(lambda l: l.startswith('B2_')))])]

    def _panel(ax, ttl, idx):
        if not idx:
            ax.axis('off'); return
        x = np.arange(len(idx))
        for xi in range(len(idx)):
            if xi % 2:
                ax.axvspan(xi - 0.5, xi + 0.5, color='black', alpha=0.05)
        ax.errorbar(x, off[idx], yerr=rms[idx], fmt='o', ms=5, color='steelblue',
                    ecolor='gray', elinewidth=0.9, capsize=2)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels([labels[i] for i in idx], rotation=45, ha='right', fontsize=7)
        u = units[idx[0]] if units is not None else ''
        ax.set_ylabel(f'CWFS−FAM [{u}]'); ax.set_title(ttl); ax.grid(axis='y', alpha=0.3)

    for pg, panels in pages:
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True, squeeze=False)
        for ax, (ttl, idx) in zip(axes[:, 0], panels):
            _panel(ax, ttl, idx)
        fig.suptitle(f'{title} — CWFS−FAM DoF recovery ({pg}): point = fit offset, bar = robust RMS',
                     fontsize=11)
        pdf.savefig(fig); plt.close(fig)


def fp_grid(R, step):
    """Area-uniform grid of field points (deg) within radius R (deg)."""
    g = np.arange(-R, R + 1e-9, step)
    xx, yy = np.meshgrid(g, g)
    m = (xx ** 2 + yy ** 2) <= R ** 2
    return np.column_stack([xx[m], yy[m]])


def zj_to_fwhm(Z, noll, conv):
    """Z (..., nj) pupil-Zernike vectors (µm) at Noll ``noll`` -> AOS FWHM (arcsec).

    Pads into a Noll-4-start contiguous vector (Z1-3 excluded), applies
    ts_wep convertZernikesToPsfWidth (per-Zernike arcsec) and quadrature-sums.
    """
    Z = np.atleast_2d(np.asarray(Z, float))
    jmax = max(noll)
    full = np.zeros((Z.shape[0], jmax - 3))                # column 0 == Noll 4
    src = [i for i, j in enumerate(noll) if j >= 4]
    full[:, [noll[i] - 4 for i in src]] = Z[:, src]
    dpsf = np.asarray(conv(full), float)                   # (n, jmax-3) arcsec per Zernike
    return np.sqrt(np.nansum(dpsf ** 2, axis=1))


def fwhm_from_dW(svd, noll, dW, pos, grid_pos, conv):
    """corner + fp AOS FWHM (arcsec) from a residual double-Zernike wavefront dW
    (nt, n_kj) on ``svd.kj_grid``.  Evaluate the pupil-Zernike residual at field
    points (dW @ Bᵀ), convert to FWHM, average over the 4 rotated CWFS corner
    positions (ConsDB analog) and over the focal-plane grid (area-average)."""
    nt, nj = dW.shape[0], len(noll)
    zg_all = dW @ corner_matrix_at(svd, noll, grid_pos).T   # fixed grid -> one basis
    fp = np.full(nt, np.nan); cor = np.full(nt, np.nan)
    for t in range(nt):
        if not np.all(np.isfinite(dW[t])):
            continue
        fp[t] = np.nanmean(zj_to_fwhm(zg_all[t].reshape(-1, nj), noll, conv))
        if np.all(np.isfinite(pos[t])):
            zc = (dW[t] @ corner_matrix_at(svd, noll, pos[t]).T).reshape(len(CORNERS), nj)
            cor[t] = np.nanmean(zj_to_fwhm(zc, noll, conv))
    return dict(corner=cor, fp=fp)


def miw_fwhm_series(triplets, fd_grp, fmi, noll, conv):
    """Per-image focal-plane MIW FWHM (arcsec): mean over the FAM visit's donuts of
    the per-donut measured-intrinsic-wavefront FWHM (row-aligned sidecar ``fmi``)."""
    out = np.full(len(triplets), np.nan)
    for t, key in enumerate(triplets):
        idx = fd_grp.get(key)
        if idx is None or len(idx) == 0:
            continue
        out[t] = np.nanmean(zj_to_fwhm(fmi[np.asarray(idx, int)], noll, conv))
    return out


# ---------------------------------------------------------------------------
# Per-visit "observed" wavefront FWHM on the focal-plane grid.
#
# The DZ fit is band-limited to k<=6, so the per-visit high-spatial-frequency
# content is not in Wf/Wc (it lands, averaged, in the MIW).  To get the true
# delivered image quality we bin the per-donut wavefront onto the grid and work
# with per-cell Zernike fields (ncell, nj) rather than kj-DZ vectors:
#   MIW_grid   = binned per-donut MIW sidecar
#   dev_grid   = binned per-donut (zk - MIW)          (full spatial content)
#   Wobserved  = MIW_grid + dev_grid                  (the per-visit "truth")
# Every delivered-FWHM level is then FWHM(Wobserved - correction), where a
# correction (Wf/Wc) is a kj-DZ residual evaluated on the same grid points.
# ---------------------------------------------------------------------------
def bin_to_grid(vals, thx, thy, grid_pos, cell):
    """Median per-cell Zernike field on ``grid_pos``.

    ``vals`` (ndon, nj) per-donut Zernikes at field positions (``thx``,``thy``,
    deg); assign each donut to its nearest grid cell (square cell of side
    ``cell`` deg == the fp_grid step) and take the per-cell median.  Cells with
    no donut are NaN (dropped from the focal-plane average downstream)."""
    ncell, nj = len(grid_pos), vals.shape[1]
    out = np.full((ncell, nj), np.nan)
    finite = np.isfinite(thx) & np.isfinite(thy)
    if not finite.any():
        return out
    # nearest grid cell by half-cell box around each grid point
    for c, (gx, gy) in enumerate(grid_pos):
        m = finite & (np.abs(thx - gx) <= 0.5 * cell) & (np.abs(thy - gy) <= 0.5 * cell)
        if m.any():
            out[c] = np.nanmedian(vals[m], axis=0)
    return out


def dW_to_grid_zernikes(svd, noll, dW_row, grid_pos):
    """Per-cell Zernike field (ncell, nj) for a single kj-DZ residual on grid."""
    nj = len(noll)
    return (dW_row @ corner_matrix_at(svd, noll, grid_pos).T).reshape(len(grid_pos), nj)


def fwhm_of_cellfield(Zcell, noll, conv):
    """Focal-plane-average AOS FWHM (arcsec) of a per-cell Zernike field
    (ncell, nj); NaN cells are ignored."""
    return np.nanmean(zj_to_fwhm(Zcell, noll, conv))


def raft_centers(camera):
    """Field-angle centre (deg) of each science raft — the mean over its science
    CCDs' bbox centres.  ~21 points spanning the focal plane; used as the GalSim
    delivered-FWHM sampling (one PSF per raft)."""
    from lsst.afw import cameraGeom
    from lsst.geom import Point2D
    from collections import defaultdict
    pts = defaultdict(list)
    for d in camera:
        if d.getType() != cameraGeom.DetectorType.SCIENCE:
            continue
        bb = d.getBBox()
        fa = d.getTransform(cameraGeom.PIXELS, cameraGeom.FIELD_ANGLE).applyForward(
            Point2D(bb.getCenterX(), bb.getCenterY()))
        pts[d.getName().split('_')[0]].append((np.rad2deg(fa.getX()), np.rad2deg(fa.getY())))
    return np.array([np.mean(v, axis=0) for _, v in sorted(pts.items())])


def obs_fields_at(pos, cell, svd, noll, fmi, fam_dev, fam_thx, fam_thy, idx,
                  t, r50, r22):
    """Per-cell Zernike fields (ncell, nj) at field positions ``pos`` for one
    triplet ``t`` (donut row indices ``idx``): MIW, Wobserved, and the four
    'Wobserved - correction' residuals.  Shared by the formula + GalSim reducers.

    ``pos`` (ncell, 2) field-angle deg; ``cell`` the binning half-box side (deg).
    r50/r22 hold per-triplet DZ reconstructions Wf (FAM) and Wc (CWFS)."""
    miw = bin_to_grid(fmi[idx], fam_thx[idx], fam_thy[idx], pos, cell)
    dev = bin_to_grid(fam_dev[idx], fam_thx[idx], fam_thy[idx], pos, cell)
    Wobs = miw + dev
    def gz(dW_row):
        return dW_to_grid_zernikes(svd, noll, dW_row, pos)
    return dict(miw=miw, obs=Wobs,
                floor=Wobs - gz(r50['Wf'][t]),
                fam22=Wobs - gz(r22['Wf'][t]),
                cwfs50=Wobs - gz(r50['Wc'][t]),
                cwfs22=Wobs - gz(r22['Wc'][t]))


def _fwhm_page(ordn, lines, title, pdf):
    """Shared time-history + histogram page.  ``lines`` = list of (y, label, color)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for y, lab, c in lines:
        if np.isfinite(y).any():
            axes[0].plot(ordn, y, '.', ms=3, color=c,
                         label=f'{lab}  (median {np.nanmedian(y):.3f}″)')
    axes[0].set_xlabel('image ordinal'); axes[0].set_ylabel('AOS FWHM [arcsec]')
    axes[0].set_title(title); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for y, lab, c in lines:
        yf = y[np.isfinite(y)]
        if yf.size:
            axes[1].hist(yf, bins=40, histtype='step', color=c, label=lab)
    axes[1].set_xlabel('AOS FWHM [arcsec]'); axes[1].set_ylabel('N')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    pdf.savefig(fig); plt.close(fig)


def aos_fwhm_raw_page(ordn, fwhm_obs, fwhm_miw, pdf, method='formula'):
    """Page A: the raw uncorrected delivered FWHM, FWHM(Wobserved), on its own
    page (its excursion dwarfs the corrected levels), with the MIW baseline as a
    reference line.  ``method`` labels the Zj->FWHM route (formula | GalSim+HSM)."""
    lines = [(fwhm_obs, 'FWHM(FAM) raw — uncorrected observed optics', 'firebrick'),
             (fwhm_miw, 'MIW baseline', 'black')]
    _fwhm_page(ordn, lines, f'AOS FWHM [{method}] (focal-plane average) — raw '
               'uncorrected FAM (Wobserved = MIW + per-donut deviation)', pdf)


def aos_fwhm_levels_page(ordn, levels, pdf, method='formula'):
    """Page B: cumulative delivered FWHM levels, each = FWHM(Wobserved - correction),
    all including the MIW and the per-visit uncorrectable high-order content.
    ``method`` labels the Zj->FWHM route (formula = convertZernikesToPsfWidth;
    GalSim+HSM = rendered OpticalPSF⊗Kolmogorov, HSM, one PSF per raft centre)."""
    _fwhm_page(ordn, levels, f'AOS FWHM delivered [{method}] (focal-plane average) — '
               'FWHM(Wobserved - correction): MIW baseline, uncorrectable floor, '
               'FAM 22/12, CWFS 50/34, CWFS 22/12', pdf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', default='pathA_50_34_i_5rot')
    ap.add_argument('--wfs-name', required=True,
                    help='CWFS variant key; reads/writes under output/<ps>/wfs/<wfs-name>/ '
                         'and output/<ps>/<mi>/wfs/<wfs-name>/')
    ap.add_argument('--coord', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--dz-prefix', default='z1toz6')
    ap.add_argument('--rcond', type=float, default=1e-2, help='pinv cutoff for the corner OFC inverse')
    ap.add_argument('--reject-k', type=float, default=5.0, help='scatter outlier: drop >K·nMAD from the mass')
    ap.add_argument('--no-offsets', action='store_true', help='disable the CWFS-FAM per-(Zj,corner) offsets')
    ap.add_argument('--max-triplets', type=int, default=0, help='cap triplets (quick test); 0=all')
    ap.add_argument('--galsim-fwhm', action='store_true',
                    help='add a GalSim+HSM delivered-FWHM cross-check of the formula pages '
                         '(one rendered PSF per raft centre; parallel, slow — RSP + galsim)')
    ap.add_argument('--galsim-workers', type=int, default=16,
                    help='process-pool workers for the GalSim renders (--galsim-fwhm)')
    ap.add_argument('--band', default='i', choices=['u', 'g', 'r', 'i', 'z', 'y'],
                    help='band for the GalSim OpticalPSF wavelength')
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()
    from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd
    try:
        from lsst.ts.wep.utils import convertZernikesToPsfWidth as conv_fwhm
    except Exception as e:
        conv_fwhm = None
        print(f'[wfs_dof_compare] AOS_FWHM disabled (no convertZernikesToPsfWidth): {e}')
    coord = args.coord
    base = Path(args.output_root) / args.param_set
    bmi = base / args.mi_name
    zc, txc, tyc = f'zk_{coord}', f'thx_{coord}', f'thy_{coord}'
    offsets = {} if args.no_offsets else DEFAULT_OFFSETS

    noll = [int(x) for x in np.asarray(
        pq.read_table(str(base / 'visits.parquet'), columns=['nollIndices']).to_pandas()['nollIndices'].iloc[0])]

    # ---- FAM per-visit DZ fits (W built per SVD from its kj_grid below) ----
    fits = pd.read_parquet(bmi / 'fits.parquet')
    fam_key = {(int(d), int(s)): i for i, (d, s) in enumerate(zip(fits.day_obs, fits.seq_num))}

    # ---- CWFS corner donuts ----
    cw = pq.read_table(str(base / 'wfs' / args.wfs_name / 'donuts.parquet'),
                       columns=['detector', 'day_obs', 'seq_num', 'fam_seq_num', zc, txc, tyc]).to_pandas()
    cw_zk = np.stack(cw[zc].values).astype(float)
    cw_thx = np.rad2deg(cw[txc].astype(float).values); cw_thy = np.rad2deg(cw[tyc].astype(float).values)
    cw_det = cw.detector.astype(str).values
    cw_grp = cw.groupby(['day_obs', 'fam_seq_num']).indices

    # ---- CWFS MIW sidecar (row-aligned to wfs/donuts.parquet; built by
    #      run_make_intrinsic_sidecar --wfs-corner-height, i.e. the identical
    #      reconstruct_at path as the FAM sidecar, with the SW1/SW0 half-sensor Z4 height) ----
    scf = bmi / 'wfs' / args.wfs_name / 'zk_intrinsic.parquet'
    mi_sc = np.stack(pq.read_table(str(scf), columns=['zk_intrinsic_MI']).to_pandas()
                     ['zk_intrinsic_MI'].values).astype(float)
    md = pq.read_schema(str(scf)).metadata or {}
    noll_i = (np.frombuffer(md[b'nollIndices'], dtype=int).tolist() if b'nollIndices' in md else noll)
    mi_cw = mi_sc[:, [noll_i.index(j) for j in noll]]          # MIW per donut, aligned to noll

    # ---- FAM per-donut MIW sidecar + per-donut measured wavefront (row-aligned
    #      to the full FAM donuts.parquet).  Used for (a) the per-donut MIW FWHM
    #      series and (b) the binned-to-grid Wobserved: MIW_grid + dev_grid where
    #      dev = per-donut (zk - MIW), carrying the full spatial content the k<=6
    #      DZ fit smooths into the visit-averaged MIW. ----
    fd_grp = fmi = None
    fam_dev = fam_thx = fam_thy = None
    try:
        fd = pq.read_table(str(base / 'donuts.parquet'),
                           columns=['day_obs', 'seq_num', zc, txc, tyc]).to_pandas()
        scf_f = bmi / 'zk_intrinsic.parquet'
        fmi_raw = np.stack(pq.read_table(str(scf_f), columns=['zk_intrinsic_MI']).to_pandas()
                           ['zk_intrinsic_MI'].values).astype(float)
        mdf = pq.read_schema(str(scf_f)).metadata or {}
        noll_f = (np.frombuffer(mdf[b'nollIndices'], dtype=int).tolist() if b'nollIndices' in mdf else noll)
        fmi = fmi_raw[:, [noll_f.index(j) for j in noll]]
        fd_grp = {(int(d), int(s)): idx for (d, s), idx in
                  fd.groupby(['day_obs', 'seq_num']).indices.items()}
        # per-donut measured wavefront (aligned to noll) minus per-donut MIW
        fam_zk = np.stack(fd[zc].values).astype(float)             # (ndon, nZk_file)
        # align the measured zk columns to noll via the donut file's own nollIndices
        ndon_md = pq.read_schema(str(base / 'donuts.parquet')).metadata or {}
        noll_d = (np.frombuffer(ndon_md[b'nollIndices'], dtype=int).tolist()
                  if b'nollIndices' in ndon_md else noll_f)
        fam_zk = fam_zk[:, [noll_d.index(j) for j in noll]]
        fam_dev = fam_zk - fmi                                     # per-donut (zk - MIW)
        fam_thx = np.rad2deg(np.asarray(fd[txc], float))
        fam_thy = np.rad2deg(np.asarray(fd[tyc], float))
    except Exception as e:
        print(f'[wfs_dof_compare] MIW FWHM disabled (no FAM sidecar/donuts): {e}')

    triplets = sorted(set(cw_grp.keys()) & set(fam_key.keys()), key=lambda k: (k[0], k[1]))
    if args.max_triplets:
        triplets = triplets[:args.max_triplets]
    print(f'[wfs_dof_compare] {args.param_set}/{args.mi_name}/wfs:{args.wfs_name}: '
          f'{len(triplets)} matched triplets, coord={coord}, '
          f'offsets={"off" if args.no_offsets else "on"}, rcond={args.rcond}')

    # per-corner CWFS deviation (per-donut MIW-subtracted, then per-(Zj,corner) offset) + positions
    off_by_det = {det: np.array([offsets.get(j, {}).get(det, 0.0) for j in noll]) for det in CORNERS}
    zdev = np.full((len(triplets), 4, len(noll)), np.nan)
    pos = np.full((len(triplets), 4, 2), np.nan)
    for t, key in enumerate(triplets):
        ci = np.asarray(cw_grp[key], int)
        for c, det in enumerate(CORNERS):
            cc = ci[cw_det[ci] == det]
            if len(cc) == 0:
                continue
            pos[t, c] = [np.median(cw_thx[cc]), np.median(cw_thy[cc])]
            zdev[t, c] = np.nanmedian(cw_zk[cc] - mi_cw[cc], axis=0) - off_by_det[det]

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    out = bmi / 'wfs' / args.wfs_name / f"wfs_dof_compare_{'nooffset' if args.no_offsets else 'offsets'}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    ordn = np.arange(len(triplets))
    grid_pos = fp_grid(FP_RADIUS, GRID_STEP)                       # ~89-point focal-plane grid
    recon = {}                                                     # per-scheme DZ reconstructions for the hybrid page
    with PdfPages(str(out)) as pdf:
        for name, (n_keep, n_dof) in [('50DOF-34vmode', (34, None)), ('22DOF-12vmode', (12, DOF22))]:
            svd = build_ofc_svd(list(noll), k_min=1, k_max=6, n_keep=n_keep, n_dof=n_dof)
            W = np.column_stack([fits[f'{args.dz_prefix}_z{j}_c{k}'].values if
                                 f'{args.dz_prefix}_z{j}_c{k}' in fits.columns else np.full(len(fits), np.nan)
                                 for (k, j) in svd.kj_grid])
            A_fam_all = svd.project_amplitudes(W)                      # (nfits, n_keep)
            fam_rows = np.array([fam_key[key] for key in triplets])
            A_fam = A_fam_all[fam_rows]
            A_cwfs = np.full_like(A_fam, np.nan)
            for t in range(len(triplets)):
                if not np.all(np.isfinite(zdev[t])):
                    continue
                B = corner_matrix_at(svd, noll, pos[t])
                A_cwfs[t] = np.linalg.pinv(B @ svd.U_eff, rcond=args.rcond) @ zdev[t].ravel()
            vmode_lab = [f'v{m+1}' for m in range(n_keep)]
            dof_lab, dof_units = svd.dof_labels()
            vfam, vcw = svd.vmodes(A_fam), svd.vmodes(A_cwfs)
            dfam, dcw = svd.dof(A_fam), svd.dof(A_cwfs)
            th_pages(vmode_lab, ordn, vfam, vcw, f'{name} v-modes', pdf)
            scatter_pages(vmode_lab, vfam, vcw, f'{name} v-modes', pdf, args.reject_k)
            summary_page(vmode_lab, vfam, vcw, f'{name} v-modes', pdf, args.reject_k)
            th_pages(dof_lab, ordn, dfam, dcw, f'{name} DoF', pdf)
            scatter_pages(dof_lab, dfam, dcw, f'{name} DoF', pdf, args.reject_k)
            dof_summary_pages(dof_lab, dof_units, dfam, dcw, f'{name} DoF', pdf, args.reject_k)
            med_r = np.nanmedian([robust_fit(vfam[:, i], vcw[:, i], args.reject_k)[1]['r'] for i in range(n_keep)])
            print(f'  {name}: median v-mode CWFS-vs-FAM r = {med_r:.3f}')
            if conv_fwhm is not None:
                # CWFS-FAM recovery-error FWHM (fp + ConsDB-corner analog) — kept
                # as a printed diagnostic; the standalone per-scheme page is
                # superseded by the delivered-FWHM levels page below.
                dW_rec = ((vcw - vfam) * svd.Sigma[svd._keep()][None, :]) @ svd.U_eff.T
                fw = fwhm_from_dW(svd, noll, dW_rec, pos, grid_pos, conv_fwhm)
                print(f'  {name}: median recovery-error FWHM  fp={np.nanmedian(fw["fp"]):.3f}″  '
                      f'corner={np.nanmedian(fw["corner"]):.3f}″')
            recon[name] = dict(svd=svd, Wf=A_fam @ svd.U_eff.T, Wc=A_cwfs @ svd.U_eff.T)

        # ---- delivered-FWHM pages: per-visit Wobserved (binned to grid) minus
        #      each correction.  Requires the per-donut FAM deviation (fam_dev) +
        #      positions and both SVD schemes. ----
        have_obs = (conv_fwhm is not None and fam_dev is not None
                    and {'50DOF-34vmode', '22DOF-12vmode'} <= set(recon))
        if have_obs:
            r50, r22 = recon['50DOF-34vmode'], recon['22DOF-12vmode']
            svd = r50['svd']
            nt = len(triplets)
            FIELDS = ['miw', 'obs', 'floor', 'fam22', 'cwfs50', 'cwfs22']
            # valid triplets (have FAM donuts) + their per-cell fields on the
            # FORMULA grid.  Cache each triplet's field dict for reuse.
            fields_by_t = {}
            for t, key in enumerate(triplets):
                idx = fd_grp.get(key) if fd_grp is not None else None
                if idx is None or len(idx) == 0:
                    continue
                fields_by_t[t] = obs_fields_at(
                    grid_pos, GRID_STEP, svd, noll, fmi, fam_dev, fam_thx, fam_thy,
                    np.asarray(idx, int), t, r50, r22)

            def series(reducer, fkey):
                y = np.full(nt, np.nan)
                for t, fd in fields_by_t.items():
                    y[t] = reducer(fd[fkey])
                return y

            form = lambda Z: fwhm_of_cellfield(Z, noll, conv_fwhm)   # noqa: E731
            fwhm_obs = series(form, 'obs')
            L = {k: series(form, k) for k in ['miw', 'floor', 'fam22', 'cwfs50', 'cwfs22']}
            aos_fwhm_raw_page(ordn, fwhm_obs, L['miw'], pdf, method='formula')  # Page A
            levels = [
                (L['miw'],    'MIW baseline', 'black'),
                (L['floor'],  'uncorrectable floor  Wobs - FAM 50/34', 'gray'),
                (L['fam22'],  'FAM 22/12  Wobs - FAM 22/12', 'darkorange'),
                (L['cwfs50'], 'CWFS 50/34  Wobs - CWFS 50/34', 'purple'),
                (L['cwfs22'], 'CWFS 22/12  Wobs - CWFS 22/12', 'steelblue'),
            ]
            aos_fwhm_levels_page(ordn, levels, pdf, method='formula')          # Page B
            print(f'  [formula] raw FWHM(Wobserved) median {np.nanmedian(fwhm_obs):.3f}″')
            for y, lab, _ in levels:
                print(f'  [formula] delivered FWHM median {np.nanmedian(y):.3f}″  — {lab}')

            # ---- GalSim+HSM parallel cross-check (one PSF per raft centre) ----
            if args.galsim_fwhm:
                import psf_render as pr
                from lsst.obs.lsst import LsstCam
                rpos = raft_centers(LsstCam.getCamera())
                # recompute the six fields on the raft-centre positions, gather
                # every (triplet, field, raft) Zernike vector into one big batch.
                gfields = {t: obs_fields_at(rpos, RAFT_CELL, svd, noll, fmi, fam_dev,
                                            fam_thx, fam_thy,
                                            np.asarray(fd_grp[triplets[t]], int),
                                            t, r50, r22)
                           for t in fields_by_t}
                lam_nm = pr.LAM_NM[args.band]
                batch, keyindex = [], []
                for t, fd in gfields.items():
                    for fk in FIELDS:
                        Z = fd[fk]                           # (nraft, nj)
                        for c in range(len(rpos)):
                            keyindex.append((t, fk, c)); batch.append(Z[c])
                print(f'  [galsim] rendering {len(batch)} PSFs '
                      f'({len(gfields)} triplets x {len(FIELDS)} fields x {len(rpos)} rafts), '
                      f'workers={args.galsim_workers} …')
                fw = pr.optics_fwhm_batch(batch, noll, lam_nm, workers=args.galsim_workers)
                # median over rafts -> per-(triplet, field)
                acc = {(t, fk): [] for t in gfields for fk in FIELDS}
                for (t, fk, _), v in zip(keyindex, fw):
                    acc[(t, fk)].append(v)
                gser = {fk: np.full(nt, np.nan) for fk in FIELDS}
                for (t, fk), vals in acc.items():
                    gser[fk][t] = np.nanmedian(vals)
                aos_fwhm_raw_page(ordn, gser['obs'], gser['miw'], pdf, method='GalSim+HSM')
                glevels = [
                    (gser['miw'],    'MIW baseline', 'black'),
                    (gser['floor'],  'uncorrectable floor  Wobs - FAM 50/34', 'gray'),
                    (gser['fam22'],  'FAM 22/12  Wobs - FAM 22/12', 'darkorange'),
                    (gser['cwfs50'], 'CWFS 50/34  Wobs - CWFS 50/34', 'purple'),
                    (gser['cwfs22'], 'CWFS 22/12  Wobs - CWFS 22/12', 'steelblue'),
                ]
                aos_fwhm_levels_page(ordn, glevels, pdf, method='GalSim+HSM')
                print(f'  [galsim] raw FWHM(Wobserved) median {np.nanmedian(gser["obs"]):.3f}″')
                for y, lab, _ in glevels:
                    print(f'  [galsim] delivered FWHM median {np.nanmedian(y):.3f}″  — {lab}')
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
