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

os.environ.setdefault('NUMEXPR_MAX_THREADS', '8')   # silence galsim/numexpr thread warning

sys.path.insert(0, str(Path(__file__).resolve().parent))          # same-study siblings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # aos/code (shared + other studies)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))      # repo root -> common/
from common.psf_render import (   # noqa: E402  GalSim render + HSM
    LAM_NM, build_psf_tools, render_measure, optics_fwhm)
from psf_maps_lib import (   # noqa: E402  shared with the closed_loop study
    FP_RADIUS, sample_science_stars, miw_zernikes, measure_zk, psf_page,
    fwhm_optics_hist, build_svd, residual_W, eval_dz_field,
    mimic_corner_matrix, residual_W_mimic, mimic_measurements,
    load_fam_visits, multi_fwhm_hist)

FP_RADIUS = 1.75        # deg, field normalization + FoV edge

# fixed plot scales (same across all pages so cases are comparable)
SCALE_ELLIP, KEY_ELLIP = 0.5, 0.2         # quiver scale (e per 0.4deg) + reference key
SCALE_COMA,  KEY_COMA = 0.125, 0.05
TREFOIL_AREA, KEY_TREFOIL = 1500.0, 0.1   # marker area per unit amplitude + reference key


# render_measure / build_psf_tools / _higher_moments live in common/psf_render.py
# (shared with run_wfs_dof_compare); imported at the top.


# 22-DOF subset: 5 M2 hex (0-4) + 5 Cam hex (5-9) + first 7 M1M3 (10-16) + first 5 M2 (30-34)
from aos_state import DOF22  # noqa: E402  canonical 22-DOF index list


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
    ap.add_argument('--mi', default='pathA_50_34_i_5rot',
                    help='measured-intrinsic build supplying both the MIW split maps and '
                         'the per-visit FAM fits')
    ap.add_argument('--case', default='miw',
                    choices=['miw', 'fam50', 'fam22', 'all', 'mimic50', 'mimic22',
                             'mimic', 'validate'],
                    help="'validate' = check convertZernikesToPsfWidth vs GalSim+HSM")
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
    cases = {'all': ['miw', 'fam50', 'fam22'],
             'mimic': ['mimic50', 'mimic22']}.get(args.case, [args.case])
    # 'validate' checks the convertZernikesToPsfWidth formula vs GalSim+HSM truth.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    # shared: stars, MIW base wavefront, psf tools, atm reference
    stars = sample_science_stars(camera, args.n_stars, FP_RADIUS, args.seed)
    miw_zk, noll = miw_zernikes(stars, base / args.mi / 'intrinsic_split_maps.parquet', camera, hmap)
    atm, aper = build_psf_tools(lam_nm)
    fwhm_atm = render_measure(None, noll, lam_nm, atm, aper, with_optics=False)['fwhm']
    print(f'[atm] pure Kolmogorov({ATM_FWHM}") HSM FWHM = {fwhm_atm:.4f}"  (subtracted in step 4)')

    # WFS-mimic real-donut corner measurements (loaded once, shared by mimic50/mimic22)
    Zc = None
    if any(c.startswith('mimic') for c in cases):
        from run_wfs_mimic import DEFAULT as MIMIC_DEFAULT
        sec = {**MIMIC_DEFAULT, 'delta_deg': args.mimic_delta}
        mvis = load_fam_visits(base / args.mi / 'fits.parquet', args.day_obs, args.rot_lim, args.max_visits)
        Zc = mimic_measurements(base, base / args.mi, args.coord, noll, sec, mvis)

    for case in cases:
        out = base / args.mi / 'psf' / f'psf_fp_maps_{case}_{args.band}.pdf'
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
            else:
                mimic = case.startswith('mimic')
                n_keep, n_dof = (34, None) if case.endswith('50') else (12, DOF22)
                svd = build_svd(noll, n_keep, n_dof)
                B = mimic_corner_matrix(svd, noll, args.mimic_delta) if mimic else None
                visits = load_fam_visits(base / args.mi / 'fits.parquet',
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
