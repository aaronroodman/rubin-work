#!/usr/bin/env python
"""Leakage matrix L and direct sparse sensitivity from Danish fits to model donuts.

For each field point and defocus side, a noise-free donut is drawn with the
Danish forward model from the defocused Batoid wavefront (``zernikeTA`` to Noll
78, as ts_wep builds its reference). It is then fit as ts_wep's Danish
estimator fits a sparse set: the listed Noll terms free, every other term held
at the nominal defocused Batoid model, so the unfitted terms sit at the Batoid
intrinsic. Per point:

- ``A_dense``: in-focus pupil Zernike change per unit DOF (Noll 4..78, µm of
  wavefront per µm or degree), from ``batoid.zernikeGQ``;
- ``A_direct``, ``A_direct_opd``, ``A_direct_rt``: the change in the sparse fit
  per unit DOF, from donuts of the perturbed telescope drawn by Danish from the
  defocused TA fit, drawn by Danish from the defocused OPD, and ray-traced by
  batoid with a Kolmogorov blur;
- ``L``: the change in the sparse fit per µm injected into one unfitted term
  (Noll 4..78 outside the fit set), dimensionless;
- ``G_rt``: fit response to each defocused TA term (Noll 4..78), dimensionless,
  from ray-traced donuts with a Zernike OPD screen on the stop plane;
  ``G_rt[:, other]`` is the ray-traced L;
- ``dTA``, ``dOPD``: change per unit DOF of the defocused ``zernikeTA`` fit and of
  the defocused wavefront (``zernikeGQ``), Noll 4..78, µm per µm or degree.

The check is ``A_direct = D[fit] + L @ D[other]`` with D one of A_dense, dTA, dOPD,
and ``A_direct_rt = G_rt @ dTA``. Danish and the ray trace both mask the
ts_wep ``Spider_3D`` vanes at SPIDER_ANGLE.

Usage::

    python smatrix/sparse_fit/code/donut_leakage.py --jobs 16
    python smatrix/sparse_fit/code/donut_leakage.py --configs sci_p0_p0_extra --dofs 0 3 --label smoke
"""
import argparse
import json
import os
import pathlib
import time
import warnings

import numpy as np
import batoid
import danish
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
import galsim

from rubinwork.common import study_run
from pupil_masks import danish_clear, stop_uv
import rubinwork.smatrix.compute_smatrix as cs

warnings.simplefilter("ignore")

BAND = "r"
JMAX_MODEL = 78                  # ts_wep nollIndicesModel = 0..78
NOLL_A = np.arange(4, JMAX_MODEL + 1)  # bending modes reach above Noll 28; their leakage counts
FIT_SETS = {
    "prod": [4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 20, 21, 27, 28],      # fam_tables d13-sparse
    "prod_z22": [4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 20, 21, 22, 27, 28],
    "dense21": [j for j in range(4, 27) if j not in (20, 21)],      # standard dense fit
}
BINNING = 2                      # d13-sparse runs bin x2
NPIX = 81                        # binned stamp; 1.5 mm defocus donut is ~61 px across
FWHM_TRUE = 1.0                  # arcsec
FLUX = 1e5
DZ_TARGET_UM = 0.01              # RMS in-focus Noll 4..78 change per DOF step or injection;
                                 # at 0.05 the corner fits are already nonlinear

# Corner half-chips: cameraGeom FIELD_ANGLE centres (deg), SW0 extra, SW1 intra.
CORNERS = {
    "R00_SW0": (-1.1897, -1.1251), "R00_SW1": (-1.1897, -1.2543),
    "R04_SW0": (1.1251, -1.1897), "R04_SW1": (1.2543, -1.1897),
    "R40_SW0": (-1.1251, 1.1897), "R40_SW1": (-1.2543, 1.1897),
    "R44_SW0": (1.1897, 1.1251), "R44_SW1": (1.1897, 1.2543),
}
SCI_GRID_DEG = (-0.75, 0.0, 0.75)
SPIDER_ANGLE = 0.0               # deg; Danish and the ray trace both mask the Spider_3D vanes


CWFS_GRID = (6, 3)               # field points per half-chip along pixel x (4072), y (2000)


def _cwfs_grid():
    """6 x 3 cell centres on each corner half-chip, cameraGeom FIELD_ANGLE (deg)."""
    from lsst.obs.lsst import LsstCam
    from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE
    import lsst.geom as geom
    cam = LsstCam.getCamera()
    out = []
    for name in CORNERS:
        det = cam[name]
        bb = det.getBBox()
        tr = det.getTransform(PIXELS, FIELD_ANGLE)
        nx, ny = CWFS_GRID
        for i in range(nx):
            for j in range(ny):
                px = bb.getMinX() - 0.5 + (i + 0.5) * bb.getWidth() / nx
                py = bb.getMinY() - 0.5 + (j + 0.5) * bb.getHeight() / ny
                f = tr.applyForward(geom.Point2D(px, py))
                out.append((f"{name}_x{i}y{j}", *np.rad2deg([f.getX(), f.getY()])))
    return out


def configs(which="check"):
    """Field point and defocus configurations: (name, thx deg, thy deg, side, instrument).

    ``check``: the 8 half-chip centres and 9 science points (both sides);
    ``fam``: the dense build's Gauss-Legendre field grid (11 rings x 35 spokes to
    1.75 deg), camera shifted, both sides; ``cwfs``: a 6 x 3 grid on each corner
    half-chip, Detector shifted, on that half-chip's side; ``all``: all three.
    """
    out = []
    if which in ("check", "all"):
        for name, (x, y) in CORNERS.items():
            side = "extra" if name.endswith("SW0") else "intra"
            out.append((name, x, y, side, "LsstCam"))
        tag = {-0.75: "m", 0.0: "0", 0.75: "p"}
        for x in SCI_GRID_DEG:
            for y in SCI_GRID_DEG:
                for side in ("intra", "extra"):
                    out.append((f"sci_{tag[x]}{tag[y]}_{side}", x, y, side, "LsstFamCam"))
    if which in ("fam", "all"):
        li, _ = np.polynomial.legendre.leggauss(cs.RINGS)
        radii = np.sqrt((1 + li) / 2) * cs.FIELD_RADIUS_DEG
        azs = np.linspace(0, 2 * np.pi, cs.SPOKES, endpoint=False)
        for ir, r in enumerate(radii):
            for ia, a in enumerate(azs):
                for side in ("intra", "extra"):
                    out.append((f"fam_r{ir:02d}_a{ia:02d}_{side}", r * np.cos(a),
                                r * np.sin(a), side, "LsstFamCam"))
    if which in ("cwfs", "all"):
        for name, x, y in _cwfs_grid():
            side = "extra" if "_SW0_" in name else "intra"
            out.append((name, x, y, side, "LsstCam"))
    return out


_W = {}


def _init(dofs, opts=None):
    global FIT_SETS
    opts = opts or {}
    if opts.get("fit_sets"):
        FIT_SETS = {k: FIT_SETS[k] for k in opts["fit_sets"]}
    _W["opts"] = opts
    from lsst.ts.wep.instrument import Instrument
    os.environ.setdefault("BATOID_RUBIN_DATA_DIR", cs.DEFAULT_DATA_DIR)
    cs._init_worker(BAND, cs.DEFAULT_DATA_DIR, "bend", 50)
    _W["instr"] = {n: Instrument(configFile=f"policy:instruments/{n}.yaml")
                   for n in ("LsstCam", "LsstFamCam")}
    _W["dofs"] = dofs
    _W["optics"] = {}


def _optic(idof, step):
    """Perturbed in-focus telescope; idof < 0 is the fiducial."""
    if idof < 0:
        return cs._W["fiducial"]
    key = (idof, step)
    if key not in _W["optics"]:
        from batoid_rubin import LSSTBuilder
        dof = np.zeros(50)
        dof[idof] = step
        _W["optics"][key] = LSSTBuilder(cs._W["fiducial"], **cs._W["kwargs"]).with_aos_dof(
            dof.tolist()).build()
    return _W["optics"][key]


def zk_infocus(optic, th, wl):
    """In-focus pupil Zernikes (m), Noll 0..JMAX_MODEL; ts_wep's intrinsic is the same to 28."""
    return batoid.zernikeGQ(optic, *th, wl, jmax=JMAX_MODEL, eps=optic.pupilObscuration,
                            rings=12) * wl


def zk_ta(instr, optic, th, wl, sign):
    """Defocused transverse-aberration Zernikes (m), Noll 0..78, as ts_wep's model."""
    shifted = instr._applyBatoidOffsets(optic, sign)
    return batoid.zernikeTA(shifted, *th, wl, jmax=JMAX_MODEL,
                            eps=shifted.pupilObscuration, focal_length=instr.focalLength,
                            nrad=10, naz=int(2 * np.pi * 10)) * wl


def zk_defocus_opd(instr, optic, th, wl, sign):
    """Defocused wavefront (OPD) Zernikes (m), Noll 0..78."""
    shifted = instr._applyBatoidOffsets(optic, sign)
    return batoid.zernikeGQ(shifted, *th, wl, jmax=JMAX_MODEL,
                            eps=shifted.pupilObscuration, rings=12) * wl


def factory_for(instr):
    return danish.DonutFactory(
        R_outer=instr.radius, R_inner=instr.radius * instr.obscuration,
        mask_params=instr.maskParams, focal_length=instr.focalLength,
        pixel_scale=instr.pixelSize * BINNING, spider_angle=SPIDER_ANGLE)


RT_NX = 1024                     # pupil ray grid; ~0.47M unvignetted rays
RT_OS = 5                        # deposit oversampling per binned pixel


def raytrace(instr, optic, th, wl, sign, spider=SPIDER_ANGLE):
    """Noise-free ray-traced donut, centred on its own chief ray (relative flux).

    A fixed pupil grid (not random photons), so 0.01 µm steps give smooth
    differences. Rays are deposited with linear (cloud-in-cell) weights on an
    ``RT_OS``-times oversampled grid, convolved with a Kolmogorov FWHM_TRUE
    profile and summed to binned pixels. Batoid has no spider, so the
    ts_wep ``Spider_3D`` vanes are masked on the stop plane as Danish masks them.
    Batoid focal-plane x, y map to
    Danish image columns, rows with no flip (checked on difference images).
    """
    shifted = instr._applyBatoidOffsets(optic, sign)
    rv = batoid.RayVector.asGrid(optic=shifted, wavelength=wl, theta_x=th[0],
                                 theta_y=th[1], nx=RT_NX)
    spider_ok = danish_clear(instr, *stop_uv(shifted, rv), *th, spider_angle=spider,
                             circles=False)
    rv = shifted.trace(rv)
    chief = shifted.trace(batoid.RayVector.fromStop(0.0, 0.0, optic=shifted, wavelength=wl,
                                                    theta_x=th[0], theta_y=th[1]))
    ok = ~rv.vignetted & spider_ok
    fpix = instr.pixelSize * BINNING / RT_OS
    nf = NPIX * RT_OS
    u = (rv.x[ok] - chief.x[0]) / fpix + nf / 2 - 0.5
    v = (rv.y[ok] - chief.y[0]) / fpix + nf / 2 - 0.5
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i0, v - j0
    fine = np.zeros((nf, nf))
    for di, wu in ((0, 1 - fu), (1, fu)):
        for dj, wv in ((0, 1 - fv), (1, fv)):
            ii, jj = i0 + di, j0 + dj
            m = (ii >= 0) & (ii < nf) & (jj >= 0) & (jj < nf)
            np.add.at(fine, (jj[m], ii[m]), (wu * wv)[m])
    if "kolm" not in _W:
        scale = 3600 * np.rad2deg(fpix / instr.focalLength)      # arcsec per fine pixel
        _W["kolm"] = galsim.Kolmogorov(fwhm=FWHM_TRUE).drawImage(
            nx=121, ny=121, scale=scale, method="no_pixel").array
    fine = fftconvolve(fine, _W["kolm"], mode="same")
    return fine.reshape(NPIX, RT_OS, NPIX, RT_OS).sum((1, 3))


def with_screen(optic, j, coef, instr):
    """Optic with a stop-plane OPD screen of ``coef`` m in annular Noll ``j``."""
    c = np.zeros(j + 1)
    c[j] = coef
    scr = batoid.OPDScreen(batoid.Plane(), batoid.Zernike(
        c, R_outer=instr.radius, R_inner=instr.radius * instr.obscuration),
        name="InjScreen", coordSys=optic.stopSurface.coordSys,
        inMedium=optic.inMedium, outMedium=optic.inMedium)
    return optic.withInsertedOptic(before="M1", item=scr)


def draw(factory, z_ref, th):
    m = danish.SingleDonutModel(factory, z_ref=z_ref, z_terms=(), thx=th[0], thy=th[1],
                                npix=NPIX)
    return m.model(flux=FLUX, dx=0.0, dy=0.0, fwhm=FWHM_TRUE, z_fit=[])


def fit(factory, z_ref_fit, fit_idx, th, img):
    """Danish sparse fit as in ts_wep; returns fitted Zernikes (m) on fit_idx."""
    m = danish.SingleDonutModel(factory, z_ref=z_ref_fit, z_terms=fit_idx, thx=th[0],
                                thy=th[1], npix=NPIX)
    x0 = m.pack_params(flux=float(np.sum(img)), dx=0.0, dy=0.0, fwhm=1.0,
                       bkg=[], z_fit=[0.0] * len(fit_idx))
    r = least_squares(m.chi, jac=m.jac, x0=x0, args=(img, 1.0),
                      ftol=1e-12, xtol=1e-12, gtol=1e-12)
    return np.array(m.unpack_params(r.x)["z_fit"])


def save_images(directory, out, saved, rt_scale, instr):
    """Ray-traced donuts of one configuration as multi-HDU FITS.

    HDUs: NOMINAL (npix, npix); PLUS and MINUS (ndof, npix, npix), the donuts at
    +step and -step of DOF ``DOFS['idof']``; DOFS table (idof, step in µm or deg).
    Images are noise-free, in counts with total flux matching the Danish model.
    """
    from astropy.io import fits
    hdr = fits.Header()
    for key, val, com in (
            ("CONFIG", out["name"], "configuration name"),
            ("THX_DEG", out["thx_deg"], "field angle x, cameraGeom FIELD_ANGLE (deg)"),
            ("THY_DEG", out["thy_deg"], "field angle y, cameraGeom FIELD_ANGLE (deg)"),
            ("SIDE", out["side"], "defocus side"),
            ("INSTR", out["instrument"], "ts_wep instrument"),
            ("WL_M", out["wavelength_m"], "wavelength (m)"),
            ("BINNING", BINNING, "pixel binning"),
            ("PIXSCALE", instr.pixelSize * BINNING, "binned pixel size (m)"),
            ("SPIDER", SPIDER_ANGLE, "spider angle (deg)"),
            ("FWHM", FWHM_TRUE, "Kolmogorov FWHM (arcsec)"),
            ("FLUX", FLUX, "Danish model flux"),
            ("RTSCALE", rt_scale, "flux per ray"),
            ("RT_NX", RT_NX, "pupil ray grid"),
            ("RT_OS", RT_OS, "deposit oversampling")):
        hdr[key] = (val, com)
    hdus = [fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(np.asarray(saved["nominal"], np.float32), name="NOMINAL"),
            fits.ImageHDU(np.asarray(saved["plus"], np.float32), name="PLUS"),
            fits.ImageHDU(np.asarray(saved["minus"], np.float32), name="MINUS"),
            fits.BinTableHDU.from_columns([
                fits.Column(name="idof", format="J", array=np.array(out["dofs"])),
                fits.Column(name="step", format="D", array=np.array(out["steps"]))],
                name="DOFS")]
    fits.HDUList(hdus).writeto(pathlib.Path(directory) / f"donuts_{out['name']}.fits",
                               overwrite=True)


def run_config(cfg):
    name, thx, thy, side, iname = cfg
    t0 = time.time()
    instr = _W["instr"][iname]
    wl = instr.wavelength[BAND] if isinstance(instr.wavelength, dict) else \
        instr.wavelength[instr.refBand]
    th = np.deg2rad([thx, thy])
    sign = +1 if side == "extra" else -1
    fac = factory_for(instr)
    fid = _optic(-1, 0)

    zf0 = zk_infocus(fid, th, wl)
    ta0 = zk_ta(instr, fid, th, wl, sign)
    opd0 = zk_defocus_opd(instr, fid, th, wl, sign)
    refs = {}
    for fs, idx in FIT_SETS.items():
        zr = ta0.copy()
        zr[idx] -= zf0[idx]           # ts_wep getOffAxisCoeff(nollIndicesIntr=idx)
        refs[fs] = zr
    # Test donuts: Danish drawn from the defocused TA fit, Danish drawn from the
    # defocused OPD, and batoid ray-traced (the ground truth).
    img_ta0 = draw(fac, ta0, th)
    rt_scale = img_ta0.sum() / raytrace(instr, fid, th, wl, sign).sum()
    truths = {
        "ta": lambda opt: draw(fac, zk_ta(instr, opt, th, wl, sign), th),
        "opd": lambda opt: draw(fac, zk_defocus_opd(instr, opt, th, wl, sign), th),
        "rt": lambda opt: rt_scale * raytrace(instr, opt, th, wl, sign),
    }
    truths = {k: v for k, v in truths.items() if k in _W["opts"].get("truths", truths)}
    fit0_truth = {}
    saved = {"nominal": None, "plus": [], "minus": []}
    for tr, mk in truths.items():
        img = img_ta0 if tr == "ta" else mk(fid)
        if tr == "rt":
            saved["nominal"] = img
        fit0_truth[tr] = {fs: fit(fac, refs[fs], idx, th, img) for fs, idx in FIT_SETS.items()}

    out = dict(name=name, thx_deg=thx, thy_deg=thy, side=side, instrument=iname,
               wavelength_m=wl, fit_sets=FIT_SETS,
               nominal_fit_minus_intrinsic_um={
                   tr: {fs: ((f0[fs] - zf0[idx]) * 1e6).tolist()
                        for fs, idx in FIT_SETS.items()}
                   for tr, f0 in fit0_truth.items()},
               dofs=_W["dofs"], steps=[], A_dense=[], dTA=[], dOPD=[],
               A_direct={fs: [] for fs in FIT_SETS},
               A_direct_opd={fs: [] for fs in FIT_SETS},
               A_direct_rt={fs: [] for fs in FIT_SETS},
               A_direct_rt_2x={fs: [] for fs in FIT_SETS},
               L={fs: [] for fs in FIT_SETS}, L_cols={})

    def central(f, h):
        """Central difference (f(+h) - f(-h)) / 2h; cancels the second-order term."""
        return (f(+h) - f(-h)) / (2 * h)

    for d in _W["dofs"]:
        unit = 1e-3 if d in cs.ANGLE_DOF else 1.0
        s1 = (zk_infocus(_optic(d, unit), th, wl) - zf0)[NOLL_A] / unit
        step = unit * DZ_TARGET_UM * 1e-6 / max(np.sqrt(np.mean(s1 ** 2)) * unit, 1e-15)
        tas = {h: zk_ta(instr, _optic(d, h), th, wl, sign) for h in (step, -step)}
        imgs = {tr: {h: (draw(fac, tas[h], th) if tr == "ta" else mk(_optic(d, h)))
                     for h in (step, -step)} for tr, mk in truths.items()}
        out["steps"].append(step)
        if "rt" in truths:
            saved["plus"].append(imgs["rt"][step])
            saved["minus"].append(imgs["rt"][-step])
        out["A_dense"].append((central(lambda h: zk_infocus(_optic(d, h), th, wl), step)
                               [NOLL_A] * 1e6).tolist())
        out["dTA"].append((central(lambda h: tas[h], step)[NOLL_A] * 1e6).tolist())
        out["dOPD"].append((central(lambda h: zk_defocus_opd(instr, _optic(d, h), th, wl,
                                                             sign), step)[NOLL_A] * 1e6).tolist())
        for tr, key in (("ta", "A_direct"), ("opd", "A_direct_opd"), ("rt", "A_direct_rt")):
            if tr not in truths:
                continue
            for fs, idx in FIT_SETS.items():
                out[key][fs].append((central(
                    lambda h: fit(fac, refs[fs], idx, th, imgs[tr][h]), step) * 1e6).tolist())
        if _W["opts"].get("step_check") and "rt" in truths:
            # Same response at twice the step: separates fit noise from nonlinearity.
            im2 = {h: truths["rt"](_optic(d, h)) for h in (2 * step, -2 * step)}
            for fs, idx in FIT_SETS.items():
                out["A_direct_rt_2x"][fs].append((central(
                    lambda h: fit(fac, refs[fs], idx, th, im2[h]), 2 * step) * 1e6).tolist())
            for h in (2 * step, -2 * step):
                _W["optics"].pop((d, h), None)
        for h in (step, -step):
            _W["optics"].pop((d, h), None)
    if _W["opts"].get("image_dir") and saved["nominal"] is not None:
        save_images(_W["opts"]["image_dir"], out, saved, rt_scale, instr)
    if not _W["opts"].get("leakage", True):
        out["seconds"] = time.time() - t0
        return out

    delta = DZ_TARGET_UM * 1e-6
    for fs, idx in FIT_SETS.items():
        others = [j for j in NOLL_A if j not in idx]
        out["L_cols"][fs] = [int(j) for j in others]
        for j in others:
            def fit_inj(h):
                ta = ta0.copy()
                ta[j] += h
                return fit(fac, refs[fs], idx, th, draw(fac, ta, th))
            out["L"][fs].append(central(fit_inj, delta).tolist())

    # Ray-traced injections: one stop-plane OPD Zernike per Noll 4..78. With
    # M = dTA per unit injection and F = d(sparse fit) per unit injection,
    # G = F M^-1 is the fit response to each defocused TA term; G[:, fit] ~ I and
    # G[:, other] is the ray-traced L.
    M = np.zeros((len(NOLL_A), len(NOLL_A)))
    F = {fs: np.zeros((len(idx), len(NOLL_A))) for fs, idx in FIT_SETS.items()}
    for c, j in enumerate(NOLL_A):
        tas, fits = {}, {}
        for h in (delta, -delta):
            opt = with_screen(fid, j, h, instr)
            tas[h] = zk_ta(instr, opt, th, wl, sign)[NOLL_A]
            img = rt_scale * raytrace(instr, opt, th, wl, sign)
            fits[h] = {fs: fit(fac, refs[fs], idx, th, img) for fs, idx in FIT_SETS.items()}
        M[:, c] = (tas[delta] - tas[-delta]) / (2 * delta)
        for fs in FIT_SETS:
            F[fs][:, c] = (fits[delta][fs] - fits[-delta][fs]) / (2 * delta)
    Minv = np.linalg.inv(M)
    out["M_rt"] = M.tolist()
    out["F_rt"] = {fs: F[fs].tolist() for fs in FIT_SETS}
    out["G_rt"] = {fs: (F[fs] @ Minv).tolist() for fs in FIT_SETS}
    out["seconds"] = time.time() - t0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--configs", nargs="*", default=None, help="config names; default all")
    p.add_argument("--dofs", type=int, nargs="*", default=list(range(50)))
    p.add_argument("--label", default="donut_leakage")
    p.add_argument("--fit-sets", nargs="*", default=None, help="default all")
    p.add_argument("--truths", nargs="*", default=None, help="ta opd rt; default all")
    p.add_argument("--no-leakage", action="store_true", help="skip L and G_rt")
    p.add_argument("--step-check", action="store_true", help="also rt fits at 2x step")
    p.add_argument("--config-set", default="check", choices=["check", "fam", "cwfs", "all"])
    p.add_argument("--save-images", action="store_true", help="write ray-traced donuts")
    args = p.parse_args()
    opts = dict(fit_sets=args.fit_sets, leakage=not args.no_leakage,
                step_check=args.step_check)
    if args.truths:
        opts["truths"] = args.truths

    cfgs = [c for c in configs(args.config_set)
            if args.configs is None or c[0] in args.configs]
    run = study_run("smatrix/sparse_fit", args.label)
    if args.save_images:
        opts["image_dir"] = str(run.path)
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs, initializer=_init, initargs=(args.dofs, opts)) as pool:
            for r in pool.imap_unordered(run_config, cfgs):
                (run.path / f"leak_{r['name']}.json").write_text(json.dumps(r))
                print(f"{r['name']}: {r['seconds']:.0f} s", flush=True)
    else:
        _init(args.dofs, opts)
        for c in cfgs:
            r = run_config(c)
            (run.path / f"leak_{r['name']}.json").write_text(json.dumps(r))
            print(f"{r['name']}: {r['seconds']:.0f} s", flush=True)
    print("wrote", run.path)


if __name__ == "__main__":
    main()
