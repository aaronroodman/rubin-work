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
- ``A_direct``: the change in the sparse fit per unit DOF, from donuts of the
  perturbed telescope;
- ``L``: the change in the sparse fit per µm injected into one unfitted term
  (Noll 4..78 outside the fit set), dimensionless;
- ``dTA``, ``dOPD``: change per unit DOF of the defocused ``zernikeTA`` fit and of
  the defocused wavefront (``zernikeGQ``), Noll 4..78, µm per µm or degree.

The check is ``A_direct = D[fit] + L @ D[other]`` with D one of A_dense, dTA, dOPD.

Usage::

    python smatrix/sparse_fit/code/donut_leakage.py --jobs 16
    python smatrix/sparse_fit/code/donut_leakage.py --configs sci_p0_p0_extra --dofs 0 3 --label smoke
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import batoid
import danish
from scipy.optimize import least_squares

from rubinwork.common import study_run
import rubinwork.smatrix.compute_smatrix as cs

warnings.simplefilter("ignore")

BAND = "r"
JMAX_MODEL = 78                  # ts_wep nollIndicesModel = 0..78
NOLL_A = np.arange(4, JMAX_MODEL + 1)  # bending modes reach above Noll 28; their leakage counts
FIT_SETS = {
    "prod": [4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 20, 21, 27, 28],      # fam_tables d13-sparse
    "prod_z22": [4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 20, 21, 22, 27, 28],
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


def configs():
    """Field point and defocus configurations: (name, thx deg, thy deg, side, instrument)."""
    out = []
    for name, (x, y) in CORNERS.items():
        side = "extra" if name.endswith("SW0") else "intra"
        out.append((name, x, y, side, "LsstCam"))
    tag = {-0.75: "m", 0.0: "0", 0.75: "p"}
    for x in SCI_GRID_DEG:
        for y in SCI_GRID_DEG:
            for side in ("intra", "extra"):
                out.append((f"sci_{tag[x]}{tag[y]}_{side}", x, y, side, "LsstFamCam"))
    return out


_W = {}


def _init(dofs):
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
        pixel_scale=instr.pixelSize * BINNING, spider_angle=0.0)


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
    img0 = draw(fac, ta0, th)
    fit0 = {fs: fit(fac, refs[fs], idx, th, img0) for fs, idx in FIT_SETS.items()}

    out = dict(name=name, thx_deg=thx, thy_deg=thy, side=side, instrument=iname,
               wavelength_m=wl, fit_sets=FIT_SETS,
               nominal_fit_minus_intrinsic_um={
                   fs: ((fit0[fs] - zf0[idx]) * 1e6).tolist()
                   for fs, idx in FIT_SETS.items()},
               dofs=_W["dofs"], steps=[], A_dense=[], dTA=[], dOPD=[],
               A_direct={fs: [] for fs in FIT_SETS},
               L={fs: [] for fs in FIT_SETS}, L_cols={})

    def central(f, h):
        """Central difference (f(+h) - f(-h)) / 2h; cancels the second-order term."""
        return (f(+h) - f(-h)) / (2 * h)

    for d in _W["dofs"]:
        unit = 1e-3 if d in cs.ANGLE_DOF else 1.0
        s1 = (zk_infocus(_optic(d, unit), th, wl) - zf0)[NOLL_A] / unit
        step = unit * DZ_TARGET_UM * 1e-6 / max(np.sqrt(np.mean(s1 ** 2)) * unit, 1e-15)
        tas = {h: zk_ta(instr, _optic(d, h), th, wl, sign) for h in (step, -step)}
        imgs = {h: draw(fac, tas[h], th) for h in tas}
        out["steps"].append(step)
        out["A_dense"].append((central(lambda h: zk_infocus(_optic(d, h), th, wl), step)
                               [NOLL_A] * 1e6).tolist())
        out["dTA"].append((central(lambda h: tas[h], step)[NOLL_A] * 1e6).tolist())
        out["dOPD"].append((central(lambda h: zk_defocus_opd(instr, _optic(d, h), th, wl,
                                                             sign), step)[NOLL_A] * 1e6).tolist())
        for fs, idx in FIT_SETS.items():
            out["A_direct"][fs].append((central(
                lambda h: fit(fac, refs[fs], idx, th, imgs[h]), step) * 1e6).tolist())
        for h in (step, -step):
            _W["optics"].pop((d, h), None)

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
    out["seconds"] = time.time() - t0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--configs", nargs="*", default=None, help="config names; default all")
    p.add_argument("--dofs", type=int, nargs="*", default=list(range(50)))
    p.add_argument("--label", default="donut_leakage")
    args = p.parse_args()

    cfgs = [c for c in configs() if args.configs is None or c[0] in args.configs]
    run = study_run("smatrix/sparse_fit", args.label)
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs, initializer=_init, initargs=(args.dofs,)) as pool:
            results = list(pool.imap_unordered(run_config, cfgs))
    else:
        _init(args.dofs)
        results = [run_config(c) for c in cfgs]
    for r in results:
        (run.path / f"leak_{r['name']}.json").write_text(json.dumps(r))
        print(f"{r['name']}: {r['seconds']:.0f} s", flush=True)
    print("wrote", run.path)


if __name__ == "__main__":
    main()
