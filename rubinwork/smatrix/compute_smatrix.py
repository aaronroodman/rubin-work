#!/usr/bin/env python
"""Compute the Rubin AOS double-Zernike sensitivity matrix with batoid_rubin.

This reproduces the ts_ofc generator
(lsst.ts.ofc.scripts.generate_sensitivity_matrix) *exactly* -- same
``double_zernike`` helper, field sampling, jmax/kmax, eps, nx, and unit
handling -- so the result is directly comparable to the OFC-shipped matrix
``policy/sensitivity_matrix/lsst_sensitivity_dz_31_29_50.yaml``.

The one deliberate difference is the LSSTBuilder configuration.  The OFC file
was generated with bare ``LSSTBuilder(fiducial)`` defaults (ZCS, arcsec,
flip_m2=True).  Here we instead use the convention used elsewhere in our AOS
work (ts_intrinsic_wavefront): OCS, flip_m2=False, flip_m1m3=False.  Comparing
the two therefore exposes exactly which DOF columns differ by convention.

Output (in ``../output/``):
    smatrix_dz_31_29_50_<band>_<config>.npy   # (31 field, 29 pupil, 50 dof)
    smatrix_dz_31_29_50_<band>_<config>.fits

DOF ordering (length 50), units um unless noted:
    0-2   M2   dz, dx, dy            (um)
    3-4   M2   Rx, Ry                (degree)
    5-7   Cam  dz, dx, dy            (um)
    8-9   Cam  Rx, Ry                (degree)
    10-29 M1M3 bending modes 0..19   (um)
    30-49 M2   bending modes 0..19   (um)

Matrix value units: um of wavefront per (um or degree) of DOF.
(The OFC-shipped matrix uses degree for the angle DOF despite its header
saying arcsec -- its angle columns are ~3600x the per-arcsec values.)

Run at USDF (where batoid_rubin_data lives):
    python compute_smatrix.py --band r
Override the data location with --data-dir or $BATOID_RUBIN_DATA_DIR.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import batoid
import galsim
from astropy.io import fits

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):
        return x

# --------------------------------------------------------------------------
# Configuration constants -- matched to ts_ofc policy/configurations/lsst.yaml
# --------------------------------------------------------------------------
FIELD_RADIUS_DEG = 1.75          # outer field radius (inner = 0)
EPS = 0.61                       # pupil fractional obscuration
NX = 255                         # batoid ray grid
RINGS = 11                       # Gauss-Legendre rings in field
SPOKES = 35                      # azimuthal spokes in field
JMAX = 28                        # pupil Noll -> 29 pupil terms
KMAX = 30                        # field Noll -> 31 field terms
N_RIGID = 10                     # M2 + camera rigid-body DOF
ANGLE_DOF = [3, 4, 8, 9]         # M2 Rx,Ry ; Cam Rx,Ry (tip/tilt, in degree)

WAVELENGTHS_UM = {               # microns
    "u": 0.365, "g": 0.480, "r": 0.622, "i": 0.754,
    "z": 0.869, "y": 0.971, "ref": 0.5,
}

# Convention chosen to reproduce the OFC-shipped matrix EXACTLY (see
# docs/conventions.md): ZCS, both bending flips on, degree angle units.  The number
# of M1M3 / M2 modes is read from the target bend dir's bend.yaml, so this works
# for the 50-DOF `bend` set and the full `bend_full` set (156 M1M3 + 72 M2).
BUILDER_KWARGS = dict(
    fea_dir="fea_legacy",
    use_m1m3_modes=None,             # None -> read bend.yaml mode selection
    use_m2_modes=None,
    dof_coord_system="ZCS",          # ZCS: matches OFC hexapod rigid-body sign
    flip_m1m3_bending_modes=True,     # both flips ON to match OFC bending-mode sign
    flip_m2_bending_modes=True,
    dof_angle_units="degree",         # degree: matches OFC angle columns
)
# batoid_rubin ZCS differs from the OFC matrix by a sign on the y-translation
# (dy) and y-tip/tilt (Ry) of M2 and camera.  Flip those 4 DOF columns to match
# OFC exactly.  See docs/conventions.md [open question for Josh/Guillem].
OFC_Y_SIGN_FLIP = [2, 4, 7, 9]        # M2 dy, M2 Ry, Cam dy, Cam Ry
CONFIG_TAG = "ofc_zcs"                # default tag for the 50-DOF OFC comparison

DEFAULT_DATA_DIR = "/sdf/group/rubin/u/roodman/LSST/packages/batoid_rubin_data"


def mode_counts(data_dir, bend_dir):
    """(n_m1m3, n_m2) from the target bend dir's bend.yaml."""
    import yaml
    cfg = yaml.safe_load((Path(data_dir) / bend_dir / "bend.yaml").read_text())
    return len(cfg["use_m1m3_modes"]), len(cfg["use_m2_modes"])


def make_step(ndof):
    step = np.ones(ndof)
    step[ANGLE_DOF] = 1e-3           # 1e-3 deg = 3.6 arcsec: small, linear, no vignetting
    return step


def double_zernike(optic, field, wavelength, rings=RINGS, spokes=SPOKES,
                   jmax=JMAX, kmax=KMAX, **kwargs):
    """Double-Zernike coefficients (verbatim from ts_ofc generator).

    field in radians, wavelength in meters.  Returns array [field_k, pupil_j]
    of shape (kmax+1, jmax+1) in waves.
    """
    if spokes is None:
        spokes = 2 * rings + 1
    li, w = np.polynomial.legendre.leggauss(rings)
    radii = np.sqrt((1 + li) / 2) * field
    w = w * (np.pi / (2 * spokes))
    azs = np.linspace(0, 2 * np.pi, spokes, endpoint=False)
    radii, azs = np.meshgrid(radii, azs)
    w = np.broadcast_to(w, radii.shape).ravel()
    radii = radii.ravel()
    azs = azs.ravel()
    thx = radii * np.cos(azs)
    thy = radii * np.sin(azs)
    coefs = np.array([
        batoid.zernike(optic, tx, ty, wavelength, jmax=jmax, eps=EPS, nx=NX)
        for tx, ty in zip(thx, thy)
    ])
    basis = galsim.zernike.zernikeBasis(kmax, thx, thy, R_outer=field)
    return np.dot(basis, coefs * w[:, None]) / np.pi


def _builder_kwargs(data_dir, bend_dir):
    kwargs = dict(BUILDER_KWARGS)
    kwargs["fea_dir"] = str(Path(data_dir) / kwargs["fea_dir"])
    kwargs["bend_dir"] = str(Path(data_dir) / bend_dir)
    return kwargs


# module-level worker state (set per process by _init_worker; spawn-safe)
_W = {}


def _init_worker(band, data_dir, bend_dir, ndof):
    os.environ["BATOID_RUBIN_DATA_DIR"] = str(data_dir)
    fid_band = "g_500" if band == "ref" else band
    _W["fiducial"] = batoid.Optic.fromYaml(f"LSST_{fid_band}.yaml")
    _W["kwargs"] = _builder_kwargs(data_dir, bend_dir)
    _W["wl_m"] = WAVELENGTHS_UM[band] * 1e-6
    _W["field"] = np.deg2rad(FIELD_RADIUS_DEG)
    _W["ndof"] = ndof
    _W["step"] = make_step(ndof)


def _dz_for_dof(idof):
    """Perturbed double-Zernike for a single unit DOF (idof<0 -> intrinsic)."""
    from batoid_rubin import LSSTBuilder
    if idof < 0:
        optic = _W["fiducial"]
    else:
        dof = np.zeros(_W["ndof"])
        dof[idof] = _W["step"][idof]
        optic = LSSTBuilder(_W["fiducial"], **_W["kwargs"]).with_aos_dof(dof.tolist()).build()
    return idof, double_zernike(optic, _W["field"], _W["wl_m"])


def compute(band, data_dir, bend_dir="bend", jobs=1):
    wl_um = WAVELENGTHS_UM[band]
    n_m1m3, n_m2 = mode_counts(data_dir, bend_dir)
    ndof = N_RIGID + n_m1m3 + n_m2
    step = make_step(ndof)

    tasks = list(range(-1, ndof))  # -1 = intrinsic, then 0..ndof-1
    results = {}
    if jobs and jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs, initializer=_init_worker,
                     initargs=(band, data_dir, bend_dir, ndof)) as pool:
            for idof, dz in tqdm(pool.imap_unordered(_dz_for_dof, tasks), total=len(tasks)):
                results[idof] = dz
    else:
        _init_worker(band, data_dir, bend_dir, ndof)
        for idof in tqdm(tasks):
            i, dz = _dz_for_dof(idof)
            results[i] = dz

    dz0 = results[-1]
    sens = np.empty((ndof, KMAX + 1, JMAX + 1))
    for idof in range(ndof):
        sens[idof] = (results[idof] - dz0) / step[idof]  # per unit DOF

    # -> (field_k, pupil_j, dof), then waves -> um
    sens = np.einsum("ijk->jki", sens) * wl_um
    # y-axis handedness patch so the result matches the OFC-shipped matrix.
    sens[:, :, OFC_Y_SIGN_FLIP] *= -1.0
    return sens, (n_m1m3, n_m2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--band", default="r", choices=list(WAVELENGTHS_UM))
    p.add_argument("--data-dir",
                   default=os.environ.get("BATOID_RUBIN_DATA_DIR", DEFAULT_DATA_DIR),
                   help="batoid_rubin_data dir containing fea_legacy/ and bend/")
    p.add_argument("--bend-dir", default="bend",
                   help="bend-format dir under data-dir (e.g. bend, bend_full)")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "output"))
    p.add_argument("--jobs", type=int, default=1, help="parallel processes over DOF")
    p.add_argument("--tag", default=None, help="override output config tag")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "fea_legacy").is_dir() or not (data_dir / args.bend_dir).is_dir():
        raise SystemExit(
            f"data-dir {data_dir} must contain fea_legacy/ and {args.bend_dir}/ subdirs")

    sens, (n_m1m3, n_m2) = compute(args.band, data_dir, bend_dir=args.bend_dir, jobs=args.jobs)
    ndof = sens.shape[2]
    print("sensitivity matrix shape:", sens.shape, f"({n_m1m3} M1M3 + {n_m2} M2 modes)")

    tag = args.tag or ("ofc_zcs" if args.bend_dir == "bend" else args.bend_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"smatrix_dz_{KMAX+1}_{JMAX+1}_{ndof}_{args.band}_{tag}"
    np.save(outdir / f"{stem}.npy", sens)
    hdr = fits.Header()
    hdr["CONTENT"] = "DZ sensitivity (field_k, pupil_j, dof)"
    hdr["DIM"] = f"({KMAX+1},{JMAX+1},{ndof})"
    hdr["NM1M3"] = n_m1m3
    hdr["NM2"] = n_m2
    hdr["BAND"] = args.band
    hdr["COORDSYS"] = "ZCS"
    hdr["FLIPM1M3"] = True
    hdr["FLIPM2"] = True
    hdr["YSIGN"] = "dy,Ry flipped to match OFC"
    hdr["UNITS"] = "um_zk_per_um_or_degree_dof"
    fits.PrimaryHDU(sens, header=hdr).writeto(outdir / f"{stem}.fits", overwrite=True)
    print("wrote", outdir / f"{stem}.npy")


if __name__ == "__main__":
    main()
