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
NDOF = 50
N_M1M3 = 20
N_M2 = 20

WAVELENGTHS_UM = {               # microns
    "u": 0.365, "g": 0.480, "r": 0.622, "i": 0.754,
    "z": 0.869, "y": 0.971, "ref": 0.5,
}

# Convention chosen to reproduce the OFC-shipped matrix EXACTLY (see
# CONVENTIONS.md).  use_m1m3_modes/use_m2_modes=None -> read bend.yaml, the
# standard AOS 20-of-30 raw SVD modes: M1M3=[0..18, 26], M2=[0..16, 25, 26, 27].
BUILDER_KWARGS = dict(
    fea_dir="fea_legacy",
    bend_dir="bend",
    use_m1m3_modes=None,
    use_m2_modes=None,
    dof_coord_system="ZCS",          # ZCS: matches OFC hexapod rigid-body sign
    flip_m1m3_bending_modes=True,     # both flips ON to match OFC bending-mode sign
    flip_m2_bending_modes=True,
    dof_angle_units="degree",         # degree: matches OFC angle columns
)
# batoid_rubin ZCS differs from the OFC matrix by a sign on the y-translation
# (dy) and y-tip/tilt (Ry) of M2 and camera.  Flip those 4 DOF columns to match
# OFC exactly.  See CONVENTIONS.md [open question for Josh/Guillem].
OFC_Y_SIGN_FLIP = [2, 4, 7, 9]        # M2 dy, M2 Ry, Cam dy, Cam Ry
CONFIG_TAG = "ofc_zcs"                # identifies the convention in output names

# Per-DOF finite-difference step, in each DOF's native unit (um or degree).
# Microns/bending: 1.0 is safe and linear.  Angle DOF (3,4,8,9) are in DEGREES
# here; a 1-degree tilt vignettes the telescope (rank-0 Zernike fit), so use a
# small angle step and divide it out -> per-degree sensitivity, matching OFC.
STEP = np.ones(NDOF)
STEP[[3, 4, 8, 9]] = 1e-3         # 1e-3 deg = 3.6 arcsec: small, linear, no vignetting

DEFAULT_DATA_DIR = "/sdf/group/rubin/u/roodman/LSST/packages/batoid_rubin_data"


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


def _builder_kwargs(data_dir):
    kwargs = dict(BUILDER_KWARGS)
    kwargs["fea_dir"] = str(Path(data_dir) / kwargs["fea_dir"])
    kwargs["bend_dir"] = str(Path(data_dir) / kwargs["bend_dir"])
    return kwargs


# module-level worker state (set per process by _init_worker)
_W = {}


def _init_worker(band, data_dir):
    os.environ["BATOID_RUBIN_DATA_DIR"] = str(data_dir)
    fid_band = "g_500" if band == "ref" else band
    _W["fiducial"] = batoid.Optic.fromYaml(f"LSST_{fid_band}.yaml")
    _W["kwargs"] = _builder_kwargs(data_dir)
    _W["wl_m"] = WAVELENGTHS_UM[band] * 1e-6
    _W["field"] = np.deg2rad(FIELD_RADIUS_DEG)


def _dz_for_dof(idof):
    """Perturbed double-Zernike for a single unit DOF (idof<0 -> intrinsic)."""
    from batoid_rubin import LSSTBuilder
    if idof < 0:
        optic = _W["fiducial"]
    else:
        dof = np.zeros(NDOF)
        dof[idof] = STEP[idof]
        optic = LSSTBuilder(_W["fiducial"], **_W["kwargs"]).with_aos_dof(dof.tolist()).build()
    return idof, double_zernike(optic, _W["field"], _W["wl_m"])


def compute(band, data_dir, jobs=1):
    wl_um = WAVELENGTHS_UM[band]

    tasks = list(range(-1, NDOF))  # -1 = intrinsic, then 0..49
    results = {}
    if jobs and jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs, initializer=_init_worker, initargs=(band, data_dir)) as pool:
            for idof, dz in tqdm(pool.imap_unordered(_dz_for_dof, tasks), total=len(tasks)):
                results[idof] = dz
    else:
        _init_worker(band, data_dir)
        for idof in tqdm(tasks):
            i, dz = _dz_for_dof(idof)
            results[i] = dz

    dz0 = results[-1]
    sens = np.empty((NDOF, KMAX + 1, JMAX + 1))
    for idof in range(NDOF):
        sens[idof] = (results[idof] - dz0) / STEP[idof]  # per unit DOF

    # -> (field_k=31, pupil_j=29, dof=50), then waves -> um
    sens = np.einsum("ijk->jki", sens) * wl_um
    # y-axis handedness patch so the result matches the OFC-shipped matrix.
    sens[:, :, OFC_Y_SIGN_FLIP] *= -1.0
    return sens


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--band", default="r", choices=list(WAVELENGTHS_UM))
    p.add_argument("--data-dir",
                   default=os.environ.get("BATOID_RUBIN_DATA_DIR", DEFAULT_DATA_DIR),
                   help="batoid_rubin_data dir containing fea_legacy/ and bend/")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "output"))
    p.add_argument("--jobs", type=int, default=1, help="parallel processes over DOF")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "fea_legacy").is_dir() or not (data_dir / "bend").is_dir():
        raise SystemExit(
            f"data-dir {data_dir} must contain fea_legacy/ and bend/ subdirs")

    sens = compute(args.band, data_dir, jobs=args.jobs)
    print("sensitivity matrix shape:", sens.shape)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"smatrix_dz_31_29_50_{args.band}_{CONFIG_TAG}"
    np.save(outdir / f"{stem}.npy", sens)
    hdr = fits.Header()
    hdr["CONTENT"] = "DZ sensitivity (field_k, pupil_j, dof)"
    hdr["DIM"] = "(31,29,50)"
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
