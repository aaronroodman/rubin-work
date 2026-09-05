#!/usr/bin/env python
"""Camera-gravity wavefront model for the MIW astig/coma investigation.

Builds the batoid_rubin telescope with gravitational flexure of the camera and
computes the induced double-Zernike (DZ) wavefront and per-Zernike focal-plane
maps, to compare against the measured intrinsic wavefront (MIW) OCS maps.

Two gravity ingredients (see smatrix/docs/miw_astig_coma_investigation.md and
smatrix/docs/status/future_issues.md #4):

1. LENS SURFACE FIGURE (what batoid_rubin ships) -- ``with_camera_gravity``
   applies the ``fea_legacy/L?S?zer.fits.gz`` files as a surface-figure
   perturbation (``surface + Zernike``) to the 6 lens optical surfaces
   (L1/L2/L3 entrance+exit).  This carries the lens surface DEFORMATION plus a
   per-surface piston/tilt (≈ rigid axial shift of each lens).  It does NOT
   move the focal plane and does NOT decenter the lenses.

2. RIGID-BODY SAG (``include_rb=True``; NOT applied by batoid_rubin) -- the
   ``fea_legacy/{L1,L2,L3,F,FP}RB.fits.gz`` files give the gravity rigid-body
   displacement [dx,dy,dz,Rx,Ry,Rz] of each lens, the filter, and the FOCAL
   PLANE.  These are dead data in the shipped model; here they are optionally
   applied via ``withGloballyShiftedOptic`` / ``withLocallyRotatedOptic`` (the
   same mechanism the AOS hexapod uses), so we can compare surface-figure-only
   vs the complete gravity picture.

   NOTE ON RB CONVENTION: the RB column order is taken as [dx,dy,dz,Rx,Ry,Rz]
   in mm / rad, with the same ZCS->batoid x,z flip as the zer files.  The lens
   decenters are large and mostly common-mode (bulk camera decenter, which the
   hexapod compensates); the DIFFERENTIAL part aberrates.  Signs/units should
   be sanity-checked against the IM source -- the validation script prints the
   incremental effect for exactly this reason.

Gravity convention (batoid_rubin): axial term scales as (cos z - 1)
[rotation-independent -> assigned to OCS]; lateral term scales as sin z and is
projected onto the rotating camera x/y by the rotator angle.  z = zenith angle;
we parameterise everything by ELEVATION = 90 - z.

Runs on the laptop (batoid_rubin) for the gravity model; the 50/34 correction
(camera_gravity_maps.py) additionally needs the LSST stack on the USDF RSP.
"""
from pathlib import Path
import os
import sys

import numpy as np
import galsim
import batoid

# reuse the exact OFC/smatrix DZ convention (field sampling, eps, nx, jmax, kmax)
_SMATRIX_CODE = Path(__file__).resolve().parents[2] / "smatrix" / "code"
sys.path.insert(0, str(_SMATRIX_CODE))
import compute_smatrix as C  # noqa: E402

FIELD_RAD = np.deg2rad(C.FIELD_RADIUS_DEG)
WL = {"u": 365, "g": 480, "r": 622, "i": 754, "z": 869, "y": 971}

# Builder kwargs matching smatrix/compute_smatrix (OFC ZCS convention).  For a
# pure-gravity perturbation (no AOS dof) these mostly do not matter, but we keep
# them identical so the DZ is in the same frame as the sensitivity matrix and
# the 50/34 SVD.
CAM_ELEMENTS = {  # RB file stem -> batoid compound-optic name
    "L1RB": "LSSTCamera.L1",
    "L2RB": "LSSTCamera.L2",
    "L3RB": "LSSTCamera.L3",
    "FRB": "LSSTCamera.Filter",
    "FPRB": "LSSTCamera.Detector",
}


def data_dir():
    d = os.environ.get("BATOID_RUBIN_DATA_DIR")
    if not d:
        raise RuntimeError("set BATOID_RUBIN_DATA_DIR")
    return Path(d)


def _resolve_dir(dd, name):
    """Full path under the data dir if it exists, else the bare name so
    batoid_rubin can infer its bundled default (envs differ: the laptop has the
    custom bend_zemax/bend_full, the USDF cvmfs install ships only its default)."""
    p = dd / name
    return str(p) if p.exists() else name


def base_builder(band="i", bend_dir=None):
    """Fiducial LSSTBuilder in the smatrix/OFC convention (no perturbations).

    Gravity does NOT use the bending-mode basis, so bend_dir is irrelevant to the
    result; we just need a valid one.  Prefer an existing custom basis, else fall
    back to batoid_rubin's bundled default.
    """
    from batoid_rubin import LSSTBuilder
    dd = data_dir()
    if bend_dir is None:
        for cand in ("bend_zemax", "bend_full", "bend"):
            if (dd / cand).exists():
                bend_dir = str(dd / cand)
                break
        else:
            bend_dir = "bend"   # batoid_rubin bundled default
    fid = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
    return LSSTBuilder(
        fid,
        fea_dir=_resolve_dir(dd, "fea_legacy"),
        bend_dir=bend_dir,
        use_m1m3_modes=None, use_m2_modes=None,
        dof_coord_system="ZCS",
        flip_m1m3_bending_modes=True, flip_m2_bending_modes=True,
        dof_angle_units="degree",
    )


def rb_gravity_displacements(fea_dir, zenith_rad, rotation_rad=0.0):
    """Gravity rigid-body displacement per camera element.

    Returns dict elem_name -> (shift[3] in meters, (rx,ry,rz) in rad), using the
    same gravity scaling as the zer files:
        g = row0*(cos z - 1) + (row1*cos rot + row2*sin rot)*sin z
    Column order assumed [dx,dy,dz,Rx,Ry,Rz]; ZCS->batoid flips x and z.
    """
    from astropy.io import fits
    ax = np.cos(zenith_rad) - 1.0
    lat = np.sin(zenith_rad)
    out = {}
    for stem, name in CAM_ELEMENTS.items():
        d = np.asarray(fits.getdata(f"{fea_dir}/{stem}.fits.gz"))
        g = d[0, 3:9] * ax + (d[1, 3:9] * np.cos(rotation_rad)
                              + d[2, 3:9] * np.sin(rotation_rad)) * lat
        dx, dy, dz, rx, ry, rz = g
        # mm -> m; ZCS->batoid flip of x and z (matching the zer convention)
        shift = np.array([-dx, dy, -dz]) * 1e-3
        rot = (-rx, ry, -rz)
        out[name] = (shift, rot)
    return out


def apply_rb_gravity(optic, disp):
    """Apply rigid-body gravity displacements (from rb_gravity_displacements)."""
    for name, (shift, (rx, ry, rz)) in disp.items():
        if np.any(shift):
            optic = optic.withGloballyShiftedOptic(name, shift)
        if rx or ry:
            optic = optic.withLocallyRotatedOptic(
                name, batoid.RotX(rx) @ batoid.RotY(ry))
    return optic


def gravity_telescope(builder, elevation_deg, rotation_deg=0.0, include_rb=False):
    """Built telescope with camera gravity at the given elevation/rotation."""
    zen = np.deg2rad(90.0 - elevation_deg)
    b = builder.with_camera_gravity(zenith=zen * galsim.radians,
                                    rotation=np.deg2rad(rotation_deg) * galsim.radians)
    optic = b.build()
    if include_rb:
        disp = rb_gravity_displacements(builder.fea_dir, zen, np.deg2rad(rotation_deg))
        optic = apply_rb_gravity(optic, disp)
    return optic


def gravity_dz(builder, elevation_deg, band="i", rotation_deg=0.0,
               include_rb=False, dz_design=None):
    """DZ change (waves) induced by camera gravity, differenced from design.

    Returns (dz_change, dz_design) each shape (kmax+1, jmax+1).
    """
    wl = WL[band] * 1e-9
    if dz_design is None:
        dz_design = C.double_zernike(builder.build(), FIELD_RAD, wl)
    tel = gravity_telescope(builder, elevation_deg, rotation_deg, include_rb)
    dz = C.double_zernike(tel, FIELD_RAD, wl)
    return dz - dz_design, dz_design


def field_basis(thx_deg, thy_deg, kmax=C.KMAX):
    """Field-Zernike basis on the DZ field disk (matches double_zernike)."""
    thx = np.deg2rad(np.asarray(thx_deg, float))
    thy = np.deg2rad(np.asarray(thy_deg, float))
    return galsim.zernike.zernikeBasis(kmax, thx, thy, R_outer=FIELD_RAD)


def dz_to_maps(dz, thx_deg, thy_deg, band="i", nolls=(4, 5, 6, 7, 8, 11)):
    """Evaluate DZ (waves) into per-pupil-Noll focal-plane maps in microns."""
    B = field_basis(thx_deg, thy_deg)          # (kmax+1, npts)
    wl_um = WL[band] * 1e-9 * 1e6
    return {j: (dz[:, j] @ B) * wl_um for j in nolls}


def map_stats(zmap, thx_deg, thy_deg, fr=C.FIELD_RADIUS_DEG):
    """RMS (um) and RMS-weighted mean field radial order of a focal-plane map."""
    v = np.asarray(zmap, float)
    good = np.isfinite(v)
    x = np.asarray(thx_deg)[good] / fr
    y = np.asarray(thy_deg)[good] / fr
    B = galsim.zernike.zernikeBasis(20, x, y, R_outer=1.0)
    c, *_ = np.linalg.lstsq(B.T, v[good], rcond=None)
    ns = np.array([galsim.zernike.noll_to_zern(j)[0] for j in range(1, len(c))])
    p = c[1:] ** 2
    return float(np.std(v[good])), float(np.sum(ns * p) / (p.sum() + 1e-30))


# --- Calculation 2 (revived): Z5-Z8 amplitude & field order vs elevation ------
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band", default="i")
    ap.add_argument("--elevations", type=float, nargs="+", default=[70, 45, 30])
    ap.add_argument("--rotation", type=float, default=0.0)
    ap.add_argument("--rings", type=int, default=6, help="DZ field rings (speed)")
    ap.add_argument("--spokes", type=int, default=15)
    ap.add_argument("--nx", type=int, default=63)
    args = ap.parse_args()

    # coarser DZ sampling for the amplitude scan (full-res for maps elsewhere)
    C.RINGS, C.SPOKES, C.NX = args.rings, args.spokes, args.nx

    b = base_builder(args.band)
    wl = WL[args.band] * 1e-9
    dz_design = C.double_zernike(b.build(), FIELD_RAD, wl)

    # a modest field grid for the stats
    t = np.linspace(-1.6, 1.6, 25)
    X, Y = np.meshgrid(t, t)
    inside = np.hypot(X, Y) <= 1.6
    xg, yg = X[inside], Y[inside]

    print(f"Camera-gravity Z5-Z8 (um RMS over field; field-order n) vs elevation, "
          f"band={args.band}, rot={args.rotation}")
    for rb in (False, True):
        tag = "surface+RB" if rb else "surface-only"
        print(f"\n--- {tag} ---")
        print(f"{'elev':>5} " + " ".join(f"{'Z%d' % j:>13}" for j in (4, 5, 6, 7, 8, 11)))
        for el in args.elevations:
            dz, _ = gravity_dz(b, el, args.band, args.rotation, rb, dz_design)
            maps = dz_to_maps(dz, xg, yg, args.band)
            cells = []
            for j in (4, 5, 6, 7, 8, 11):
                rms, nbar = map_stats(maps[j], xg, yg)
                cells.append(f"{rms:.3f}(n{nbar:.1f})")
            print(f"{el:5.0f} " + " ".join(f"{c:>13}" for c in cells))


if __name__ == "__main__":
    main()
