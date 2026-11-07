#!/usr/bin/env python
"""Per-field-point probe of batoid_rubin against the ts_ofc sensitivity matrix.

For each DOF, the pupil Zernike change per unit DOF is computed directly with
batoid at a few field points and compared with the ts_ofc double-Zernike (DZ)
matrix evaluated at the same points. This is cheap (seconds per DOF) and is
used to test hypotheses for the small residuals before a full DZ rebuild.

Usage::

    python smatrix/ofc_reproduction/code/probe_field_points.py --label baseline
    python smatrix/ofc_reproduction/code/probe_field_points.py --label angstep \
        --dofs 3 4 8 9 --angle-step 2.7778e-4
"""
import argparse
import json

import numpy as np
import batoid
import galsim

from rubinwork.common import study_run
import rubinwork.smatrix.compute_smatrix as cs
from compare_ofc import DOF_LABELS, find_ofc_file, load_ofc

# Field points (deg, OCS-free: only radius and angle matter for the DZ basis).
OFC_INTRINSIC = ("/sdf/home/r/roodman/u/LSST/packages/ts_ofc/python/lsst/ts/ofc/"
                 "policy/intrinsic_zernikes/lsst/intrinsic_zk_{band}_31_29.yaml")
FIELD_PTS_DEG = np.array([[0.0, 0.0], [0.75, 0.0], [0.0, -0.75],
                          [-0.75, 0.75], [1.18, 1.18]])


def ofc_at(ofc, thx_deg, thy_deg):
    """OFC DZ evaluated at field points; returns (npts, 29 pupil, 50 dof) in um."""
    basis = galsim.zernike.zernikeBasis(
        ofc.shape[0] - 1, np.deg2rad(thx_deg), np.deg2rad(thy_deg),
        R_outer=np.deg2rad(cs.FIELD_RADIUS_DEG))          # (31, npts)
    return np.einsum("kp,kjd->pjd", basis, ofc)


def zk_at(optic, pts_deg, wl_m, nx):
    return np.array([batoid.zernike(optic, *np.deg2rad(p), wl_m, jmax=cs.JMAX,
                                    eps=cs.EPS, nx=nx) for p in pts_deg])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True)
    p.add_argument("--band", default="r")
    p.add_argument("--dofs", type=int, nargs="*", default=list(range(50)))
    p.add_argument("--angle-step", type=float, default=1e-3, help="deg")
    p.add_argument("--lin-step", type=float, default=1.0, help="um")
    p.add_argument("--nx", type=int, default=cs.NX)
    p.add_argument("--fiducial", default=None,
                   help="batoid yaml for the fiducial optic (default LSST_<band>.yaml)")
    p.add_argument("--central", action="store_true",
                   help="central difference (+/- step) instead of one-sided")
    args = p.parse_args()

    run = study_run("smatrix/ofc_reproduction", f"probe_{args.label}")
    cs._init_worker(args.band, cs.DEFAULT_DATA_DIR, "bend", 50)
    if args.fiducial:
        cs._W["fiducial"] = batoid.Optic.fromYaml(args.fiducial)
    from batoid_rubin import LSSTBuilder
    wl_um = cs.WAVELENGTHS_UM[args.band]
    wl_m = wl_um * 1e-6
    ofc = load_ofc(find_ofc_file(None))
    ofc_pts = ofc_at(ofc, FIELD_PTS_DEG[:, 0], FIELD_PTS_DEG[:, 1])

    def build(idof, s):
        dof = np.zeros(50)
        dof[idof] = s
        return LSSTBuilder(cs._W["fiducial"], **cs._W["kwargs"]).with_aos_dof(
            dof.tolist()).build()

    z0 = zk_at(cs._W["fiducial"], FIELD_PTS_DEG, wl_m, args.nx)
    intr = ofc_at(load_ofc(OFC_INTRINSIC.format(band="" if args.band == "ref" else args.band))[:, :, None],
                  FIELD_PTS_DEG[:, 0], FIELD_PTS_DEG[:, 1])[:, :, 0]   # waves
    di = (z0 - intr)[:, 4:] * wl_um
    print(f"intrinsic vs OFC: rms diff {np.sqrt(np.mean(di**2)):.3e} um of wavefront "
          f"(OFC intrinsic rms {np.sqrt(np.mean((intr[:, 4:] * wl_um)**2)):.3e} um)")
    rows = []
    print(f"{'idof':>4} {'DOF':<10} {'rms_ofc_um':>10} {'rms_diff_um':>11} "
          f"{'rel':>8} {'fit_scale':>9}")
    for d in args.dofs:
        s = args.angle_step if d in cs.ANGLE_DOF else args.lin_step
        zp = zk_at(build(d, s), FIELD_PTS_DEG, wl_m, args.nx)
        if args.central:
            zm = zk_at(build(d, -s), FIELD_PTS_DEG, wl_m, args.nx)
            sens = (zp - zm) / (2 * s) * wl_um
        else:
            sens = (zp - z0) / s * wl_um
        if d in cs.OFC_Y_SIGN_FLIP:
            sens = -sens
        a, b = sens[:, 4:], ofc_pts[:, 4:, d]          # Noll 4..28
        diff = a - b
        scale = np.sum(a * b) / np.sum(b * b)          # ours = scale * OFC
        rec = dict(idof=d, dof=DOF_LABELS[d], rms_ofc=float(np.sqrt(np.mean(b**2))),
                   rms_diff=float(np.sqrt(np.mean(diff**2))), scale=float(scale),
                   diff_by_noll=np.sqrt(np.mean(diff**2, axis=0)).tolist(),
                   ofc_by_noll=np.sqrt(np.mean(b**2, axis=0)).tolist())
        rows.append(rec)
        print(f"{d:>4} {DOF_LABELS[d]:<10} {rec['rms_ofc']:10.3e} "
              f"{rec['rms_diff']:11.3e} {rec['rms_diff']/rec['rms_ofc']:8.2e} "
              f"{scale:9.5f}", flush=True)
    out = run.path / "probe.json"
    out.write_text(json.dumps(dict(args=vars(args), field_pts_deg=FIELD_PTS_DEG.tolist(),
                                   rows=rows), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
