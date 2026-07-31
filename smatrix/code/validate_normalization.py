#!/usr/bin/env python
"""Validate the geometric normalization weights (f, r) against the official
ts_config_mttcs v13 file, for the nominal 50 DOF.

Official v13 weights: w_i = r_i^0.5 * f_i^-0.5 (sqrt(r/f)), field-averaged
(quadrature) FWHM over pupil Noll Z4..Z22.  File:
  ts_config_mttcs/MTAOS/v13/ofc/normalization_weights/range0.5_fwhm-0.15.yaml
(the "-0.15" in the filename is a typo; the header states beta = -0.5).

We reproduce it standalone from our batoid_rubin OFC-basis DZ sensitivity matrix:
  f = compute_f_quadrature(sens, Noll 4-22)         [arcsec / DOF-unit]
  r = (force_range/20)/max|force-per-um|, rb_stroke [DOF-unit]
  w = sqrt(r/f)
Result: all 50 DOF agree with the official weights to < 0.1%.
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

import normalization_weights as NW

# default locations (edit for USDF: /sdf/group/rubin/u/roodman/LSST/packages/...)
TS_OFC = "/Users/roodman/Astrophysics/Claude/packages/ts_ofc/python/lsst/ts/ofc/policy"
V13 = "/Users/roodman/Astrophysics/Claude/packages/ts_config_mttcs/MTAOS/v13/ofc"

# ts_ofc compute_range_weights uses the FIRST 20 raw force columns
# (BendModeToForce.rot_mat = force_data[:, 3:23]) -- NOT the bend.yaml sensitivity
# selection [0..18,26].  So the shipped range for B20 uses raw mode 19's force
# while the B20 sensitivity uses raw mode 26 (a known ts_ofc quirk).  Use the
# first-20 columns here to reproduce the official v13 weights exactly.
SEL_M1M3 = list(range(20))
SEL_M2 = list(range(20))

DOF_LABELS = (["M2 dz", "M2 dx", "M2 dy", "M2 Rx", "M2 Ry",
               "Cam dz", "Cam dx", "Cam dy", "Cam Rx", "Cam Ry"]
              + [f"M1M3 B{i}" for i in range(1, 21)]
              + [f"M2 B{i}" for i in range(1, 21)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smatrix", default=None,
                   help="50-DOF OFC-basis DZ matrix .npy")
    p.add_argument("--ts-ofc", default=TS_OFC)
    p.add_argument("--v13", default=V13)
    args = p.parse_args()

    outdir = Path(__file__).resolve().parent.parent / "output"
    sm = args.smatrix or (outdir / "smatrix_dz_31_29_50_r_ofc_zcs.npy")
    sens = np.load(sm)

    # f (FWHM, arcsec/DOF), field-averaged quadrature, Noll 4-22
    f = NW.compute_f_quadrature(sens, rings=5, spokes=6, znmin=4, znmax=22)

    # r (range)
    fm1 = np.array(yaml.safe_load(open(f"{args.ts_ofc}/M1M3/M1M3_1um_156_force.yaml")))[:, 3:]
    fm2 = np.array(yaml.safe_load(open(f"{args.ts_ofc}/M2/M2_1um_72_force.yaml")))[:, 3:]
    r = NW.compute_range_weights(fm1, fm2, SEL_M1M3, SEL_M2)

    # geometric weights
    w = NW.geometric_weights(r, f)               # sqrt(r/f)

    # official reference
    w_off = np.array(yaml.safe_load(
        open(f"{args.v13}/normalization_weights/range0.5_fwhm-0.15.yaml")))

    ratio = w / w_off
    print("Geometric normalization sqrt(r/f) vs official v13 (50 DOF):")
    print(f"  max deviation = {np.max(np.abs(ratio - 1)) * 100:.3f}%   "
          f"median ratio = {np.median(ratio):.5f}")
    print(f"  within 0.2%: {np.sum(np.abs(ratio - 1) < 0.002)}/50")
    print(f"\n  {'DOF':<9}{'f(asec/DOF)':>13}{'r':>12}{'w=sqrt(r/f)':>13}{'official':>12}{'ratio':>8}")
    for i in [0, 3, 5, 8, 10, 28, 29, 30, 45, 49]:
        print(f"  {DOF_LABELS[i]:<9}{f[i]:>13.4f}{r[i]:>12.4f}{w[i]:>13.4f}"
              f"{w_off[i]:>12.4f}{ratio[i]:>8.4f}")

    np.savez(outdir / "normalization_50dof.npz", f=f, r=r, w=w, w_official=w_off,
             labels=DOF_LABELS)
    print("\nsaved", outdir / "normalization_50dof.npz")


if __name__ == "__main__":
    main()
