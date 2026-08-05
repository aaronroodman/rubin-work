#!/usr/bin/env python
"""DZ wavefront sensitivity to M1M3 / M2 thermal gradients (force-free).

batoid_rubin's thermal model applies FEA-derived surface displacement per unit
of each temperature term.  We compute the double-Zernike sensitivity the same
way as the bending modes, for the 7 thermal DOF:

  M1M3: TBulk (C), TxGrad, TyGrad, TzGrad, TrGrad (C/m)
  M2:   TzGrad, TrGrad (C/m)

Output: ../output/thermal_sensitivity.npz  (sens (31,29,7), labels, units)

Motivation: thermal gradients produce spherical (Z11/Z22) and, when
non-axisymmetric (x/y), astigmatism/coma -- with NO actuator force.  This is a
natural source of the MIW Z5-Z8/Z11/Z22 excess that force-limited bending modes
cannot reach.
"""
from pathlib import Path
import numpy as np
import batoid
from batoid_rubin import LSSTBuilder

import compute_smatrix as C

OUT = Path(__file__).resolve().parent.parent / "output"
DD = "/Users/roodman/LSST/batoid_rubin_data"
WL = 0.622e-6
WL_UM = 0.622

LABELS = ["M1M3 TBulk", "M1M3 TxGrad", "M1M3 TyGrad", "M1M3 TzGrad",
          "M1M3 TrGrad", "M2 TzGrad", "M2 TrGrad"]
UNITS = ["C", "C/m", "C/m", "C/m", "C/m", "C/m", "C/m"]


def builder_kwargs():
    kw = dict(C.BUILDER_KWARGS)
    kw["fea_dir"] = str(Path(DD) / kw["fea_dir"])
    kw["bend_dir"] = str(Path(DD) / "bend_zemax")
    return kw


def main():
    fid = batoid.Optic.fromYaml("LSST_r.yaml")
    field = np.deg2rad(C.FIELD_RADIUS_DEG)
    kw = builder_kwargs()
    dz0 = C.double_zernike(fid, field, WL)

    def sens_for(**kwargs):
        which = kwargs.pop("which")
        b = LSSTBuilder(fid, **kw)
        if which == "m1m3":
            b = b.with_m1m3_temperature(
                m1m3_TBulk=kwargs.get("TBulk", 0.0),
                m1m3_TxGrad=kwargs.get("Tx", 0.0), m1m3_TyGrad=kwargs.get("Ty", 0.0),
                m1m3_TzGrad=kwargs.get("Tz", 0.0), m1m3_TrGrad=kwargs.get("Tr", 0.0))
        else:
            b = b.with_m2_temperature(m2_TzGrad=kwargs.get("Tz", 0.0),
                                      m2_TrGrad=kwargs.get("Tr", 0.0))
        return (C.double_zernike(b.build(), field, WL) - dz0) * WL_UM

    perturb = [dict(which="m1m3", TBulk=1.0), dict(which="m1m3", Tx=1.0),
               dict(which="m1m3", Ty=1.0), dict(which="m1m3", Tz=1.0),
               dict(which="m1m3", Tr=1.0), dict(which="m2", Tz=1.0),
               dict(which="m2", Tr=1.0)]
    sens = np.stack([sens_for(**p) for p in perturb], axis=-1)   # (31,29,7)
    print("thermal sensitivity shape:", sens.shape)

    np.savez(OUT / "thermal_sensitivity.npz", sens=sens, labels=LABELS, units=UNITS)
    print("saved", OUT / "thermal_sensitivity.npz")
    # quick summary: field-constant dominant pupil Noll per thermal DOF
    for k, lab in enumerate(LABELS):
        fc = sens[1, :, k]
        top = np.argsort(-np.abs(fc))[:4]
        print(f"  {lab:12s} ({UNITS[k]}): "
              + ", ".join(f"Z{j}={fc[j]:+.4f}" for j in top))


if __name__ == "__main__":
    main()
