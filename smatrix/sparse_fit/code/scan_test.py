"""Fitted Zernikes against DOF amplitude h (k x step, k = -8..8), ray-traced and Danish-drawn.

Prints, per truth, the quadratic term and the RMS scatter about the best-fit
quadratic in h, each over the linear change at the largest h (vector norms over
the fitted terms).

Usage::

    python scan_test.py --config R44_SW0 --dof 23 --fit-set dense21
"""
import argparse

import numpy as np

import donut_leakage as dl

p = argparse.ArgumentParser()
p.add_argument("--config", default="R44_SW0")
p.add_argument("--dof", type=int, default=23)
p.add_argument("--fit-set", default="dense21")
p.add_argument("--nx", type=int, default=1024)
p.add_argument("--kmax", type=int, default=8)
p.add_argument("--step-mult", type=float, default=1.0)
a = p.parse_args()
dl.RT_NX = a.nx
dl._init([a.dof])
fid = dl._optic(-1, 0)
name, thx, thy, side, iname = [c for c in dl.configs() if c[0] == a.config][0]
I = dl._W["instr"][iname]
wl = I.wavelength[dl.BAND]
th = np.deg2rad([thx, thy])
sign = 1 if side == "extra" else -1
fac = dl.factory_for(I)
idx = dl.FIT_SETS[a.fit_set]
zf0 = dl.zk_infocus(fid, th, wl)
ta0 = dl.zk_ta(I, fid, th, wl, sign)
ref = ta0.copy()
ref[idx] -= zf0[idx]
sc = dl.draw(fac, ta0, th).sum() / dl.raytrace(I, fid, th, wl, sign).sum()
unit = 1e-3 if a.dof in dl.cs.ANGLE_DOF else 1.0
s1 = (dl.zk_infocus(dl._optic(a.dof, unit), th, wl) - zf0)[dl.NOLL_A] / unit
step = a.step_mult * dl.DZ_TARGET_UM * 1e-6 / np.sqrt(np.mean(s1 ** 2))
ks = np.arange(-a.kmax, a.kmax + 1)
F = {"rt": [], "ta": []}
for k in ks:
    opt = dl._optic(a.dof, k * step) if k else fid
    F["rt"].append(dl.fit(fac, ref, idx, th, sc * dl.raytrace(I, opt, th, wl, sign)))
    F["ta"].append(dl.fit(fac, ref, idx, th, dl.draw(fac, dl.zk_ta(I, opt, th, wl, sign), th)))
    dl._W["optics"].clear()
h = ks * step
for tr, f in F.items():
    f = np.array(f) * 1e6                                    # (k, term) µm
    V = np.vander(h, 3)
    coef, *_ = np.linalg.lstsq(V, f, rcond=None)
    dev = f - V @ coef
    lin = np.linalg.norm(coef[1]) * h[-1]          # |linear change| at the largest h, µm
    quad = np.linalg.norm(coef[0]) * h[-1] ** 2      # |quadratic term| there, µm
    print(f"{tr}: step {step:.3g}; |slope| {np.linalg.norm(coef[1]):.3f} um per unit; "
          f"at h_max: quadratic / linear {quad / lin:.3f}, "
          f"RMS scatter about quadratic / linear {np.sqrt(np.mean(np.sum(dev ** 2, 1))) / lin:.3f} "
          "(dimensionless, vector norms over fitted terms)")
