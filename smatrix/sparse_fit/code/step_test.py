"""Ray-traced Danish fit response per DOF against step size and ray-grid density.

Prints |A(s) - A(ref)| / |A(ref)| (dimensionless) for step multiples s of the
nominal step, ref being 1x at RT_NX 1024; the central difference uses +-s.

Usage::

    python step_test.py --config R44_SW0 --dofs 1 7 23 46 --fit-set dense21
"""
import argparse

import numpy as np

import donut_leakage as dl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="R44_SW0")
    p.add_argument("--dofs", type=int, nargs="+", default=[1, 7, 23, 46])
    p.add_argument("--fit-set", default="dense21")
    p.add_argument("--mult", type=float, nargs="+", default=[0.5, 1, 2, 4, 8])
    p.add_argument("--nx", type=int, nargs="+", default=[1024, 2048])
    args = p.parse_args()
    dl._init(args.dofs)
    fid = dl._optic(-1, 0)
    name, thx, thy, side, iname = [c for c in dl.configs() if c[0] == args.config][0]
    instr = dl._W["instr"][iname]
    wl = instr.wavelength[dl.BAND]
    th = np.deg2rad([thx, thy])
    sign = 1 if side == "extra" else -1
    fac = dl.factory_for(instr)
    idx = dl.FIT_SETS[args.fit_set]
    zf0 = dl.zk_infocus(fid, th, wl)
    ta0 = dl.zk_ta(instr, fid, th, wl, sign)
    ref = ta0.copy()
    ref[idx] -= zf0[idx]
    tot = dl.draw(fac, ta0, th).sum()
    for d in args.dofs:
        unit = 1e-3 if d in dl.cs.ANGLE_DOF else 1.0
        s1 = (dl.zk_infocus(dl._optic(d, unit), th, wl) - zf0)[dl.NOLL_A] / unit
        step = unit * dl.DZ_TARGET_UM * 1e-6 / np.sqrt(np.mean(s1 ** 2)) / unit
        res = {}
        for nx in args.nx:
            dl.RT_NX = nx
            scale = tot / dl.raytrace(instr, fid, th, wl, sign).sum()
            for m in (args.mult if nx == args.nx[0] else [1]):
                h = m * step
                f = {s: dl.fit(fac, ref, idx, th,
                               scale * dl.raytrace(instr, dl._optic(d, s * h), th, wl, sign))
                     for s in (1, -1)}
                res[(nx, m)] = (f[1] - f[-1]) / (2 * h)
                dl._W["optics"].clear()
        dl.RT_NX = args.nx[0]
        A0 = res[(args.nx[0], 1)]
        n = np.linalg.norm(A0)
        print(f"dof {d}: |A| {n * 1e6:.3f} um per unit; |A(s) - A(1x, nx {args.nx[0]})| / |A|: "
              + ", ".join(f"{m}x/nx{nx} {np.linalg.norm(a - A0) / n:.4f}"
                          for (nx, m), a in res.items() if (nx, m) != (args.nx[0], 1)),
              flush=True)


if __name__ == "__main__":
    main()
