"""Diagnose the closure floor of A_sparse = G_rt @ dTA against ray-traced fits.

For each DOF, an "equivalent screen" is a stop-plane OPD screen tuned so its change in
the defocused TA fit equals the DOF's, Noll 4..J. Ray-traced sparse fits per unit DOF:

- ``A_rt``: the real DOF;
- ``A_scr[J]``: the equivalent screen to Noll J.

A_rt - A_scr[J] is what TA to Noll J misses (higher terms, vignetting); A_scr[78]
against G_rt @ dTA tests the linear model itself. Also prints the RMS of dTA above
Noll 78 and the change in dTA with the denser TA sampling.

Usage::

    python diag_floor.py --configs R44_SW0 sci_pp_extra --dofs 1 3 15 35 36
"""
import argparse
import json
import pathlib

import numpy as np
import batoid

import donut_leakage as dl
from rubinwork.common import study_run

NRAD_HI = 20


def zk_ta_hi(instr, optic, th, wl, sign, jmax):
    sh = instr._applyBatoidOffsets(optic, sign)
    return batoid.zernikeTA(sh, *th, wl, jmax=jmax, eps=sh.pupilObscuration,
                            focal_length=instr.focalLength, nrad=NRAD_HI,
                            naz=int(2 * np.pi * NRAD_HI)) * wl


def screen_optic(optic, coef, instr):
    scr = batoid.OPDScreen(batoid.Plane(), batoid.Zernike(
        coef, R_outer=instr.radius, R_inner=instr.radius * instr.obscuration),
        name="InjScreen", coordSys=optic.stopSurface.coordSys,
        inMedium=optic.inMedium, outMedium=optic.inMedium)
    return optic.withInsertedOptic(before="M1", item=scr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["R44_SW0", "sci_pp_extra"])
    p.add_argument("--dofs", type=int, nargs="+", default=[1, 3, 15, 35, 36])
    p.add_argument("--jmax", type=int, nargs="+", default=[78, 153])
    p.add_argument("--label", default="diag_floor")
    p.add_argument("--rt-run", default="../output/20261110_donut_leakage_rt_l")
    args = p.parse_args()
    run = study_run("smatrix/sparse_fit", args.label)
    dl._init(args.dofs)
    fid = dl._optic(-1, 0)
    fs = "prod"
    idx = dl.FIT_SETS[fs]
    jtop = max(args.jmax)
    out = {}
    for cfg in dl.configs():
        name, thx, thy, side, iname = cfg
        if name not in args.configs:
            continue
        prev = json.load(open(pathlib.Path(args.rt_run) / f"leak_{name}.json"))
        G = np.array(prev["G_rt"][fs])
        instr = dl._W["instr"][iname]
        wl = instr.wavelength[dl.BAND]
        th = np.deg2rad([thx, thy])
        sign = 1 if side == "extra" else -1
        fac = dl.factory_for(instr)
        zf0 = dl.zk_infocus(fid, th, wl)
        ref = dl.zk_ta(instr, fid, th, wl, sign)
        ref[idx] -= zf0[idx]
        img0 = dl.draw(fac, dl.zk_ta(instr, fid, th, wl, sign), th)
        rt_scale = img0.sum() / dl.raytrace(instr, fid, th, wl, sign).sum()
        ta_fid = zk_ta_hi(instr, fid, th, wl, sign, jtop)

        def rtfit(opt):
            return dl.fit(fac, ref, idx, th, rt_scale * dl.raytrace(instr, opt, th, wl, sign))

        print(f"== {name}", flush=True)
        for d in args.dofs:
            k = prev["dofs"].index(d)
            step = prev["steps"][k]
            res = dict(step=step)
            A_rt = (rtfit(dl._optic(d, step)) - rtfit(dl._optic(d, -step))) / (2 * step)
            dta_hi = (zk_ta_hi(instr, dl._optic(d, step), th, wl, sign, jtop)
                      - zk_ta_hi(instr, dl._optic(d, -step), th, wl, sign, jtop)) / (2 * step)
            dta78 = np.array(prev["dTA"][k]) * 1e-6
            pred = G @ dta78
            n = np.linalg.norm(A_rt)
            res["above78_rms_over_4to78_rms"] = float(
                np.sqrt(np.mean(dta_hi[79:] ** 2)) / np.sqrt(np.mean(dta_hi[4:79] ** 2)))
            res["dTA_sampling_change"] = float(
                np.linalg.norm(dta_hi[4:79] - dta78) / np.linalg.norm(dta78))
            res["A_rt_vs_saved"] = float(
                np.linalg.norm(A_rt - np.array(prev["A_direct_rt"][fs][k]) * 1e-6) / n)
            res["pred_G_vs_A_rt"] = float(np.linalg.norm(A_rt - pred) / n)
            for J in args.jmax:
                target = dta_hi[:J + 1].copy()
                target[:4] = 0.0
                c = -target.copy()
                for _ in range(3):
                    got = (zk_ta_hi(instr, screen_optic(fid, step * c, instr), th, wl, sign, J)
                           - zk_ta_hi(instr, screen_optic(fid, -step * c, instr), th, wl,
                                      sign, J)) / (2 * step)
                    got[:4] = 0.0
                    c = c - (target - got)
                A_scr = (rtfit(screen_optic(fid, step * c, instr))
                         - rtfit(screen_optic(fid, -step * c, instr))) / (2 * step)
                res[f"A_rt_vs_scr{J}"] = float(np.linalg.norm(A_rt - A_scr) / n)
                res[f"scr{J}_vs_pred_G"] = float(np.linalg.norm(A_scr - pred) / n)
                res[f"screen_match_resid{J}"] = float(
                    np.linalg.norm(target - got) / np.linalg.norm(target))
            out[f"{name}_dof{d}"] = res
            print(f"dof {d}: |A_rt| {n * 1e6:.3f} um per unit; " + ", ".join(
                f"{key} {v:.4f}" for key, v in res.items() if key != "step"), flush=True)
        (run.path / "diag_floor.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
