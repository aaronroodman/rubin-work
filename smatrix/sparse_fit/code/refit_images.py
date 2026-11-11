"""Refit saved ray-traced donuts (``donuts_<config>.fits``) with Danish, logging fit
quality, and redo the five-point slope with a retry for false minima.

Every image is fit from the ts_wep cold start. With ``--warm all`` it is also fit
warm-started from the nominal image's solution for the same fit set; with
``--warm retry`` only when the cold fit looks lost: fwhm more than ``FWHM_TOL`` from
the nominal fit's, or cost above ``RETRY_FACTOR`` x the nominal cost.
The lower-cost fit is kept. Output per configuration: ``refit_<config>.npz``.

Usage::

    python refit_images.py --runs 20261110_bf_dof20_24 --configs fam_r00_a08_intra \
        --dofs 22 --warm all --label refit_test
"""
import argparse
import glob
import time

import numpy as np
from astropy.io import fits

import donut_leakage as dl
from rubinwork.common import study_run

# False minima seen: fwhm 2.5-3.5 arcsec, cost ~1000x nominal; good fits at +-2 steps
# keep fwhm within 0.02 arcsec and reach ~50x nominal cost (dense21).
FWHM_TOL = 0.2        # arcsec
RETRY_FACTOR = 200.0  # cold cost / nominal cost
OUT_ROOT = "/sdf/group/rubin/u/roodman/LSST/rubin-work/studies/smatrix/sparse_fit"


def _setup(name):
    """Field, factory and per-fit-set reference Zernikes for one configuration."""
    cfg = [c for c in dl.configs("all") if c[0] == name][0]
    _, thx, thy, side, iname = cfg
    instr = dl._W["instr"][iname]
    wl = instr.wavelength[dl.BAND] if isinstance(instr.wavelength, dict) else \
        instr.wavelength[instr.refBand]
    th = np.deg2rad([thx, thy])
    sign = +1 if side == "extra" else -1
    fid = dl._optic(-1, 0)
    zf0 = dl.zk_infocus(fid, th, wl)
    ta0 = dl.zk_ta(instr, fid, th, wl, sign)
    refs = {}
    for fs, idx in dl.FIT_SETS.items():
        zr = ta0.copy()
        zr[idx] -= zf0[idx]
        refs[fs] = zr
    return th, dl.factory_for(instr), refs


def _one(fac, ref, idx, th, img, x0=None):
    m, r = dl.fit_full(fac, ref, idx, th, img, x0=x0)
    p = m.unpack_params(r.x)
    return dict(z=np.array(p["z_fit"]) * 1e6, x=r.x, cost=r.cost, fwhm=p["fwhm"],
                nfev=r.nfev)


def refit(path, dofs, warm):
    """Refit one donuts file; returns arrays keyed by fit set."""
    with fits.open(path) as h:
        name = h[0].header["CONFIG"]
        ks = np.array([int(k) for k in h[0].header["STEPS_K"].split()])
        nom = h["NOMINAL"].data.astype(float)
        cube = h["STEPS"].data.astype(float)
        tab = h["DOFS"].data
        idofs, steps = np.array(tab["idof"]), np.array(tab["step"])
    sel = [i for i, d in enumerate(idofs) if dofs is None or d in dofs]
    th, fac, refs = _setup(name)
    out = dict(name=name, idof=idofs[sel], step=steps[sel], k=ks)
    for fs, idx in dl.FIT_SETS.items():
        f0 = _one(fac, refs[fs], idx, th, nom)
        nd, nk, nz = len(sel), len(ks), len(idx)
        res = {key: np.full((nd, nk) + ((nz,) if key == "z" else ()), np.nan)
               for key in ("z", "cost", "fwhm", "nfev", "z_warm", "cost_warm",
                           "fwhm_warm")}
        res["z_warm"] = np.full((nd, nk, nz), np.nan)
        for a, i in enumerate(sel):
            for b in range(nk):
                c = _one(fac, refs[fs], idx, th, cube[i, b])
                for key in ("z", "cost", "fwhm", "nfev"):
                    res[key][a, b] = c[key]
                lost = (abs(c["fwhm"] - f0["fwhm"]) > FWHM_TOL
                        or c["cost"] > RETRY_FACTOR * f0["cost"])
                if warm == "all" or (warm == "retry" and lost):
                    w = _one(fac, refs[fs], idx, th, cube[i, b], x0=f0["x"])
                    res["z_warm"][a, b] = w["z"]
                    res["cost_warm"][a, b] = w["cost"]
                    res["fwhm_warm"][a, b] = w["fwhm"]
        # Best fit per image, then the five-point quadratic as in donut_leakage.
        use_w = np.nan_to_num(res["cost_warm"], nan=np.inf) < res["cost"]
        zb = np.where(use_w[..., None], res["z_warm"], res["z"])
        slope, curv, scat = [], [], []
        for a in range(len(sel)):
            h = np.r_[0.0, ks * steps[sel][a]]
            V = np.vander(h, 3)
            f = np.vstack([f0["z"], zb[a]])
            c, *_ = np.linalg.lstsq(V, f, rcond=None)
            lin = np.linalg.norm(c[1]) * 2 * steps[sel][a]
            slope.append(c[1])
            curv.append(np.linalg.norm(c[0]) * (2 * steps[sel][a]) ** 2 / lin)
            scat.append(np.sqrt(np.mean(np.sum((f - V @ c) ** 2, 1))) / lin)
        out[fs] = dict(res, z0=f0["z"], cost0=f0["cost"], fwhm0=f0["fwhm"],
                       use_warm=use_w, slope=np.array(slope), curv=np.array(curv),
                       scatter=np.array(scat))
    return out


def _save(directory, out):
    flat = {k: v for k, v in out.items() if not isinstance(v, dict)}
    for fs in dl.FIT_SETS:
        flat.update({f"{fs}__{k}": v for k, v in out[fs].items()})
    np.savez(f"{directory}/refit_{out['name']}.npz", **flat)


def _work(args):
    path, dofs, warm = args
    t0 = time.time()
    out = refit(path, dofs, warm)
    out["seconds"] = time.time() - t0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run directory names")
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--dofs", type=int, nargs="*", default=None)
    p.add_argument("--warm", choices=["none", "retry", "all"], default="retry")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--label", default="refit")
    args = p.parse_args()
    files = sorted(f for r in args.runs for f in glob.glob(f"{OUT_ROOT}/{r}/donuts_*.fits"))
    if args.configs:
        files = [f for f in files
                 if any(f.endswith(f"donuts_{c}.fits") for c in args.configs)]
    run = study_run("smatrix/sparse_fit", args.label)
    work = [(f, args.dofs, args.warm) for f in files]
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs, initializer=dl._init, initargs=(list(range(50)), {})) as pool:
            for out in pool.imap_unordered(_work, work):
                _save(run.path, out)
                print(f"{out['name']}: {out['seconds']:.0f} s", flush=True)
    else:
        dl._init(list(range(50)), {})
        for w in work:
            out = _work(w)
            _save(run.path, out)
            print(f"{out['name']}: {out['seconds']:.0f} s", flush=True)
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
