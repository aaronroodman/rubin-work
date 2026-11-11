"""Merge the per-configuration refit files into one response table per fit set.

For each configuration, DOF and fit set the slope dz/dDOF (µm per DOF unit; DOF units
µm or deg, ts_ofc sign convention) comes from a quadratic through the nominal fit and the scan points. Points
whose best fit is lost (fwhm more than ``FWHM_TOL`` from the nominal fit's) are dropped:
there the wavefront is beyond the fitted terms and no start recovers it. With 2 points
left the fit is linear. ``ndrop`` records how many scan points were dropped.

Output ``responses.npz`` in a run directory: ``name``, ``thx``, ``thy`` (deg, OCS),
``side``, ``instrument``, ``kind`` (check, fam, cwfs), ``dofs``, ``step`` (n_cfg, 50),
and per fit set ``{fs}__noll``, ``{fs}__z0`` (µm), ``{fs}__slope`` (n_cfg, 50, n_z),
``{fs}__curv``, ``{fs}__scatter`` (dimensionless, over the linear change at +2 steps),
``{fs}__ndrop``.

Usage::

    python merge_refit.py --runs 20261111_refit_dof00_04 ... --label responses
"""
import argparse
import glob

import numpy as np

import donut_leakage as dl
from refit_images import FWHM_TOL, OUT_ROOT
from rubinwork.common import study_run


def _kind(name):
    return "fam" if name.startswith("fam_") else ("cwfs" if "_x" in name else "check")


def _fit(h, f):
    """Slope, curvature and scatter over the linear change at the largest |h|."""
    deg = 2 if len(h) >= 3 else 1
    V = np.vander(h, deg + 1)
    c, *_ = np.linalg.lstsq(V, f, rcond=None)
    hm = np.max(np.abs(h))
    lin = np.linalg.norm(c[-2]) * hm
    curv = np.linalg.norm(c[0]) * hm ** 2 / lin if deg == 2 else np.nan
    scat = np.sqrt(np.mean(np.sum((f - V @ c) ** 2, 1))) / lin if len(h) > deg + 1 \
        else np.nan
    return c[-2], curv, scat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--label", default="responses")
    args = p.parse_args()
    cfgs = dl.configs("all")
    index = {c[0]: i for i, c in enumerate(cfgs)}
    n, ndof = len(cfgs), 50
    out = dict(name=np.array([c[0] for c in cfgs]),
               thx=np.array([c[1] for c in cfgs]), thy=np.array([c[2] for c in cfgs]),
               side=np.array([c[3] for c in cfgs]),
               instrument=np.array([c[4] for c in cfgs]),
               kind=np.array([_kind(c[0]) for c in cfgs]),
               dofs=np.arange(ndof), step=np.full((n, ndof), np.nan))
    for fs, idx in dl.FIT_SETS.items():
        nz = len(idx)
        out[f"{fs}__noll"] = np.array(idx)
        out[f"{fs}__z0"] = np.full((n, nz), np.nan)
        out[f"{fs}__slope"] = np.full((n, ndof, nz), np.nan)
        for key in ("curv", "scatter"):
            out[f"{fs}__{key}"] = np.full((n, ndof), np.nan)
        out[f"{fs}__ndrop"] = np.full((n, ndof), -1, dtype=int)
    files = [f for r in args.runs for f in glob.glob(f"{OUT_ROOT}/{r}/refit_*.npz")]
    for f in files:
        d = np.load(f)
        i = index[str(d["name"])]
        ks = d["k"]
        # Refits made before the sign fix carry no dof_sign; their donuts are batoid ZCS.
        sgn = d["dof_sign"] if "dof_sign" in d else dl.DOF_SIGN[d["idof"]]
        for a, idof in enumerate(d["idof"]):
            step = d["step"][a]
            out["step"][i, idof] = step
            for fs in dl.FIT_SETS:
                uw = d[f"{fs}__use_warm"][a]
                zb = np.where(uw[:, None], d[f"{fs}__z_warm"][a], d[f"{fs}__z"][a])
                fb = np.where(uw, d[f"{fs}__fwhm_warm"][a], d[f"{fs}__fwhm"][a])
                good = np.abs(fb - float(d[f"{fs}__fwhm0"])) <= FWHM_TOL
                h = np.r_[0.0, ks[good] * step * sgn[a]]
                fz = np.vstack([d[f"{fs}__z0"], zb[good]])
                s, c, sc = _fit(h, fz)
                out[f"{fs}__z0"][i] = d[f"{fs}__z0"]
                out[f"{fs}__slope"][i, idof] = s
                out[f"{fs}__curv"][i, idof] = c
                out[f"{fs}__scatter"][i, idof] = sc
                out[f"{fs}__ndrop"][i, idof] = np.sum(~good)
    missing = np.isnan(out["step"]).sum()
    run = study_run("smatrix/sparse_fit", args.label)
    np.savez(f"{run.path}/responses.npz", **out)
    for fs in dl.FIT_SETS:
        nd = out[f"{fs}__ndrop"]
        print(f"{fs}: scans with dropped points {np.sum(nd > 0)}, "
              f"with 2+ dropped {np.sum(nd > 1)} (of {nd.size})")
    print(f"missing (configuration, DOF) {missing}; wrote {run.path}/responses.npz")


if __name__ == "__main__":
    main()
