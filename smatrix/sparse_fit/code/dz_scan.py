"""Per-point fitted Zernike changes along each DOF scan, FAM points only.

From the refit files, the best fit per image (cold or warm, lower cost) minus the
nominal fit, at k = -2, -1, 0, +1, +2 steps. Points whose fit is lost (fwhm more than
``FWHM_TOL`` from the nominal fit's) are NaN. Output ``dz_scan.npz``: ``name``, ``thx``,
``thy`` (deg, OCS), ``side``, ``step`` (n, 50; DOF units, µm or arcsec), ``k``, and per
fit set ``{fs}__noll`` and ``{fs}__dz`` (n, 50, 5, n_z; µm).

Usage::

    python dz_scan.py --runs 20261111_refit_dof00_04 20261111_refit_dof05_09 \
        --label dz_scan
"""
import argparse
import glob

import numpy as np

import donut_leakage as dl
from refit_images import FWHM_TOL, OUT_ROOT
from rubinwork.common import study_run

K = np.array([-2, -1, 0, 1, 2])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--label", default="dz_scan")
    args = p.parse_args()
    cfgs = [c for c in dl.configs("all") if c[0].startswith("fam_")]
    index = {c[0]: i for i, c in enumerate(cfgs)}
    n = len(cfgs)
    out = dict(name=np.array([c[0] for c in cfgs]), thx=np.array([c[1] for c in cfgs]),
               thy=np.array([c[2] for c in cfgs]), side=np.array([c[3] for c in cfgs]),
               step=np.full((n, 50), np.nan), k=K)
    for fs, idx in dl.FIT_SETS.items():
        out[f"{fs}__noll"] = np.array(idx)
        out[f"{fs}__dz"] = np.full((n, 50, len(K), len(idx)), np.nan)
    for r in args.runs:
        for f in glob.glob(f"{OUT_ROOT}/{r}/refit_fam_*.npz"):
            d = np.load(f)
            i = index[str(d["name"])]
            col = [list(K).index(k) for k in d["k"]]
            for a, idof in enumerate(d["idof"]):
                out["step"][i, idof] = d["step"][a]
                for fs in dl.FIT_SETS:
                    uw = d[f"{fs}__use_warm"][a]
                    zb = np.where(uw[:, None], d[f"{fs}__z_warm"][a], d[f"{fs}__z"][a])
                    fb = np.where(uw, d[f"{fs}__fwhm_warm"][a], d[f"{fs}__fwhm"][a])
                    dz = zb - d[f"{fs}__z0"]
                    dz[np.abs(fb - float(d[f"{fs}__fwhm0"])) > FWHM_TOL] = np.nan
                    out[f"{fs}__dz"][i, idof, col] = dz
                    out[f"{fs}__dz"][i, idof, 2] = 0.0
    print(f"missing (point, DOF): {np.isnan(out['step']).sum()}")
    run = study_run("smatrix/sparse_fit", args.label)
    np.savez(f"{run.path}/dz_scan.npz", **out)
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
