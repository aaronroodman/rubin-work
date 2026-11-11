"""Nonlinearity of the brute-force responses scaled to the DOF range.

``curv`` in the responses is |quadratic| / |linear| change at +2 steps (dimensionless).
Both scale with amplitude h as h^2 and h, so at the full DOF range the ratio is
curv x range / (2 step). Printed per fit set and DOF group: median, 99th percentile and
maximum over (configuration, DOF), and the number of (configuration, DOF) above
``FLAG`` (dimensionless). Scans with dropped points are excluded. Output
``linearity_range.npz``.

Usage::

    python linearity_range.py --label linearity_range
"""
import argparse

import numpy as np

from kmax_test import GROUPS, dof_range
from refit_images import OUT_ROOT
from rubinwork.common import study_run

FLAG = 0.05
FIT_SETS = ("prod", "prod_z22", "dense21")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", default="20261111_responses")
    p.add_argument("--label", default="linearity_range")
    args = p.parse_args()
    d = np.load(f"{OUT_ROOT}/{args.responses}/responses.npz")
    rng = dof_range()
    out = {}
    print(f"quadratic / linear change at the full DOF range (dimensionless): "
          f"median | p99 | max | n > {FLAG}")
    print("fit set   " + "".join(f"{g:>30s}" for g in GROUPS))
    for fs in FIT_SETS:
        q = d[f"{fs}__curv"] * rng[None, :] / (2 * d["step"])
        q[d[f"{fs}__ndrop"] > 0] = np.nan
        out[fs] = q
        cells = []
        for g in GROUPS.values():
            e = q[:, list(g)]
            e = e[np.isfinite(e)]
            cells.append(f"{np.median(e):.4f} | {np.percentile(e, 99):.3f} | "
                         f"{e.max():.3f} | {np.sum(e > FLAG):d}")
        print(f"{fs:9s} " + "".join(f"{c:>30s}" for c in cells))
        worst = np.nanmax(q, 0)
        top = np.argsort(worst)[::-1][:5]
        print("          worst DOF (max): " + ", ".join(f"{i} {worst[i]:.3f}" for i in top))
    run = study_run("smatrix/sparse_fit", args.label)
    np.savez(f"{run.path}/linearity_range.npz", name=d["name"], **out)
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
