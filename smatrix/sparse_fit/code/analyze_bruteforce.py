"""Ray-traced Danish response per DOF against the dense Smatrix rows, and its step stability.

For each fit set: median over configurations of |A_rt - A_dense[fit]| / |A_rt|
(``A_dense`` is the in-focus zernikeGQ change, what the dense Smatrix holds), of the
same with dTA[fit], and of |A_rt(step) - A_rt(2 step)| / |A_rt|.

Usage::

    python smatrix/sparse_fit/code/analyze_bruteforce.py <run directory>
"""
import glob
import json
import sys

import numpy as np

from analyze_rt_l import GROUPS, NA

DOF_NAMES = (["M2 dz", "M2 dx", "M2 dy", "M2 Rx", "M2 Ry",
              "Cam dz", "Cam dx", "Cam dy", "Cam Rx", "Cam Ry"]
             + [f"M1M3 B{i}" for i in range(1, 21)] + [f"M2 B{i}" for i in range(1, 21)])


def main(run):
    rs = [json.load(open(f)) for f in sorted(glob.glob(f"{run}/leak_*.json"))]
    for fs, idx in rs[0]["fit_sets"].items():
        fi = [list(NA).index(j) for j in idx]
        res = {k: {"corner": [], "sci": []} for k in ("A_dense", "dTA", "step")}
        for r in rs:
            where = "sci" if r["name"].startswith("sci") else "corner"
            A = np.array(r["A_direct_rt"][fs]).T
            n = np.linalg.norm(A, axis=0)
            for k in ("A_dense", "dTA"):
                P = np.array(r[k]).T[fi]
                res[k][where].append(np.linalg.norm(A - P, axis=0) / n)
            if r["A_direct_rt_2x"][fs]:
                A2 = np.array(r["A_direct_rt_2x"][fs]).T
                res["step"][where].append(np.linalg.norm(A - A2, axis=0) / n)
        print(f"\n== fit set {fs} ({len(idx)} terms): median over configurations, "
              "dimensionless; corners (n=8) | science (n=18)")
        print(f"{'DOF group':<16}{'A_rt vs A_dense':>18}{'A_rt vs dTA':>18}"
              f"{'step vs 2x step':>18}")
        for g, ds in GROUPS.items():
            row = f"{g:<16}"
            for k in res:
                c = np.median(np.array(res[k]["corner"])[:, list(ds)])
                s = np.median(np.array(res[k]["sci"])[:, list(ds)])
                row += f"{c:>9.3f} |{s:>6.3f}"
            print(row)
        for k in ("A_dense", "step"):
            allv = np.vstack(res[k]["corner"] + res[k]["sci"])
            med = np.median(allv, axis=0)
            worst = np.argsort(med)[::-1][:6]
            print(f"worst DOF, {k} (median over {len(rs)}): "
                  + ", ".join(f"{DOF_NAMES[d]} {med[d]:.3f}" for d in worst))


if __name__ == "__main__":
    main(sys.argv[1])
