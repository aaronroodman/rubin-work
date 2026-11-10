"""Ray-traced Danish response at one field point: camera shift (FAM) vs Detector shift (CWFS).

Usage::

    python geom_test.py --point R44_SW0 --dofs 0 1 3 5 6 8 15 20 23 35 36 46
"""
import argparse

import numpy as np

import donut_leakage as dl
from analyze_bruteforce import DOF_NAMES

p = argparse.ArgumentParser()
p.add_argument("--point", default="R44_SW0")
p.add_argument("--dofs", type=int, nargs="+", default=[0, 1, 3, 5, 6, 8, 15, 20, 23, 35, 36, 46])
args = p.parse_args()
dl._init(args.dofs, dict(truths=["rt"], leakage=False))
x, y = dl.CORNERS[args.point]
side = "extra" if args.point.endswith("SW0") else "intra"
res = {i: dl.run_config((f"{args.point}_{i}", x, y, side, i)) for i in ("LsstCam", "LsstFamCam")}
print(f"{args.point} ({side}): |A_FAM - A_CWFS| / |A_CWFS| (dimensionless), per fit set")
for fs in dl.FIT_SETS:
    a = np.array(res["LsstCam"]["A_direct_rt"][fs]).T
    b = np.array(res["LsstFamCam"]["A_direct_rt"][fs]).T
    e = np.linalg.norm(b - a, axis=0) / np.linalg.norm(a, axis=0)
    print(f"  {fs}: " + ", ".join(f"{DOF_NAMES[d]} {v:.3f}" for d, v in zip(args.dofs, e)))
