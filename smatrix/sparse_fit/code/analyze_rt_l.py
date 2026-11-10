"""Closure of A_sparse = G_rt @ dTA against ray-traced A_direct_rt, and field symmetry of G_rt.

Usage::

    python smatrix/sparse_fit/code/analyze_rt_l.py <run directory>
"""
import glob
import json
import sys

import numpy as np
from galsim.zernike import noll_to_zern

NA = np.arange(4, 79)
GROUPS = {"M2 hexapod": range(0, 5), "camera hexapod": range(5, 10),
          "M1M3 B1-B10": range(10, 20), "M1M3 B11-B20": range(20, 30),
          "M2 B1-B10": range(30, 40), "M2 B11-B20": range(40, 50)}


def rot(nolls, phi):
    """Coefficient rotation for f(theta) -> f(theta - phi) on the listed Noll terms."""
    nolls = list(nolls)
    R = np.eye(len(nolls))
    for a, j in enumerate(nolls):
        n, m = noll_to_zern(j)
        if m <= 0:
            continue
        b = nolls.index(next(k for k in nolls if noll_to_zern(k) == (n, -m)))
        c, s = np.cos(m * phi), np.sin(m * phi)
        R[a, a], R[a, b], R[b, a], R[b, b] = c, -s, s, c
    return R


def load(run):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{run}/leak_*.json"))]


def main(run):
    rs = load(run)
    for fs in rs[0]["fit_sets"]:
        idx = rs[0]["fit_sets"][fs]
        fi = [list(NA).index(j) for j in idx]
        oi = [c for c in range(len(NA)) if c not in fi]
        res = {k: {"corner": [], "sci": []} for k in ("G_rt", "L_danish", "dense")}
        for r in rs:
            where = "sci" if r["name"].startswith("sci") else "corner"
            G = np.array(r["G_rt"][fs])
            L = np.array(r["L"][fs]).T
            dTA, Ad = np.array(r["dTA"]).T, np.array(r["A_dense"]).T
            A = np.array(r["A_direct_rt"][fs]).T
            n = np.linalg.norm(A, axis=0)
            res["G_rt"][where].append(np.linalg.norm(A - G @ dTA, axis=0) / n)
            res["L_danish"][where].append(
                np.linalg.norm(A - (dTA[fi] + L @ dTA[oi]), axis=0) / n)
            res["dense"][where].append(np.linalg.norm(A - Ad[fi], axis=0) / n)
        print(f"\n== fit set {fs}: median residual / |A_direct_rt| (dimensionless), "
              "corners (n=8) | science (n=18)")
        print(f"{'DOF group':<16}" + "".join(f"{k:>18}" for k in res))
        for g, ds in GROUPS.items():
            row = f"{g:<16}"
            for k in res:
                c = np.median(np.array(res[k]["corner"])[:, list(ds)])
                s = np.median(np.array(res[k]["sci"])[:, list(ds)])
                row += f"{c:>9.3f} |{s:>6.3f}"
            print(row)
        allg = np.vstack(res["G_rt"]["corner"] + res["G_rt"]["sci"])
        med = np.median(allg, axis=0)
        worst = np.argsort(med)[::-1][:6]
        print("worst DOF, G_rt (median over 26):",
              ", ".join(f"dof{d} {med[d]:.3f}" for d in worst))
        print("max |G_rt[:, fit] - I| (dimensionless), median over 26:",
              f"{np.median([np.abs(np.array(r['G_rt'][fs])[:, fi] - np.eye(len(fi))).max() for r in rs]):.3f}")

        # Field symmetry: rotate G into the frame with the field point on +x.
        print("G_rt spread at equal radius (dimensionless; RMS deviation from group mean "
              "over RMS mean), raw | rotated:")
        groups = {}
        for r in rs:
            rad = np.hypot(r["thx_deg"], r["thy_deg"])
            groups.setdefault((round(rad, 2), r["side"]), []).append(r)
        for (rad, side), gr in sorted(groups.items()):
            if len(gr) < 2:
                continue
            raw = [np.array(r["G_rt"][fs]) for r in gr]
            out = []
            for sgn in (1, -1):
                al = []
                for r, G in zip(gr, raw):
                    a = np.arctan2(r["thy_deg"], r["thx_deg"])
                    al.append(rot(idx, -sgn * a) @ G @ rot(NA, sgn * a))
                out.append(al)

            def spread(Gs):
                Gs = np.array(Gs)[:, :, oi]
                m = Gs.mean(0)
                return np.sqrt(np.mean((Gs - m) ** 2)) / np.sqrt(np.mean(m ** 2))
            print(f"  r = {rad:.2f} deg {side:<6} n={len(gr)}: {spread(raw):.3f} | "
                  f"{min(spread(o) for o in out):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
