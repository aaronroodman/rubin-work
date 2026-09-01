#!/usr/bin/env python
"""Piece 1 of the sparse-fit study: analyze the ts_ofc DoubleZernike sensitivity
matrix to find which DOF drive the SECONDARY/TERTIARY aberrations of each
azimuthal family, and whether those same DOF also drive a field-CORRELATED
PRIMARY of the same family.

Motivation: the MIW shows residual correlations between radial orders of the same
azimuthal aberration (astig Z5/6<->Z12/13<->Z23/24, coma Z7/8<->Z16/17, ...).  A
sparse donut fit (fit only the primary; fix secondary/tertiary to nominal) would
push real secondary/tertiary content into the primary.  The DOF that produce
BOTH a secondary/tertiary AND a field-correlated primary of the same family are
the ones whose mis-separation would create exactly this MIW correlation -- and
that a revised (sparse) sensitivity matrix must handle.

Sensitivity S[k, j, d]  (31 field-Zernike k, 29 pupil-Noll j, 50 DOF d), a galsim
DoubleZernike: wavefront Zernike j at field (u,v) from DOF d = sum_k S[k,j,d]
Z_k^field(u,v).  k=1 is the field-constant; pupil index = Noll j.  Because the
field basis is orthonormal, the field-MAP correlation of two pupil Zernikes'
responses to a DOF equals the correlation of their field-coefficient vectors
S[1:, ja, d] vs S[1:, jb, d] -- so no simulation is needed here.

Output: per-family page (primary/secondary/tertiary field power per DOF +
primary<->secondary field correlation) and a printed shortlist of coupling DOF.
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_SENSDIR = ("/Users/roodman/Astrophysics/Claude/packages/ts_config_mttcs/MTAOS/"
            "ofc/sensitivity_matrix/")
# theoretical matrix has ALL radial orders populated; the production "measured"
# one zeroes coma2/tref2/astig3/tet2 -- compared on the first page.
DEFAULT_S = _SENSDIR + "lsst_sensitivity_dz_31_29_50.yaml"
MEASURED_S = _SENSDIR + "new_measured_lsst_sensitivity_dz_31_29_50.yaml"

DOF_NAMES = (["M2:dZ", "M2:dX", "M2:dY", "M2:rX", "M2:rY"]
             + ["Cam:dZ", "Cam:dX", "Cam:dY", "Cam:rX", "Cam:rY"]
             + [f"M13b{i+1}" for i in range(20)]
             + [f"M2b{i+1}" for i in range(20)])
DOF_GROUPS = [(0, 5, "M2 hex"), (5, 10, "Cam hex"), (10, 30, "M1M3 bend"),
              (30, 50, "M2 bend")]

# family -> {order: (cos_Noll, sin_Noll)}; sin=None for m=0
FAMILIES = [
    ("Astigmatism", "m=2", {"1st": (6, 5), "2nd": (12, 13), "3rd": (24, 23)}),
    ("Coma", "m=1", {"1st": (8, 7), "2nd": (16, 17)}),
    ("Trefoil", "m=3", {"1st": (10, 9), "2nd": (18, 19)}),
    ("Tetrafoil", "m=4", {"1st": (14, 15), "2nd": (26, 25)}),
    ("Spherical", "m=0", {"1st": (11, None), "2nd": (22, None)}),
]


def power(S, j, d):
    """Total field-response power (const + varying) of pupil Noll j to DOF d."""
    return float(np.sqrt(np.sum(S[1:, j, d] ** 2)))


def pair_power(S, oc, os_, d):
    p = power(S, oc, d) ** 2
    if os_ is not None:
        p += power(S, os_, d) ** 2
    return float(np.sqrt(p))


def field_corr(S, ja, jb, d):
    """Field-map correlation of two pupil Zernikes' responses to DOF d."""
    a, b = S[1:, ja, d], S[1:, jb, d]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", default=DEFAULT_S)
    ap.add_argument("--out", default="output/sensitivity_sparse_analysis.pdf")
    ap.add_argument("--min-frac", type=float, default=0.05,
                    help="secondary power threshold (fraction of the family's max) to "
                    "consider a DOF a secondary driver")
    args = ap.parse_args()

    S = np.nan_to_num(np.array(yaml.safe_load(open(args.sens)), dtype=float))
    nk, nj, nd = S.shape
    print(f"sensitivity (structural): {args.sens}\n  shape (field {nk}, pupil {nj}, DOF {nd})")
    Sm = np.nan_to_num(np.array(yaml.safe_load(open(MEASURED_S)), dtype=float))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    x = np.arange(nd)
    with PdfPages(args.out) as pdf:
        # page 0: which pupil-Noll terms are populated -- theoretical vs measured
        fam_of = {5: "as1", 6: "as1", 7: "co1", 8: "co1", 9: "tr1", 10: "tr1", 11: "sp1",
                  12: "as2", 13: "as2", 14: "te1", 15: "te1", 16: "co2", 17: "co2",
                  18: "tr2", 19: "tr2", 22: "sp2", 23: "as3", 24: "as3", 25: "te2", 26: "te2"}
        jj = np.arange(4, 29)
        pth = np.array([np.sqrt(np.sum(S[1:, j, :] ** 2)) for j in jj])
        pme = np.array([np.sqrt(np.sum(Sm[1:, j, :] ** 2)) for j in jj])
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(jj - 0.2, pth, 0.4, label="theoretical (lsst)", color="tab:blue")
        ax.bar(jj + 0.2, pme, 0.4, label="production (measured)", color="tab:orange")
        ax.set_yscale("log"); ax.set_xticks(jj)
        ax.set_xticklabels([f"Z{j}\n{fam_of.get(j,'')}" for j in jj], fontsize=7)
        ax.set_ylabel("total field-response power over all DOF")
        ax.set_title("Sensitivity population per pupil Noll: production ZEROES "
                     "coma2/tref2/astig3/tet2 (Z16-19,23-26) -- kept for astig2/tet1")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        zeroed = [f"Z{j}" for j in jj if pth[j - 4] > 0 and pme[j - 4] == 0]
        print(f"  production 'measured' matrix zeroes: {zeroed}")
        for name, ml, orders in FAMILIES:
            keys = list(orders)
            # per-order per-DOF power
            pw = {o: np.array([pair_power(S, *orders[o], d) for d in range(nd)])
                  for o in keys}
            # primary<->secondary (and tertiary) field correlation per DOF (cos, sin)
            corr = {}
            for hi in keys[1:]:
                cc = np.array([field_corr(S, orders["1st"][0], orders[hi][0], d)
                               for d in range(nd)])
                sc = (np.array([field_corr(S, orders["1st"][1], orders[hi][1], d)
                                for d in range(nd)])
                      if orders[hi][1] is not None else np.full(nd, np.nan))
                corr[hi] = (cc, sc)

            fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 9))
            # panel 1: field power per DOF for each radial order
            w = 0.8 / len(keys)
            for i, o in enumerate(keys):
                a1.bar(x + i * w, pw[o], w, label=f"{o} (Z{orders[o][0]}"
                       + (f"/{orders[o][1]}" if orders[o][1] else "") + ")")
            a1.set_yscale("log"); a1.set_ylabel("field-response power (native units)")
            a1.set_title(f"{name} ({ml}): per-DOF field response by radial order")
            a1.legend(fontsize=8, ncol=len(keys))
            for lo, hi, lab in DOF_GROUPS:
                a1.axvline(hi - 0.5, color="0.7", lw=0.6, ls=":")
                a1.text((lo + hi) / 2, a1.get_ylim()[1], lab, ha="center", va="top",
                        fontsize=8, color="0.4")

            # panel 2: primary<->secondary field correlation, only where secondary
            # is a real driver (power above threshold)
            thr = args.min_frac * pw[keys[1]].max()
            drv = pw[keys[1]] > thr
            for hi, (cc, sc) in corr.items():
                a2.scatter(x[drv], cc[drv], s=36, label=f"1st-{hi} cos", marker="o")
                ok = drv & np.isfinite(sc)
                a2.scatter(x[ok], sc[ok], s=36, label=f"1st-{hi} sin", marker="^")
            a2.axhline(0, color="k", lw=0.5); a2.set_ylim(-1.05, 1.05)
            a2.set_ylabel("primary<->higher field-map correlation")
            a2.set_xlabel("DOF index")
            a2.set_title(f"{name}: does a secondary-driving DOF also drive a "
                         f"field-correlated primary?  (only DOF with 2nd power > "
                         f"{args.min_frac:.0%} of max)")
            a2.legend(fontsize=8, ncol=4)
            for ax in (a1, a2):
                ax.set_xticks(x[::2]); ax.set_xticklabels([DOF_NAMES[i] for i in x[::2]],
                                                          rotation=90, fontsize=6)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            # shortlist: DOF that drive secondary AND a field-correlated primary
            print(f"\n{name} ({ml}) -- DOF driving 2nd (>{args.min_frac:.0%} max) "
                  f"and their primary coupling:")
            order2 = np.argsort(pw[keys[1]])[::-1]
            for d in order2:
                if pw[keys[1]][d] <= thr:
                    break
                cc, sc = corr[keys[1]]
                ratio = pw[keys[1]][d] / (pw["1st"][d] + 1e-12)
                print(f"  {DOF_NAMES[d]:8s}: 2nd={pw[keys[1]][d]:7.2f}  1st={pw['1st'][d]:7.2f}"
                      f"  2nd/1st={ratio:5.2f}  corr(1st,2nd) cos={cc[d]:+.2f} "
                      f"sin={sc[d] if np.isfinite(sc[d]) else float('nan'):+.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
