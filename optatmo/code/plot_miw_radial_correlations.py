#!/usr/bin/env python
"""Scatter MIW values of 1st vs 2nd (vs 3rd) RADIAL-ORDER terms of each aberration
family, at the native MIW field bins (no interpolation).

Hypothesis: a residual leakage in the separation of radial orders of the same
azimuthal aberration (e.g. astigmatism Z5/6 vs Z12/13 vs Z23/24) would show as a
linear correlation across the field -- and could source much of the MIW pattern.

The cos (m>0) and sin (m<0) components are matched to the SAME trig function
across radial orders (Noll: m>0 -> cos m*theta, m<0 -> sin |m|*theta):
  Astig  m=2: 1st (Z6c,Z5s)  2nd (Z12c,Z13s)  3rd (Z24c,Z23s)
  Coma   m=1: 1st (Z8c,Z7s)  2nd (Z16c,Z17s)
  Trefoil m=3:1st (Z10c,Z9s) 2nd (Z18c,Z19s)
  Tetra  m=4: 1st (Z14c,Z15s)2nd (Z26c,Z25s)
  Spherical m=0: 1st Z11      2nd Z22   (no cos/sin)
Each panel overlays the cos and sin components (same physical relation by
rotational symmetry) and prints the combined Pearson r and best-fit slope.

Output: 5 pages (Astig has 3 order-pairs on one page; the rest one panel each).

  python optatmo/code/plot_miw_radial_correlations.py            # default 5rot i-band MIW
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DEFAULT_MIW = ("aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/"
               "pathA_50_34_i_5rot/intrinsic_split_maps.parquet")

# family -> {order: (cos_noll, sin_noll)}; sin=None for m=0 (spherical)
FAMILIES = [
    ("Astigmatism", "m=2", {"1st": (6, 5), "2nd": (12, 13), "3rd": (24, 23)}),
    ("Coma", "m=1", {"1st": (8, 7), "2nd": (16, 17)}),
    ("Trefoil", "m=3", {"1st": (10, 9), "2nd": (18, 19)}),
    ("Tetrafoil", "m=4", {"1st": (14, 15), "2nd": (26, 25)}),
    ("Spherical", "m=0", {"1st": (11, None), "2nd": (22, None)}),
]


def col(df, j):
    return df[f"Z{j}_OCS"].to_numpy(float)


def scatter_pair(ax, df, oa, ob, la, lb):
    """oa, ob = (cos_j, sin_j) for the two radial orders; overlay cos & sin."""
    xs, ys = [], []
    for comp, cc, (ja, jb) in [("cos (X)", "tab:blue", (oa[0], ob[0])),
                               ("sin (Y)", "tab:red", (oa[1], ob[1]))]:
        if ja is None or jb is None:
            continue
        x, y = col(df, ja), col(df, jb)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=6, alpha=0.35, color=cc, label=comp)
        xs.append(x[ok]); ys.append(y[ok])
    X, Y = np.concatenate(xs), np.concatenate(ys)
    r = np.corrcoef(X, Y)[0, 1]
    sl, ic = np.polyfit(X, Y, 1)
    xx = np.array([X.min(), X.max()])
    ax.plot(xx, sl * xx + ic, "k--", lw=1.2, label=f"fit slope={sl:+.2f}")
    ax.axhline(0, color="0.8", lw=0.5); ax.axvline(0, color="0.8", lw=0.5)
    ax.set_xlabel(f"{la} [µm]"); ax.set_ylabel(f"{lb} [µm]")
    ax.set_title(f"{la} vs {lb}    r={r:+.2f}", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miw", default=DEFAULT_MIW)
    ap.add_argument("--out", default="optatmo/output/miw_radial_correlations.pdf")
    args = ap.parse_args()

    df = pd.read_parquet(args.miw)
    n = int(np.isfinite(col(df, 6)).sum())
    print(f"MIW: {args.miw}\n  {len(df)} field bins ({n} finite); native binning, no interpolation")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with PdfPages(args.out) as pdf:
        for name, mlabel, orders in FAMILIES:
            keys = list(orders)
            pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
            fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 5.0),
                                     squeeze=False)
            rs = []
            for ax, (a, b) in zip(axes[0], pairs):
                rs.append(scatter_pair(ax, df, orders[a], orders[b],
                                       f"{name} {a}", f"{name} {b}"))
            js = ", ".join(f"{o}:Z{orders[o][0]}"
                           + (f"/Z{orders[o][1]}" if orders[o][1] else "")
                           for o in keys)
            fig.suptitle(f"{name} ({mlabel})  radial-order cross-correlation  [{js}]",
                         fontsize=12)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
            print(f"  {name:11s}: " + "  ".join(f"{a}-{b} r={r:+.2f}"
                                                for (a, b), r in zip(pairs, rs)))
    print(f"wrote {args.out} (5 pages)")


if __name__ == "__main__":
    main()
