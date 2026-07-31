#!/usr/bin/env python
"""Visualize agreement between our DZ sensitivity matrix and the OFC matrix.

Produces ../output/smatrix_ofc_comparison_<band>.png with:
  (top)    per-DOF signed correlation ours vs OFC (after global sign)
  (bottom) per-DOF rms ratio ours/OFC
Mismatched DOF (|corr|<0.9) are highlighted.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from compute_smatrix import CONFIG_TAG
from compare_ofc import DOF_LABELS, find_ofc_file, load_ofc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--band", default="r")
    p.add_argument("--ofc-file", default=None)
    args = p.parse_args()

    outdir = Path(__file__).resolve().parent.parent / "output"
    ours = np.load(outdir / f"smatrix_dz_31_29_50_{args.band}_{CONFIG_TAG}.npy")
    ofc = load_ofc(find_ofc_file(args.ofc_file))

    ndof = 50
    corr = np.full(ndof, np.nan)
    ratio = np.full(ndof, np.nan)
    for d in range(ndof):
        a, b = ours[:, :, d].ravel(), ofc[:, :, d].ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na and nb:
            corr[d] = np.dot(a, b) / (na * nb)
            ratio[d] = np.sqrt(np.mean(a**2)) / np.sqrt(np.mean(b**2))
    bad = np.abs(corr) < 0.9

    x = np.arange(ndof)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    c = np.where(bad, "crimson", "steelblue")

    ax0.bar(x, corr, color=c)
    ax0.axhline(-1, ls=":", c="gray"); ax0.axhline(1, ls=":", c="gray")
    ax0.set_ylabel("corr(ours, OFC)")
    ax0.set_title(f"batoid_rubin DZ sensitivity vs OFC "
                  f"(band {args.band}, {CONFIG_TAG}) — "
                  f"{int((~bad).sum())}/50 agree (global sign: ours = -OFC)")
    ax0.set_ylim(-1.15, 1.15)

    ax1.bar(x, ratio, color=c)
    ax1.axhline(1, ls=":", c="gray")
    ax1.set_ylabel("rms ratio ours/OFC")
    ax1.set_ylim(0, 1.3)
    for d in np.where(bad)[0]:
        ax1.annotate(DOF_LABELS[d], (d, ratio[d]), rotation=90,
                     ha="center", va="bottom", fontsize=8, color="crimson")

    # DOF group boundaries
    for ax in (ax0, ax1):
        for xb in (9.5, 29.5):
            ax.axvline(xb, color="k", lw=0.6, alpha=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(DOF_LABELS, rotation=90, fontsize=6)
    ax1.set_xlabel("DOF  |  0-9 rigid body · 10-29 M1M3 bend · 30-49 M2 bend")

    fig.tight_layout()
    out = outdir / f"smatrix_ofc_comparison_{args.band}.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
