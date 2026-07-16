#!/usr/bin/env python3
"""Focal-plane whisker summary for a combined guider moment table.

    python run_guider_whiskers.py output/<dataset>/moments.parquet \
        --output output/<dataset>/plots/whiskers.pdf

Aggregates the dataset (median over exposures per detector) and draws the
traceless shape moment (Q1, Q2) = (Mxx-Myy, 2Mxy) as whiskers across the focal
plane: left = static per-stamp (optics bound), right = image motion
(turbulence). Both panels share one whisker scale.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Combined moments parquet.")
    p.add_argument("--output", required=True, help="Output PDF path.")
    return p.parse_args(argv)


def drawWhiskers(ax, x, y, q1, q2, scale, color):
    mag = np.hypot(q1, q2)
    ang = 0.5 * np.arctan2(q2, q1)
    dx = scale * mag * np.cos(ang)
    dy = scale * mag * np.sin(ang)
    for xi, yi, dxi, dyi in zip(x, y, dx, dy):
        ax.plot([xi - dxi, xi + dxi], [yi - dyi, yi + dyi], color=color, lw=2)
    ax.scatter(x, y, s=10, color="k", zorder=3)


def main(argv=None):
    args = parseArgs(argv)
    df = pd.read_parquet(args.input)
    if df.empty:
        raise SystemExit(f"{args.input} is empty; nothing to plot.")

    # median over exposures, per detector and kind
    agg = (df.groupby(["detector", "kind"])[["xfp", "yfp", "Q1", "Q2"]]
             .median().reset_index())

    panels = [("stamp_mean", "C0", "Static per-stamp (optics bound)"),
              ("motion", "C3", "Image motion (turbulence)")]
    mags = []
    for kind, _, _ in panels:
        sub = agg[agg["kind"] == kind]
        mags.append(np.hypot(sub["Q1"], sub["Q2"]).to_numpy())
    mmax = np.nanmax(np.concatenate(mags)) if mags else 1.0

    xy = agg.drop_duplicates("detector")[["xfp", "yfp"]].to_numpy()
    span = np.nanmax(np.hypot(xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean()))
    scale = 0.15 * span / mmax if mmax > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (kind, color, title) in zip(axes, panels):
        sub = agg[agg["kind"] == kind]
        drawWhiskers(ax, sub["xfp"], sub["yfp"], sub["Q1"], sub["Q2"], scale, color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("xfp")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("yfp")
    nExp = df["expId"].nunique()
    fig.suptitle(f"Guider shape moments (arcsec^2, median of {nExp} exposures)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"wrote {args.output} ({nExp} exposures, {agg['detector'].nunique()} sensors)")


if __name__ == "__main__":
    main()
