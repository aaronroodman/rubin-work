#!/usr/bin/env python3
"""Focal-plane whisker summary from the combined guider moment dataset.

    python run_guider_whiskers.py output/<dataset>/moments/ \
        --output output/<dataset>/plots/whiskers.pdf

Reads the (Hive-partitioned) summary dataset, aggregates per detector (median
over exposures), and draws the traceless shape moment (Q1, Q2) = (Mxx-Myy,
2Mxy) as whiskers across the focal plane: left = static per-stamp (optics
bound), right = image motion (turbulence). Both panels share one scale.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as pds  # noqa: E402


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Combined moments dataset dir (or single parquet).")
    p.add_argument("--output", required=True, help="Output PDF path.")
    return p.parse_args(argv)


def loadMoments(path: str) -> pd.DataFrame:
    """Load the summary table from a partitioned dataset dir or a parquet file."""
    if os.path.isdir(path):
        return pds.dataset(path, format="parquet", partitioning="hive").to_table().to_pandas()
    return pd.read_parquet(path)


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
    df = loadMoments(args.input)
    if df.empty:
        raise SystemExit(f"{args.input} is empty; nothing to plot.")

    agg = (df.groupby("detector")[["xfp", "yfp", "Q1_stamp", "Q2_stamp", "Q1_motion", "Q2_motion"]]
             .median().reset_index())

    panels = [("Q1_stamp", "Q2_stamp", "C0", "Static per-stamp (optics bound)"),
              ("Q1_motion", "Q2_motion", "C3", "Image motion (turbulence)")]
    mmax = np.nanmax([np.hypot(agg[q1], agg[q2]) for q1, q2, _, _ in panels])
    xy = agg[["xfp", "yfp"]].to_numpy()
    span = np.nanmax(np.hypot(xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean()))
    scale = 0.15 * span / mmax if mmax > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (q1, q2, color, title) in zip(axes, panels):
        drawWhiskers(ax, agg["xfp"], agg["yfp"], agg[q1], agg[q2], scale, color)
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
