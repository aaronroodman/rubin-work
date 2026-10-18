#!/usr/bin/env python
"""Validation plots for the BLOCK-T539 closed-loop AOS table.

Reads the parquet from ``build_t539_table.py`` and writes a small multi-page PDF:
  1. v-mode (v1..v12) histograms
  2. telemetry sanity -- column finite-fraction bar + key-quantity histograms
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TELE = ["cam_air_temp", "m2_air_temp", "m1m3_air_temp", "outside_temp",
        "dome_delta_t", "m2_delta_t", "cam_m1m3_delta_t",
        "x_gradient", "y_gradient", "z_gradient", "radial_gradient",
        "tma_truss_temp_pxpy", "tma_truss_temp_mxmy",
        "wind_speed_inside", "wind_speed_outside", "wind_dir_outside"]
KEY = ["z_gradient", "radial_gradient", "dome_delta_t", "m2_delta_t",
       "wind_speed_inside", "wind_speed_outside", "outside_temp",
       "tma_truss_temp_pxpy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    vcols = [f"v{i}" for i in range(1, 13) if f"v{i}" in df.columns]
    tele = [c for c in TELE if c in df.columns]

    with PdfPages(args.out) as pdf:
        # --- page 1: v-mode histograms ---
        if vcols:
            fig, axs = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
            for ax, c in zip(axs.ravel(), vcols):
                x = df[c].dropna().to_numpy()
                if len(x):
                    ax.hist(x, bins=30, color="#4c72b0", edgecolor="white", lw=0.3)
                    ax.axvline(0, color="k", lw=0.6, alpha=0.5)
                    ax.set_title(f"{c}  mean={np.nanmean(x):+.3f} std={np.nanstd(x):.3f}",
                                 fontsize=9)
                else:
                    ax.set_title(f"{c} (empty)", fontsize=9)
                ax.tick_params(labelsize=7)
            for ax in axs.ravel()[len(vcols):]:
                ax.axis("off")
            fig.suptitle(f"BLOCK-T539 converged v-modes (n={len(df)})", fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)

        # --- page 2: telemetry coverage + key histograms ---
        if tele:
            fig = plt.figure(figsize=(16, 10), constrained_layout=True)
            gs = fig.add_gridspec(3, 4)
            axb = fig.add_subplot(gs[0, :])
            frac = [df[c].notna().mean() for c in tele]
            axb.bar(range(len(tele)), frac, color="#55a868")
            axb.set_xticks(range(len(tele)))
            axb.set_xticklabels(tele, rotation=60, ha="right", fontsize=7)
            axb.set_ylabel("finite fraction")
            axb.set_ylim(0, 1.05)
            axb.set_title("telemetry column coverage")
            key = [c for c in KEY if c in df.columns][:8]
            for k, c in enumerate(key):
                ax = fig.add_subplot(gs[1 + k // 4, k % 4])
                x = df[c].dropna().to_numpy()
                if len(x):
                    ax.hist(x, bins=25, color="#c44e52", edgecolor="white", lw=0.3)
                ax.set_title(f"{c}  (n={len(x)})", fontsize=9)
                ax.tick_params(labelsize=7)
            fig.suptitle("BLOCK-T539 telemetry validation", fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
