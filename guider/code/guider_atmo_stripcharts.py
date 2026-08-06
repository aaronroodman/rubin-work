#!/usr/bin/env python
"""
Strip-charts of guider PSF centroid + second moments vs stamp number, from
the moments parquet written by guider_atmo_sim.py.

  python guider_atmo_stripcharts.py [--outdir output/atmo_sim]

Produces guider_atmo_stripcharts_visit{v}.png in --outdir: one row per tracked
quantity, one line per guide sensor, x-axis = stamp number (time = stamp / rate).
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUANTS = [
    ("cen_x_asec", "centroid x [arcsec]"),
    ("cen_y_asec", "centroid y [arcsec]"),
    ("fwhm_asec",  "FWHM [arcsec]"),
    ("e1",         "e1"),
    ("e2",         "e2"),
    ("T_asec2",    "T = Ixx+Iyy [arcsec^2]"),
]


def main(outdir="output/atmo_sim"):
    path = os.path.join(outdir, "guider_atmo_moments.parquet")
    df = pd.read_parquet(path)
    df = df[df["ok"]] if "ok" in df.columns else df
    guiders = sorted(df["guider"].unique())
    visits = sorted(df["visit"].unique())

    for v in visits:
        dv = df[df["visit"] == v]
        fig, axes = plt.subplots(len(QUANTS), 1, figsize=(11, 2.1 * len(QUANTS)),
                                 sharex=True)
        for ax, (col, label) in zip(axes, QUANTS):
            for g in guiders:
                dg = dv[dv["guider"] == g].sort_values("stamp")
                ax.plot(dg["stamp"], dg[col], lw=0.9, label=g)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].set_title(f"Guider PSF strip-charts — visit {v}", fontsize=11)
        axes[-1].set_xlabel("stamp number")
        axes[0].legend(ncol=4, fontsize=7, loc="upper right")
        fig.tight_layout()
        out = os.path.join(outdir, f"guider_atmo_stripcharts_visit{v:02d}.png")
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")

        # quick numeric summary of the jitter each guider sees
        print(f"visit {v} centroid RMS [arcsec]:")
        for g in guiders:
            dg = dv[dv["guider"] == g]
            print(f"  {g}: x={dg['cen_x_asec'].std():.4f}  "
                  f"y={dg['cen_y_asec'].std():.4f}  "
                  f"<FWHM>={dg['fwhm_asec'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="output/atmo_sim",
                    help="Directory holding guider_atmo_moments.parquet.")
    args = ap.parse_args()
    main(outdir=args.outdir)
