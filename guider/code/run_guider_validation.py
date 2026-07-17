#!/usr/bin/env python3
"""Lightweight per-night validation QA plot from the guider moment dataset.

    python run_guider_validation.py output/<dataset>/moments/ \
        --output output/<dataset>/plots/validation.png

Reads the (Hive-partitioned) summary dataset and draws four QA panels vs
sequence number: sensors tracked per exposure, median FWHM, Alt/Az jitter, and
the additivity residual (coadd - [stamp+motion]) as a fraction of the coadd
size -- a check that the moment decomposition closes.
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
    p.add_argument("--output", required=True, help="Output PNG path.")
    return p.parse_args(argv)


def loadMoments(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        return pds.dataset(path, format="parquet", partitioning="hive").to_table().to_pandas()
    return pd.read_parquet(path)


def main(argv=None):
    args = parseArgs(argv)
    df = loadMoments(args.input)
    if df.empty:
        raise SystemExit(f"{args.input} is empty; nothing to plot.")

    perExp = df.groupby("seqNum")
    counts = perExp.size()
    fwhm = perExp["fwhm_med"].median()
    jitter = perExp["jitter_altaz"].median()
    resid_frac = (df["resid_T"] / df["T_coadd"]).replace([np.inf, -np.inf], np.nan).dropna()

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(counts.index, counts.values, ".", ms=4)
    ax[0, 0].set_title("sensors tracked per exposure")
    ax[0, 0].set_xlabel("seqNum"); ax[0, 0].set_ylabel("n sensors")

    ax[0, 1].plot(fwhm.index, fwhm.values, ".", ms=4, color="C1")
    ax[0, 1].set_title("median FWHM per exposure")
    ax[0, 1].set_xlabel("seqNum"); ax[0, 1].set_ylabel("FWHM [arcsec]")

    ax[1, 0].plot(jitter.index, jitter.values, ".", ms=4, color="C2")
    ax[1, 0].set_title("Alt/Az centroid jitter per exposure")
    ax[1, 0].set_xlabel("seqNum"); ax[1, 0].set_ylabel("jitter [arcsec]")

    lim = np.nanpercentile(np.abs(resid_frac), 99) if len(resid_frac) else 1.0
    ax[1, 1].hist(resid_frac, bins=np.linspace(-lim, lim, 41), color="C3", alpha=0.8)
    ax[1, 1].axvline(0.0, color="k", lw=1)
    ax[1, 1].set_title("additivity residual  (T_coadd - [T_stamp+T_motion]) / T_coadd")
    ax[1, 1].set_xlabel("fractional residual"); ax[1, 1].set_ylabel("sensor-exposures")

    nExp = df["expId"].nunique()
    nights = sorted(df["dayObs"].unique()) if "dayObs" in df else []
    fig.suptitle(f"Guider validation  {nights}  ({nExp} exposures, {len(df)} rows)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=110)
    print(f"wrote {args.output} ({nExp} exposures, {len(df)} rows)")


if __name__ == "__main__":
    main()
