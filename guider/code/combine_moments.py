#!/usr/bin/env python3
"""Combine per-seq guider products into per-day single-file parquets.

Concatenates the per-seq tables in --seqdir into three per-day files under
--outdir:

  guider_moments_<dayObs>.parquet   (your decomposition, per detector)
  guider_stars_<dayObs>.parquet     (summit_utils per-stamp tracker table)
  guider_metrics_<dayObs>.parquet   (summit_utils per-exposure metrics)

and writes skipped_visits.txt listing visits with missing data (from the
per-seq SKIPPED markers) and visits with no tracked stars.

    python combine_moments.py --seqdir <night>/seq --outdir <night> --dayobs 20260709
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

PRODUCTS = ("moments", "stars", "metrics")


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqdir", required=True, help="Dir with per-seq <seqNum>_*.parquet.")
    p.add_argument("--outdir", required=True, help="Per-day output dir.")
    p.add_argument("--dayobs", type=int, required=True)
    return p.parse_args(argv)


def combineProduct(seqdir, product):
    """Concat non-empty per-seq files for one product; return (df, empties)."""
    files = sorted(glob.glob(os.path.join(seqdir, f"*_{product}.parquet")))
    frames, empties = [], []
    for f in files:
        d = pd.read_parquet(f)
        if d.empty:
            empties.append(os.path.basename(f).split("_")[0])   # seqNum
        else:
            frames.append(d)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df, empties


def main(argv=None):
    args = parseArgs(argv)
    os.makedirs(args.outdir, exist_ok=True)

    counts = {}
    for product in PRODUCTS:
        df, _ = combineProduct(args.seqdir, product)
        out = os.path.join(args.outdir, f"guider_{product}_{args.dayobs}.parquet")
        df.to_parquet(out, index=False)
        counts[product] = len(df)

    # skip report: missing-data markers vs no-star empties (from the moments product)
    missing = []
    for m in sorted(glob.glob(os.path.join(args.seqdir, "*_SKIPPED.txt"))):
        missing.append(open(m).read().strip())
    _, nostar = combineProduct(args.seqdir, "moments")
    nostar = [s for s in nostar if s not in {mm.split("\t")[2] for mm in missing if "\t" in mm}]

    lines = []
    if missing:
        lines.append(f"# {len(missing)} visit(s) SKIPPED -- missing/insufficient data:")
        lines += [f"  {m}" for m in missing]
    if nostar:
        lines.append(f"# {len(nostar)} visit(s) with no tracked stars: {', '.join(sorted(nostar))}")
    report = "\n".join(lines) if lines else "no skipped visits"
    print(report)
    with open(os.path.join(args.outdir, "skipped_visits.txt"), "w") as fh:
        fh.write(report + "\n")

    print(f"dayObs {args.dayobs}: "
          + ", ".join(f"{k}={counts[k]}" for k in PRODUCTS) + f" -> {args.outdir}")


if __name__ == "__main__":
    main()
