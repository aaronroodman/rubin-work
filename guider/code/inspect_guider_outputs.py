#!/usr/bin/env python3
"""Sanity-check the three per-seq guider outputs for one visit.

Loads <seqNum>_{moments,stars,metrics}.parquet from --seqdir and prints row
counts, the detectors present in each, key moment columns, and basic finiteness
/ additivity checks -- a quick "do the outputs look good?" after a test run,
especially for partial-guider visits.

    python inspect_guider_outputs.py --seqdir <dir> --seq-num 53
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqdir", required=True, help="Dir with per-seq parquets.")
    p.add_argument("--seq-num", type=int, required=True)
    return p.parse_args(argv)


def load(seqdir, seqNum, product):
    path = os.path.join(seqdir, f"{seqNum}_{product}.parquet")
    if not os.path.exists(path):
        print(f"  [{product}] MISSING: {path}")
        return None
    return pd.read_parquet(path)


def main(argv=None):
    args = parseArgs(argv)
    sd, sn = args.seqdir, args.seq_num

    marker = os.path.join(sd, f"{sn}_SKIPPED.txt")
    if os.path.exists(marker):
        print(f"SKIPPED marker present:\n  {open(marker).read().strip()}")

    moments = load(sd, sn, "moments")
    stars = load(sd, sn, "stars")
    metrics = load(sd, sn, "metrics")

    print(f"\n=== rows ===")
    for name, df in [("moments", moments), ("stars", stars), ("metrics", metrics)]:
        print(f"  {name:8s}: {0 if df is None else len(df)} rows")

    if stars is not None and not stars.empty:
        det = sorted(stars["detector"].unique())
        print(f"\n=== stars (tracker) === {len(det)} guiders: {det}")
        print("  per-detector stamps:",
              dict(stars.groupby("detector")["stamp"].nunique()))

    if moments is not None and not moments.empty:
        det = sorted(moments["detector"].unique())
        print(f"\n=== moments (decomposition) === {len(det)} guiders: {det}")
        cols = ["detector", "nStamps", "Q1_coadd", "Q2_coadd", "T_coadd",
                "Q1_stamp", "Q1_motion", "resid_Q1", "resid_T",
                "coma1_static", "coma1_dyn", "tau_x", "tau_y"]
        show = [c for c in cols if c in moments.columns]
        with pd.option_context("display.width", 160, "display.max_columns", 40):
            print(moments[show].to_string(index=False))

        # additivity: coadd ~ stamp + motion (2nd order should nearly close)
        for q in ("Q1", "T"):
            r = moments.get(f"resid_{q}")
            c = moments.get(f"{q}_coadd")
            if r is not None and c is not None:
                frac = np.abs(r) / np.abs(c).replace(0, np.nan)
                print(f"  additivity {q}: |resid|/|coadd| median={np.nanmedian(frac):.3f}")

        # finiteness: flag columns that are entirely NaN
        allnan = [c for c in moments.columns
                  if moments[c].dtype.kind == "f" and not np.isfinite(moments[c]).any()]
        print(f"  all-NaN float columns: {allnan or 'none'}")
        print(f"  schema_version: {moments.get('schema_version', pd.Series(['?'])).iloc[0]}")

    if metrics is not None and not metrics.empty:
        print(f"\n=== metrics (GuiderMetricsBuilder) === {metrics.shape[1]} cols")
        print("  columns:", list(metrics.columns)[:20],
              "..." if metrics.shape[1] > 20 else "")


if __name__ == "__main__":
    main()
