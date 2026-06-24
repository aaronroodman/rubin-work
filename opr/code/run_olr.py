#!/usr/bin/env python3
"""CLI: compute the Open Loop Reproduction for one night -> parquet.

    python run_olr.py --in output/20260329/nightly_aos_table.parquet \
                      --out output/20260329/olr.parquet \
                      --ofc-config-dir ~/WORK/ts_config_mttcs/MTAOS/ofc

Reads a nightly AOS table, builds the OFC sensitivity matrix, and writes one row
per usable seq with the open-loop-reproduced OPD and deviation Zernikes (plus
the original measured values and the intrinsic, for reference).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from olr import (
    DEFAULT_DOF_INDICES,
    DEFAULT_TRUNCATION,
    DEFAULT_ZN_SELECTED,
    CORNER_NAMES,
    build_olr_sensitivity_matrix,
    extract_olr,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="infile", required=True, help="Input nightly table parquet")
    ap.add_argument("--out", required=True, help="Output OLR parquet")
    ap.add_argument(
        "--ofc-config-dir",
        required=True,
        help="Path to ts_config_mttcs/MTAOS/ofc",
    )
    ap.add_argument("--truncation", type=int, default=DEFAULT_TRUNCATION)
    ap.add_argument("--zn-selected", type=int, nargs="+", default=DEFAULT_ZN_SELECTED)
    ap.add_argument("--dof-indices", type=int, nargs="+", default=DEFAULT_DOF_INDICES)
    ap.add_argument("--seq-min", type=int, default=None)
    ap.add_argument("--seq-max", type=int, default=None)
    args = ap.parse_args()

    ofc_config_dir = os.path.expanduser(args.ofc_config_dir)
    table = pd.read_parquet(args.infile)
    print(f"Loaded {args.infile}: {len(table)} rows")

    sens_mat = build_olr_sensitivity_matrix(
        ofc_config_dir,
        truncation=args.truncation,
        zn_selected=args.zn_selected,
    )
    print(f"Sensitivity matrix: {sens_mat.shape}")

    records = extract_olr(
        table,
        sens_mat,
        indices=args.dof_indices,
        seq_min=args.seq_min,
        seq_max=args.seq_max,
    )
    if not records:
        print(f"ERROR: no usable seqs in {args.infile}", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(records)

    # Self-check: olr_deviation == olr_opd - intrinsic (where intrinsic present).
    _check_identity(out_df)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_df.to_parquet(args.out)
    print(f"Wrote {args.out}: {len(out_df)} rows, {len(out_df.columns)} cols")


def _check_identity(df, atol=1e-4):
    """Assert olr_deviation == olr_opd - intrinsic for every corner-row."""
    n_checked = 0
    for corner in CORNER_NAMES:
        for _, row in df.iterrows():
            intr = row.get(f"intrinsic_{corner}")
            if intr is None:
                continue
            intr = np.asarray(intr, dtype=float)
            if not np.isfinite(intr).all():
                continue
            opd = np.asarray(row[f"olr_opd_{corner}"], dtype=float)
            dev = np.asarray(row[f"olr_deviation_{corner}"], dtype=float)
            if not np.allclose(dev, opd - intr, atol=atol):
                raise AssertionError(
                    f"OLR identity violated at seq={row['seq']} corner={corner}"
                )
            n_checked += 1
    print(f"  identity check passed: olr_deviation == olr_opd - intrinsic ({n_checked} corner-rows)")


if __name__ == "__main__":
    main()
