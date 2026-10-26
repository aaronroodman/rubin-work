"""Concatenate the per-night parquet files of one table.

Command line, used by the Snakefile::

    python -m rubinwork.products.cwfs_science.builders.combine --table ccd_visits \\
        --out ccd_visits.parquet ccd_20260315.parquet ccd_20260316.parquet
"""

import argparse
import sys

import pandas as pd

from ..reader import TABLE_KEYS

__all__ = ["combine", "main"]


def combine(paths, table="ccd_visits"):
    """One table from the per-night files, sorted on its key.

    Raises
    ------
    `ValueError`
        If the key is not unique.
    """
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    key = list(TABLE_KEYS[table])
    df = df.sort_values(key, ignore_index=True)
    dup = df.duplicated(key)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated {key} rows (count)")
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--table", required=True, choices=sorted(TABLE_KEYS))
    p.add_argument("--out", required=True)
    p.add_argument("nights", nargs="+")
    a = p.parse_args(argv)
    df = combine(a.nights, a.table)
    df.to_parquet(a.out, index=False)
    print(f"wrote {len(df)} rows (count) to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
