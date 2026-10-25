#!/usr/bin/env python3
"""Compare two ``wfs_corner_compare`` builds, table by table, at zero tolerance by default.

Usage::

    python -m rubinwork.products.wfs_corner_compare.compare_builds <dir-a> <dir-b> [--atol X]
"""

import argparse
import pathlib
import sys

from ..fam_tables.compare_builds import compare_table
from . import reader


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args(argv)
    failed = False
    for table, key in reader.TABLE_KEYS.items():
        pa, pb = pathlib.Path(args.a) / f"{table}.parquet", pathlib.Path(args.b) / f"{table}.parquet"
        print(f"--- {table}")
        if not pa.exists() or not pb.exists():
            print(f"    MISSING: {pa if not pa.exists() else pb}")
            failed = True
            continue
        rep = compare_table(pa, pb, list(key), rtol=args.rtol, atol=args.atol)
        print(f"    rows: a={rep['rows_a']} b={rep['rows_b']} (counts)")
        if rep["rows_a"] != rep["rows_b"] or rep["differing"]:
            failed = True
        for col, mx in rep["differing"]:
            print(f"    {col}: max abs difference "
                  f"{'(non-numeric)' if mx is None else f'{mx:.6g}'}")
        if not rep["differing"]:
            print("    values: identical within tolerance")
    print("\nRESULT:", "DIFFERENCES FOUND" if failed else "builds match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
