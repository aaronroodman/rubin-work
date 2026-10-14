#!/usr/bin/env python3
"""Acceptance check for a `fam_tables` build that only re-derived the M1M3 gradients.

The gradient fix of 2026-10-13 (see ``notes/status/organization_plan.md`` section 10)
replaced the build-time recomputation of the four M1M3 bulk thermal gradients with a join
against the value-added database. A build made with that fix, reusing an earlier build's
``mktable`` and ``fit`` outputs, must differ from that earlier build in **exactly** those
four columns and nothing else. This checks both halves of that claim:

1. the four gradient columns equal ``visit_telemetry`` at **zero tolerance** on every
   visit the database has a gradient for, with the NaN count reported;
2. every other column of ``visits``, ``fits`` and ``donuts`` is identical to the reference
   build at zero tolerance, aligned on each table's verified-unique key.

Usage::

    python -m rubinwork.products.fam_tables.refbuild.check_gradient_rebuild \\
        --variant danish_1_2 --reference 20261013 --build 20261014
"""

import argparse
import sys

import numpy as np
import pyarrow.parquet as pq

from .. import reader
from ..builders.run_attach_telemetry import GRADIENT_COLS, GRADIENT_DB_COLS
from ..compare_builds import TABLE_KEYS, compare_table, key_is_unique


def check_gradients(build_dir):
    """Compare a build's gradient columns with the value-added database.

    Parameters
    ----------
    build_dir : `pathlib.Path`
        The build directory to check.

    Returns
    -------
    ok : `bool`
        Whether every column matches at zero tolerance.
    """
    from ...value_added import reader as va

    v = pq.read_table(str(build_dir / "visits.parquet"),
                      columns=["day_obs", "seq_num"] + GRADIENT_COLS).to_pandas()
    db = va.visits(columns=list(GRADIENT_DB_COLS.values()))
    m = v.merge(db[["day_obs", "seq_num"] + list(GRADIENT_DB_COLS.values())],
                on=["day_obs", "seq_num"], how="left")
    print(f"--- gradients vs the value-added database, {len(m)} visits")
    ok = True
    for col, dbcol in GRADIENT_DB_COLS.items():
        a = m[col].to_numpy(dtype=float)
        b = m[dbcol].to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        exact = bool(np.array_equal(a[both], b[both]))
        n_nan_build = int((~np.isfinite(a)).sum())
        n_nan_db = int((~np.isfinite(b)).sum())
        worst = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        print(f"    {col}: {int(both.sum())} visits compared, "
              f"max abs difference {worst:.3e} degC/m, exact: {exact}; "
              f"NaN in build {n_nan_build}, NaN in database {n_nan_db}")
        if not exact or n_nan_build != n_nan_db:
            ok = False
    return ok


def main(argv=None):
    """Command line entry point.

    Returns
    -------
    status : `int`
        0 when both halves of the check pass.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="danish_1_2")
    ap.add_argument("--reference", required=True,
                    help="the build whose mktable outputs were reused")
    ap.add_argument("--build", required=True, help="the build to check")
    args = ap.parse_args(argv)

    ref = reader.build_dir(args.variant, args.reference)
    new = reader.build_dir(args.variant, args.build)
    print(f"reference {ref}\nbuild     {new}\n")

    grad_ok = check_gradients(new)

    print(f"\n--- every other column vs {args.reference}, zero tolerance")
    other_ok = True
    for table in ("visits", "fits", "donuts"):
        pa_, pb = ref / f"{table}.parquet", new / f"{table}.parquet"
        if not (pa_.exists() and pb.exists()):
            print(f"  {table}: MISSING ({pa_.exists()}, {pb.exists()})")
            other_ok = False
            continue
        key = TABLE_KEYS[table]
        for path in (pa_, pb):
            if not key_is_unique(path, key):
                print(f"  {table}: key {'+'.join(key)} is NOT unique on {path}")
                other_ok = False
        rep = compare_table(pa_, pb, key, rtol=0.0, atol=0.0)
        differing = [(c, d) for c, d in rep["differing"] if c not in GRADIENT_COLS]
        grads = [(c, d) for c, d in rep["differing"] if c in GRADIENT_COLS]
        print(f"  {table}: {rep['rows_a']} vs {rep['rows_b']} rows, "
              f"{rep['compared']} columns compared, "
              f"{len(rep['only_a'])} only in the reference, "
              f"{len(rep['only_b'])} only in the new build")
        print(f"    differing: {len(differing)} non-gradient, "
              f"{len(grads)} gradient "
              + (f"(max abs {max(d for _, d in grads if d is not None):.3e} degC/m)"
                 if any(d is not None for _, d in grads) else ""))
        if differing:
            print(f"    UNEXPECTED: {differing[:20]}")
            other_ok = False
        if rep["only_a"] or rep["only_b"]:
            print(f"    UNEXPECTED column-set change: only_a={rep['only_a'][:10]} "
                  f"only_b={rep['only_b'][:10]}")
            other_ok = False
        if rep["rows_a"] != rep["rows_b"]:
            print("    UNEXPECTED row-count change")
            other_ok = False

    print(f"\nRESULT: gradients {'OK' if grad_ok else 'FAILED'}, "
          f"other columns {'OK' if other_ok else 'FAILED'}")
    return 0 if (grad_ok and other_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
