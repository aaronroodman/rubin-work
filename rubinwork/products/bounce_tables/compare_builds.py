#!/usr/bin/env python3
"""Compare two ``bounce_tables`` builds: the four parquet tables, and optionally the
nine study PDFs the same analysis renders.

The regression check for the phase 3a code move (``notes/status/organization_plan.md``):
build a reference before the move and the same one after, then compare. Values are
expected to match exactly, so the tolerances default to zero.

The parquet value comparison is
`rubinwork.products.fam_tables.compare_builds.compare_table`, reused rather than copied.
Each table has its own key, verified unique on the reference: ``bounce_kj_stats`` on
``(bounce, night, comparison, k, j)`` (1,890 of 1,890 rows), ``bounce_dof_stats`` on
``(bounce, comparison, night, kind, index)`` (2,550 of 2,550), ``bounce_fwhm_metric`` on
``(bounce, comparison)`` (6 of 6), ``bounce_fwhm_vs_bvalue`` on
``(bounce, comparison, night)`` (9 of 9; counts dimensionless). Uniqueness is asserted on
both sides, so a build that breaks it fails loudly instead of silently mis-aligning.

**``night`` has to be in the ``bounce_kj_stats`` key.** Without it only 756 of the 1,890
rows are distinct, because each leg carries a per-night row per (k, j) plus a pooled one
— so the plan-shaped guess would have compared the wrong rows against each other. It is a
**string** there and on ``bounce_dof_stats``, where the pooled row carries the literal
``'all'``, and an **int64** on ``bounce_fwhm_vs_bvalue``, which has per-night rows only.

**The nine PDFs are the study's half**, and after the split they are written by
``aos/code/bounce/run_bounce.py`` into a different directory from the tables. Pass
``--plots-a`` / ``--plots-b`` to size-check them; with neither, only the tables are
compared. They are never byte-compared: matplotlib embeds a creation timestamp, so two
runs of identical code produce different bytes.

Unlike ``dof_lut`` there is no schema metadata and no frozen provenance file to compare:
this builder stamps neither. What a build ran with is in its ``manifest.json``.

Usage::

    python -m rubinwork.products.bounce_tables.compare_builds <dir-a> <dir-b>
    python -m rubinwork.products.bounce_tables.compare_builds <dir-a> <dir-b> \\
        --plots-a <ref-dir> --plots-b <moved-dir>
"""

import argparse
import pathlib
import sys

from ..fam_tables.compare_builds import compare_table, key_is_unique
from . import reader

TABLE_KEYS = dict(reader.TABLE_KEYS)
"""Row key per table, from the reader so the two cannot drift apart."""

PLOTS = ("bounce_summary.pdf", "bounce_dz_vs_ordinal.pdf",
         "bounce_vmode_vs_ordinal.pdf", "bounce_dof_vs_ordinal.pdf",
         "bounce_dof_night_scatter.pdf", "bounce_dof_night_values.pdf",
         "bounce_5x5_camera_hexapod.pdf", "bounce_fwhm_metric.pdf",
         "bounce_fwhm_vs_bvalue.pdf")
"""The nine study PDFs, size-checked rather than byte-compared.

Seven are conditional on what the run finds — five on the Optical Feedback Control
singular-value decomposition building, one on a bounce having a ``camera_hexapod_only``
leg, and the two FWHM figures on ``lsst.ts.wep`` — so absent from BOTH sides is
agreement. Absent from one side is a real difference.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a", help="first build directory (the reference)")
    ap.add_argument("dir_b", help="second build directory")
    ap.add_argument("--tables", nargs="+", default=None,
                    help="restrict to tables whose name contains one of these strings "
                         "(default: all four)")
    ap.add_argument("--plots-a", default=None,
                    help="directory holding the reference's nine study PDFs "
                         "(default: dir_a)")
    ap.add_argument("--plots-b", default=None,
                    help="directory holding the second build's nine study PDFs; "
                         "after the phase 3a split the study writes these somewhere "
                         "other than the product build directory")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args(argv)

    a, b = pathlib.Path(args.dir_a), pathlib.Path(args.dir_b)
    failed = False

    tables = list(TABLE_KEYS)
    if args.tables:
        tables = [t for t in tables if any(s in t for s in args.tables)]
    print(f"comparing {len(tables)} table(s) (count, dimensionless)\n")

    for name in tables:
        key = list(TABLE_KEYS[name])
        rel = f"{name}.parquet"
        pa, pb = a / rel, b / rel
        print(f"--- {name}  ({rel})")
        if not pa.exists() or not pb.exists():
            print(f"    MISSING: {pa if not pa.exists() else pb}")
            failed = True
            continue

        for side, path in (("a", pa), ("b", pb)):
            unique, n_unique, n_rows = key_is_unique(path, key)
            print(f"    key {'+'.join(key)} on {side}: {n_unique} unique of {n_rows} rows"
                  f" (counts, dimensionless) -> {'unique' if unique else 'NOT UNIQUE'}")
            if not unique:
                print("    ERROR: the key does not separate every row, so a value "
                      "comparison would be meaningless")
                failed = True

        rep = compare_table(pa, pb, key, rtol=args.rtol, atol=args.atol)
        print(f"    rows: a={rep['rows_a']:,} b={rep['rows_b']:,}"
              f"{'  MISMATCH' if rep['rows_a'] != rep['rows_b'] else ''}")
        print(f"    columns compared: {rep['compared']} (count, dimensionless)"
              f" | only in a: {len(rep['only_a'])} | only in b: {len(rep['only_b'])}")
        if rep["only_a"]:
            print(f"      only in a: {rep['only_a']}")
        if rep["only_b"]:
            print(f"      only in b: {rep['only_b']}")
        if rep["key_aligned"] is False:
            print("    WARNING: key columns do not align, so value diffs are unreliable")
        if rep["rows_a"] != rep["rows_b"] or rep["differing"]:
            failed = True
        if rep["differing"]:
            print(f"    differing: {len(rep['differing'])} columns")
            for col, mx in rep["differing"]:
                print(f"      {col}: max abs difference "
                      f"{'(non-numeric)' if mx is None else f'{mx:.6g}'}")
        else:
            print("    values: identical")

    # The nine study PDFs, by size: matplotlib embeds a creation timestamp.
    plots_a = pathlib.Path(args.plots_a) if args.plots_a else a
    plots_b = pathlib.Path(args.plots_b) if args.plots_b else b
    print("\n--- study plots (size only; matplotlib embeds a timestamp)")
    print(f"    a: {plots_a}\n    b: {plots_b}")
    n_present = 0
    for rel in PLOTS:
        pa, pb = plots_a / rel, plots_b / rel
        if not pa.exists() and not pb.exists():
            print(f"    {rel}: absent from both (conditional output) -- nothing to "
                  f"compare")
            continue
        if not pa.exists() or not pb.exists():
            print(f"    MISSING: {pa if not pa.exists() else pb}")
            failed = True
            continue
        n_present += 1
        sa, sb = pa.stat().st_size, pb.stat().st_size
        frac = abs(sa - sb) / sa if sa else 0.0
        print(f"    {rel}: a={sa:,} b={sb:,} bytes (difference {frac:.2%} of a)")
    print(f"    {n_present} of {len(PLOTS)} PDFs present on both sides "
          f"(counts, dimensionless)")

    print("\nRESULT:", "DIFFERENCES FOUND" if failed else "builds match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
