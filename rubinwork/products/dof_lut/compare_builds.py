#!/usr/bin/env python3
"""Compare two ``dof_lut`` builds: three parquet tables, their schema metadata and the
frozen provenance.

The regression check for the phase 3a code move (``notes/status/organization_plan.md``):
build a reference before the move and the same one after, then compare. Values are
expected to match exactly, so the tolerances default to zero.

The parquet value comparison is
`rubinwork.products.fam_tables.compare_builds.compare_table`, reused rather than copied.
Each table has its own key, verified unique on the reference: ``lut`` on ``dof_index``
(50 of 50 rows), ``lut_dz`` on ``(k, j)`` (126 of 126), ``lut_by_rotbin`` on
``(rotbin_center, dof_index)`` (450 of 450; counts dimensionless). Uniqueness is asserted
on both sides, so a build that breaks it fails loudly instead of silently mis-aligning.

**Two things no earlier product's comparison checked.**

``lut.parquet`` and ``lut_dz.parquet`` carry eight fields of parquet **schema metadata**
-- ``param_set``, ``mi_name``, ``reduce``, ``n_dof``, ``n_keep``, ``n_visits``, ``k_min``,
``k_max``. The mode counts and the visit count behind a LUT live only there, so a build
that silently changed ``n_keep`` would compare clean on every row while being a different
measurement. They are compared field by field.

``lut_config.yaml`` is the builder's own frozen provenance, and is compared key by key for
the same reason. It overlaps the schema metadata but is not identical to it: it also
carries ``bands``, ``programs`` and ``use_alt_window``.

The PDF is **not** compared: matplotlib embeds a creation timestamp, so two runs of
identical code produce different bytes. It is checked for existence and comparable size.

Usage::

    python -m rubinwork.products.dof_lut.compare_builds <dir-a> <dir-b>
    python -m rubinwork.products.dof_lut.compare_builds <dir-a> <dir-b> --tables lut_dz
"""

import argparse
import pathlib
import sys

import pyarrow.parquet as pq
import yaml

from ..fam_tables.compare_builds import compare_table, key_is_unique
from . import reader

TABLE_KEYS = dict(reader.TABLE_KEYS)
"""Row key per table, from the reader so the two cannot drift apart."""

METADATA_TABLES = ("lut", "lut_dz")
"""Tables carrying parquet schema metadata, compared field by field."""

PDF_GLOBS = ("*.pdf",)
"""Plots to size-check rather than byte-compare."""




def schema_metadata(path):
    """The parquet schema metadata of one table, decoded.

    Parameters
    ----------
    path : `pathlib.Path`
        A parquet file.

    Returns
    -------
    metadata : `dict` [`str`, `str`]
        Every schema-metadata field, keys and values decoded to `str`. Empty when the
        file carries none.
    """
    raw = pq.read_schema(str(path)).metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def compare_mappings(map_a, map_b):
    """Compare two flat mappings key by key.

    Parameters
    ----------
    map_a, map_b : `dict`
        The two mappings, from schema metadata or parsed YAML.

    Returns
    -------
    report : `dict`
        ``n_keys`` compared, ``only_a`` and ``only_b`` key names, and ``differing`` as a
        list of ``(key, value in a, value in b)``.
    """
    keys_a, keys_b = set(map_a), set(map_b)
    shared = sorted(keys_a & keys_b)
    differing = [(k, map_a[k], map_b[k]) for k in shared if map_a[k] != map_b[k]]
    return {
        "n_keys": len(shared),
        "only_a": sorted(keys_a - keys_b),
        "only_b": sorted(keys_b - keys_a),
        "differing": differing,
    }


def _report_mapping(label, rep):
    """Print one mapping comparison. Returns `True` when it found a difference."""
    print(f"    {label}: {rep['n_keys']} keys compared (count, dimensionless)"
          f" | only in a: {len(rep['only_a'])} | only in b: {len(rep['only_b'])}")
    bad = bool(rep["only_a"] or rep["only_b"] or rep["differing"])
    if rep["only_a"]:
        print(f"      only in a: {rep['only_a']}")
    if rep["only_b"]:
        print(f"      only in b: {rep['only_b']}")
    if rep["differing"]:
        print(f"      differing: {len(rep['differing'])} keys")
        for key, va, vb in rep["differing"]:
            print(f"        {key}: a={va!r} b={vb!r}")
    elif not bad:
        print(f"      {label}: identical")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a", help="first build directory (the reference)")
    ap.add_argument("dir_b", help="second build directory")
    ap.add_argument("--tables", nargs="+", default=None,
                    help="restrict to tables whose name contains one of these strings "
                         "(default: all of them)")
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
        # lut_by_rotbin is conditional on the fit table having a rotator-angle column.
        # Absent from BOTH sides is agreement, not a difference; absent from one is a
        # real difference between the builds.
        if not pa.exists() and not pb.exists():
            print("    absent from both builds (conditional output) -- nothing to "
                  "compare")
            continue
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

        # The mode counts and visit count live ONLY here, so a build that changed them
        # would otherwise compare clean on every row.
        if name in METADATA_TABLES:
            meta_rep = compare_mappings(schema_metadata(pa), schema_metadata(pb))
            if _report_mapping("schema metadata", meta_rep):
                failed = True

    # The builder's own frozen provenance, compared key by key.
    print(f"\n--- provenance  ({reader.PROVENANCE})")
    pa, pb = a / reader.PROVENANCE, b / reader.PROVENANCE
    if not pa.exists() or not pb.exists():
        print(f"    MISSING: {pa if not pa.exists() else pb}")
        failed = True
    else:
        with open(pa) as f:
            prov_a = yaml.safe_load(f) or {}
        with open(pb) as f:
            prov_b = yaml.safe_load(f) or {}
        if _report_mapping("provenance", compare_mappings(prov_a, prov_b)):
            failed = True

    # The PDF carries a matplotlib creation timestamp, so compare sizes, not bytes.
    print("\n--- plots (size only; matplotlib embeds a timestamp)")
    for pattern in PDF_GLOBS:
        rels = sorted({str(p.relative_to(a)) for p in a.glob(pattern)}
                      | {str(p.relative_to(b)) for p in b.glob(pattern)})
        for rel in rels:
            pa, pb = a / rel, b / rel
            if not pa.exists() or not pb.exists():
                print(f"    MISSING: {pa if not pa.exists() else pb}")
                failed = True
                continue
            sa, sb = pa.stat().st_size, pb.stat().st_size
            frac = abs(sa - sb) / sa if sa else 0.0
            print(f"    {rel}: a={sa:,} b={sb:,} bytes (difference {frac:.2%} of a)")

    print("\nRESULT:", "DIFFERENCES FOUND" if failed else "builds match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
