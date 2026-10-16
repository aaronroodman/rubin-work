#!/usr/bin/env python3
"""Move the `fam_variant` and `state_variant` path and name columns to catalog names.

Item c3 of the phase-3c `value_added` plan, and plan section 6. Three columns record
names that also exist in the filesystem, so a rename on disk silently breaks the join
between the database and the files -- the breakage ``value_added/docs/schema.md`` warns
about:

* ``fam_variant.fits_path`` -- an **absolute** filesystem path, which breaks whenever the
  tree it names moves. For the one registered variant it was a **dead** path until the
  3c rebuild: the long-key layout ``aos/output/<param_set>/fits.parquet`` has named
  nothing on disk since ``dir_name`` arrived;
* ``state_variant.intrinsic_ref`` -- the joined MIW build directory name, which resolves
  old-tree-first and so names a build ambiguously once the same name exists in both trees.

Each becomes a catalog reference of the form ``<product>:<variant>@<build>``, which
`rubinwork.products.catalog` resolves and which survives a rename of the old tree.

WHAT THIS DELIBERATELY DOES NOT TOUCH
-------------------------------------
**``fam_variant.param_set`` is left alone, although plan section 6 lists it.** It is not
merely descriptive: `reader.fam_variant_id` *derives* the primary key from it, as
``fam__<param_set>__<route>__<prefix>__<scheme>``, and `register_fam_variant`'s
``ON CONFLICT (fam_variant_id) DO UPDATE SET param_set = excluded.param_set`` writes it
back from the CLI argument. Rewriting the stored value to a catalog reference would leave
it inconsistent with the id that was computed from it, and the next
``build_fam_dz --param-set <long key>`` would silently restore the old value. The path and
the build reference are what a reader needs to resolve files; the ``param_set`` is the
variant's identity and stays as it is.

``fam_variant.fam_variant_id`` and ``state_variant.variant_id`` are likewise untouched:
`fam_dz` and `optical_state` rows are **keyed on them**, so rewriting an id would orphan
every row that referenced it (2,528 `fam_dz` rows and ~98,800 MIW `optical_state` rows at
the time of writing).

Usage
-----
Report what would change and write nothing::

    python -m rubinwork.products.value_added.builders.migrate_registry_to_catalog \\
        --db <v1>/live/aos_efd.duckdb --dry-run

Apply::

    python -m rubinwork.products.value_added.builders.migrate_registry_to_catalog \\
        --db <v1>/live/aos_efd.duckdb \\
        --fam-build danish_1_2@20261014 --miw-build d12-A_50_34_i_5rot@20261014

The database is opened read-write, so no reader may hold it. Re-running after a successful
pass is a no-op: a value already in catalog form is left alone.
"""
import argparse
import pathlib
import sys

import duckdb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

CATALOG_SEP = ':'
"""Separator marking a value as already-migrated, as in ``fam_tables:danish_1_2@...``."""


def catalog_ref(product, build_spec):
    """A ``<product>:<variant>@<build>`` reference, with the build resolved if omitted.

    Parameters
    ----------
    product : `str`
        Catalog product name, e.g. ``'fam_tables'`` or ``'miw'``.
    build_spec : `str`
        ``variant`` or ``variant@build``. A bare variant resolves to its ``current``
        build, so the stored reference always names a concrete dated build rather than a
        moving target.

    Returns
    -------
    ref : `str`
        The reference, e.g. ``'miw:d12-A_50_34_i_5rot@20261014'``.

    Raises
    ------
    `ValueError`
        If `build_spec` is empty.
    """
    from rubinwork.products import catalog

    if not build_spec:
        raise ValueError(f'empty build spec for product {product!r}')
    variant, _, build = build_spec.partition('@')
    if not build:
        build = catalog.path(product, variant).name
    else:
        catalog.path(product, variant, build)
    return f'{product}{CATALOG_SEP}{variant}@{build}'


def plan_updates(con, fam_ref=None, miw_ref=None):
    """The rows this pass would change, as ``(table, key, column, old, new)`` tuples.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Open connection; only read here.
    fam_ref : `str`, optional
        Catalog reference for the FAM tables, from `catalog_ref`. When `None`,
        `fam_variant` is skipped.
    miw_ref : `str`, optional
        Catalog reference for the MIW build. When `None`, `state_variant` is skipped.

    Returns
    -------
    updates : `list`
        One tuple per column that would change. A column already holding a catalog
        reference is omitted, which is what makes the pass idempotent.
    """
    updates = []
    if fam_ref:
        rows = con.execute(
            'SELECT fam_variant_id, fits_path FROM fam_variant').fetchall()
        for vid, fits_path in rows:
            if fits_path is not None and CATALOG_SEP not in str(fits_path):
                updates.append(('fam_variant', vid, 'fits_path', fits_path, fam_ref))
    if miw_ref:
        rows = con.execute(
            "SELECT variant_id, intrinsic_ref FROM state_variant "
            "WHERE intrinsic_route = 'miw'").fetchall()
        for vid, ref in rows:
            if ref is not None and CATALOG_SEP not in str(ref):
                updates.append(('state_variant', vid, 'intrinsic_ref', ref, miw_ref))
    return updates


def apply_updates(con, updates):
    """Apply `plan_updates` output, one parameterized statement per row.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Read-write connection.
    updates : `list`
        Tuples from `plan_updates`.

    Returns
    -------
    n : `int`
        Number of column values written.
    """
    key_col = {'fam_variant': 'fam_variant_id', 'state_variant': 'variant_id'}
    for table, key, col, _old, new in updates:
        con.execute(f'UPDATE {table} SET {col} = ? WHERE {key_col[table]} = ?',
                    [new, key])
    return len(updates)


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--db', required=True, help='database to migrate')
    p.add_argument('--fam-build', default=None, metavar='VARIANT[@BUILD]',
                   help="fam_tables build the registered fam_variant now refers to, "
                        "e.g. danish_1_2@20261014")
    p.add_argument('--miw-build', default=None, metavar='VARIANT[@BUILD]',
                   help="miw build the miw-route state_variant rows now refer to, "
                        "e.g. d12-A_50_34_i_5rot@20261014")
    p.add_argument('--dry-run', action='store_true',
                   help='print the updates and write nothing')
    a = p.parse_args(argv)

    if not (a.fam_build or a.miw_build):
        p.error('give --fam-build, --miw-build, or both')

    fam_ref = catalog_ref('fam_tables', a.fam_build) if a.fam_build else None
    miw_ref = catalog_ref('miw', a.miw_build) if a.miw_build else None
    if fam_ref:
        print(f'fam_tables -> {fam_ref}')
    if miw_ref:
        print(f'miw        -> {miw_ref}')

    con = duckdb.connect(a.db, read_only=a.dry_run)
    try:
        updates = plan_updates(con, fam_ref, miw_ref)
        if not updates:
            print('nothing to do: every target column already holds a catalog reference')
            return 0
        for table, key, col, old, new in updates:
            print(f'  {table}.{col}  [{key}]')
            print(f'    old: {old}')
            print(f'    new: {new}')
        if a.dry_run:
            print(f'--dry-run: {len(updates)} column value(s) would change, '
                  f'nothing written')
            return 0
        n = apply_updates(con, updates)
        print(f'wrote {n} column value(s)')
    finally:
        con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
