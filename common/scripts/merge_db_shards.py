#!/usr/bin/env python3
"""Merge sharded value-added database files into the main database.

A build pass is parallelized by splitting its nights across shards, each writing its own
DuckDB file, because DuckDB's file lock is process-wide and excludes readers as well as
writers — several processes cannot write one file. This script folds the shard files back
into the main database, table by table, with the same conflict resolution the builders use so
a shard never clobbers a column another pass owns.

Usage
-----
Merge every shard of one build and leave the shard files in place::

    python common/scripts/merge_db_shards.py --shards output/value_added/shards/telemetry_*.duckdb

Report what would be merged without writing::

    python common/scripts/merge_db_shards.py --shards 'output/value_added/shards/*.duckdb' --dry-run

Notes
-----
Merging is idempotent: every table is written with ``ON CONFLICT ... DO UPDATE``, so
re-merging the same shard produces the same rows. `visit_telemetry` merges on ``visit_id``
and `optical_state` on ``(visit_id, variant_id)``.

`visit_telemetry` is merged **column by column from the shard's non-NULL values only**. A
shard that ran one group leaves the other groups' columns NULL, and a blind row-wise upsert
would overwrite a sibling shard's real values with those NULLs. `COALESCE` on the excluded
value keeps whichever side has data.

`column_coverage` is **not** merged: it is a derived inventory rebuilt from the merged data
by `efd_db.refresh_coverage`, which this script calls once at the end.
"""
import argparse
import glob
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from common import efd_db                                            # noqa: E402

#: Tables merged, with their conflict keys. Order matters: `state_variant` must precede
#: `optical_state`, whose rows reference it.
MERGE_ORDER = [
    ('state_variant', ('variant_id',)),
    ('visit_telemetry', ('visit_id',)),
    ('optical_state', ('visit_id', 'variant_id')),
    ('fetch_log', ('day_obs', 'group_name')),
]


def table_columns(con, table):
    """Column names of `table` in the order the table declares them.

    Returns
    -------
    cols : `list` [`str`]
        Empty if the table does not exist.
    """
    rows = con.execute(
        'SELECT column_name FROM information_schema.columns '
        'WHERE table_name = ? ORDER BY ordinal_position', [table]).fetchall()
    return [r[0] for r in rows]


def merge_table(con, table, keys, shard_alias, coalesce=False, dry_run=False):
    """Merge one table from an attached shard into the main database.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Writable connection to the main database, with the shard already attached.
    table : `str`
    keys : `tuple` [`str`]
        Conflict key columns.
    shard_alias : `str`
        ATTACH alias of the shard.
    coalesce : `bool`, optional
        Update non-key columns as ``COALESCE(excluded.c, table.c)`` rather than
        ``excluded.c``, so a NULL in the shard does not erase an existing value. Required
        for the wide `visit_telemetry`, where each group owns a disjoint column subset.
    dry_run : `bool`, optional
        Count the shard's rows and return without writing.

    Returns
    -------
    n : `int`
        Rows the shard offered for this table (not the number that changed, which DuckDB
        does not report for an upsert).
    """
    main_cols = table_columns(con, table)
    # `information_schema` is not qualified per attached database, so the shard's columns come
    # from `duckdb_columns()`, which carries a `database_name`.
    shard_cols = [r[0] for r in con.execute(
        'SELECT column_name FROM duckdb_columns() '
        'WHERE database_name = ? AND table_name = ? ORDER BY column_index',
        [shard_alias, table]).fetchall()]
    if not shard_cols:
        return 0
    # Only columns present on both sides can move; a shard built by an older schema is
    # merged for what it has rather than refused.
    cols = [c for c in shard_cols if c in main_cols]
    missing = [c for c in main_cols if c not in shard_cols]
    if missing:
        print(f'    {table}: shard lacks {len(missing)} column(s) present in the main '
              f'database ({", ".join(missing[:4])}{" ..." if len(missing) > 4 else ""}); '
              f'those stay as the main database has them')
    n = con.execute(f'SELECT COUNT(*) FROM {shard_alias}.{table}').fetchone()[0]
    if dry_run or not n:
        return int(n)
    sel = ', '.join(cols)
    upd = [c for c in cols if c not in keys]
    if upd:
        if coalesce:
            sets = ', '.join(f'{c} = COALESCE(excluded.{c}, {table}.{c})' for c in upd)
        else:
            sets = ', '.join(f'{c} = excluded.{c}' for c in upd)
        action = f'DO UPDATE SET {sets}'
    else:
        action = 'DO NOTHING'
    con.execute(f'INSERT INTO {table} ({sel}) SELECT {sel} FROM {shard_alias}.{table} '
                f'ON CONFLICT ({", ".join(keys)}) {action}')
    return int(n)


def merge_shard(con, path, dry_run=False):
    """Merge every mergeable table of one shard file.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Writable main-database connection.
    path : `pathlib.Path`
    dry_run : `bool`, optional

    Returns
    -------
    counts : `dict` [`str`, `int`]
        Rows offered per table.
    """
    alias = 'shard'
    con.execute(f"ATTACH '{path}' AS {alias} (READ_ONLY)")
    try:
        counts = {}
        for table, keys in MERGE_ORDER:
            counts[table] = merge_table(con, table, keys, alias,
                                        coalesce=(table == 'visit_telemetry'),
                                        dry_run=dry_run)
        return counts
    finally:
        con.execute(f'DETACH {alias}')


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shards', nargs='+', required=True,
                   help='shard database files, or glob patterns matching them')
    p.add_argument('--db', default=None, help='main database; default from efd_db')
    p.add_argument('--dry-run', action='store_true',
                   help='report the rows each shard offers without writing')
    a = p.parse_args(argv)

    paths = []
    for spec in a.shards:
        hits = sorted(glob.glob(spec))
        if not hits:
            if pathlib.Path(spec).exists():
                hits = [spec]
            else:
                print(f'warning: no shard matches {spec!r}')
        paths.extend(pathlib.Path(h).resolve() for h in hits)
    paths = sorted(set(paths))
    if not paths:
        p.error('no shard files found')

    main_db = pathlib.Path(a.db) if a.db else efd_db.default_db_path()
    print(f'main database: {main_db}')
    print(f'{len(paths)} shard(s) to merge'
          + (' (dry run, nothing written)' if a.dry_run else ''))

    con = efd_db.open_db(main_db, create=True)
    totals = {}
    try:
        for path in paths:
            print(f'  {path.name}')
            counts = merge_shard(con, path, dry_run=a.dry_run)
            for table, n in counts.items():
                if n:
                    print(f'    {table}: {n} rows offered')
                totals[table] = totals.get(table, 0) + n
        if not a.dry_run:
            print('rebuilding column_coverage from the merged data')
            efd_db.refresh_coverage(con)
        print('\nrows offered per table across all shards:')
        for table, n in totals.items():
            print(f'  {table:18s} {n}')
        if not a.dry_run:
            print('\nmain database row counts after the merge:')
            for table, _keys in MERGE_ORDER:
                n = con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                print(f'  {table:18s} {n}')
    finally:
        con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
