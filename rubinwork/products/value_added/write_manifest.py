"""Write the ``manifest.json`` that registers a ``value_added`` build in the catalog.

The last step of a build, so a build directory becomes a catalog entry only after the
database is on disk. Nothing here moves the variant's ``current`` symlink; that is
``python -m rubinwork.products.catalog set-current``.

**A scratch build writes no manifest.** This product's builders take a database path —
``--db``, or ``AOS_EFD_DB`` — and that is how a reference build, an archive copy and a
shard all stay out of the catalog. Only a build written into the catalog's own build
directory gets a manifest, so a scratch database can never be mistaken for a registered
build. `is_catalog_build` is the test the builders apply.

**No upstream is recorded for the telemetry tables**: they come from the Engineering
Facility Database (EFD) and the Consolidated Database (ConsDB), which are external
services rather than products of this repository. The MIW-route ``optical_state`` rows and
the ``fam_dz`` rows *do* have repository upstreams — a ``miw`` build and a ``fam_tables``
build — but per-variant rather than per-build, so they are recorded in ``config`` from the
registry rows rather than in ``inputs``, which names one build per product. Phase 3c
changes the registry rows to catalog names (plan section 6), after which they can move to
``inputs``.

Command line::

    python -m rubinwork.products.value_added.write_manifest \\
        --variant v1 --build 20261013
"""

import argparse
import pathlib
import sys

from .. import manifest
from . import reader

__all__ = ["build_config", "is_catalog_build", "write", "main"]


def is_catalog_build(db_path, variant=reader.PRODUCT_VARIANT):
    """Whether `db_path` lies inside a catalog build directory of this product.

    Parameters
    ----------
    db_path : `str` or `pathlib.Path`
        The database a builder is writing.
    variant : `str`, optional
        Product variant, ``'v1'`` by default.

    Returns
    -------
    catalog_build : `bool`
        `True` when the path is under the catalog's ``<products>/value_added/<variant>/``
        directory, so the build should end by writing a manifest. `False` for a scratch
        database — a reference build, a shard, an archive copy — which must not be
        registered.

    Notes
    -----
    Decided by the path alone, not by a flag, so there is no way to register a scratch
    build by passing the wrong argument.

    The variant's ``_work`` subtree is excluded even though it lies inside the variant
    directory: it holds shards and other intermediates, and a shard is scratch. Without
    this, every shard of one sharded run wrote the same ``manifest.json`` in the shared
    shard directory and raced.
    """
    db_path = pathlib.Path(db_path).resolve()
    try:
        root = reader.build_dir(variant).resolve()
    except Exception:
        return False
    if not (root == db_path.parent or root in db_path.parents):
        return False
    return reader.WORK_DIR not in db_path.relative_to(root).parts


def build_config(variant, con=None, extra=None):
    """The manifest ``config``: the schema version and what the registries hold.

    Parameters
    ----------
    variant : `str`
        Product variant, e.g. ``'v1'``.
    con : `duckdb.DuckDBPyConnection`, optional
        An open connection to the build, used to summarise its registry rows. When
        `None` the registry summary is omitted.
    extra : `dict`, optional
        Build-time settings to record, such as the night range and the groups built.

    Returns
    -------
    config : `dict`
        ``variant`` (the schema version), the ``tables`` and ``registries`` the schema
        defines, and when `con` is given a ``state_variants`` / ``fam_variants`` list of
        the variant ids present with their row counts (counts dimensionless), plus the
        ``day_obs`` span. A ``build_options`` key holds `extra`.
    """
    config = {"variant": variant,
              "tables": list(reader.TABLES),
              "registries": list(reader.REGISTRIES)}
    if con is not None:
        config["state_variants"] = _variant_rows(con, "optical_state", "variant_id")
        config["fam_variants"] = _variant_rows(con, "fam_dz", "fam_variant_id")
        span = con.execute(
            "SELECT MIN(day_obs), MAX(day_obs), COUNT(*) FROM visit_telemetry").fetchone()
        config["day_obs_min"], config["day_obs_max"] = span[0], span[1]
        config["n_visit_telemetry_rows"] = span[2]
    if extra:
        config["build_options"] = dict(extra)
    return config


def _variant_rows(con, table, col):
    """``{variant: n_rows}`` for one long table, or ``{}`` when it is empty."""
    try:
        rows = con.execute(
            f"SELECT {col}, COUNT(*) FROM {table} GROUP BY 1 ORDER BY 1").fetchall()
    except Exception:
        return {}
    return {str(v): int(n) for v, n in rows}


def write(variant=reader.PRODUCT_VARIANT, build=None, build_dir=None,
          status="complete", build_options=None, inputs=None):
    """Write the manifest of one ``value_added`` build.

    Parameters
    ----------
    variant : `str`, optional
        Product variant, ``'v1'`` by default.
    build : `str`, optional
        Build name, normally the date (``"20261013"``). Taken from `build_dir`'s name when
        that is given instead.
    build_dir : `str` or `pathlib.Path`, optional
        The build directory. By default the catalog directory of ``variant``/``build``.
    status : `str`, optional
        Manifest ``status``, ``"complete"`` by default. Anything else keeps the build out
        of the catalog.
    build_options : `dict`, optional
        Build-time settings, recorded under ``config.build_options``.
    inputs : `dict`, optional
        Upstream builds read, as ``{product: "variant@build"}``. Normally empty: the
        telemetry comes from the EFD and ConsDB, which are external services, and the
        per-variant repository upstreams are recorded in ``config`` instead.

    Returns
    -------
    manifest_path : `pathlib.Path`
        The manifest that was written.

    Raises
    ------
    `ValueError`
        If `build_dir` and `build` are both absent.

    Notes
    -----
    Row counts come from the database itself, not from a builder's counters, so the
    manifest reports what is on disk.
    """
    if build_dir is None:
        if not build:
            raise ValueError("write() needs `build` or `build_dir`")
        build_dir = reader.build_dir(variant, build)
    build_dir = pathlib.Path(build_dir)
    build = build or build_dir.name

    con = None
    db = build_dir / reader.DB_FILE
    if db.exists():
        con = reader.open_db(db, readonly=True)
    try:
        config = build_config(variant, con=con, extra=build_options)
    finally:
        if con is not None:
            con.close()

    return manifest.write(
        build_dir, product=reader.PRODUCT, variant=variant, build=build,
        config=config, inputs=inputs or None, status=status)


def main(argv=None):
    """Command line entry point.

    Returns
    -------
    status : `int`
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rubinwork.products.value_added.write_manifest",
        description="Write manifest.json for a value_added build, as its last step.")
    parser.add_argument("--variant", default=reader.PRODUCT_VARIANT,
                        help="Schema version (default: v1)")
    parser.add_argument("--build", default=None,
                        help="Build name (YYYYMMDD); taken from --build-dir if omitted")
    parser.add_argument("--build-dir", default=None,
                        help="Build directory (default: the catalog directory of "
                             "<variant>/<build>)")
    parser.add_argument("--status", default="complete",
                        help="Manifest status (default: complete)")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Build-time setting to record under config.build_options; "
                             "repeatable")
    args = parser.parse_args(argv)

    options = {}
    for item in args.option:
        key, _, value = item.partition("=")
        options[key] = value
    path = write(args.variant, build=args.build, build_dir=args.build_dir,
                 status=args.status, build_options=options or None)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def finish_build(db_path, variant=reader.PRODUCT_VARIANT, build_options=None):
    """Write the manifest if `db_path` is a catalog build; do nothing if it is scratch.

    The one line every builder ends with, so the scratch-versus-catalog test lives in one
    place rather than being repeated per builder.

    Parameters
    ----------
    db_path : `str` or `pathlib.Path` or `None`
        The database the builder just wrote. `None` means the builder used the default
        path, which is resolved here.
    variant : `str`, optional
        Product variant, ``'v1'`` by default.
    build_options : `dict`, optional
        Build-time settings to record, such as the night range built.

    Returns
    -------
    manifest_path : `pathlib.Path` or `None`
        The manifest written, or `None` for a scratch build.
    """
    path = pathlib.Path(db_path) if db_path else reader.default_db_path()
    if not is_catalog_build(path, variant):
        return None
    return write(variant, build_dir=path.parent, build_options=build_options)
