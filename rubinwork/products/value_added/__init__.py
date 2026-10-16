"""The ``value_added`` product: per-visit engineering telemetry and derived quantities.

An **incremental** product. The Consolidated Database (ConsDB), including its
transformed-Engineering-Facility-Database (EFD) tables, is fast, so **nothing that lives
there is copied here**. This database holds exactly two kinds of thing:

* **EFD quantities.** Direct EFD access is slow and was scattered across several topics'
  scripts. Each quantity is fetched once into this database, which then becomes the single
  place EFD access happens.
* **Value-added quantities** — expensive or intricate to *compute* rather than to fetch.
  The M1M3 bulk thermal gradients and the recovered optical state are the two.

Four data tables and four registries:

``visit_telemetry``
    **Wide**, one row per exposure, 197 columns. Its groups — Trim, hexapod look-up table
    (LUT), Tweak, M1M3 gradients, camera-body temperatures, air turbulence, hexapod motion
    history, twilight — are one value per visit per quantity, arrive together from the
    same per-night fetch, and have no variants.
``optical_state``
    **Long**, keyed ``(visit_id, variant_id)``. The recovered optical state is a *family*
    of variants along three independent axes — reduction scheme (22/12, 50/34),
    intrinsic-wavefront route (batoid, measured intrinsic wavefront (MIW)), and optical
    path difference (OPD) source version — so a new variant is rows, not schema.
``fam_dz``
    **Long**, keyed ``(visit_id, fam_variant_id)``: the full-array-mode Double-Zernike
    fits projected onto the sensitivity-matrix singular value decomposition (SVD).
``m1m3_thermal_r2``
    One row per exposure: the M1M3 bulk thermal gradient fit, in deg C per unit
    normalized ``r^2`` amplitude.
``state_variant`` / ``fam_variant``
    The registries describing each ``optical_state`` / ``fam_dz`` variant.
``column_coverage``
    Every column's group, source, **units** and non-null span. A bare number in this
    database is a bug, so units are never empty.
``fetch_log``
    The telemetry builder's per-night EFD fetch outcome, keyed ``(day_obs, group_name)``.

**A variant must be named on every ``optical_state`` and ``fam_dz`` read.** A long table
with a forgotten variant filter silently multiplies the sample by the variant count,
which is the one failure mode this shape introduces.

Read it as the product reader::

    from rubinwork.products import value_added as efd_db
    df = efd_db.visits(day_obs_range=(20251023, None))
    df = efd_db.join_consdb(df)            # band, pointing, temperatures from ConsDB
    st = efd_db.optical_state('v50_34__miw__consdb_v1')

``default_db_path`` resolves **the catalog's current build first, then the old tree**
``value_added/output/aos_efd.duckdb``, with ``AOS_EFD_DB`` overriding both. So a plain
`open_db` and `connect` both open the current build.

Build it with the builders under `builders`, driven by ``builders/run_build.sh``, which
shards a long backfill because DuckDB's file lock is process-wide and excludes readers as
well as writers. The telemetry builder reaches the EFD only from an interactive node
(``slacrd``, ``sdfiana*``), never from a Slurm compute node, so ``run_build.sh`` refuses
``--mode batch`` for it.

A **scratch** build — one whose database path comes from ``AOS_EFD_DB`` or ``--db`` rather
than the catalog — writes no manifest, so it cannot be mistaken for a catalog build. Only
a build into the catalog's own build directory is registered.

Variants here are schema versions, ``v1`` today (plan section 5): a column renamed,
removed or redefined starts ``v2``, while added columns, added registry rows and appended
nights do not.
"""

from .reader import (  # noqa: F401
    DB_FILE, PRODUCT, PRODUCT_VARIANT, REGISTRIES, TABLE_KEYS, TABLES,
    build_dir, catalog_db_path, connect, coverage, default_db_path, fam_dz, fam_variants,
    fetch_status, join_consdb, load, m1m3_thermal_r2, old_tree_db_path, open_db,
    optical_state, variants, visits,
)

__all__ = ["DB_FILE", "PRODUCT", "PRODUCT_VARIANT", "REGISTRIES", "TABLE_KEYS", "TABLES",
           "build_dir", "catalog_db_path", "connect", "coverage", "default_db_path",
           "fam_dz", "fam_variants", "fetch_status", "join_consdb", "load",
           "m1m3_thermal_r2", "old_tree_db_path", "open_db", "optical_state", "variants",
           "visits"]
