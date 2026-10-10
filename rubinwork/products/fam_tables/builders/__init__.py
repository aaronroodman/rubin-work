"""Builders for the ``fam_tables`` product.

``run_attach_telemetry`` attaches per-visit engineering telemetry to the visits
tables; ``run_blitz_mktable`` (with ``blitz_reader``) builds the Danish 1.3 blitz
recast tables in one pass. ``backfill_visit_sides`` is a one-off migration.

The heavy ``mktable``, ``fit`` and ``combine_*`` steps are **not** here: they are
runners in the external ``ts_intrinsic_wavefront`` package, called by the
product's ``Snakefile`` exactly as ``aos/Snakefile`` called them.

Each module is a command line, run with ``-m``::

    python -m rubinwork.products.fam_tables.builders.run_blitz_mktable --help
"""
