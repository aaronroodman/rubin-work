"""The ``fam_tables`` product: per-donut, per-visit and Double-Zernike-fit tables
from Full Array Mode (FAM) wavefront processing.

Three tables per build, all keyed on the FAM visit (the **extra-focal** exposure of the
triplet):

``donuts.parquet``
    One row per donut per visit: the measured wavefront Zernike coefficients (microns of
    wavefront), the donut's field position and its quality flags.
``visits.parquet``
    One row per visit: pointing, rotator and filter, the visit quality cuts, and — where
    ``attach_telemetry`` has run — the per-visit engineering telemetry.
``fits.parquet``
    One row per visit: the Double Zernike (DZ) fit to that visit's donuts.

Variants are defined in ``variants.yaml``; the variant name is the short ``dir_name``
from ``aos/param_sets.yaml``, and each entry carries the long ``param_set`` key that
remains the recorded identity elsewhere.

Read a build with `load`, which returns the three tables::

    from rubinwork.products.fam_tables import load
    donuts, visits, fits = load("danish_1_2")

Build one with the module's command line, which drives the same steps the
``aos/Snakefile`` rules used to::

    python -m rubinwork.products.fam_tables.build --variant danish_1_2 --chunk 20251116_20251130
"""

from .reader import (TABLES, build_dir, load, load_table, variant_config, variants,
                     variant_for_param_set)

__all__ = ["TABLES", "build_dir", "load", "load_table", "variant_config", "variants",
           "variant_for_param_set"]
