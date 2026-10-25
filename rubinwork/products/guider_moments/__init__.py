"""The ``guider_moments`` product: per-night guider moment tables.

Each of the eight guide sensors reads a small region of interest (ROI) at 5 Hz through an
exposure. A build tracks the guide stars in those stamps and decomposes each sensor's shape
into coadd, static-per-stamp and image-motion second moments (arcsec**2), with the third-order
static/dynamic/cross terms, the temporal covariances and the centroid power spectral
densities (PSDs) beside them. Three tables come out per exposure:

``<seqNum>_moments.parquet``
    this repository's decomposition, one row per (exposure, guide sensor).
``<seqNum>_stars.parquet``
    the ``summit_utils`` `GuiderStarTracker` per-stamp table, one row per
    (exposure, sensor, tracked star).
``<seqNum>_metrics.parquet``
    the ``summit_utils`` `GuiderMetricsBuilder` per-exposure row.

and the combine step concatenates a night's per-exposure files into
``guider_{moments,stars,metrics}_<night>.parquet`` beside ``skipped_visits.txt`` and
``detector_counts_<night>.txt``.

**An incremental product** (plan section 5): the variant is the schema version,
``v6``, taken from `guiderMoments.SCHEMA_VERSION`, and nights are **partitions** under
``products/guider_moments/v6/nights/<YYYYMMDD>/``, each with its own ``manifest.json``.
There is no dated build and no single ``current``: a night is built once and appended, and
an incompatible schema change starts ``v7``. ``v4`` holds the nights built before the
reorganization (20260512-20260712), copied from the old tree rather than rebuilt; the
builders write only ``v6``.

Import the reader -- no ``sys.path`` work::

    from rubinwork.products import guider_moments
    df = guider_moments.load_night(20260706)              # the three combined tables
    mom = guider_moments.load_table('moments', 20260706)  # just one

**The frames are the recurring trap.** Guider moments and the adjacent-CCD moments are each
in their own detector's Data Visualization Coordinate System (DVCS) pixel frame, neither
rotated to the Observatory Coordinate System (OCS). The trace ``T`` compares 1:1 between the
two sides; ``Q1``/``Q2`` are frame-dependent and agree only after an OCS rotation. See
``guider/docs/guider_vs_ccd_moments.md``.

Notes
-----
``summit_utils`` is **external** -- a clone on branch ``u/aaronroodman/guider-fixes`` --
and supplies the raw stamps, the star tracker and the metrics. It is not part of this
repository and is not changed by a build; each night's manifest records the commit used.

The adjacent-CCD science side (``guider_psfmoments_<night>.parquet``, from
``preliminary_visit_image`` stamps) is **not** part of this product: it is the science half
of the guider-vs-CCD comparison study and stays in ``guider/``.
"""

from .reader import (  # noqa: F401
    DETECTOR_COUNTS_FILE,
    NIGHTS_DIR,
    PRODUCT,
    PRODUCT_VARIANT,
    SKIPPED_FILE,
    TABLES,
    TABLE_KEYS,
    load_night,
    load_table,
    night_dir,
    nights,
    schema_version,
    variant_config,
    variants,
)

__all__ = [
    "DETECTOR_COUNTS_FILE",
    "NIGHTS_DIR",
    "PRODUCT",
    "PRODUCT_VARIANT",
    "SKIPPED_FILE",
    "TABLES",
    "TABLE_KEYS",
    "load_night",
    "load_table",
    "night_dir",
    "nights",
    "schema_version",
    "variant_config",
    "variants",
]
