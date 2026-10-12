"""Read a ``guider_moments`` build: per-night partitions of guider moment tables.

An **incremental** product (plan section 5), so the variant is a schema version and a
night is a partition rather than a dated build::

    products/guider_moments/v6/nights/<YYYYMMDD>/
        guider_moments_<night>.parquet
        guider_stars_<night>.parquet
        guider_metrics_<night>.parquet
        skipped_visits.txt  detector_counts_<night>.txt  manifest.json
        seq/<seqNum>_{moments,stars,metrics}.parquet

Each night carries its own ``manifest.json``, so a night's provenance -- the code commit,
the ``summit_utils`` commit and the tracker settings -- is recorded per night rather than
per build. `rubinwork.products.catalog.path` resolves a night the same way it resolves a
build, with ``build="nights/<night>"``.

Typical read::

    from rubinwork.products import guider_moments
    mom = guider_moments.load_table('moments', 20260706)
    tables = guider_moments.load_night(20260706)      # all three, as a dict
"""
import functools
import pathlib

import pandas as pd
import yaml

from .. import catalog

VARIANTS_FILE = pathlib.Path(__file__).with_name('variants.yaml')
"""The variant definitions: one entry per schema version."""

PRODUCT = 'guider_moments'
"""Product name, as it appears in the catalog and in every manifest."""

PRODUCT_VARIANT = 'v6'
"""The only variant today: the schema version `guiderMoments.SCHEMA_VERSION` writes.

An **incremental** product carries a schema version rather than dated builds (plan
section 5). A column renamed, removed or redefined starts ``v7``; added columns and
appended nights do not. A test asserts this equals ``SCHEMA_VERSION``, so the two cannot
drift.
"""

NIGHTS_DIR = 'nights'
"""The subdirectory of a variant that holds the per-night partitions."""

TABLES = ('moments', 'stars', 'metrics')
"""The three tables a night holds, in the order the builder writes them."""

TABLE_KEYS = {
    'moments': ('expId', 'detector'),
    'stars': ('expid', 'detector', 'trackid'),
    'metrics': ('expId',),
}
"""Columns that make a row of each table unique, verified on the 3a reference build.

Note the case split: this repository's ``moments`` table uses ``expId`` and the
``summit_utils`` ``stars`` table uses ``expid``. Both are as the writers produce them.
"""

SKIPPED_FILE = 'skipped_visits.txt'
"""Visits with missing guider data or no tracked stars, written by the combine step."""

DETECTOR_COUNTS_FILE = 'detector_counts_{night}.txt'
"""Per-visit guide-sensor count distribution, written by the combine step."""


def _table_file(table, night):
    """File name of one combined table of a night.

    Parameters
    ----------
    table : `str`
        One of `TABLES`.
    night : `int` or `str`
        ``day_obs`` as ``YYYYMMDD``.

    Returns
    -------
    name : `str`
        The file name, without a directory.
    """
    if table not in TABLES:
        raise ValueError(f'unknown table {table!r}; known tables: {list(TABLES)}')
    return f'guider_{table}_{int(night)}.parquet'


def night_dir(night, variant=PRODUCT_VARIANT):
    """Where a night's partition is written, whether or not it exists yet.

    `rubinwork.products.catalog.path` resolves only nights already on disk and complete;
    a builder needs the path to write into.

    Parameters
    ----------
    night : `int` or `str`
        ``day_obs`` as ``YYYYMMDD``.
    variant : `str`, optional
        Product variant, ``'v6'`` by default.

    Returns
    -------
    night_dir : `pathlib.Path`
        A directory under the products root. Not created.
    """
    return catalog.products_root() / PRODUCT / variant / NIGHTS_DIR / str(int(night))


def nights(variant=PRODUCT_VARIANT, complete_only=True):
    """The nights this variant holds.

    Parameters
    ----------
    variant : `str`, optional
        Product variant, ``'v6'`` by default.
    complete_only : `bool`, optional
        Keep only nights whose manifest says ``status: "complete"``. `True` by default, so
        an interrupted night is invisible, which is the point of the manifest.

    Returns
    -------
    nights : `list` [`int`]
        ``day_obs`` values, ascending. Empty when the variant has no nights yet.
    """
    root = catalog.products_root() / PRODUCT / variant / NIGHTS_DIR
    if not root.is_dir():
        return []
    found = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        if complete_only and not catalog.is_complete(entry):
            continue
        found.append(int(entry.name))
    return found


def load_table(table, night, variant=PRODUCT_VARIANT, night_root=None):
    """One combined table of one night.

    Parameters
    ----------
    table : `str`
        One of `TABLES`: ``'moments'``, ``'stars'`` or ``'metrics'``.
    night : `int` or `str`
        ``day_obs`` as ``YYYYMMDD``.
    variant : `str`, optional
        Product variant, ``'v6'`` by default.
    night_root : `str` or `pathlib.Path`, optional
        Read from this directory instead of the catalog. For a scratch or reference build,
        which has no manifest and so is not in the catalog.

    Returns
    -------
    table : `pandas.DataFrame`
        The table as written. Moments are in arcsec**2 (second order) and arcsec**3
        (third order); see ``guiderMoments`` for the column definitions.
    """
    if night_root is not None:
        directory = pathlib.Path(night_root)
    else:
        directory = catalog.path(PRODUCT, variant, f'{NIGHTS_DIR}/{int(night)}')
    return pd.read_parquet(directory / _table_file(table, night))


def load_night(night, variant=PRODUCT_VARIANT, tables=TABLES, night_root=None):
    """Every combined table of one night.

    Parameters
    ----------
    night : `int` or `str`
        ``day_obs`` as ``YYYYMMDD``.
    variant : `str`, optional
        Product variant, ``'v6'`` by default.
    tables : `tuple` [`str`], optional
        Which tables to read; all three by default.
    night_root : `str` or `pathlib.Path`, optional
        Read from this directory instead of the catalog.

    Returns
    -------
    tables : `dict` [`str`, `pandas.DataFrame`]
        Keyed by the names in `tables`.
    """
    return {t: load_table(t, night, variant=variant, night_root=night_root)
            for t in tables}


@functools.lru_cache(maxsize=1)
def _variants_doc():
    """Parsed ``variants.yaml``, read once per process."""
    with open(VARIANTS_FILE) as handle:
        return yaml.safe_load(handle) or {}


def variants():
    """The defined variants.

    Returns
    -------
    variants : `list` [`str`]
        Variant names, in ``variants.yaml`` order. A defined variant need not have any
        nights on disk; use `nights` for that.
    """
    return list((_variants_doc().get('variants') or {}))


def variant_config(variant=PRODUCT_VARIANT):
    """One variant's full ``variants.yaml`` entry.

    Parameters
    ----------
    variant : `str`, optional
        Product variant, ``'v6'`` by default.

    Returns
    -------
    config : `dict`
        The entry: the schema version, the Butler query, the guider readout rate in Hz,
        and the tracker and moment settings the pipeline runs with.

    Raises
    ------
    `KeyError`
        If the variant is not defined.
    """
    entries = _variants_doc().get('variants') or {}
    if variant not in entries:
        raise KeyError(f'no variant {variant!r} of {PRODUCT}; '
                       f'known variants: {list(entries)}')
    return entries[variant]


def schema_version(variant=PRODUCT_VARIANT):
    """The schema version a variant writes.

    Parameters
    ----------
    variant : `str`, optional
        Product variant, ``'v6'`` by default.

    Returns
    -------
    version : `int`
        ``guiderMoments.SCHEMA_VERSION`` as the variant records it.
    """
    return int(variant_config(variant)['schema_version'])
