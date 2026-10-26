"""Read ``cwfs_science`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import yaml

from .. import catalog

PRODUCT = "cwfs_science"
"""Product name, as it appears in the catalog and in every manifest."""

TABLES = ("ccd_visits", "raft_visits")
"""The parquet tables a build holds."""

TABLE_KEYS = {"ccd_visits": ("day_obs", "seq_num", "detector"),
              "raft_visits": ("day_obs", "seq_num", "raft")}
"""Columns that make a row of each table unique."""

VARIANTS_FILE = pathlib.Path(__file__).with_name("variants.yaml")


@functools.lru_cache(maxsize=1)
def _variants_doc():
    """Parsed ``variants.yaml``, read once per process."""
    with open(VARIANTS_FILE) as f:
        return yaml.safe_load(f) or {}


def variants(registered_only=True):
    """Names of every defined variant, sorted.

    Parameters
    ----------
    registered_only : `bool`, optional
        Omit entries carrying ``registered: false`` (the default).

    Returns
    -------
    names : `list` [`str`]
    """
    entries = _variants_doc().get("variants") or {}
    if registered_only:
        entries = {k: v for k, v in entries.items() if (v or {}).get("registered", True)}
    return sorted(entries)


def variant_config(variant):
    """The ``variants.yaml`` entry of one variant.

    Raises
    ------
    `KeyError`
        If `variant` is not defined.
    """
    entries = _variants_doc().get("variants") or {}
    if variant not in entries:
        raise KeyError(f"{variant!r} is not a {PRODUCT} variant; defined: {sorted(entries)}")
    return entries[variant]


def build_dir(variant, build):
    """Where a build of `variant` is written, whether or not it exists yet."""
    return catalog.products_root() / PRODUCT / variant / build


def load_table(table, variant, build=catalog.CURRENT):
    """Read one table of a build.

    Parameters
    ----------
    table : `str`
        One of `TABLES`.
    variant : `str`
        Variant name, e.g. ``"d13-blitz_rso1071"``.
    build : `str`, optional
        Build name, or ``"current"``.

    Returns
    -------
    data : `pandas.DataFrame`
        ``ccd_visits``: one row per (visit, corner CCD); ``raft_visits``: one per
        (visit, corner raft). Columns in the package docstring.

    Raises
    ------
    `ValueError`
        If `table` is not one of `TABLES`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build or the table is missing.
    """
    if table not in TABLES:
        raise ValueError(f"{table!r} is not a {PRODUCT} table; expected one of {TABLES}")
    return catalog.load(PRODUCT, variant, build=build, file=f"{table}.parquet")


def load(variant, build=catalog.CURRENT, tables=TABLES):
    """Read a build's tables, as a tuple in the order of `tables`."""
    return tuple(load_table(t, variant, build=build) for t in tables)
