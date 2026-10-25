"""Read ``wfs_corner_compare`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import yaml

from .. import catalog
from ..cwfs_tables import reader as _cwfs

PRODUCT = "wfs_corner_compare"
"""Product name, as it appears in the catalog and in every manifest."""

TABLES = ("wfs_corner_compare", "corner_stats")
"""The parquet tables a build holds, in the order `load` returns them."""

TABLE_KEYS = {
    "wfs_corner_compare": ("day_obs", "seq_num", "corner", "j"),
    "corner_stats": ("corner", "j"),
}
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


def cwfs_variant(variant):
    """The ``cwfs_tables`` variant this comparison reads."""
    return variant_config(variant)["cwfs_variant"]


def fam_variant(variant):
    """The ``fam_tables`` variant, through the ``cwfs_tables`` variant it is paired to."""
    return _cwfs.fam_variant(cwfs_variant(variant))


def settings(variant):
    """`rubinwork.corner_compare.DEFAULTS` with this variant's overrides applied."""
    from rubinwork.corner_compare import DEFAULTS

    return {**DEFAULTS, **(variant_config(variant).get("settings") or {})}


def dir_name(variant):
    """The variant's directory under ``aos/output/wfs_corner_compare/<fam>/`` in the old tree."""
    return variant_config(variant)["dir_name"]


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
        Variant name, e.g. ``"d12-refitWcs"``.
    build : `str`, optional
        Build name, or ``"current"``.

    Returns
    -------
    data : `pandas.DataFrame`
        ``wfs_corner_compare``: one row per (triplet, corner, Noll j), with
        ``fam_interp`` and ``cwfs_median`` in µm of wavefront.
        ``corner_stats``: per (corner, j) the pair count ``n``, Pearson ``r`` and
        Spearman ``rho`` (dimensionless), Huber ``slope`` (dimensionless) and ``off``
        [µm], and ``rms``, the nMAD of the residuals [µm].

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
