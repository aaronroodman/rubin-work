"""Read ``cwfs_tables`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import yaml

from .. import catalog

PRODUCT = "cwfs_tables"
"""Product name, as it appears in the catalog and in every manifest."""

TABLES = ("donuts", "visits")
"""The two tables a build holds, in the order `load` returns them."""

DONUT_KEY = ("day_obs", "seq_num", "detector", "thx_OCS", "thy_OCS")
"""Columns that make a ``donuts.parquet`` row unique, for comparison alignment.

The tables carry no donut-id column, so the ``fam_tables`` key does not apply here, and
``(day_obs, seq_num, detector)`` alone is **not** unique on the paired variants -- several
donuts share a corner sensor in one exposure. ``thx_OCS``/``thy_OCS`` are field angles in
radians (Optical Coordinate System). The centroid columns would also separate the rows,
but the ``unpaired`` variants do not carry them.
"""

VISIT_KEY = ("day_obs", "seq_num")
"""Columns that make a ``visits.parquet`` row unique: one row per in-focus exposure."""

VARIANTS_FILE = pathlib.Path(__file__).with_name("variants.yaml")

# The fields a variant entry may carry beyond the bookkeeping ones, in the order the
# generated aos/param_sets.yaml writes them -- `reader` before `dataset_type`, which is
# the order that file has today; byte-identity across the phase 3a move depends on it.
# `collection` is handled separately: an entry with no other field is emitted as a bare
# collection string, which is also the shape that file has today.
BUILDER_FIELDS = ("collection", "seq_offset", "reader", "dataset_type",
                  "combine_offsets", "defocused_only")


@functools.lru_cache(maxsize=1)
def _variants_doc():
    """Parsed ``variants.yaml``, read once per process."""
    with open(VARIANTS_FILE) as f:
        return yaml.safe_load(f) or {}


def variants(registered_only=True):
    """Names of every defined variant.

    Parameters
    ----------
    registered_only : `bool`, optional
        Omit entries carrying ``registered: false`` (the default). Those are emitted into
        the generated ``aos/param_sets.yaml`` so recorded provenance keeps resolving, but
        they are not product builds.

    Returns
    -------
    names : `list` [`str`]
        Variant names, sorted. A defined variant need not have a build on disk; use
        `rubinwork.products.catalog.list_products` for what is built.
    """
    entries = _variants_doc().get("variants") or {}
    if registered_only:
        entries = {k: v for k, v in entries.items()
                   if (v or {}).get("registered", True)}
    return sorted(entries)


def variant_config(variant):
    """The full configuration of one variant.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"d12-refitWcs"``.

    Returns
    -------
    config : `dict`
        The ``variants.yaml`` entry: the upstream ``fam_variant`` and
        ``upstream_name``, the Butler ``collection``, and any reader fields.

    Raises
    ------
    `KeyError`
        If `variant` is not defined in ``variants.yaml``.
    """
    entries = _variants_doc().get("variants") or {}
    if variant not in entries:
        raise KeyError(
            f"{variant!r} is not a cwfs_tables variant; defined: {sorted(entries)}")
    return entries[variant]


def fam_variant(variant):
    """The ``fam_tables`` variant whose visits drive this CWFS variant's walk.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    fam_variant : `str`
        A ``fam_tables`` variant name, e.g. ``"danish_1_2"``.
    """
    return variant_config(variant)["fam_variant"]


def _builder_entry(cfg):
    """The ``wfs_collections`` value for one variant: a bare string, or a dict.

    A variant carrying only a collection becomes the bare collection string, which is
    the shape ``aos/param_sets.yaml`` has today; byte-identity of the generated file
    depends on it.
    """
    extra = {k: cfg[k] for k in BUILDER_FIELDS if k != "collection" and k in cfg}
    if not extra:
        return cfg["collection"]
    return {"collection": cfg["collection"], **extra}


def wfs_collections_for(fam_variant_name):
    """The ``wfs_collections`` map of one ``fam_tables`` variant.

    This is the FAM/CWFS triplet link, which this product owns.
    `rubinwork.products.fam_tables.gen_param_sets` calls it to rebuild the
    ``wfs_collections`` block of the generated ``aos/param_sets.yaml``, keyed by each
    entry's ``upstream_name`` so the generated file is unchanged by the move.

    Parameters
    ----------
    fam_variant_name : `str`
        A ``fam_tables`` variant name, e.g. ``"danish_1_2"``.

    Returns
    -------
    collections : `dict`
        ``upstream_name`` -> a bare collection `str`, or a `dict` carrying
        ``collection`` plus the reader fields. Empty if no CWFS variant names this FAM
        variant. Insertion order follows ``variants.yaml``.
    """
    entries = _variants_doc().get("variants") or {}
    return {cfg["upstream_name"]: _builder_entry(cfg)
            for cfg in entries.values()
            if (cfg or {}).get("fam_variant") == fam_variant_name}


def build_dir(variant, build):
    """Where a build of `variant` is written, whether or not it exists yet.

    `rubinwork.products.catalog.path` resolves only builds already on disk; a builder
    needs the path to write into.

    Parameters
    ----------
    variant : `str`
        Variant name.
    build : `str`
        Build name, usually the date the data is produced (``"20261010"``).

    Returns
    -------
    build_dir : `pathlib.Path`
        The build directory under the products root. Not created.
    """
    return catalog.products_root() / PRODUCT / variant / build


def load_table(table, variant, build=catalog.CURRENT):
    """Read one table of a build.

    Parameters
    ----------
    table : `str`
        One of `TABLES`.
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name, or ``"current"``.

    Returns
    -------
    data : `pandas.DataFrame`
        The table.

    Raises
    ------
    `ValueError`
        If `table` is not one of `TABLES`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build or the table is missing.
    """
    if table not in TABLES:
        raise ValueError(
            f"{table!r} is not a cwfs_tables table; expected one of {TABLES}")
    return catalog.load(PRODUCT, variant, build=build, file=f"{table}.parquet")


def load(variant, build=catalog.CURRENT, tables=TABLES):
    """Read a build's tables.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"d12-refitWcs"``.
    build : `str`, optional
        Build name, or ``"current"`` (the default) for the variant's default build.
    tables : `tuple` [`str`], optional
        Which tables to read, in the order they are returned. `TABLES` by default.

    Returns
    -------
    tables : `tuple` [`pandas.DataFrame`]
        One DataFrame per requested table: ``(donuts, visits)`` by default.

    Raises
    ------
    `ValueError`
        If `tables` names something that is not in `TABLES`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build does not exist, or a requested table is missing from it.

    Notes
    -----
    These tables are small -- tens of thousands of donut rows for a full variant, against
    millions for ``fam_tables`` -- so reading both is cheap.

    Examples
    --------
    >>> donuts, visits = load("d12-refitWcs")                  # doctest: +SKIP
    >>> visits, = load("d12-tarts", tables=("visits",))         # doctest: +SKIP
    """
    bad = [t for t in tables if t not in TABLES]
    if bad:
        raise ValueError(f"not cwfs_tables tables: {bad}; expected from {TABLES}")
    return tuple(load_table(t, variant, build=build) for t in tables)
