"""Read ``fam_tables`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import yaml

from .. import catalog

PRODUCT = "fam_tables"
"""Product name, as it appears in the catalog and in every manifest."""

TABLES = ("donuts", "visits", "fits")
"""The three tables a build holds, in the order `load` returns them."""

VARIANTS_FILE = pathlib.Path(__file__).with_name("variants.yaml")


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
        Omit entries carrying ``registered: false`` (the default). Those are
        emitted into the generated ``aos/param_sets.yaml`` so recorded
        provenance keeps resolving, but they are not product builds.

    Returns
    -------
    names : `list` [`str`]
        Variant names, sorted. A defined variant need not have a build on disk;
        use `rubinwork.products.catalog.list_products` for what is built.
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
        Variant name, e.g. ``"danish_1_2"``.

    Returns
    -------
    config : `dict`
        The ``variants.yaml`` entry, including the long ``param_set`` key, the
        Butler collections and (for a ``snakemake`` variant) its date chunks.

    Raises
    ------
    `KeyError`
        If `variant` is not defined in ``variants.yaml``.
    """
    entries = _variants_doc().get("variants") or {}
    if variant not in entries:
        raise KeyError(
            f"{variant!r} is not a fam_tables variant; defined: {sorted(entries)}")
    return entries[variant]


def variant_for_param_set(param_set):
    """The variant whose entry carries this long ``param_set`` key.

    The two names differ: ``danish_1_2`` is the variant of param_set
    ``fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x``. They coincide for the
    ``danish_1_3`` variants, so a caller must not assume either way.

    Parameters
    ----------
    param_set : `str`
        A long ``param_set`` key, or a variant name (returned unchanged).

    Returns
    -------
    variant : `str`
        The variant name.

    Raises
    ------
    `KeyError`
        If no variant carries this ``param_set`` and it is not itself a variant.
    """
    entries = _variants_doc().get("variants") or {}
    if param_set in entries:
        return param_set
    for name, cfg in entries.items():
        if (cfg or {}).get("param_set") == param_set:
            return name
    raise KeyError(
        f"{param_set!r} is neither a fam_tables variant nor a registered "
        f"param_set; variants: {sorted(entries)}")


def build_dir(variant, build):
    """Where a build of `variant` is written, whether or not it exists yet.

    `rubinwork.products.catalog.path` resolves only builds already on disk; a
    builder needs the path to write into.

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
        raise ValueError(f"{table!r} is not a fam_tables table; expected one of {TABLES}")
    return catalog.load(PRODUCT, variant, build=build, file=f"{table}.parquet")


def load(variant, build=catalog.CURRENT, tables=TABLES):
    """Read a build's tables.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"danish_1_2"``.
    build : `str`, optional
        Build name, or ``"current"`` (the default) for the variant's default build.
    tables : `tuple` [`str`], optional
        Which tables to read, in the order they are returned. `TABLES` by default.

    Returns
    -------
    tables : `tuple` [`pandas.DataFrame`]
        One DataFrame per requested table: ``(donuts, visits, fits)`` by default.

    Raises
    ------
    `ValueError`
        If `tables` names something that is not in `TABLES`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build does not exist, or a requested table is missing from it.

    Notes
    -----
    ``donuts.parquet`` is large — 12.4 GB for ``danish_1_2`` — so reading all three
    tables pulls the whole donut table into memory. Pass ``tables=("visits", "fits")``
    when the donuts are not needed, or use `load_table` for one table, or
    `rubinwork.products.catalog.path` to get the directory and read the parquet with
    column or row-group selection.

    Examples
    --------
    >>> donuts, visits, fits = load("danish_1_2")        # doctest: +SKIP
    >>> visits, fits = load("danish_1_2", tables=("visits", "fits"))   # doctest: +SKIP
    """
    bad = [t for t in tables if t not in TABLES]
    if bad:
        raise ValueError(f"not fam_tables tables: {bad}; expected from {TABLES}")
    return tuple(load_table(t, variant, build=build) for t in tables)
