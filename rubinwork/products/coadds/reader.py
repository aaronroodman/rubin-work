"""Read ``coadds`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import numpy
import yaml

from .. import catalog
# The one sibling-product import in a reader: a coadd's FAM variant follows from its MIW
# variant, and `miw` owns that mapping.  Reading it here keeps the fact in one place
# rather than restating it per coadd variant, where the two could drift apart.
from ..miw import reader as _miw

PRODUCT = "coadds"
"""Product name, as it appears in the catalog and in every manifest."""

TABLES = ("blocks_summary", "coadd_metrics")
"""The parquet tables a build holds, in the order `load` returns them."""

GRIDS = ("block_grids", "block_grids_ext")
"""The npz grid files a build holds, in the order `load_grids` returns them."""

PLOTS = ("coadd_blocks_miw_perblock.pdf", "coadd_blocks_miw_timeseries.pdf")
"""The build's plots. Artifacts of the build, not data anything reads back."""

BLOCK_KEY = ("block",)
"""Column that makes a row of either table unique: one row per detected block.

``block`` is the detection index, assigned in time order over the whole FAM table, so it
is unique within a build but means nothing across builds -- a build over more nights
renumbers. ``coadd_metrics`` additionally carries ``ordinal``, the position among the
blocks actually built, which is why both are kept.
"""

# Settings each variant carries beyond its two bookkeeping fields. The builder's argparse
# default stands in for a null, so an entry need not pin every one.
BUILD_FIELDS = ("bands", "min_visits", "max_blur", "pointing_tol", "wide_tol",
                "max_seq_span", "bounce_round", "bounce_programs", "wide_programs",
                "ext_terms", "cwfs_inner", "cwfs_outer", "n_bins", "n_iter", "pct",
                "instrument")

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
        Omit entries carrying ``registered: false`` (the default).

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
        Variant name, e.g. ``"d12-A_50_34_i_5rot"``.

    Returns
    -------
    config : `dict`
        The ``variants.yaml`` entry: the upstream ``miw_variant``, the old-tree
        ``dir_name``, and the block-selection settings of `BUILD_FIELDS`.

    Raises
    ------
    `KeyError`
        If `variant` is not defined in ``variants.yaml``.
    """
    entries = _variants_doc().get("variants") or {}
    if variant not in entries:
        raise KeyError(
            f"{variant!r} is not a coadds variant; defined: {sorted(entries)}")
    return entries[variant]


def miw_variant(variant):
    """The ``miw`` variant whose per-donut sidecar each block's remnant is compared to.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    miw_variant : `str`
        A ``miw`` variant name, e.g. ``"d12-A_50_34_i_5rot"``.
    """
    return variant_config(variant)["miw_variant"]


def fam_variant(variant):
    """The ``fam_tables`` variant whose donuts and visits are blocked up.

    Follows from the MIW variant -- a MIW is built from one FAM variant -- so it is not
    stated twice in ``variants.yaml``.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    fam_variant : `str`
        A ``fam_tables`` variant name, e.g. ``"danish_1_2"``.
    """
    return _miw.fam_variant(miw_variant(variant))


def dir_name(variant):
    """The variant's subdirectory in the old ``aos/output/coadd/<P>/`` tree.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    dir_name : `str`
        e.g. ``"50_34"``.
    """
    return variant_config(variant)["dir_name"]


def build_settings(variant):
    """The block-selection settings of one variant, nulls dropped.

    A null means "use the builder's own default", so it is omitted rather than passed on
    as ``None``.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    settings : `dict`
        A subset of `BUILD_FIELDS`, in `BUILD_FIELDS` order.
    """
    cfg = variant_config(variant)
    return {k: cfg[k] for k in BUILD_FIELDS
            if k in cfg and cfg[k] is not None}


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
    """Read one parquet table of a build.

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
        The table, one row per block.

    Raises
    ------
    `ValueError`
        If `table` is not one of `TABLES`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build or the table is missing.
    """
    if table not in TABLES:
        raise ValueError(f"{table!r} is not a coadds table; expected one of {TABLES}")
    return catalog.load(PRODUCT, variant, build=build, file=f"{table}.parquet")


def load_grids(grids="block_grids", variant=None, build=catalog.CURRENT):
    """Read one npz grid file of a build.

    Parameters
    ----------
    grids : `str`, optional
        One of `GRIDS`. ``"block_grids"`` by default -- the Z5-Z8 coadd and MIW maps
        plus the Double Zernike (DZ) primitives and the subspace bases.
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name, or ``"current"``.

    Returns
    -------
    data : `numpy.lib.npyio.NpzFile`
        The loaded archive, opened with ``allow_pickle=True`` because the metadata
        arrays include object dtypes. Keyed by array name; ``grids`` and ``miw`` carry
        the maps in µm of wavefront, indexed ``(block, term, y, x)``.

    Raises
    ------
    `ValueError`
        If `grids` is not one of `GRIDS`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build or the file is missing.
    """
    if grids not in GRIDS:
        raise ValueError(f"{grids!r} is not a coadds grid file; expected one of {GRIDS}")
    if variant is None:
        raise ValueError("load_grids() needs `variant`")
    path = catalog.path(PRODUCT, variant, build=build) / f"{grids}.npz"
    if not path.exists():
        raise catalog.ProductNotFound(f"{path} is missing from the build")
    return numpy.load(path, allow_pickle=True)


def load(variant, build=catalog.CURRENT, tables=TABLES):
    """Read a build's parquet tables.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"d12-A_50_34_i_5rot"``.
    build : `str`, optional
        Build name, or ``"current"`` (the default) for the variant's default build.
    tables : `tuple` [`str`], optional
        Which tables to read, in the order they are returned. `TABLES` by default.

    Returns
    -------
    tables : `tuple` [`pandas.DataFrame`]
        One DataFrame per requested table: ``(blocks_summary, coadd_metrics)`` by
        default.

    Raises
    ------
    `ValueError`
        If `tables` names something that is not in `TABLES`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build does not exist, or a requested table is missing from it.

    Notes
    -----
    The maps themselves are in the npz files, not these tables; read them with
    `load_grids`.

    Examples
    --------
    >>> summary, metrics = load("d12-A_50_34_i_5rot")            # doctest: +SKIP
    >>> grids = load_grids(variant="d12-A_50_34_i_5rot")         # doctest: +SKIP
    """
    bad = [t for t in tables if t not in TABLES]
    if bad:
        raise ValueError(f"not coadds tables: {bad}; expected from {TABLES}")
    return tuple(load_table(t, variant, build=build) for t in tables)
