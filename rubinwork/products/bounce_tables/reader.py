"""Read ``bounce_tables`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import yaml

from .. import catalog
# A bounce_tables variant names one `miw` variant, and the miw product owns the mapping
# from that to the (param_set, mi_name) the configuration is keyed on and to the upstream
# FAM variant.  Reading it here keeps the mapping in one place -- as the dof_lut and
# coadds readers do -- rather than restating it per entry, where the two could drift from
# the generated mi_config.yaml.
from ..miw import reader as _miw

PRODUCT = "bounce_tables"
"""Product name, as it appears in the catalog and in every manifest."""

TABLES = ("bounce_kj_stats", "bounce_dof_stats", "bounce_fwhm_metric",
          "bounce_fwhm_vs_bvalue")
"""The parquet tables a build holds, in the order `load` returns them."""

TABLE_KEYS = {
    "bounce_kj_stats": ("bounce", "night", "comparison", "k", "j"),
    "bounce_dof_stats": ("bounce", "comparison", "night", "kind", "index"),
    "bounce_fwhm_metric": ("bounce", "comparison"),
    "bounce_fwhm_vs_bvalue": ("bounce", "comparison", "night"),
}
"""Columns that make a row of each table unique, verified on the 3a reference build.

Counts are of unique rows over total rows (dimensionless): ``bounce_kj_stats``
1,890 / 1,890, ``bounce_dof_stats`` 2,550 / 2,550, ``bounce_fwhm_metric`` 6 / 6,
``bounce_fwhm_vs_bvalue`` 9 / 9. No key column holds a null on any table.

``night`` has to be in the ``bounce_kj_stats`` key: without it only 756 of the 1,890 rows
are distinct, because each leg carries a per-night row per (k, j) plus a pooled one. It
is a **string** on ``bounce_kj_stats`` and ``bounce_dof_stats``, where the pooled row
carries the literal ``'all'``, and an **int64** on ``bounce_fwhm_vs_bvalue``, which has
per-night rows only.
"""

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
        Variant name, e.g. ``"d12-A_50_34_i_5rot_july"``.

    Returns
    -------
    config : `dict`
        The ``variants.yaml`` entry: the ``miw_variant`` whose settings entry the build
        uses, the ``fits`` table name within that MIW build, and the old-tree
        ``dir_name``.

    Raises
    ------
    `KeyError`
        If `variant` is not defined in ``variants.yaml``.
    """
    entries = _variants_doc().get("variants") or {}
    if variant not in entries:
        raise KeyError(
            f"{variant!r} is not a bounce_tables variant; defined: {sorted(entries)}")
    return entries[variant]


def miw_variant(variant):
    """The ``miw`` variant this bounce takes its fit table and settings from.

    The upstream: a build reads one Double-Zernike fit table from a ``miw`` build, and
    that is what the manifest records under ``inputs``. The settings entry behind the
    same variant also supplies the singular-value-decomposition mode counts and the
    focal order range.

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


def fits_name(variant):
    """The fit table file name this variant reduces, within its MIW build.

    Two variants can share a MIW settings entry and still be different measurements,
    because the fit table is a separate choice: ``d12-A_50_34_i_5rot_july`` reduces the
    hand-made ``fits_july.parquet`` at 3,385 visits, the plain variant the MIW build's
    own ``fits.parquet`` at 1,126.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    fits : `str`
        A file name, e.g. ``"fits_july.parquet"``.
    """
    return variant_config(variant)["fits"]


def fam_variant(variant):
    """The ``fam_tables`` variant the fit table ultimately comes from.

    Recorded in the manifest's ``config`` rather than its ``inputs``: a build reads the
    MIW build's fit table, not the FAM tables directly. Derived through the MIW variant,
    since a MIW is built from one FAM variant.

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


def mi_keys(variant):
    """The ``(param_set, mi_name)`` pair the configuration is keyed on.

    The ``bounce:`` override in ``aos/analysis_config.yaml`` is keyed on the **long**
    param_set, and for ``danish_1_2`` it redefines the BLOCK-T720 legs as a five-point
    elevation sweep, so resolving the wrong key would silently measure one leg.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    param_set : `str`
        The long ``aos/param_sets.yaml`` key of the upstream FAM variant.
    mi_name : `str`
        The settings entry's ``name`` in the generated ``aos/mi_config.yaml``.
    """
    return _miw.mi_keys(miw_variant(variant))


def dir_name(variant):
    """The variant's subdirectory in the old ``aos/output/bounce/`` tree.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    dir_name : `str`
        e.g. ``"danish_1_2_A_50_34_i_5rot_july"``.
    """
    return variant_config(variant)["dir_name"]


def build_dir(variant, build):
    """Where a build of `variant` is written, whether or not it exists yet.

    `rubinwork.products.catalog.path` resolves only builds already on disk; a builder
    needs the path to write into.

    Parameters
    ----------
    variant : `str`
        Variant name.
    build : `str`
        Build name, usually the date the data is produced (``"20261012"``).

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
            f"{table!r} is not a bounce_tables table; expected one of {TABLES}")
    return catalog.load(PRODUCT, variant, build=build, file=f"{table}.parquet")


def load(variant, build=catalog.CURRENT, tables=TABLES):
    """Read a build's parquet tables.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"d12-A_50_34_i_5rot_july"``.
    build : `str`, optional
        Build name, or ``"current"`` (the default) for the variant's default build.
    tables : `tuple` [`str`], optional
        Which tables to read, in the order they are returned. `TABLES` by default.

    Returns
    -------
    tables : `tuple` [`pandas.DataFrame`]
        One DataFrame per requested table:
        ``(bounce_kj_stats, bounce_dof_stats, bounce_fwhm_metric,
        bounce_fwhm_vs_bvalue)`` by default. The Δ columns of ``bounce_kj_stats`` are in
        µm of wavefront; ``bounce_dof_stats`` ``delta`` is in the unit its own ``unit``
        column names per row (µm for translations and bending-mode amplitudes, arcsec for
        hexapod rotations, dimensionless for v-modes); every FWHM column of the two
        ``fwhm`` tables is in arcsec.

    Raises
    ------
    `ValueError`
        If `tables` names something that is not in `TABLES`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build does not exist, or a requested table is missing from it.

    Examples
    --------
    >>> kj, dof, fwhm, bval = load("d12-A_50_34_i_5rot_july")   # doctest: +SKIP
    """
    bad = [t for t in tables if t not in TABLES]
    if bad:
        raise ValueError(f"not bounce_tables tables: {bad}; expected from {TABLES}")
    return tuple(load_table(t, variant, build=build) for t in tables)
