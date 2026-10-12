"""Read ``dof_lut`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import pyarrow.parquet as pq
import yaml

from .. import catalog
# A dof_lut variant names one `miw` variant, and the miw product owns the mapping from
# that to the (param_set, mi_name) the builder takes and to the upstream FAM variant.
# Reading it here keeps the mapping in one place -- as the coadds reader does -- rather
# than restating it per entry, where the two could drift from the generated mi_config.
from ..miw import reader as _miw

PRODUCT = "dof_lut"
"""Product name, as it appears in the catalog and in every manifest."""

TABLES = ("lut", "lut_dz", "lut_by_rotbin")
"""The parquet tables a build holds, in the order `load` returns them.

``lut_by_rotbin`` is written only when the Double Zernike (DZ) fit table carries a
rotator-angle column, so a build may legitimately lack it; `load` skips a missing one
rather than failing.
"""

PROVENANCE = "lut_config.yaml"
"""The builder's frozen provenance file, read by `build_provenance`."""

PLOTS = ("lut.pdf",)
"""The build's plot. An artifact of the build, not data anything reads back."""

TABLE_KEYS = {
    "lut": ("dof_index",),
    "lut_dz": ("k", "j"),
    "lut_by_rotbin": ("rotbin_center", "dof_index"),
}
"""Columns that make a row of each table unique, verified on the 3a reference build.

One row per recovered degree of freedom (DOF) in ``lut`` (50 of 50 rows unique), per
(focal order k, pupil Noll index j) pair in ``lut_dz`` (126 of 126), and per
(rotator bin centre, DOF) pair in ``lut_by_rotbin`` (450 of 450; counts dimensionless).
No key column holds a null on any table.
"""

SCHEMA_METADATA_FIELDS = ("param_set", "mi_name", "reduce", "n_dof", "n_keep",
                          "n_visits", "k_min", "k_max")
"""Parquet schema-metadata keys the builder stamps on ``lut`` and ``lut_dz``.

Part of the product, not incidental: the mode counts and the visit count that produced
the table are recorded only here, so `build_metadata` reads them back and the build
comparison checks them. No earlier product carried metadata this way.
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
        Variant name, e.g. ``"d12-A_50_34_i_5rot"``.

    Returns
    -------
    config : `dict`
        The ``variants.yaml`` entry: the ``miw_variant`` whose settings entry the build
        uses, and the old-tree ``dir_name``.

    Raises
    ------
    `KeyError`
        If `variant` is not defined in ``variants.yaml``.
    """
    entries = _variants_doc().get("variants") or {}
    if variant not in entries:
        raise KeyError(
            f"{variant!r} is not a dof_lut variant; defined: {sorted(entries)}")
    return entries[variant]


def miw_variant(variant):
    """The ``miw`` variant supplying this LUT's settings.

    Settings only -- the singular-value-decomposition mode counts, the focal order
    range, the coordinate system, the band and program cuts and the rotator bin edges.
    The LUT fits the plain Full Array Mode (FAM) Double Zernike table and never reads a
    measured intrinsic wavefront map, so this is not an upstream build.

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
    """The ``fam_tables`` variant whose DZ fits and visits the LUT averages.

    The real upstream: a build reads ``fits.parquet`` and ``visits.parquet`` from a
    ``fam_tables`` build, and this is what the manifest records under ``inputs``.
    Derived through the MIW variant, since a MIW is built from one FAM variant.

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
    """The ``(param_set, mi_name)`` pair the builder's command line takes.

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
    """The variant's subdirectory in the old ``aos/output/lut/`` tree.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    dir_name : `str`
        e.g. ``"danish_1_2_A_50_34_i_5rot"``.
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
        Build name, usually the date the data is produced (``"20261011"``).

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
        raise ValueError(f"{table!r} is not a dof_lut table; expected one of {TABLES}")
    return catalog.load(PRODUCT, variant, build=build, file=f"{table}.parquet")


def build_metadata(variant, build=catalog.CURRENT, table="lut"):
    """The parquet schema metadata stamped on a build's table.

    The mode counts and the visit count behind a LUT are recorded only in the parquet
    schema metadata, so they are read back from there rather than recomputed.

    Parameters
    ----------
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name, or ``"current"``.
    table : `str`, optional
        Which table's metadata to read: ``"lut"`` (the default) or ``"lut_dz"``. Both
        carry the same eight fields.

    Returns
    -------
    metadata : `dict` [`str`, `str`]
        The `SCHEMA_METADATA_FIELDS`, decoded to `str`. Values are the builder's own
        strings: ``n_dof``, ``n_keep`` and ``n_visits`` are counts (dimensionless),
        ``k_min`` and ``k_max`` the inclusive focal-order range, ``reduce`` the
        reduction used over visits (``"median"`` or ``"mean"``).

    Raises
    ------
    `ValueError`
        If `table` carries no schema metadata.
    `rubinwork.products.catalog.ProductNotFound`
        If the build or the table is missing.
    """
    if table not in ("lut", "lut_dz"):
        raise ValueError(
            f"{table!r} carries no schema metadata; expected 'lut' or 'lut_dz'")
    path = catalog.path(PRODUCT, variant, build=build) / f"{table}.parquet"
    if not path.exists():
        raise catalog.ProductNotFound(f"{path} is missing from the build")
    raw = pq.read_schema(str(path)).metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def build_provenance(variant, build=catalog.CURRENT):
    """The builder's frozen ``lut_config.yaml``.

    Parameters
    ----------
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name, or ``"current"``.

    Returns
    -------
    provenance : `dict`
        The builder's resolved settings: ``param_set``, ``mi_name``, ``n_dof``,
        ``n_keep``, ``reduce``, ``n_visits``, ``k_min``, ``k_max``, ``bands``,
        ``programs`` and ``use_alt_window``.

    Raises
    ------
    `rubinwork.products.catalog.ProductNotFound`
        If the build or the file is missing.
    """
    path = catalog.path(PRODUCT, variant, build=build) / PROVENANCE
    if not path.exists():
        raise catalog.ProductNotFound(f"{path} is missing from the build")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load(variant, build=catalog.CURRENT, tables=TABLES):
    """Read a build's parquet tables.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"d12-A_50_34_i_5rot"``.
    build : `str`, optional
        Build name, or ``"current"`` (the default) for the variant's default build.
    tables : `tuple` [`str`], optional
        Which tables to read, in the order they are returned. `TABLES` by default,
        whose third entry ``lut_by_rotbin`` is skipped when the build does not have it.

    Returns
    -------
    tables : `tuple` [`pandas.DataFrame`]
        One DataFrame per table present: ``(lut, lut_dz, lut_by_rotbin)`` by default, or
        ``(lut, lut_dz)`` for a build whose fit table had no rotator-angle column.
        Values in ``lut`` are DOF amplitudes in µm for translations and bending modes
        and arcsec for hexapod tip-tilt, as its ``dof_unit`` column states per row; the
        DZ columns of ``lut_dz`` are in µm of wavefront.

    Raises
    ------
    `ValueError`
        If `tables` names something that is not in `TABLES`.
    `rubinwork.products.catalog.ProductNotFound`
        If the build does not exist, or a requested table other than
        ``lut_by_rotbin`` is missing from it.

    Notes
    -----
    An explicitly requested ``lut_by_rotbin`` is **not** skipped -- asking for it by name
    and silently getting nothing back would hide a missing diagnostic. It is skipped only
    when it comes from the `TABLES` default.

    Examples
    --------
    >>> lut, lut_dz, by_rotbin = load("d12-A_50_34_i_5rot")       # doctest: +SKIP
    >>> build_metadata("d12-A_50_34_i_5rot")["n_keep"]            # doctest: +SKIP
    '34'
    """
    bad = [t for t in tables if t not in TABLES]
    if bad:
        raise ValueError(f"not dof_lut tables: {bad}; expected from {TABLES}")
    default = tuple(tables) == tuple(TABLES)
    out = []
    for table in tables:
        if table == "lut_by_rotbin" and default:
            path = catalog.path(PRODUCT, variant, build=build) / f"{table}.parquet"
            if not path.exists():
                continue
        out.append(load_table(table, variant, build=build))
    return tuple(out)
