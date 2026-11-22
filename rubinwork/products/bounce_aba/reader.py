"""Read ``bounce_aba`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import pandas as pd
import yaml

from .. import catalog
from ..miw import reader as _miw

PRODUCT = "bounce_aba"
"""Product name, as it appears in the catalog and in every manifest."""

TABLES = ("aba_legs", "aba_dz", "aba_state", "aba_noise")
"""The parquet tables a build holds, in the order `load` returns them."""

TABLE_KEYS = {
    "aba_legs": ("night", "leg"),
    "aba_dz": ("night", "leg", "k", "j"),
    "aba_state": ("night", "leg", "kind", "index"),
    "aba_noise": ("k", "j"),
}
"""Columns that make a row of each table unique."""

VARIANTS_FILE = pathlib.Path(__file__).with_name("variants.yaml")


@functools.lru_cache(maxsize=1)
def _variants_doc():
    with open(VARIANTS_FILE) as f:
        return yaml.safe_load(f) or {}


def variants():
    """Names of every defined variant."""
    return list((_variants_doc().get("variants") or {}).keys())


def variant_config(variant):
    """One variant's settings: the ``defaults`` block with its own entry over it.

    Raises
    ------
    `KeyError`
        If `variant` is not defined.
    """
    doc = _variants_doc()
    entries = doc.get("variants") or {}
    if variant not in entries:
        raise KeyError(f"unknown {PRODUCT} variant {variant!r}; defined: {sorted(entries)}")
    return {**(doc.get("defaults") or {}), **entries[variant]}


def mi_keys(variant):
    """The ``(param_set, mi_name)`` the bounce configuration is resolved with.

    ``mi_name`` is `None` for a batoid variant, which takes the param_set-level settings.
    """
    cfg = variant_config(variant)
    if cfg["intrinsic"] == "miw":
        return _miw.mi_keys(cfg["miw_variant"])
    from ..fam_tables import reader as _fam
    return _fam.variant_config(cfg["fam_variant"])["param_set"], None


def fits_input(variant):
    """The ``(product, variant)`` whose ``fits.parquet`` a build reduces."""
    cfg = variant_config(variant)
    if cfg["intrinsic"] == "miw":
        return "miw", cfg["miw_variant"]
    if cfg["intrinsic"] == "batoid":
        return "fam_tables", cfg["fam_variant"]
    raise ValueError(f"variant {variant!r}: unknown intrinsic {cfg['intrinsic']!r}")


def build_dir(variant, build):
    """Where a build of `variant` is written, whether or not it exists yet."""
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
        Build name; the variant's ``current`` build by default.

    Returns
    -------
    table : `pandas.DataFrame`
    """
    if table not in TABLES:
        raise KeyError(f"unknown table {table!r}; tables: {TABLES}")
    return pd.read_parquet(catalog.path(PRODUCT, variant, build) / f"{table}.parquet")


def load(variant, build=catalog.CURRENT):
    """All four tables of a build, in `TABLES` order."""
    return tuple(load_table(t, variant, build) for t in TABLES)
