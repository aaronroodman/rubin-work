"""Read an ``atmo_sim_moments`` visit partition.

Typical read::

    from rubinwork.products import atmo_sim_moments
    mom = atmo_sim_moments.load_table("moments", 20260706, 688)
    ccd = atmo_sim_moments.load_table("ccd_psf", 20260706, 688)
"""
import functools
import pathlib

import pandas as pd
import yaml

from .. import catalog

__all__ = ["PRODUCT", "PRODUCT_VARIANT", "VISITS_DIR", "TABLES", "visit_dir", "visits",
           "load_table", "variants", "variant_config"]

VARIANTS_FILE = pathlib.Path(__file__).with_name("variants.yaml")

PRODUCT = "atmo_sim_moments"

PRODUCT_VARIANT = "v1"

VISITS_DIR = "visits"

TABLES = ("moments", "stars", "metrics", "ccd_psf")
"""``moments``, ``stars``, ``metrics``: the ``guider_moments`` tables, same schema.
``ccd_psf``: one row per science CCD."""


def _exp_id(day_obs, seq_num):
    return int(day_obs) * 100000 + int(seq_num)


def visit_dir(day_obs, seq_num, variant=PRODUCT_VARIANT):
    """Where a visit's partition is written, whether or not it exists yet."""
    return (catalog.products_root() / PRODUCT / variant / VISITS_DIR
            / str(_exp_id(day_obs, seq_num)))


def visits(variant=PRODUCT_VARIANT, complete_only=True):
    """``(day_obs, seq_num)`` of every visit in a variant, ascending."""
    root = catalog.products_root() / PRODUCT / variant / VISITS_DIR
    if not root.is_dir():
        return []
    return [(int(e.name) // 100000, int(e.name) % 100000) for e in sorted(root.iterdir())
            if e.is_dir() and e.name.isdigit()
            and (not complete_only or catalog.is_complete(e))]


def table_file(table, seq_num):
    """File name of one table in a visit partition."""
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}; known tables: {list(TABLES)}")
    return "ccd_psf.parquet" if table == "ccd_psf" else f"{int(seq_num)}_{table}.parquet"


def load_table(table, day_obs, seq_num, variant=PRODUCT_VARIANT):
    """One table of one visit.

    Parameters
    ----------
    table : `str`
        One of `TABLES`.
    day_obs, seq_num : `int`

    Returns
    -------
    table : `pandas.DataFrame`
        The guider tables are as ``guider_moments`` writes them (arcsec**2, each sensor's
        DVCS frame). ``ccd_psf`` has ``detector``, ``det_name``, ``psf_sigma`` (pix,
        ``det(M)**0.25``), ``psf_ixx``, ``psf_iyy``, ``psf_ixy`` (pix**2, CCD pixel
        frame, HSM adaptive moments, as ConsDB ``ccdvisit1_quicklook``), ``fwhm_asec``
        (arcsec), ``e1``, ``e2`` (dimensionless, ``(Ixx - Iyy) / T``, ``2 Ixy / T``),
        ``xc``, ``yc`` (CCD pix) and ``ok``.
    """
    directory = catalog.path(PRODUCT, variant, f"{VISITS_DIR}/{_exp_id(day_obs, seq_num)}")
    return pd.read_parquet(directory / table_file(table, seq_num))


@functools.lru_cache(maxsize=1)
def _variants_doc():
    with open(VARIANTS_FILE) as handle:
        return yaml.safe_load(handle) or {}


def variants():
    """Variant names, in ``variants.yaml`` order."""
    return list(_variants_doc().get("variants") or {})


def variant_config(variant=PRODUCT_VARIANT):
    """One variant's ``variants.yaml`` entry."""
    entries = _variants_doc().get("variants") or {}
    if variant not in entries:
        raise KeyError(f"no variant {variant!r} of {PRODUCT}; known variants: {list(entries)}")
    return entries[variant]
