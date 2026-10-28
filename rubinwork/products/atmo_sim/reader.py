"""Read an ``atmo_sim`` visit partition.

Layout::

    products/atmo_sim/<variant>/visits/<expId>/
        guider_raw/*.fits  science_stars.fits  truth.parquet  atmosphere.json
        manifest.json

Typical read::

    from rubinwork.products import atmo_sim
    truth = atmo_sim.load_truth(20260706, 688)
    hdus = atmo_sim.science_stars(20260706, 688)       # astropy HDUList
    atmo_sim.collection()                              # Butler chain for GuiderReader
"""
import functools
import json
import pathlib

import pandas as pd
import yaml

from .. import catalog

__all__ = ["PRODUCT", "PRODUCT_VARIANT", "VISITS_DIR", "exp_id", "visit_dir", "visits",
           "partition", "load_truth", "load_atmosphere", "science_stars", "collection",
           "run_collection", "variants", "variant_config"]

VARIANTS_FILE = pathlib.Path(__file__).with_name("variants.yaml")

PRODUCT = "atmo_sim"

PRODUCT_VARIANT = "v1"

VISITS_DIR = "visits"
"""The subdirectory of a variant that holds the per-visit partitions."""


def exp_id(day_obs, seq_num):
    """Exposure ID, ``day_obs * 100000 + seq_num``."""
    return int(day_obs) * 100000 + int(seq_num)


def visit_dir(day_obs, seq_num, variant=PRODUCT_VARIANT):
    """Where a visit's partition is written, whether or not it exists yet."""
    return (catalog.products_root() / PRODUCT / variant / VISITS_DIR
            / str(exp_id(day_obs, seq_num)))


def partition(day_obs, seq_num, variant=PRODUCT_VARIANT):
    """A complete visit's directory, through the catalog.

    Raises
    ------
    `rubinwork.products.catalog.ProductNotFound`
        If the visit is not built or not complete.
    """
    return catalog.path(PRODUCT, variant, f"{VISITS_DIR}/{exp_id(day_obs, seq_num)}")


def visits(variant=PRODUCT_VARIANT, complete_only=True):
    """``(day_obs, seq_num)`` of every visit in a variant, ascending."""
    root = catalog.products_root() / PRODUCT / variant / VISITS_DIR
    if not root.is_dir():
        return []
    found = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        if complete_only and not catalog.is_complete(entry):
            continue
        found.append((int(entry.name) // 100000, int(entry.name) % 100000))
    return found


def load_truth(day_obs, seq_num, variant=PRODUCT_VARIANT):
    """Per-star truth table.

    Returns
    -------
    truth : `pandas.DataFrame`
        One row per star: ``kind`` (``guider`` or ``science``), ``detector``,
        ``det_name``, ``xccd``, ``yccd`` (CCD pix), ``theta_e_asec``, ``theta_n_asec``
        (field angle East and North of the boresight, arcsec), ``flux_adu`` (ADU),
        ``pedestal_adu`` (ADU), ``read_noise_adu`` (ADU rms), ``n_stamps`` (count) and
        ``jac`` (CCD pixel to (East, North) Jacobian, arcsec/pix, row-major 2x2).
    """
    return pd.read_parquet(partition(day_obs, seq_num, variant) / "truth.parquet")


def load_atmosphere(day_obs, seq_num, variant=PRODUCT_VARIANT):
    """The visit's atmosphere record (``atmosphere.json``) as a `dict`."""
    with open(partition(day_obs, seq_num, variant) / "atmosphere.json") as fh:
        return json.load(fh)


def science_stars(day_obs, seq_num, variant=PRODUCT_VARIANT):
    """The science-CCD stars, one image HDU per CCD, in the CCD pixel frame.

    Each HDU header carries ``DETECTOR``, ``X0``, ``Y0`` (CCD pixel of array ``[0, 0]``),
    ``XSTAR``, ``YSTAR`` (true star CCD pixel) and ``THETA_E``, ``THETA_N`` (arcsec).

    Returns
    -------
    hdus : `astropy.io.fits.HDUList`
    """
    from astropy.io import fits

    return fits.open(partition(day_obs, seq_num, variant) / "science_stars.fits")


def collection(variant=PRODUCT_VARIANT):
    """The Butler CHAINED collection to read simulated ``guider_raw`` from."""
    return variant_config(variant)["butler"]["chain"]


def run_collection(day_obs, variant=PRODUCT_VARIANT):
    """The Butler RUN collection a night's simulated ``guider_raw`` is ingested into."""
    return f"{variant_config(variant)['butler']['run_prefix']}/{int(day_obs)}"


@functools.lru_cache(maxsize=1)
def _variants_doc():
    with open(VARIANTS_FILE) as handle:
        return yaml.safe_load(handle) or {}


def variants():
    """Variant names, in ``variants.yaml`` order."""
    return list(_variants_doc().get("variants") or {})


def variant_config(variant=PRODUCT_VARIANT):
    """One variant's ``variants.yaml`` entry.

    Raises
    ------
    `KeyError`
        If the variant is not defined.
    """
    entries = _variants_doc().get("variants") or {}
    if variant not in entries:
        raise KeyError(f"no variant {variant!r} of {PRODUCT}; known variants: {list(entries)}")
    return entries[variant]
