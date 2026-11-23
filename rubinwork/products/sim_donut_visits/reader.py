"""Read ``sim_donut_visits`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from .. import catalog

PRODUCT = "sim_donut_visits"
"""Product name, as it appears in the catalog and in every manifest."""

INDEX = "donuts.parquet"

VARIANTS_FILE = pathlib.Path(__file__).with_name("variants.yaml")


@functools.lru_cache(maxsize=1)
def _variants_doc():
    with open(VARIANTS_FILE) as f:
        return yaml.safe_load(f) or {}


def variants():
    """Names of every defined variant, sorted."""
    return sorted(_variants_doc().get("variants") or {})


def variant_config(variant):
    """The ``variants.yaml`` entry of one variant, with the defaults filled in.

    Raises
    ------
    `KeyError`
        If `variant` is not defined.
    """
    entries = _variants_doc().get("variants") or {}
    if variant not in entries:
        raise KeyError(f"{variant!r} is not a {PRODUCT} variant; defined: {sorted(entries)}")
    return {**(_variants_doc().get("defaults") or {}), **(entries[variant] or {})}


def build_dir(variant, build):
    """Where a build of `variant` is written, whether or not it exists yet."""
    return catalog.products_root() / PRODUCT / variant / build


def stamp_file(ccd):
    """Name of the FITS file holding every visit of one corner half-chip."""
    return f"stamps_{ccd}.fits"


def extname(visit, point_id):
    """HDU name of one donut."""
    return f"v{int(visit):03d}_{point_id}"


def load(variant, build=catalog.CURRENT):
    """The donut table: one row per visit and point.

    Returns
    -------
    donuts : `pandas.DataFrame`
        ``visit``; ``ccd`` (e.g. ``R00_SW0``), ``corner``, ``side``, ``point_id``;
        ``thx_CCS_deg``, ``thy_CCS_deg``, ``thx_OCS_deg``, ``thy_OCS_deg``,
        ``field_r_deg`` (field angle, deg); ``mag`` (AB); ``flux_e`` (donut electrons);
        ``sky_mag_arcsec2``; ``sky_e_per_px`` and ``read_noise_e`` (e- per 10 µm
        pixel); ``seeing_fwhm_arcsec``, ``seeing_e`` (distortion, dimensionless),
        ``seeing_beta_deg`` (major axis from CCS +x), ``r0_m``, ``L0_m``;
        ``offset_x_px``, ``offset_y_px`` (donut shift from the chief ray, pixels, CCS x
        and y); ``peak_e`` (brightest pixel of the expectation, e-); ``file``,
        ``extname``.
    """
    return pd.read_parquet(catalog.path(PRODUCT, variant, build) / INDEX)


def stamp(variant, ccd, visit, point_id, build=catalog.CURRENT):
    """One donut.

    Returns
    -------
    observed, expected : `numpy.ndarray`
        ``(n, n)`` sky-subtracted noisy stamp and its noise-free expectation, electrons
        per pixel; row = CCS y, column = CCS x, centred on the chief ray.
    header : `astropy.io.fits.Header`
    """
    path = catalog.path(PRODUCT, variant, build) / stamp_file(ccd)
    with fits.open(path) as hdul:
        h = hdul[extname(visit, point_id)]
        cube = np.asarray(h.data, dtype=float)
        return cube[0], cube[1], h.header.copy()
