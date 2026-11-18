"""Read ``giant_donuts`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import pandas as pd
import yaml

from .. import catalog

PRODUCT = "giant_donuts"
"""Product name, as it appears in the catalog and in every manifest."""

TABLES = ("exposures", "sides", "fits")
"""The parquet tables of a build."""

DET_OFFSET_MM = {"science": 0.0, "SW0": 1.5, "SW1": -1.5}
"""Detector z offset of each sensor class (mm): the corner sensors sit ±1.5 mm."""

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


def load(variant, build=catalog.CURRENT):
    """The exposures, sides and fits tables of a build.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"v402s_fitc"``.
    build : `str`, optional
        Build name, or ``"current"``.

    Returns
    -------
    tables : `dict` [`str`, `pandas.DataFrame`]
        ``exposures``, ``sides`` and ``fits``; ``sides`` and ``fits`` share ``key``.
    """
    d = catalog.path(PRODUCT, variant, build)
    return {t: pd.read_parquet(d / f"{t}.parquet") for t in TABLES}


def stamp(key, variant, build=catalog.CURRENT):
    """One side's stamp.

    Returns
    -------
    image : `numpy.ndarray`
        Electrons per pixel, background subtracted; CCS pixels, origin lower left.
    mask : `numpy.ndarray` [`bool`]
        True where a pixel is masked (defects, neighbours).
    header : `astropy.io.fits.Header`
        The primary header: key, visit, detector, side, field and rotator angle.
    """
    import numpy as np
    from astropy.io import fits
    d = catalog.path(PRODUCT, variant, build)
    with fits.open(d / "stamps" / f"{key}.fits") as h:
        return (np.asarray(h["IMAGE"].data, float), np.asarray(h["MASK"].data, bool),
                h[0].header.copy())


def mask_name(sensor_class, side, mask="v4.02s", step_mm=4.0):
    """File name of the danish mask of a side: ``RubinObsc_<mask>_r_det.._cam.._m2...yaml``."""
    s = 1.0 if side == "extra" else -1.0
    d, c = DET_OFFSET_MM[sensor_class], s * abs(step_mm)
    return f"RubinObsc_{mask}_r_det{d:+.1f}_cam{c:+.1f}_m2{c:+.1f}.yaml"


def mask_params(sensor_class, side, variant, build=catalog.CURRENT):
    """The danish mask parameters of a side, as `danish.DonutFactory` takes them."""
    cfg = variant_config(variant)
    d = catalog.path(PRODUCT, variant, build)
    name = mask_name(sensor_class, side, cfg["mask"], cfg["step_mm"])
    return yaml.safe_load((d / "masks" / name).read_text())
