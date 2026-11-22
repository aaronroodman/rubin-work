"""Read ``sim_donut_fits`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from .. import catalog

PRODUCT = "sim_donut_fits"
"""Product name, as it appears in the catalog and in every manifest."""

INDEX = "fits.parquet"

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


def stamp_file(sim_variant, band, config, side):
    """Name of the FITS file holding one sim variant, band, configuration and side."""
    return f"stamps_{sim_variant}_{band}_{config}_{side}.fits"


def dz_columns(noll, frame):
    """Zernike-deviation (OPD - intrinsic) column names, ``dz<j>_<frame>_um``."""
    return [f"dz{j}_{frame}_um" for j in noll]


def load(variant, build=catalog.CURRENT):
    """The fit table: one row per fitted stamp.

    Returns
    -------
    fits : `pandas.DataFrame`
        Columns ``sim_variant``, ``band``, ``config``, ``side``, ``point_id``,
        ``file``, ``extname``; ``thx_CCS_deg``, ``thy_CCS_deg``, ``thx_OCS_deg``,
        ``thy_OCS_deg``, ``field_r_deg`` (field angle, deg); ``npix_binned``;
        ``flux_ratio`` (fitted flux over stamp sum, dimensionless); ``dx_px``,
        ``dy_px`` (fitted shift from the chief ray, binned pixels, CCS x and y);
        ``fwhm_arcsec`` (fitted Kolmogorov FWHM, arcsec); ``chi2_dof`` (chi² over
        pixels minus parameters, at the variant's flux and sky, noise-free stamp);
        ``l1`` (fraction of the flux misplaced) and ``pow`` (residual power over sim
        power), both dimensionless; ``nfev``, ``status``, ``success``, ``cost``,
        ``optimality`` (`scipy.optimize.least_squares`); ``dz<j>_CCS_um`` and
        ``dz<j>_OCS_um``: Zernike deviation, OPD (the fitted wavefront) minus the
        batoid intrinsic (``batoid_intrinsic`` kind ``gq_infocus``), for each fitted Noll
        ``j`` (µm of wavefront), NaN where ``j`` was not fitted for that configuration.
        Danish's model wavefront is the intrinsicTA with, for the fitted terms, the
        intrinsic replaced by the OPD (ts_wep ``getOffAxisCoeff`` plus ``zkStart``). The
        sims are traced through the same batoid model, so the expected deviation is zero.
    """
    return pd.read_parquet(catalog.path(PRODUCT, variant, build) / INDEX)


def stamp(variant, sim_variant, band, config, side, point_id, build=catalog.CURRENT):
    """One fitted stamp and its model.

    Returns
    -------
    data, model : `numpy.ndarray`
        ``(n, n)`` binned stamp and best-fit Danish model, electrons per binned pixel
        (block mean); row = CCS y, column = CCS x.
    header : `astropy.io.fits.Header`
    """
    path = catalog.path(PRODUCT, variant, build) / stamp_file(sim_variant, band, config, side)
    with fits.open(path) as hdul:
        h = hdul[point_id]
        cube = np.asarray(h.data, dtype=float)
        return cube[0], cube[1], h.header.copy()
