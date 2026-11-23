"""Read ``sim_donut_visit_fits`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from .. import catalog

PRODUCT = "sim_donut_visit_fits"
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


def stamp_file(ccd):
    """Name of the FITS file holding the fits of one corner half-chip."""
    return f"stamps_{ccd}.fits"


def dz_columns(noll, frame):
    """Zernike-deviation (OPD - intrinsic) column names, ``dz<j>_<frame>_um``."""
    return [f"dz{j}_{frame}_um" for j in noll]


def load(variant, build=catalog.CURRENT):
    """The fit table: one row per fitted stamp.

    Returns
    -------
    fits : `pandas.DataFrame`
        The ``sim_donut_visits`` columns (visit, CCD, point, field angles, star, sky,
        seeing and offsets drawn) and: ``npix_binned``; ``flux_ratio`` (fitted flux over
        binned stamp sum, dimensionless); ``dx_px``, ``dy_px`` (fitted shift from the
        chief ray, binned pixels, CCS x and y); ``fwhm_arcsec`` (fitted Kolmogorov FWHM,
        arcsec); ``chi2_dof`` (chi² over pixels minus parameters); ``l1`` (fraction of
        the flux misplaced, dimensionless); ``nfev``, ``status``, ``success``, ``cost``,
        ``optimality`` (`scipy.optimize.least_squares`); ``dz<j>_CCS_um`` and
        ``dz<j>_OCS_um``: Zernike deviation, OPD (the fitted wavefront) minus the batoid
        intrinsic (``gq_infocus``), per fitted Noll ``j`` (µm of wavefront); the optics
        are perfectly aligned, so the expected deviation is zero; ``z<j>_intr_CCS_um``:
        that intrinsic (µm).
    """
    return pd.read_parquet(catalog.path(PRODUCT, variant, build) / INDEX)


def stamp(variant, ccd, visit, point_id, build=catalog.CURRENT):
    """One fitted stamp and its model.

    Returns
    -------
    data, model : `numpy.ndarray`
        ``(n, n)`` binned stamp and best-fit Danish model, electrons per binned pixel
        (block mean); row = CCS y, column = CCS x.
    header : `astropy.io.fits.Header`
    """
    from .. import sim_donut_visits
    path = catalog.path(PRODUCT, variant, build) / stamp_file(ccd)
    with fits.open(path) as hdul:
        h = hdul[sim_donut_visits.extname(visit, point_id)]
        cube = np.asarray(h.data, dtype=float)
        return cube[0], cube[1], h.header.copy()
