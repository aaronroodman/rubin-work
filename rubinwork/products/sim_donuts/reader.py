"""Read ``sim_donuts`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from .. import catalog

PRODUCT = "sim_donuts"
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


def stamp_file(band, config, side, dof=None):
    """Name of the FITS file holding one band, configuration and side (and DOF, for a
    variant with a DOF scan)."""
    tail = "" if dof is None else f"_dof{int(dof):02d}"
    return f"stamps_{band}_{config}_{side}{tail}.fits"


def dof_partitions(cfg):
    """``[(part, dof index, amplitude), ...]`` of a variant with a ``dof`` scan, else []."""
    d = cfg.get("dof")
    if not d:
        return []
    return list(zip(range(len(d["indices"])), d["indices"], d["amplitudes"]))


def load(variant, build=catalog.CURRENT):
    """The donut index: one row per stamp.

    Returns
    -------
    index : `pandas.DataFrame`
        Columns ``band``, ``config``, ``side``, ``point_id`` (the ``batoid_intrinsic``
        point), ``file``, ``extname``; ``thx_CCS_deg``, ``thy_CCS_deg``,
        ``thx_OCS_deg``, ``thy_OCS_deg`` (field angle, deg); ``npix``, ``pixel_m``
        (m), ``fwhm_arcsec``, ``wavelength_nm``; ``flux_m2`` (image sum, m² of
        transmitted collecting area), ``lost_m2`` (area that fell off the stamp before
        the blur, m²), ``edge_frac`` (fraction of the image sum in the outer 3 pixels,
        dimensionless); ``n_rays``, ``n_transmitted``, ``ray_spacing_m`` (m),
        ``chief_x_m``, ``chief_y_m`` (chief-ray focal-plane position, m, CCS),
        ``stop_r_min_m``, ``stop_r_max_m`` (stop radius range of transmitted rays, m).
        A variant with a DOF scan adds ``dof`` (ts_ofc DOF index), ``dof_amplitude``
        (µm or deg, ts_ofc convention) and ``opd_z<j>_CCS_um``, the in-focus OPD of the
        perturbed telescope at the field point, Noll 1-78 (µm of wavefront).
    """
    return pd.read_parquet(catalog.path(PRODUCT, variant, build) / INDEX)


def stamp(variant, band, config, side, point_id, build=catalog.CURRENT, dof=None):
    """One noise-free stamp.

    Returns
    -------
    image : `numpy.ndarray`
        ``(npix, npix)``, float64, m² of collecting area per pixel; row = CCS y,
        column = CCS x, centred on the chief ray.
    header : `astropy.io.fits.Header`
    """
    path = catalog.path(PRODUCT, variant, build) / stamp_file(band, config, side, dof)
    with fits.open(path) as hdul:
        h = hdul[point_id]
        return np.asarray(h.data, dtype=float), h.header.copy()
