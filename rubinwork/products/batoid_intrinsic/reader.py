"""Read ``batoid_intrinsic`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from .. import catalog

PRODUCT = "batoid_intrinsic"
"""Product name, as it appears in the catalog and in every manifest."""

TABLE = "zernikes.parquet"
DZ_FILE = "dz.fits"

KINDS = ("gq_infocus", "ta", "gq_defocus")
"""``gq_infocus``: in-focus optical path difference (OPD), ``batoid.zernikeGQ``;
``ta``: defocused transverse-aberration Zernikes, ``batoid.zernikeTA``, ts_wep's donut
model; ``gq_defocus``: defocused OPD."""

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


JMIN = 1
"""Lowest stored Noll index. The tilts z2, z3 place the donut against the chief ray and
are part of ts_wep's Danish reference (``nollIndicesModel`` 0..78); z1 is piston."""


def noll_columns(jmax, frame="CCS"):
    """Zernike column names ``z1_<frame>`` .. ``z<jmax>_<frame>``; `frame` is CCS or OCS."""
    return [f"z{j}_{frame}" for j in range(JMIN, jmax + 1)]


def load(variant, build=catalog.CURRENT, band=None, kind=None, config=None, side=None):
    """The Zernike table, optionally cut to one band, kind, configuration or side.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"v4_speed"``.
    build : `str`, optional
        Build name, or ``"current"``.
    band, kind, config, side : `str`, optional
        Row selections. `kind` is one of `KINDS`; `config` is ``infocus`` for
        ``gq_infocus`` rows, else a configuration name (``fam``, ``cwfs``, ``giant``);
        `side` is ``none`` for in-focus rows, else ``extra`` or ``intra``.

    Returns
    -------
    table : `pandas.DataFrame`
        One row per band, kind, configuration, side and field point. Columns
        ``band``, ``kind``, ``config``, ``side``, ``grid``, ``point_id``;
        ``thx_CCS_deg``, ``thy_CCS_deg``, ``thx_OCS_deg``, ``thy_OCS_deg`` (field
        angle, deg); ``rotator_angle_deg`` (camera rotator angle, deg; OCS =
        R(+rotator angle) CCS); ``wavelength_nm``; ``z1_CCS`` .. ``z<jmax>_CCS`` and
        ``z1_OCS`` .. ``z<jmax>_OCS`` (µm of wavefront, Noll). The grid is fixed in CCS.
    """
    df = pd.read_parquet(catalog.path(PRODUCT, variant, build) / TABLE)
    for col, val in (("band", band), ("kind", kind), ("config", config), ("side", side)):
        if val is not None:
            df = df[df[col] == val]
    return df.reset_index(drop=True)


def load_dz(variant, build=catalog.CURRENT):
    """Double-Zernike fits on the science grid, in OCS.

    Returns
    -------
    dz : `dict`
        ``{(band, kind, config, side): (coef, header)}``. `coef` has shape
        ``(kmax + 1, jmax + 1)``: field Noll k (row 0 unused) by pupil Noll j, µm of
        wavefront, OCS (field and pupil), field unit disk of radius ``RFIELD`` deg (header).
    """
    out = {}
    with fits.open(catalog.path(PRODUCT, variant, build) / DZ_FILE) as hdul:
        for h in hdul[1:]:
            key = tuple(h.header[k] for k in ("BAND", "KIND", "CONFIG", "SIDE"))
            out[key] = (np.array(h.data), h.header.copy())
    return out


def evaluate_dz(coef, thx_deg, thy_deg, radius_deg):
    """Pupil Zernikes from a double-Zernike fit at field angles.

    Parameters
    ----------
    coef : `numpy.ndarray`
        ``(kmax + 1, jmax + 1)`` from `load_dz`, µm.
    thx_deg, thy_deg : `numpy.ndarray`
        OCS field angles, deg.
    radius_deg : `float`
        Field radius of the fit (``RFIELD``), deg.

    Returns
    -------
    zk : `numpy.ndarray`
        ``(n_points, jmax + 1)``, µm.
    """
    import galsim
    kmax = coef.shape[0] - 1
    basis = galsim.zernike.zernikeBasis(kmax, np.atleast_1d(thx_deg), np.atleast_1d(thy_deg),
                                        R_outer=radius_deg)
    return basis.T @ coef
