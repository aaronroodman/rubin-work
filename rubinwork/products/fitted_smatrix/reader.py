"""Read ``fitted_smatrix`` builds, and the variant definitions they were built from."""

import functools
import pathlib

import numpy as np
import yaml

from .. import catalog

PRODUCT = "fitted_smatrix"
"""Product name, as it appears in the catalog and in every manifest."""

SENS_YAML = "lsst_sensitivity_dz_31_29_50.yaml"
"""The sensitivity-matrix file name ts_ofc ``OFCData`` looks for."""

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


def stem(variant):
    """File stem of the matrix in a build: ``smatrix_dz_31_29_50_r_<variant>``."""
    return f"smatrix_dz_31_29_50_r_{variant}"


def load(variant, build=catalog.CURRENT):
    """The sensitivity matrix and its fitted pupil Zernikes.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``"sparse_no22"``.
    build : `str`, optional
        Build name, or ``"current"``.

    Returns
    -------
    smatrix : `numpy.ndarray`
        Shape ``(31, 29, 50)``: field k, pupil Noll j, DOF in ts_ofc order and sign.
        µm of wavefront per µm of DOF, or per deg for the tilts. Rows of pupil
        Zernikes not in `noll` are zero.
    noll : `numpy.ndarray` [`int`]
        The pupil Noll indices the fit set fits.
    """
    d = catalog.path(PRODUCT, variant, build)
    return np.load(d / f"{stem(variant)}.npy"), np.loadtxt(d / "noll.txt", dtype=int)


def ofc_config_dir(variant, build=catalog.CURRENT):
    """The build's OFC config directory: the v13 one with this sensitivity matrix."""
    return catalog.path(PRODUCT, variant, build) / "ofc"


def state_estimator(variant, build=catalog.CURRENT, dof_set="standard_22", n_modes=None):
    """`rubinwork.aos_state.make_state_estimator` on this matrix.

    ``zn_selected`` is set to the variant's fitted Noll indices, so a wavefront-to-DOF
    solve uses only the rows the fit set measures. The v-mode basis ``Vh`` does not
    depend on it.

    Parameters
    ----------
    variant : `str`
        Variant name.
    build : `str`, optional
        Build name, or ``"current"``.
    dof_set : `str`, optional
        Key into `rubinwork.aos_state.DOF_SETS`.
    n_modes : `int`, optional
        Number of v-modes kept (``truncate_index``).

    Returns
    -------
    se : `lsst.ts.ofc.state_estimator.StateEstimator`
    """
    from rubinwork import aos_state

    se = aos_state.make_state_estimator(config_dir=str(ofc_config_dir(variant, build)),
                                        dof_set=dof_set, n_modes=n_modes)
    se.ofc_data.zn_selected = load(variant, build)[1]
    return se
