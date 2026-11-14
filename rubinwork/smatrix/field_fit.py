"""Double-Zernike sensitivity matrix from per-field-point fitted DOF responses.

The responses are slopes of fitted pupil Zernikes against DOF amplitude at field points
(the ``sparse_fit`` brute-force scan). Fitting those slopes over the field with field
Zernikes gives a matrix in the ts_ofc layout: (31 field k, 29 pupil Noll j, 50 DOF), µm
of wavefront per DOF unit (µm, or deg for tilts).
"""
import galsim
import numpy as np

__all__ = ["FIELD_RADIUS_DEG", "N_FIELD", "N_PUPIL", "N_DOF", "field_basis",
           "fam_smatrix"]

FIELD_RADIUS_DEG = 1.75
"""Outer radius of the field-Zernike basis, deg; the ts_ofc ``lsst.yaml`` value."""

N_FIELD, N_PUPIL, N_DOF = 31, 29, 50
"""The ts_ofc sensitivity-matrix shape: field k 0-30, pupil Noll j 0-28, DOF."""


def field_basis(kmax, thx, thy):
    """Field Zernikes 1..`kmax` at field points.

    Parameters
    ----------
    kmax : `int`
        Highest field Noll index.
    thx, thy : `numpy.ndarray`
        Field angles, deg, OCS.

    Returns
    -------
    basis : `numpy.ndarray`
        Shape ``(len(thx), kmax)``, dimensionless, on a unit disk of radius
        `FIELD_RADIUS_DEG`.
    """
    return galsim.zernike.zernikeBasis(kmax, thx, thy, R_outer=FIELD_RADIUS_DEG)[1:].T


def fam_smatrix(resp, fit_set, kmax=30):
    """Side-averaged FAM field fit of one fit set's slopes.

    Parameters
    ----------
    resp : mapping
        The ``responses.npz`` contents: ``kind``, ``side``, ``thx``, ``thy`` per field
        point and side, and per fit set ``<fit_set>__slope`` (points, 50 DOF, Zernikes)
        in µm per DOF unit and ``<fit_set>__noll``.
    fit_set : `str`
        Fit-set key, e.g. ``"prod"``.
    kmax : `int`, optional
        Highest field Noll index fitted.

    Returns
    -------
    smatrix : `numpy.ndarray`
        Shape ``(31, 29, 50)``, µm of wavefront per DOF unit. Field k 0 and pupil
        Zernikes the fit set does not fit are zero.

    Notes
    -----
    Intra- and extra-focal slopes are averaged at each FAM point before the fit; the
    intra and extra points must be listed in the same field order.
    """
    kind, side = resp["kind"], resp["side"]
    fi = (kind == "fam") & (side == "intra")
    fe = (kind == "fam") & (side == "extra")
    if not (np.allclose(resp["thx"][fi], resp["thx"][fe])
            and np.allclose(resp["thy"][fi], resp["thy"][fe])):
        raise ValueError("intra and extra FAM points are not in the same field order")
    s = resp[f"{fit_set}__slope"]
    avg = ((s[fi] + s[fe]) / 2).reshape(fi.sum(), -1)
    c, *_ = np.linalg.lstsq(field_basis(kmax, resp["thx"][fi], resp["thy"][fi]), avg,
                            rcond=None)
    out = np.zeros((N_FIELD, N_PUPIL, N_DOF))
    out[1:kmax + 1, resp[f"{fit_set}__noll"]] = (
        c.reshape(kmax, N_DOF, -1).transpose(0, 2, 1))
    return out
