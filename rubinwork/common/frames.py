"""Field-angle and Zernike frame conversions: DVCS to CCS, and CCS to OCS.

- DVCS (Data Visualization Coordinate System) is what cameraGeom ``FIELD_ANGLE`` returns.
  CCS (Camera Coordinate System) is DVCS with x and y swapped, a reflection.
- OCS (Optical Coordinate System) is CCS rotated by the camera rotator angle
  ``rotTelPos``: OCS = R(+rotTelPos) · CCS, for field angles and for Zernike
  coefficients. This is donut_viz's convention, which the FAM and CWFS tables carry.

See ``aos/docs/frame_conventions.md``. Import as
``from rubinwork.common import frames``.
"""
import numpy as np

__all__ = ["dvcs_to_ccs_field", "ccs_to_ocs_field", "ccs_to_ocs_zk"]


def dvcs_to_ccs_field(x, y):
    """DVCS field angle to CCS: swap x and y.

    Parameters
    ----------
    x, y : `float` or `numpy.ndarray`
        DVCS field angle, any angular unit.

    Returns
    -------
    x_ccs, y_ccs : `float` or `numpy.ndarray`
        CCS field angle, same unit.
    """
    return y, x


def ccs_to_ocs_field(x, y, rotator_angle_deg):
    """CCS field angle to OCS: rotate by +`rotator_angle_deg`.

    Parameters
    ----------
    x, y : `float` or `numpy.ndarray`
        CCS field angle, any angular unit.
    rotator_angle_deg : `float`
        Camera rotator angle ``rotTelPos``, deg.

    Returns
    -------
    x_ocs, y_ocs : `float` or `numpy.ndarray`
        OCS field angle, same unit.
    """
    r = np.deg2rad(rotator_angle_deg)
    c, s = np.cos(r), np.sin(r)
    return c * x - s * y, s * x + c * y


def ccs_to_ocs_zk(zk, rotator_angle_deg):
    """CCS Zernike coefficients to OCS, as donut_viz does.

    Parameters
    ----------
    zk : `numpy.ndarray`
        Shape ``(..., jmax + 1)``, Noll 0..jmax, any unit.
    rotator_angle_deg : `float`
        Camera rotator angle ``rotTelPos``, deg.

    Returns
    -------
    zk_ocs : `numpy.ndarray`
        Same shape and unit.
    """
    import galsim
    zk = np.asarray(zk)
    rot = galsim.zernike.zernikeRotMatrix(zk.shape[-1] - 1, -np.deg2rad(rotator_angle_deg))
    return zk @ rot
