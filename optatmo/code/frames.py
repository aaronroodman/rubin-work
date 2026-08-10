"""
Frame rotation between CCS (camera) and OCS (telescope) by the rotator angle.

Every moment is a definite spin under a frame rotation by angle alpha, so the
rotation is a per-spin phase on complex moment combinations:

    field position (x+iy)      spin 1   -> e^{i alpha}
    e1+i e2  = <(u+iv)^2>       spin 2   -> e^{2i alpha}
    M21+i M12 = <(u+iv) r^2>    spin 1   -> e^{i alpha}   (coma)
    M30+i M03 = <(u+iv)^3>      spin 3   -> e^{3i alpha}  (trefoil)
    M31+i M13                   spin 2   -> e^{2i alpha}
    M40+i M04                   spin 4   -> e^{4i alpha}
    e0, M22 (radial)            spin 0   -> unchanged

`sign` (+1/-1) sets the rotation sense; pin it empirically by requiring the
fitted OCS DZ to agree across exposures at different rotator angles.
"""

import numpy as np

# (index_a, index_b, spin) for the paired moments; e0=0,M22=7 are spin-0
_PAIRS = [(1, 2, 2), (3, 4, 1), (5, 6, 3), (8, 9, 2), (10, 11, 4)]


# ----------------------------------------------------------------------------
# DVCS -> CCS  (a fixed x<->y reflection; NOT the rotator rotation above)
#
# Three frames are in play (established 2026-07, verified against the ip_isr
# intrinsicZernikes calib on 20260513):
#   * DVCS  -- what lsst.afw.cameraGeom FIELD_ANGLE returns, (field_x, field_y).
#             Camera-fixed (rotates with the camera rotator). Also the frame the
#             recomputed HSM moments come out in (they're measured on the CCD
#             *pixel* stamp, whose axes map to field angle via that same
#             cameraGeom transform).  RubinTV focal-plane mosaics are DVCS.
#   * CCS   -- Camera Coordinate System, also camera-fixed; differs from DVCS by
#             a pure x<->y swap:  thx_CCS = field_y,  thy_CCS = field_x.
#             This is the frame the MIW/intrinsicZernikes calib and the
#             aggregateAOSVisitTableRaw store (confirmed: every clean per-detector
#             calib footprint = swap(cameraGeom), never cameraGeom directly).
#   * OCS   -- mirror/telescope frame, constant in Alt/Az (does NOT rotate with
#             the rotator).  OCS = R(rotTelPos) . CCS  (rotate_field/rotate_moments).
#
# So optatmo's star data (positions from cameraGeom, moments from the pixel
# stamp) starts in DVCS and must be swapped to CCS BEFORE the rotator rotation
# to OCS -- otherwise the whole dataset is a mirror image (across field_x=field_y)
# of the true OCS, while the model (MIW-OCS + AOS v-modes) is genuine OCS.  That
# reflection was the cause of good-looking moment fits but a wrong (reflected)
# wavefront and poor CWFS-corner correlation.
#
# Under the x<->y reflection the central moments transform as (from their
# integral definitions, u<->v):
#   e0 +   e1 -   e2 +   M21<->M12   M30<->-M03   M22 +   M31 -   M13 +   M40 +   M04 -
# (indices in the MOMENT_LABELS order [e0,e1,e2,M21,M12,M30,M03,M22,M31,M13,M40,M04]).
# ----------------------------------------------------------------------------

def dvcs_to_ccs_field(thx, thy):
    """DVCS field angle (cameraGeom FIELD_ANGLE) -> CCS: the x<->y swap."""
    return thy, thx


def dvcs_to_ccs_moments(mom):
    """DVCS -> CCS moment vector(s), shape (..., 12) in MOMENT_LABELS order."""
    m = np.asarray(mom, float)
    out = m.copy()
    out[..., 1] = -m[..., 1]        # e1  -> -e1
    #      2      e2 unchanged
    out[..., 3] = m[..., 4]         # M21 <- M12
    out[..., 4] = m[..., 3]         # M12 <- M21
    out[..., 5] = -m[..., 6]        # M30 <- -M03
    out[..., 6] = -m[..., 5]        # M03 <- -M30
    #      7      M22 unchanged
    out[..., 8] = -m[..., 8]        # M31 -> -M31
    #      9      M13 unchanged
    #      10     M40 unchanged
    out[..., 11] = -m[..., 11]      # M04 -> -M04
    return out


def rotate_moments(mom, alpha_rad, sign=1):
    """Rotate a (..., 12) moment array from CCS to OCS by `alpha_rad`."""
    mom = np.asarray(mom, float).copy()
    a = sign * alpha_rad
    for ia, ib, s in _PAIRS:
        z = (mom[..., ia] + 1j * mom[..., ib]) * np.exp(1j * s * a)
        mom[..., ia] = z.real
        mom[..., ib] = z.imag
    return mom


def rotate_field(thx, thy, alpha_rad, sign=1):
    """Rotate field positions (spin-1) from CCS to OCS by `alpha_rad`."""
    a = sign * alpha_rad
    c, s = np.cos(a), np.sin(a)
    return c * thx - s * thy, s * thx + c * thy
