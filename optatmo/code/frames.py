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
