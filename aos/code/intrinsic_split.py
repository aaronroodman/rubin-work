"""Decompose a rotator-dependent intrinsic field map into camera-fixed (CCS)
and telescope-fixed (OCS) components.

A measured intrinsic map (e.g. ``z4_optical_OCS``) sampled at OCS field
position ``(r, phi)`` from data taken at rotator angle ``theta`` is modelled
as the sum of a telescope-fixed map ``O`` (OCS) and a camera-fixed map ``C``
that rotates rigidly with the rotator (CCS)::

    Z_theta(r, phi) = O(r, phi) + C(r, phi - s*theta)

Rotation is diagonal in the azimuthal Fourier basis, so per radius ``r`` and
azimuthal mode ``m``::

    A_{d,m}(r) = O_m(r) + C_m(r) * exp(-i m s theta_d)

Given several rotator datasets this is a 2-unknown complex least-squares
solve per ``(r, m)``.  ``m = 0`` (the axisymmetric profile) is degenerate —
an axisymmetric pattern is identical in both frames — so it is assigned by a
flag (``'ocs'``, ``'ccs'`` or ``'split'``).

Intended for scalar (rotationally-invariant) Zernikes such as Z4 defocus,
whose coefficient value does not pick up phase under field rotation.

numpy + scipy only.
"""
from __future__ import annotations

import numpy as np


def make_polar_grid(r_min=0.06, r_max=1.70, n_r=40, n_az=180):
    """Common polar sampling grid.  Returns (R, A, X, Y) where X/Y are the
    (n_r, n_az) Cartesian field coords of each polar node."""
    R = np.linspace(r_min, r_max, n_r)
    A = np.linspace(0.0, 2 * np.pi, n_az, endpoint=False)
    RR, AA = np.meshgrid(R, A, indexing='ij')
    return R, A, RR * np.cos(AA), RR * np.sin(AA)


def sample_maps_polar(maps, X, Y):
    """Interpolate each scattered map onto the polar grid (X, Y).

    `maps` is a list of (thx, thy, val) arrays (finite values).  Returns
    ``Z`` (n_set, n_r, n_az) with NaNs outside each map's hull, and the
    NaN count.
    """
    from scipy.interpolate import LinearNDInterpolator
    out = []
    for thx, thy, val in maps:
        interp = LinearNDInterpolator(np.column_stack([thx, thy]), val)
        out.append(interp(X, Y))
    Z = np.stack(out)
    return Z, int(np.isnan(Z).sum())


def decompose_polar(Z, thetas_rad, A, s=1, m0_assignment='ocs', m_max=None):
    """Camera/telescope decomposition on a polar grid.

    Parameters
    ----------
    Z : ndarray (n_set, n_r, n_az)
        Maps sampled on the polar grid (NaNs treated as 0).
    thetas_rad : ndarray (n_set,)
        Rotator angle of each dataset (radians).
    A : ndarray (n_az,)
        Azimuth samples (used only for the roll step).
    s : int
        Rotation sense (+1 or -1).
    m0_assignment : {'ocs', 'ccs', 'split'}
        Where to put the degenerate axisymmetric (m=0) component.
    m_max : int or None
        If set, azimuthal modes above this are assigned entirely to ``O``
        (treated as rotation-independent) rather than fit — suppresses
        high-m camera noise.

    Returns
    -------
    dict with O_pol, C_pol (n_r, n_az), res (n_set, n_r, n_az), s,
    m0_assignment.
    """
    Zc = np.nan_to_num(Z, nan=0.0)
    n_set, n_r, n_az = Zc.shape
    Afft = np.fft.rfft(Zc, axis=2)                 # (n_set, n_r, M1)
    M1 = Afft.shape[2]
    O = np.zeros((n_r, M1), complex)
    C = np.zeros_like(O)
    thetas_rad = np.asarray(thetas_rad, float)
    for m in range(M1):
        if m == 0:
            tot = Afft[:, :, 0].mean(0)            # only the sum is determined
            if m0_assignment == 'ocs':
                O[:, 0] = tot
            elif m0_assignment == 'ccs':
                C[:, 0] = tot
            else:
                O[:, 0] = tot / 2
                C[:, 0] = tot / 2
            continue
        if m_max is not None and m > m_max:
            O[:, m] = Afft[:, :, m].mean(0)
            continue
        D = np.stack([np.ones_like(thetas_rad),
                      np.exp(-1j * m * s * thetas_rad)], axis=1)   # (n_set, 2)
        sol, *_ = np.linalg.lstsq(D, Afft[:, :, m], rcond=None)    # (2, n_r)
        O[:, m] = sol[0]
        C[:, m] = sol[1]
    O_pol = np.fft.irfft(O, n=n_az, axis=1)
    C_pol = np.fft.irfft(C, n=n_az, axis=1)

    dphi = A[1] - A[0]
    res = np.empty_like(Zc)
    for i, th in enumerate(thetas_rad):
        shift = int(np.round(s * th / dphi))
        res[i] = Zc[i] - (O_pol + np.roll(C_pol, shift, axis=1))
    return {'O_pol': O_pol, 'C_pol': C_pol, 'res': res,
            's': int(s), 'm0_assignment': m0_assignment}


def decompose_auto_sign(Z, thetas_rad, A, R, r_lim=(0.1, 1.6),
                        m0_assignment='ocs', m_max=None):
    """Run :func:`decompose_polar` for s=+1 and s=-1 and keep the lower
    residual.  Returns (result, {+1: rms, -1: rms})."""
    rms = {}
    best = None
    for s in (+1, -1):
        r = decompose_polar(Z, thetas_rad, A, s=s,
                            m0_assignment=m0_assignment, m_max=m_max)
        rms[s] = residual_rms(r['res'], R, r_lim)
        if best is None or rms[s] < rms[best['s']]:
            best = r
    return best, rms


def _rmask(R, r_lim):
    return (R >= r_lim[0]) & (R <= r_lim[1])


def residual_rms(field, R, r_lim=(0.1, 1.6)):
    """RMS of a (..., n_r, n_az) field over rings within ``r_lim``."""
    m = _rmask(R, r_lim)
    return float(np.sqrt(np.mean(np.asarray(field)[..., m, :] ** 2)))


def explained_variance(Z, res, R, r_lim=(0.1, 1.6)):
    """Fraction of the data variance the 2-component model explains."""
    d = residual_rms(Z, R, r_lim)
    return 1.0 - (residual_rms(res, R, r_lim) / d) ** 2 if d > 0 else np.nan


def azimuthal_amplitude(field_pol, R, r_lim=(0.1, 1.6), n_m=9):
    """RMS azimuthal-mode amplitude (m=0..n_m-1) of a polar field, averaged
    over rings within ``r_lim`` (rfft scaling)."""
    m = _rmask(R, r_lim)
    F = np.fft.rfft(field_pol, axis=1)
    return np.sqrt((np.abs(F[m]) ** 2).mean(0))[:n_m]


def polar_field_to_points(field_pol, X, Y, thx, thy):
    """Interpolate a polar field (sampled on X, Y) onto scattered points
    (thx, thy).  Used to resample O/C onto a focal-plane grid."""
    from scipy.interpolate import LinearNDInterpolator
    interp = LinearNDInterpolator(
        np.column_stack([X.ravel(), Y.ravel()]), field_pol.ravel())
    return interp(np.asarray(thx), np.asarray(thy))


def mean_rotator(fits_table, rot_range, alt_range=None,
                 rot_col='rotator_angle', alt_col='alt'):
    """Actual mean rotator angle (deg) of the visits a dataset was built
    from: rows of ``fits_table`` within ``rot_range`` (and ``alt_range``).
    Returns (mean, n, lo, hi).  ``alt`` auto-detected radians vs degrees.
    """
    import numpy as _np
    rot = _np.asarray(fits_table[rot_col], float)
    keep = (rot >= rot_range[0]) & (rot <= rot_range[1])
    if alt_range is not None and alt_col in getattr(fits_table, 'colnames',
                                                    fits_table):
        alt = _np.asarray(fits_table[alt_col], float)
        altd = _np.rad2deg(alt) if _np.nanmax(_np.abs(alt)) < 2 * _np.pi + 1e-3 else alt
        keep &= (altd >= alt_range[0]) & (altd <= alt_range[1])
    r = rot[keep]
    if r.size == 0:
        return float('nan'), 0, float('nan'), float('nan')
    return float(r.mean()), int(r.size), float(r.min()), float(r.max())
