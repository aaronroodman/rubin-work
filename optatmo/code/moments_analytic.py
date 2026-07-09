"""
Analytic (momfit-style) PSF moments from the wavefront phase gradient.

Implements the moment estimator of Zanmar Sanchez et al. (SPIE 2026,
"A wavefront sensing method based on the analysis of science images"):
image-plane moments are computed as *unweighted* central moments of the
geometric transverse-ray-aberration distribution over the (annular) pupil,

    alpha(x,y) = grad W(x,y)          [transverse ray angle, radians]

    mu_ij = < (alpha_x - abar_x)^i (alpha_y - abar_y)^j >_pupil

where <.> is the area average over the illuminated annular pupil.  This is
mathematically identical to the paper's pre-computed gradient-tensor
contractions (Eqs. 3-6); we evaluate the integral directly on a dense pupil
grid, which is exact to grid resolution and trivially extends to 4th order.

Key properties of this estimator (relevant to the accuracy study):
  * geometric optics only -- no diffraction core / rings;
  * UNWEIGHTED moments -- no HSM adaptive Gaussian weight;
  * the atmospheric kernel is NOT in the integral.  Under convolution the
    unweighted central moments (cumulants) simply ADD, so an isotropic seeing
    kernel of per-axis variance sigma_a^2 contributes only to the size
    (M11 += 2 sigma_a^2) and leaves M20, M02 and the 3rd-order moments
    unchanged.  This is exactly the momfit "seeing cancels" assumption; the
    optional atmosphere handling below implements the most charitable
    (Gaussian-equivalent) version of it.

Units: wavefront coefficients are Noll-indexed Zernikes in microns over the
annular pupil of outer radius D/2 and inner radius obscuration*D/2.  Moments
are returned in arcsec^n, matching the PIFF/HSM convention in moments_hsm.py.
"""

import numpy as np
import galsim

ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0


class AnnularPupil:
    """Dense Cartesian sampling of a uniformly illuminated annular pupil."""

    def __init__(self, diam=8.36, obscuration=0.61, ngrid=256):
        self.diam = diam
        self.obscuration = obscuration
        self.R = diam / 2.0
        self.ngrid = ngrid
        ax = np.linspace(-self.R, self.R, ngrid)
        xx, yy = np.meshgrid(ax, ax)
        r = np.hypot(xx, yy)
        mask = (r <= self.R) & (r >= obscuration * self.R)
        self.x = xx[mask]          # metres
        self.y = yy[mask]
        self.npix = self.x.size


def _zernike_gradient(coef_um, pupil):
    """Return (alpha_x, alpha_y) in arcsec at every pupil sample point.

    coef_um : 1-indexed Noll coefficient array (index 0 ignored), microns.
    """
    coef_m = np.asarray(coef_um, dtype=float) * 1e-6
    Z = galsim.zernike.Zernike(
        coef_m, R_outer=pupil.R, R_inner=pupil.obscuration * pupil.R)
    # grad W is dimensionless (m/m) = ray angle in radians.  The image-plane
    # ray displacement is -f * grad W, so the transverse-aberration coordinate
    # carries a minus sign relative to grad W; this parity is invisible to the
    # even (2nd, 4th) moments but sets the sign of all odd (3rd) moments to
    # match galsim's OpticalPSF / HSM image convention.
    ax = -Z.gradX.evalCartesian(pupil.x, pupil.y) * ARCSEC_PER_RAD
    ay = -Z.gradY.evalCartesian(pupil.x, pupil.y) * ARCSEC_PER_RAD
    return ax, ay


# Moment names in the same order/naming as moments_hsm.calculate_moments
# (e0=M11, e1=M20, e2=M02, then 3rd, then 4th order).
ANALYTIC_NAMES = ['e0', 'e1', 'e2',
                  'M21', 'M12', 'M30', 'M03',
                  'M22', 'M31', 'M13', 'M40', 'M04']


def analytic_moments(coef_um, pupil, sigma_atm_arcsec=0.0):
    """Compute momfit-style unweighted central moments from a wavefront.

    :param coef_um:          1-indexed Noll Zernike coefficients (microns).
    :param pupil:            An AnnularPupil instance.
    :param sigma_atm_arcsec: per-axis Gaussian-equivalent atmospheric sigma.
                             If > 0, its cumulants are added to the optical
                             moments (the charitable momfit seeing model).

    :returns: dict of moments (arcsec^n), keys ANALYTIC_NAMES, plus the
              normalised ellipticities e1n=M20/M11, e2n=M02/M11.
    """
    ax, ay = _zernike_gradient(coef_um, pupil)

    # centroid and central residuals
    u = ax - ax.mean()
    v = ay - ay.mean()

    usq = u * u
    vsq = v * v
    uv = u * v
    rsq = usq + vsq
    usqmvsq = usq - vsq

    # 2nd-order optical central moments (means over the pupil area)
    M11 = rsq.mean()
    M20 = usqmvsq.mean()
    M02 = 2.0 * uv.mean()

    # 3rd order
    M21 = (u * rsq).mean()
    M12 = (v * rsq).mean()
    M30 = (u * (usq - 3 * vsq)).mean()
    M03 = (v * (3 * usq - vsq)).mean()

    # 4th order
    M22 = (rsq * rsq).mean()
    M31 = (rsq * usqmvsq).mean()
    M13 = 2.0 * (rsq * uv).mean()
    M40 = (usq * usq - 6 * usq * vsq + vsq * vsq).mean()
    M04 = 4.0 * (usqmvsq * uv).mean()

    if sigma_atm_arcsec > 0.0:
        # Add an isotropic Gaussian kernel via cumulant addition.
        # Optical per-axis variances/covariance:
        s2 = sigma_atm_arcsec ** 2
        Ixx = 0.5 * (M11 + M20)
        Iyy = 0.5 * (M11 - M20)
        Ixy = 0.5 * M02
        # 2nd: variances add
        Ixx += s2
        Iyy += s2
        M11 = Ixx + Iyy
        M20 = Ixx - Iyy
        M02 = 2.0 * Ixy
        # 3rd-order central moments of an isotropic kernel are zero -> unchanged
        # 4th order: mu4_total = mu4_optics + 6 * mu2_optics * s2 + 3 s2^2 (per
        # relevant combination).  Use cumulant addition on the radial 4th moment
        # and the spin combinations.
        # M22 = <r^4>: kappa4_r + ... ; simplest exact route via <u^4>,<v^4>,<u^2v^2>
        # Recompute raw 4th central sums, then add Gaussian cumulant terms.
        u4 = (usq * usq).mean()
        v4 = (vsq * vsq).mean()
        u2v2 = (usq * vsq).mean()
        u3v = (u * usq * v).mean()
        uv3 = (u * v * vsq).mean()
        # Gaussian(0, s2) isotropic: <u^4>+=6<u^2>s2+3s4 ; <v^4> similarly ;
        # <u^2 v^2>+=<u^2>s2+<v^2>s2+s4 ; <u^3 v>+=3<uv>s2 ; <u v^3>+=3<uv>s2
        u2 = usq.mean(); v2 = vsq.mean(); uvm = uv.mean()
        s4 = s2 * s2
        u4 += 6 * u2 * s2 + 3 * s4
        v4 += 6 * v2 * s2 + 3 * s4
        u2v2 += u2 * s2 + v2 * s2 + s4
        u3v += 3 * uvm * s2
        uv3 += 3 * uvm * s2
        M22 = u4 + 2 * u2v2 + v4
        M31 = u4 - v4
        M13 = 2.0 * (u3v + uv3)
        M40 = u4 - 6 * u2v2 + v4
        M04 = 4.0 * (u3v - uv3)

    out = dict(zip(ANALYTIC_NAMES,
                   [M11, M20, M02, M21, M12, M30, M03, M22, M31, M13, M40, M04]))
    out['e1n'] = M20 / M11
    out['e2n'] = M02 / M11
    return out
