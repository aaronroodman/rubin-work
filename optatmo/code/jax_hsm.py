"""
Differentiable JAX adaptive (HSM-style) moments.

Reproduces galsim's HSM FindAdaptiveMom + PIFF calculate_moments in pure JAX:
an elliptical-Gaussian weight is found by the Bernstein-Jarvis / Hirata-Seljak
adaptive fixed point, then the flux-normalised weighted central moments (2nd,
3rd, 4th order) are evaluated with that converged weight.  The fixed point:

    w(x) = exp(-1/2 (x-c)^T M^{-1} (x-c))
    c    = <x>_{wI}                       (weighted centroid)
    M    = 2 <(x-c)(x-c)^T>_{wI}          (self-consistent covariance)

For a Gaussian image of covariance C the iteration converges to M = C, so
sigma = det(M)^{1/4} and the ellipticity of M match galsim's reported adaptive
moments.  The loop is unrolled a fixed number of times => autodiff-able.

Works on any image (model PSF or a real data star), so it can serve as both the
model estimator and a differentiable replacement for the HSM C code.
"""

import jax.numpy as jnp

LABELS = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03',
          'M22', 'M31', 'M13', 'M40', 'M04']


def adaptive_weight(image, X, Y, n_iter=25, sigma_init=0.35):
    """Find the adaptive Gaussian weight (centroid + covariance).

    :param image: 2D PSF image (jnp), need not be normalised.
    :param X, Y:  coordinate grids (arcsec), same shape as image.
    :returns: (cu, cv, Mxx, Mxy, Myy) of the converged weight.
    """
    cu = jnp.sum(image * X) / jnp.sum(image)
    cv = jnp.sum(image * Y) / jnp.sum(image)
    Mxx = sigma_init ** 2
    Myy = sigma_init ** 2
    Mxy = 0.0
    for _ in range(n_iter):
        det = Mxx * Myy - Mxy ** 2
        iXX, iYY, iXY = Myy / det, Mxx / det, -Mxy / det
        u = X - cu
        v = Y - cv
        w = jnp.exp(-0.5 * (iXX * u * u + 2 * iXY * u * v + iYY * v * v))
        wI = w * image
        A = jnp.sum(wI)
        cu = jnp.sum(wI * X) / A
        cv = jnp.sum(wI * Y) / A
        u = X - cu
        v = Y - cv
        wI = w * image  # weights at pre-update centre (BJ style); stable
        A = jnp.sum(wI)
        Cxx = jnp.sum(wI * u * u) / A
        Cyy = jnp.sum(wI * v * v) / A
        Cxy = jnp.sum(wI * u * v) / A
        Mxx, Myy, Mxy = 2.0 * Cxx, 2.0 * Cyy, 2.0 * Cxy
    return cu, cv, Mxx, Mxy, Myy


def moments_with_weight(image, X, Y, cu, cv, Mxx, Mxy, Myy):
    """Flux-normalised weighted central moments using a given Gaussian weight."""
    det = Mxx * Myy - Mxy ** 2
    iXX, iYY, iXY = Myy / det, Mxx / det, -Mxy / det
    u = X - cu
    v = Y - cv
    w = jnp.exp(-0.5 * (iXX * u * u + 2 * iXY * u * v + iYY * v * v))
    WI = w * image
    WI = WI / jnp.sum(WI)

    usq, vsq, uv = u * u, v * v, u * v
    rsq = usq + vsq
    usqmvsq = usq - vsq
    return jnp.array([
        jnp.sum(WI * rsq),
        jnp.sum(WI * usqmvsq),
        2.0 * jnp.sum(WI * uv),
        jnp.sum(WI * u * rsq),
        jnp.sum(WI * v * rsq),
        jnp.sum(WI * u * (usq - 3 * vsq)),
        jnp.sum(WI * v * (3 * usq - vsq)),
        jnp.sum(WI * rsq * rsq),
        jnp.sum(WI * rsq * usqmvsq),
        2.0 * jnp.sum(WI * rsq * uv),
        jnp.sum(WI * (usq * usq - 6 * usq * vsq + vsq * vsq)),
        4.0 * jnp.sum(WI * usqmvsq * uv),
    ])


def adaptive_moments(image, X, Y, n_iter=25, sigma_init=0.35):
    """Full pipeline: find adaptive weight, then compute the 12 moments."""
    cu, cv, Mxx, Mxy, Myy = adaptive_weight(image, X, Y, n_iter, sigma_init)
    mom = moments_with_weight(image, X, Y, cu, cv, Mxx, Mxy, Myy)
    return mom, (cu, cv, Mxx, Mxy, Myy)
