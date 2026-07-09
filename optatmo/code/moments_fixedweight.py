"""
Fixed-weight weighted moments -- the numpy twin of JaxOptAtmoPSF.moments.

A single self-consistent moment estimator applied identically to data star
images and to the JAX model PSF: an elliptical-Gaussian weight of FIXED shape
(sigma, e1, e2 -- from an HSM fit to the data), recentred on the image's
weighted-mean centroid, then weighted central moments about that centroid.

Using the same estimator on both sides makes the moment-matching fit exactly
self-consistent and lets us validate the JAX implementation.  (This differs
slightly from PIFF calculate_moments only in the centroid convention -- HSM's
Gaussian-fit centroid vs the weighted mean -- which matters at the ~3rd-moment
level for skewed PSFs; measure the data with THIS function to stay consistent.)
"""

import numpy as np

LABELS = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03',
          'M22', 'M31', 'M13', 'M40', 'M04']


def measure_fixedweight_moments(image, scale, weight, n_center_iter=3):
    """
    :param image:   2D numpy PSF image [iy, ix].
    :param scale:   arcsec/pixel.
    :param weight:  (u0, v0, sigma, e1, e2) in arcsec -- fixed weight shape and
                    starting centroid.
    :returns: dict of moments (arcsec^n), keys LABELS (+ 'cu','cv').
    """
    ny, nx = image.shape
    xs = (np.arange(nx) - nx / 2 + 0.5) * scale
    ys = (np.arange(ny) - ny / 2 + 0.5) * scale
    X, Y = np.meshgrid(xs, ys)

    u0, v0, sigma, e1, e2 = weight
    denom = np.sqrt(max(1.0 - e1 ** 2 - e2 ** 2, 1e-8))
    Mxx = sigma ** 2 * (1 + e1) / denom
    Myy = sigma ** 2 * (1 - e1) / denom
    Mxy = sigma ** 2 * e2 / denom
    det = Mxx * Myy - Mxy ** 2
    iXX, iYY, iXY = Myy / det, Mxx / det, -Mxy / det

    cu, cv = u0, v0
    for _ in range(n_center_iter):
        u, v = X - cu, Y - cv
        W = np.exp(-0.5 * (iXX * u * u + 2 * iXY * u * v + iYY * v * v))
        WI = W * image
        s = WI.sum()
        cu = (WI * X).sum() / s
        cv = (WI * Y).sum() / s

    u, v = X - cu, Y - cv
    W = np.exp(-0.5 * (iXX * u * u + 2 * iXY * u * v + iYY * v * v))
    WI = W * image
    WI = WI / WI.sum()

    usq, vsq, uv = u * u, v * v, u * v
    rsq = usq + vsq
    usqmvsq = usq - vsq
    vals = [
        (WI * rsq).sum(),
        (WI * usqmvsq).sum(),
        2.0 * (WI * uv).sum(),
        (WI * u * rsq).sum(),
        (WI * v * rsq).sum(),
        (WI * u * (usq - 3 * vsq)).sum(),
        (WI * v * (3 * usq - vsq)).sum(),
        (WI * rsq * rsq).sum(),
        (WI * rsq * usqmvsq).sum(),
        2.0 * (WI * rsq * uv).sum(),
        (WI * (usq * usq - 6 * usq * vsq + vsq * vsq)).sum(),
        4.0 * (WI * usqmvsq * uv).sum(),
    ]
    out = dict(zip(LABELS, vals))
    out['cu'], out['cv'] = cu, cv
    return out
