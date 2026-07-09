"""
Diffraction-free "weighted analytic" PSF realization.

Renders the geometric spot diagram (transverse ray aberration distribution over
the pupil, = momfit's phase-gradient point cloud) convolved with the
atmospheric kernel, WITHOUT any Fraunhofer diffraction, then measures it with
the same HSM estimator used on real data (moments_hsm.measure_hsm_moments).

This realizes the "weighted analytic variant" honestly: it applies the true
adaptive HSM weight to the analytic optical model.  Comparing its moments to
the full Fraunhofer(x)atmosphere HSM truth isolates the pure DIFFRACTION
contribution (the only physics it omits); comparing to the unweighted analytic
moments isolates the WEIGHTING effect.

The atmosphere is applied by Monte-Carlo: each rendered photon is a random
pupil ray position plus a random photon drawn from the atmospheric kernel.
"""

import numpy as np
import galsim

from moments_analytic import _zernike_gradient
import psf_model as pm


def geometric_atm_image(coef_um, pupil, fwhm, L0=25.0, kind='VonKarman',
                        scale=pm.PIXEL_SCALE, stamp=pm.STAMP,
                        n_photon=2_000_000, seed=0):
    """Render (geometric spot) (x) (atmosphere) as a galsim image.

    :returns: galsim.Image (scale set, flux ~ n_photon within stamp).
    """
    ax, ay = _zernike_gradient(coef_um, pupil)   # spot point cloud (arcsec)
    n_spot = ax.size

    rng = galsim.BaseDeviate(seed)
    atm = pm.build_atmosphere(fwhm, lam=pm.LAM, L0=L0, kind=kind)
    pa = atm.shoot(n_photon, rng)                 # atmosphere photons (arcsec)

    nr = np.random.default_rng(seed)
    idx = nr.integers(0, n_spot, size=n_photon)   # random pupil ray per photon
    u = ax[idx] + pa.x
    v = ay[idx] + pa.y

    half = stamp * scale / 2.0
    edges = np.linspace(-half, half, stamp + 1)
    H, _, _ = np.histogram2d(v, u, bins=[edges, edges])   # [iy, ix]
    img = galsim.Image(np.ascontiguousarray(H), scale=scale)
    return img
