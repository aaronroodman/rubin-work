"""
HSM adaptive-weighted PSF moments -- standalone port of PIFF's
piff.util.calculate_moments (github.com/rmjarvis/Piff util.py:337).

This is the *reference* ("truth") estimator for the accuracy study: the same
moments PIFF/OptAtmoPSF fit to real Rubin stars.  W(u,v) is the best-fit
adaptive elliptical-Gaussian HSM weight; moments are weighted sums over the
image, centred on the HSM centroid.  Naming matches PIFF:

    e0 = M11 = <r^2>,  e1 = M20 = <u^2 - v^2>,  e2 = M02 = 2<uv>
    3rd: M21, M12, M30, M03
    4th: M22, M31, M13, M40, M04
    radial: M22, M33, M44 (+ normalised M22n, M33n, M44n)

Moments are returned in arcsec^n (world units), using the image pixel scale.
Optionally returns per-moment variances given a per-pixel noise variance.
"""

import numpy as np
import galsim

MOMENT_NAMES_2 = ['Flux', 'du', 'dv', 'e0', 'e1', 'e2']
MOMENT_NAMES_3 = ['M21', 'M12', 'M30', 'M03']
MOMENT_NAMES_4 = ['M22', 'M31', 'M13', 'M40', 'M04']
MOMENT_NAMES_R = ['M22r', 'M33r', 'M44r', 'M22n', 'M33n', 'M44n']


def measure_hsm_moments(img, third_order=True, fourth_order=True, radial=True,
                        noise_var=None, weight=None):
    """Measure HSM-weighted moments of a drawn PSF image.

    :param img:          galsim.Image of the PSF (with .scale set, arcsec/pix).
    :param third_order:  include 3rd-order moments.
    :param fourth_order: include 4th-order moments.
    :param radial:       include higher radial moments.
    :param noise_var:    per-pixel variance (scalar or array) for error
                         estimates; None disables error output.
    :param weight:       optional inverse-variance weight array (0 = masked).

    :returns: (moments_dict, errors_dict_or_None).  Moments in arcsec^n;
              'e1'/'e2' entries are the *raw* M20/M02 (PIFF naming), and
              'e1n'/'e2n' are the normalised ellipticities M20/M11, M02/M11.
    """
    scale = img.scale
    try:
        amom = img.FindAdaptiveMom()
    except galsim.GalSimError:
        return None, None
    if amom.moments_status != 0:
        return None, None

    sigma_pix = amom.moments_sigma
    e1s, e2s = amom.observed_shape.e1, amom.observed_shape.e2
    cen = amom.moments_centroid  # PositionD, image (pixel) coords

    # Build the HSM Gaussian weight kernel in pixel space (method='sb').
    wt_img = galsim.ImageD(img.bounds, scale=1.0)
    prof = galsim.Gaussian(sigma=sigma_pix, flux=1.0).shear(e1=e1s, e2=e2s)
    prof.drawImage(wt_img, method='sb', center=cen, scale=1.0)
    kernel = wt_img.array.flatten()

    data = img.array.flatten().astype(float)

    # pixel-centre coordinate arrays (C order matches .flatten()); world units
    b = img.bounds
    xpix = np.arange(b.xmin, b.xmax + 1)
    ypix = np.arange(b.ymin, b.ymax + 1)
    X, Y = np.meshgrid(xpix, ypix)   # [iy, ix]
    u = (X.flatten() - cen.x) * scale
    v = (Y.flatten() - cen.y) * scale

    if weight is not None:
        w = np.asarray(weight, dtype=float).flatten()
        mask = w == 0.0
        if np.any(mask):
            good = ~mask
            data[mask] = kernel[mask] * data[good].sum() / kernel[good].sum()
    else:
        w = None
        mask = np.zeros(data.shape, dtype=bool)

    WI = kernel * data
    M00 = WI.sum()
    WI = WI / M00

    usq = u * u
    vsq = v * v
    uv = u * v
    rsq = usq + vsq
    usqmvsq = usq - vsq

    M11 = (WI * rsq).sum()
    M20 = (WI * usqmvsq).sum()
    M02 = 2.0 * (WI * uv).sum()

    mom = {'Flux': M00, 'du': (WI * u).sum(), 'dv': (WI * v).sum(),
           'e0': M11, 'e1': M20, 'e2': M02,
           'e1n': M20 / M11, 'e2n': M02 / M11,
           'sigma_arcsec': sigma_pix * scale,
           # exact HSM weight parameters (world units) so the identical
           # elliptical-Gaussian weight can be reused elsewhere (e.g. the JAX
           # model): centre in arcsec relative to the image centre, plus the
           # Gaussian sigma and its observed_shape ellipticity.
           'wt_u0': (cen.x - (b.xmin + b.xmax) / 2.0) * scale,
           'wt_v0': (cen.y - (b.ymin + b.ymax) / 2.0) * scale,
           'wt_sigma': sigma_pix * scale, 'wt_e1': e1s, 'wt_e2': e2s}

    if third_order:
        mom['M21'] = (WI * u * rsq).sum()
        mom['M12'] = (WI * v * rsq).sum()
        mom['M30'] = (WI * u * (usq - 3 * vsq)).sum()
        mom['M03'] = (WI * v * (3 * usq - vsq)).sum()
    if fourth_order:
        mom['M22'] = (WI * rsq * rsq).sum()
        mom['M31'] = (WI * rsq * usqmvsq).sum()
        mom['M13'] = 2.0 * (WI * rsq * uv).sum()
        mom['M40'] = (WI * (usq * usq - 6 * usq * vsq + vsq * vsq)).sum()
        mom['M04'] = 4.0 * (WI * usqmvsq * uv).sum()
    if radial:
        rsq2 = rsq * rsq
        M22 = (WI * rsq2).sum()
        M33 = (WI * rsq2 * rsq).sum()
        M44 = (WI * rsq2 * rsq2).sum()
        mom['M22r'], mom['M33r'], mom['M44r'] = M22, M33, M44
        mom['M22n'] = M22 / M11 ** 2
        mom['M33n'] = M33 / M11 ** 3
        mom['M44n'] = M44 / M11 ** 4

    errs = None
    if noise_var is not None:
        nv = np.asarray(noise_var, dtype=float)
        if nv.ndim == 0:
            nv = np.full(data.shape, float(nv))
        else:
            nv = nv.flatten()
        WV = kernel ** 2 * nv          # W^2 * var(I)
        varM00 = WV.sum()
        WV = WV / M00 ** 2
        errs = {'Flux': varM00,
                'du': (WV * usq).sum() * 4.0,
                'dv': (WV * vsq).sum() * 4.0,
                'e0': (WV * (rsq - M11) ** 2).sum() * 2.26 ** 2,
                'e1': (WV * (usqmvsq - M20) ** 2).sum() * 2.13 ** 2,
                'e2': (WV * (2 * uv - M02) ** 2).sum() * 2.13 ** 2}
        if third_order:
            errs['M21'] = (WV * (u * rsq - mom['M21']) ** 2).sum() * 0.66 ** 2
            errs['M12'] = (WV * (v * rsq - mom['M12']) ** 2).sum() * 0.66 ** 2
            errs['M30'] = (WV * (u * (usq - 3 * vsq) - mom['M30']) ** 2).sum()
            errs['M03'] = (WV * (v * (3 * usq - vsq) - mom['M03']) ** 2).sum()
        if fourth_order:
            errs['M22'] = (WV * (rsq * rsq - mom['M22']) ** 2).sum() * 2.62 ** 2
            errs['M31'] = (WV * (rsq * usqmvsq - mom['M31']) ** 2).sum() * 2.38 ** 2
            errs['M13'] = (WV * (2 * rsq * uv - mom['M13']) ** 2).sum() * 2.38 ** 2
            errs['M40'] = (WV * (usq * usq - 6 * usq * vsq + vsq * vsq - mom['M40']) ** 2).sum() * 1.05 ** 2
            errs['M04'] = (WV * (4 * usqmvsq * uv - mom['M04']) ** 2).sum() * 1.05 ** 2

    return mom, errs
