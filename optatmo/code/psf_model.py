"""
Galsim Fraunhofer optical PSF convolved with a VonKarman atmospheric kernel --
the "truth" forward model, matching the old PIFF optatmo2 optical_model.py.

    PSF = OpticalPSF(annular pupil, Zernike aberrations) (x) VonKarman(r0, L0)

Aberrations are Noll-indexed Zernikes in microns (1-indexed, index 0 ignored),
consistent with moments_analytic.py and the MIW maps.
"""

import numpy as np
import galsim
from scipy.optimize import brentq

# Rubin defaults (match seeing_moments_study.py / optatmo2)
DIAM = 8.36
OBSCURATION = 0.61
LAM = 750.0          # nm
PIXEL_SCALE = 0.2    # arcsec/pix
STAMP = 128

_GSP = galsim.GSParams(minimum_fft_size=64, folding_threshold=0.005)


def vonkarman_r0_for_fwhm(fwhm, lam=LAM, L0=25.0):
    """Solve for r0 (at lam) giving a VonKarman kernel of the target FWHM."""
    def f(r0):
        return galsim.VonKarman(lam=lam, r0=r0, L0=L0).calculateFWHM() - fwhm
    # r0 ~ 0.98 lam/fwhm(rad); bracket generously
    return brentq(f, 0.02, 2.0, xtol=1e-4)


def build_optical(coef_um, lam=LAM, diam=DIAM, obscuration=OBSCURATION,
                  aper=None):
    """Fraunhofer optical PSF from Zernike aberrations (microns)."""
    aber_waves = np.asarray(coef_um, dtype=float) / (lam * 1e-3)
    if aper is None:
        aper = galsim.Aperture(diam=diam, obscuration=obscuration,
                               lam=lam, gsparams=_GSP)
    return galsim.OpticalPSF(lam=lam, diam=diam, aper=aper,
                             aberrations=aber_waves.tolist(), gsparams=_GSP)


def build_atmosphere(fwhm, lam=LAM, L0=25.0, kind='VonKarman'):
    """Atmospheric kernel of a given FWHM (arcsec)."""
    if kind == 'VonKarman':
        r0 = vonkarman_r0_for_fwhm(fwhm, lam=lam, L0=L0)
        return galsim.VonKarman(lam=lam, r0=r0, L0=L0, gsparams=_GSP)
    elif kind == 'Kolmogorov':
        return galsim.Kolmogorov(fwhm=fwhm, gsparams=_GSP)
    raise ValueError(kind)


def draw_psf(coef_um, fwhm, lam=LAM, diam=DIAM, obscuration=OBSCURATION,
             L0=25.0, kind='VonKarman', g1=0.0, g2=0.0,
             scale=PIXEL_SCALE, stamp=STAMP, aper=None, flux=1.0):
    """Draw the optics (x) atmosphere PSF into a galsim image.

    :returns: galsim.Image (scale set), normalised to `flux`.
    """
    optical = build_optical(coef_um, lam=lam, diam=diam,
                            obscuration=obscuration, aper=aper)
    atm = build_atmosphere(fwhm, lam=lam, L0=L0, kind=kind)
    prof = galsim.Convolve([optical, atm])
    if g1 != 0.0 or g2 != 0.0:
        prof = prof.shear(g1=g1, g2=g2)
    prof = prof.withFlux(flux)
    img = prof.drawImage(nx=stamp, ny=stamp, scale=scale, method='auto')
    return img


def make_aperture(lam=LAM, diam=DIAM, obscuration=OBSCURATION):
    """Cache a single Aperture to reuse across draws (speed)."""
    return galsim.Aperture(diam=diam, obscuration=obscuration, lam=lam,
                           gsparams=_GSP)
