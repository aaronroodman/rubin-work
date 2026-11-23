"""Turn a noise-free donut into an observed one: seeing kernel, star flux, sky, noise.

Magnitudes are AB with a zero point for 1 e-/s (SMTN-002 ``zp_t``); sky in mag/arcsec².
Images are electrons per pixel. Import as ``from rubinwork.donut_sim import observe``.
"""
import numpy as np

__all__ = ["von_karman_r0", "seeing", "flux_e", "sky_e_per_px", "draw_mags", "add_noise"]


def von_karman_r0(fwhm_arcsec, L0_m, wavelength_nm):
    """Fried parameter (m, at `wavelength_nm`) giving a von Kármán FWHM of
    `fwhm_arcsec` with outer scale `L0_m`."""
    import galsim
    from scipy.optimize import brentq

    def fwhm(r0):
        img = galsim.VonKarman(lam=wavelength_nm, r0=r0, L0=L0_m).drawImage(
            nx=401, ny=401, scale=fwhm_arcsec / 50, method="no_pixel")
        return img.calculateFWHM()

    return brentq(lambda r0: fwhm(r0) - fwhm_arcsec, 0.02, 1.0, xtol=1e-6)


def seeing(wavelength_nm, r0_m, L0_m, e, beta_deg, shift_arcsec=(0.0, 0.0)):
    """von Kármán profile sheared to distortion `e` (second-moment ellipticity
    ``(a² - b²) / (a² + b²)``, dimensionless) with major axis at `beta_deg` from +x,
    shifted by `shift_arcsec` (x, y)."""
    import galsim
    p = galsim.VonKarman(lam=wavelength_nm, r0=r0_m, L0=L0_m)
    if e:
        p = p.shear(e=e, beta=beta_deg * galsim.degrees)
    return p.shift(*shift_arcsec)


def flux_e(mag, zeropoint_mag, exptime_s):
    """Electrons from a star of AB magnitude `mag`, zero point for 1 e-/s."""
    return exptime_s * 10 ** (-0.4 * (np.asarray(mag) - zeropoint_mag))


def sky_e_per_px(sky_mag_arcsec2, zeropoint_mag, exptime_s, pixel_arcsec):
    """Sky electrons per pixel from a surface brightness in mag/arcsec²."""
    return flux_e(sky_mag_arcsec2, zeropoint_mag, exptime_s) * pixel_arcsec ** 2


def draw_mags(rng, n, mag_min, mag_max, slope_dex):
    """`n` magnitudes from star counts ``dN/dm ∝ 10**(slope_dex * m)`` on
    [`mag_min`, `mag_max`] (inverse CDF)."""
    a, b = 10 ** (slope_dex * mag_min), 10 ** (slope_dex * mag_max)
    return np.log10(a + rng.random(n) * (b - a)) / slope_dex


def add_noise(rng, expected_e, sky_e, read_noise_e):
    """Poisson draw of `expected_e` plus sky, Gaussian read noise, sky subtracted
    (electrons per pixel)."""
    lam = expected_e + sky_e
    return (rng.poisson(lam).astype(float) + rng.normal(0.0, read_noise_e, lam.shape)
            - sky_e)
