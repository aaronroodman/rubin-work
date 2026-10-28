"""Frozen-flow atmosphere and star rendering for `atmo_sim`.

Field angles are tangent-plane offsets from the boresight, in arcsec, in the frame
``(East, North)``, as `lsst.geom.SpherePoint.getTangentPlaneOffset` returns them. Wind
velocities from psf-weather-station (psfws) are put in the same frame, so the phase
screens move in the sky direction the wind blows toward.
"""

import gc

import galsim
import numpy as np
from scipy.optimize import bisect

__all__ = [
    "D_APER",
    "OBSC",
    "WLEN_EFF",
    "Atmo",
    "psfws_layers",
    "screen_size",
    "make_atmo",
    "calibrate_atm_fwhm",
    "draw_star",
    "hsm_fwhm",
]

D_APER = 8.36
"""Rubin primary diameter (m)."""

OBSC = 0.61
"""Central obscuration, inner over outer diameter (dimensionless)."""

WLEN_EFF = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)
"""Band effective wavelength (nm)."""

GROUND_LAYER_MIN_KM = 0.2
"""Lowest layer distance (km): the ground layer is raised to it, as in imSim."""


def _vk_fwhm(r0_500, wavelength, L0):
    """von Karman FWHM (arcsec), Tokovinin 2002 eq. 19."""
    kolm = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
    r0 = r0_500 * (wavelength / 500.0)**(6.0 / 5.0)
    arg = 1.0 - 2.183 * (r0 / L0)**0.356
    return kolm * (np.sqrt(arg) if arg > 0.0 else 0.0)


def r0_500_for_fwhm(wavelength, L0, fwhm):
    """r0 at 500 nm (m) giving a von Karman FWHM (arcsec) at `wavelength` (nm)."""
    hi = min(1.0, L0 * (1.0 / 2.183)**(-0.356) * (wavelength / 500.0)**(6.0 / 5.0))
    return bisect(lambda r: _vk_fwhm(r, wavelength, L0) - fwhm, 0.01, hi)


def draw_L0(rng, median=25.0, lo=10.0, hi=100.0):
    """Outer scale (m), log-normal about `median`, limited to [`lo`, `hi`]."""
    gd = galsim.GaussianDeviate(rng)
    L0 = 0.0
    while L0 < lo or L0 > hi:
        L0 = np.exp(gd() * 0.6 + np.log(median))
    return L0


def psfws_layers(forecast_file, data_dir, when, alt_deg, az_deg, nlayers, seed):
    """Layer altitudes, weights and winds from psfws for one ERA5 time step.

    Parameters
    ----------
    forecast_file : `str`
        ECMWF ERA5 pickle name, in `data_dir`.
    data_dir : `str`
        Directory holding the forecast and psfws's ``p_and_h.p``, ``H.npy``.
    when : `pandas.Timestamp`
        Visit time (UTC); the nearest ERA5 step is used.
    alt_deg, az_deg : `float`
        Telescope altitude and azimuth (deg).
    nlayers : `int`
        Number of layers.
    seed : `int`
        psfws random seed.

    Returns
    -------
    layers : `dict`
        ``altitude`` (km, line-of-sight distance from the telescope), ``weights``
        (dimensionless, fraction of total Cn**2), ``speed`` (m/s), ``direction``
        (`galsim.Angle`, East of North, toward which the wind blows), ``era5_time``.
    """
    import pathlib

    import psfws

    pg = psfws.ParameterGenerator(forecast_file=forecast_file, telemetry_file=None,
                                  data_dir=pathlib.Path(data_dir), seed=seed)
    # ERA5 and ERA5T overlap in recent files and duplicate time steps.
    if pg.data_fa.index.has_duplicates:
        pg.data_fa = pg.data_fa[~pg.data_fa.index.duplicated(keep="first")]
        pg.data_gl = pg.data_gl[~pg.data_gl.index.duplicated(keep="first")]
        pg.N = len(pg.data_fa)
    idx = pg.data_fa.index
    pt = idx[int(np.argmin(np.abs(idx - when)))]
    p = pg.get_parameters(pt, nl=nlayers, skycoord=True, alt=alt_deg, az=az_deg,
                          location="com")
    # skycoord=True gives h * sec(z) of the height above sea level; take the
    # line-of-sight distance above the site instead.
    sec_z = 1.0 / np.sin(np.radians(alt_deg))
    dist = [max((h / sec_z - pg.h0) * sec_z, GROUND_LAYER_MIN_KM) for h in p["h"]]
    u = np.asarray(p["u"], float)     # toward East (m/s)
    v = np.asarray(p["v"], float)     # toward North (m/s)
    weights = np.asarray(p["j"], float)
    return dict(altitude=dist, weights=list(weights / weights.sum()),
                speed=list(np.hypot(u, v)),
                direction=[float(a) * galsim.radians for a in np.arctan2(u, v)],
                era5_time=str(pt))


def screen_size(layers, theta_max_deg, duration_s, scale_m, cap_m):
    """Phase-screen size (m) so no pupil footprint wraps during the exposure.

    Returns
    -------
    size_m : `float`
        Power-of-two screen size, capped at `cap_m`.
    need_m : `float`
        Size the field span, the wind drift and the pupil need.
    """
    h_max = max(layers["altitude"]) * 1e3
    need = 2.0 * np.radians(theta_max_deg) * h_max + max(layers["speed"]) * duration_s + D_APER
    npix = 1 << int(np.ceil(np.log2(min(need, cap_m) / scale_m)))
    return npix * scale_m, need


class Atmo:
    """A built atmosphere: the screens, the aperture and the second kick."""

    def __init__(self, atm, aper, second_kick, wlen, r0_500, L0, size_m):
        self.atm, self.aper, self.second_kick = atm, aper, second_kick
        self.wlen, self.r0_500, self.L0, self.size_m = wlen, r0_500, L0, size_m


def make_atmo(layers, atm_fwhm, wlen, L0, size_m, scale_m, kcrit, seed):
    """Build the screens for a target atmospheric von Karman FWHM.

    Parameters
    ----------
    layers : `dict`
        From `psfws_layers`.
    atm_fwhm : `float`
        Atmospheric von Karman FWHM (arcsec) at `wlen`.
    wlen : `float`
        Wavelength (nm).
    L0 : `float`
        Outer scale (m).
    size_m, scale_m : `float`
        Screen size and sampling (m).
    kcrit : `float`
        Split between ray-traced and second-kick scales, in units of 1/r0.
    seed : `int`
        Screen random seed.

    Returns
    -------
    atmo : `Atmo`
    """
    r0_500 = r0_500_for_fwhm(wlen, L0, atm_fwhm)
    atm = galsim.Atmosphere(
        r0_500=r0_500, L0=[L0] * len(layers["speed"]), speed=layers["speed"],
        direction=layers["direction"], altitude=layers["altitude"],
        r0_weights=layers["weights"], screen_size=size_m, screen_scale=scale_m,
        rng=galsim.BaseDeviate(seed))
    aper = galsim.Aperture(diam=D_APER, obscuration=OBSC, lam=wlen, screen_list=atm)
    r0 = atm.r0_500_effective * (wlen / 500.0)**(6.0 / 5.0)
    atm.instantiate(kmax=kcrit / r0, check="phot")
    second_kick = galsim.SecondKick(wlen, r0, aper.diam, aper.obscuration, kcrit=kcrit)
    return Atmo(atm, aper, second_kick, wlen, r0_500, L0, size_m)


def draw_star(atmo, theta_asec, t0, exptime, image, center, flux_adu, gain,
              read_noise_adu, rng):
    """Photon-shoot one star into `image`, with shot and read noise, in ADU.

    Parameters
    ----------
    atmo : `Atmo`
    theta_asec : `tuple` [`float`, `float`]
        Field angle (arcsec, East and North of the boresight).
    t0, exptime : `float`
        Integration start, relative to shutter open, and length (s).
    image : `galsim.Image`
        Target image, with a WCS from pixels to field angle; drawn into in place.
    center : `galsim.PositionD`
        Star position in the image's pixel coordinates.
    flux_adu : `float`
        Star flux (ADU).
    gain : `float`
        Gain (e-/ADU).
    read_noise_adu : `float`
        Read noise (ADU rms).
    rng : `galsim.BaseDeviate`

    Returns
    -------
    image : `galsim.Image`
        The same image, in ADU, without a pedestal.
    """
    theta = (theta_asec[0] * galsim.arcsec, theta_asec[1] * galsim.arcsec)
    psf = atmo.atm.makePSF(atmo.wlen, aper=atmo.aper, theta=theta, t0=t0,
                           exptime=exptime, second_kick=False)
    prof = galsim.Convolve(psf, atmo.second_kick).withFlux(flux_adu * gain)
    image.setZero()
    prof.drawImage(image=image, center=center, method="phot", rng=rng,
                   poisson_flux=True, add_to_image=True)
    if read_noise_adu > 0:
        image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise_adu * gain))
    image /= gain
    return image


def hsm_fwhm(image):
    """FWHM (arcsec) from HSM adaptive moments, `nan` if HSM fails."""
    try:
        m = galsim.hsm.FindAdaptiveMom(image)
    except galsim.GalSimError:
        return np.nan
    scale = np.sqrt(abs(image.wcs.pixelArea(image_pos=m.moments_centroid)))
    return 2.3548 * m.moments_sigma * scale


def calibrate_atm_fwhm(layers, target_fwhm, wlen, L0, scale_m, kcrit, seed, wcs,
                       n_stamps, exptime, cadence, flux_adu, gain, max_iter=4, tol=0.02):
    """Atmospheric FWHM (arcsec) that makes the on-axis short-stamp FWHM `target_fwhm`.

    The FWHM depends only on the turbulence across the pupil, so a 204.8 m screen with
    the same seed and winds stands in for the full one.

    Returns
    -------
    atm_fwhm : `float`
        Calibrated atmospheric von Karman FWHM (arcsec).
    record : `list` [`tuple`]
        ``(atm_fwhm, measured_fwhm)`` per pass (arcsec).
    """
    atm_fwhm, record = target_fwhm, []
    for _ in range(max_iter):
        atmo = make_atmo(layers, atm_fwhm, wlen, L0, 204.8, scale_m, kcrit, seed)
        rng = galsim.BaseDeviate(seed + 7)
        vals = []
        for k in range(n_stamps):
            img = galsim.ImageF(64, 64, wcs=wcs)
            draw_star(atmo, (0.0, 0.0), k * cadence, exptime, img, img.true_center,
                      flux_adu, gain, 0.0, rng)
            vals.append(hsm_fwhm(img))
        measured = float(np.nanmedian(vals))
        record.append((float(atm_fwhm), measured))
        del atmo
        gc.collect()
        if not np.isfinite(measured) or abs(measured - target_fwhm) / target_fwhm < tol:
            break
        atm_fwhm *= target_fwhm / measured
    return atm_fwhm, record
