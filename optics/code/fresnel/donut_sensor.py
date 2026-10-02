"""Silicon-sensor effects on defocused donuts: batoid photons into GalSim's CCD model.

Photons are drawn uniformly over the entrance pupil, given a wavelength drawn from
a bandpass and an atmospheric deflection drawn from a seeing PSF, traced through
the batoid optic, and delivered to ``galsim.SiliconSensor`` with their incidence
angles and wavelengths. GalSim converts each photon at a wavelength-dependent
depth (exponential with the silicon absorption length), moves it laterally along
its incidence direction to that depth, and drifts the electron to the pixel wells
with diffusion (and, optionally, the brighter-fatter effect).

GalSim applies the photon's ``dxdz``/``dydz`` directly inside the silicon; it does
not refract at the surface. `refract_into_silicon` applies Snell's law first, so
the in-silicon angles are used.

Units: lengths in metres unless stated, wavelengths in nm where noted, angles as
tangents (``dxdz``) or radians.
"""

import numpy as np

import fresnel_donut as fd     # imports finufft before batoid (OpenMP order)
import donut_seeing as ds
import batoid
import galsim

# Real refractive index of crystalline silicon near room temperature
# (Green & Keevers 1995, rounded); used only for the refraction angle, where a
# 1% error in n changes the in-silicon angle by 1%.
_SI_WAVE_NM = np.array([600., 650., 700., 750., 800., 850., 900., 950., 1000.])
_SI_INDEX = np.array([3.94, 3.85, 3.78, 3.73, 3.69, 3.65, 3.62, 3.60, 3.58])


def silicon_index(wavelength_nm):
    """Refractive index of silicon (dimensionless) at `wavelength_nm` (nm)."""
    return np.interp(wavelength_nm, _SI_WAVE_NM, _SI_INDEX)


def refract_into_silicon(dxdz, dydz, wavelength_nm):
    """Convert incidence tangents in air/vacuum to tangents inside silicon.

    Snell's law at a flat surface normal to z: the transverse component of the
    unit direction is divided by n_Si; the azimuth is unchanged.
    """
    norm = np.sqrt(1.0 + dxdz**2 + dydz**2)
    a, b = dxdz / norm, dydz / norm               # transverse direction cosines
    n = silicon_index(wavelength_nm)
    a_si, b_si = a / n, b / n
    c_si = np.sqrt(1.0 - a_si**2 - b_si**2)
    return a_si / c_si, b_si / c_si


def trace_photons(optic, nphot, bandpass, psf=None, seed=0, nwave_bins=16,
                  batch=4_000_000):
    """Trace photons with bandpass wavelengths and atmospheric deflections.

    Parameters
    ----------
    optic : `batoid.Optic`
    nphot : `int`
        Photons launched (vignetted ones are dropped).
    bandpass : `galsim.Bandpass`
        Throughput; wavelengths are drawn for a flat photon spectrum.
    psf : `galsim.GSObject` or `None`
        Atmospheric PSF in arcsec; each photon gets a field angle drawn from it.
    nwave_bins : `int`
        Photons are traced at the centre of their wavelength bin (bins span the
        bandpass), which bounds the chromatic error to half a bin width.

    Returns
    -------
    photons : `dict`
        ``x``, ``y`` (m, detector local frame), ``dxdz``, ``dydz`` (incidence
        tangents in air), ``wavelength`` (nm, the photon's sampled wavelength).
    """
    rng = np.random.default_rng(seed)
    gs_rng = galsim.BaseDeviate(seed + 1)
    sed = galsim.SED('1', 'nm', 'fphotons')
    edges = np.linspace(bandpass.blue_limit, bandpass.red_limit, nwave_bins + 1)
    half = optic.pupilSize / 2
    out = {k: [] for k in ('x', 'y', 'dxdz', 'dydz', 'wavelength')}
    remaining = nphot
    while remaining > 0:
        n = min(batch, remaining)
        wave = sed.sampleWavelength(n, bandpass, rng=gs_rng)
        u = rng.uniform(-half, half, n)
        v = rng.uniform(-half, half, n)
        if psf is not None:
            ph = psf.shoot(n, gs_rng)
            thx, thy = ph.x * ds.ARCSEC, ph.y * ds.ARCSEC
        else:
            thx, thy = np.zeros(n), np.zeros(n)
        ibin = np.clip(np.digitize(wave, edges) - 1, 0, nwave_bins - 1)
        for b in range(nwave_bins):
            m = ibin == b
            if not np.any(m):
                continue
            wl = 0.5 * (edges[b] + edges[b + 1]) * 1e-9
            rv = batoid.RayVector.fromFieldAngles(
                thx[m], thy[m], optic=optic, wavelength=wl, x=u[m], y=v[m])
            optic.trace(rv)
            ok = ~rv.vignetted & ~rv.failed
            out['x'].append(rv.x[ok])
            out['y'].append(rv.y[ok])
            out['dxdz'].append(rv.vx[ok] / rv.vz[ok])
            out['dydz'].append(rv.vy[ok] / rv.vz[ok])
            out['wavelength'].append(wave[m][ok])
        remaining -= n
    return {k: np.concatenate(v) for k, v in out.items()}


def sensor_image(photons, center, npix, pixel=10e-6, sensor=None,
                 angles='refracted'):
    """Accumulate photons into an `npix` x `npix` stamp, optionally via a sensor.

    Parameters
    ----------
    photons : `dict`
        Output of `trace_photons`.
    center : `array_like`
        Stamp centre on the detector (x, y), m; it falls at the centre of the
        central pixel (`npix` odd).
    sensor : `galsim.Sensor` or `None`
        None bins photon positions directly (ideal pixels, no silicon).
    angles : {'refracted', 'air', 'normal'}
        Incidence directions passed to the sensor: Snell-refracted into silicon,
        the unrefracted air directions, or normal incidence.

    Returns
    -------
    image : `numpy.ndarray`
        Stamp [y, x], unit total flux.
    """
    if sensor is None:
        return fd.bin_to_pixels(photons['x'], photons['y'], None, center, pixel,
                                npix)
    n = len(photons['x'])
    xpix = (npix + 1) / 2 + (photons['x'] - center[0]) / pixel
    ypix = (npix + 1) / 2 + (photons['y'] - center[1]) / pixel
    if angles == 'refracted':
        dxdz, dydz = refract_into_silicon(photons['dxdz'], photons['dydz'],
                                          photons['wavelength'])
    elif angles == 'air':
        dxdz, dydz = photons['dxdz'], photons['dydz']
    elif angles == 'normal':
        dxdz, dydz = np.zeros(n), np.zeros(n)
    else:
        raise ValueError(f"unknown angles mode {angles!r}")
    pa = galsim.PhotonArray(n, x=xpix, y=ypix, flux=np.ones(n),
                            dxdz=dxdz, dydz=dydz,
                            wavelength=photons['wavelength'])
    img = galsim.ImageF(npix, npix, scale=1.0)
    sensor.accumulate(pa, img)
    arr = img.array.astype(float)
    return arr / arr.sum()


def mean_radial_shift(photons, center, sensor_depth_um, angles='refracted'):
    """Mean radial displacement (m) of photons converted at a fixed depth.

    Positive is outward from `center`. Used to check the sign convention:
    converging (intra-focal) photons should move inward, diverging
    (extra-focal) photons outward.
    """
    if angles == 'refracted':
        dxdz, dydz = refract_into_silicon(photons['dxdz'], photons['dydz'],
                                          photons['wavelength'])
    else:
        dxdz, dydz = photons['dxdz'], photons['dydz']
    rx, ry = photons['x'] - center[0], photons['y'] - center[1]
    r = np.hypot(rx, ry)
    d = sensor_depth_um * 1e-6
    return np.mean(((rx + dxdz * d) * rx + (ry + dydz * d) * ry) / r - r)
