"""
Utility functions for WCS, coordinate transforms, and rotator angle calculations.

Provides functions to:
- Convert between pixel and focal plane coordinates
- Calculate Local Sidereal Time (LST) from MJD
- Calculate parallactic angle from LST, RA, DEC
- Calculate physical rotator angle from MJD, RA, DEC, and sky angle

The Rubin Observatory location is used for all sky-related calculations.
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from lsst.afw import cameraGeom

# Rubin Observatory location (Cerro Pachon)
RUBIN_LOCATION = EarthLocation.of_site('Rubin:Simonyi')


def pixel_to_focal(x, y, det):
    """Convert pixel coordinates to focal plane coordinates.

    Parameters
    ----------
    x, y : array-like
        Pixel coordinates.
    det : `lsst.afw.cameraGeom.Detector`
        Detector of interest.

    Returns
    -------
    fpx, fpy : `numpy.ndarray`
        Focal plane position in millimeters in DVCS (see LSE-349).
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def focal_to_pixel(fpx, fpy, det):
    """Convert focal plane coordinates to pixel coordinates.

    Parameters
    ----------
    fpx, fpy : array-like
        Focal plane position in millimeters in DVCS (see LSE-349).
    det : `lsst.afw.cameraGeom.Detector`
        Detector of interest.

    Returns
    -------
    x, y : `numpy.ndarray`
        Pixel coordinates.
    """
    tx = det.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
    x, y = tx.getMapping().applyForward(np.vstack((fpx, fpy)))
    return x.ravel(), y.ravel()


def calculate_lst(mjd):
    """Calculate Local Sidereal Time for the Rubin Observatory.

    Parameters
    ----------
    mjd : float or array-like
        Modified Julian Date(s).

    Returns
    -------
    lst : `astropy.coordinates.Longitude`
        Local apparent sidereal time.
    """
    time = Time(mjd, format='mjd', scale='utc')
    lst = time.sidereal_time('apparent', longitude=RUBIN_LOCATION.lon)
    return lst


def calculate_parallactic_angle(lst, ra, dec):
    """Calculate the parallactic angle.

    Uses Eqn (14.1) of Meeus' Astronomical Algorithms.

    Parameters
    ----------
    lst : `astropy.coordinates.Longitude`
        Local apparent sidereal time.
    ra : `astropy.units.Quantity`
        Right ascension with angle units.
    dec : `astropy.units.Quantity`
        Declination with angle units.

    Returns
    -------
    q : `astropy.units.Quantity`
        Parallactic angle in degrees.
    """
    H = lst.radian - ra.to(u.rad).value
    lat = RUBIN_LOCATION.lat.radian
    dec_rad = dec.to(u.rad).value

    q = np.arctan2(
        np.sin(H),
        np.tan(lat) * np.cos(dec_rad) - np.sin(dec_rad) * np.cos(H),
    ) * u.rad

    return q.to(u.deg)


def calc_rotator_angle(mjd, ra_deg, dec_deg, sky_angle_deg):
    """Calculate the physical rotator angle from sky coordinates and time.

    The physical rotator angle is derived from the sky angle and the
    parallactic angle, with a sign flip and 90-degree offset::

        physical_rotator_angle = -(sky_angle - parallactic_angle) - 90

    Parameters
    ----------
    mjd : float or array-like
        Modified Julian Date(s) of the observation midpoint.
    ra_deg : float or array-like
        Right ascension in degrees.
    dec_deg : float or array-like
        Declination in degrees.
    sky_angle_deg : float or array-like
        Sky rotation angle in degrees (``sky_rotation`` from ConsDB).

    Returns
    -------
    rotator_angle : float or `numpy.ndarray`
        Physical rotator angle in degrees, wrapped to [-180, 180].
    """
    lst = calculate_lst(mjd)
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=float)) * u.deg
    dec = np.atleast_1d(np.asarray(dec_deg, dtype=float)) * u.deg

    q = calculate_parallactic_angle(lst, ra, dec)

    # Sign flip and 90-degree offset to match physical_rotator_angle convention
    rotator_angle = -(np.atleast_1d(np.asarray(sky_angle_deg, dtype=float)) - q.value) - 90.0

    # Wrap to [-180, 180]
    rotator_angle = (rotator_angle + 180.0) % 360.0 - 180.0

    if rotator_angle.size == 1:
        return float(rotator_angle[0])
    return np.asarray(rotator_angle)
