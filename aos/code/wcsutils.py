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


def calc_rotator_from_visitinfo(par_angle_deg, rotpa_deg):
    """Calculate the physical rotator angle from visitInfo angles.

    The physical rotator angle is derived from the parallactic angle
    and the boresight rotation angle (ROTPA) from the Butler visitInfo::

        rotator_angle = par_angle - rotpa - 90

    Values near +/-360 are wrapped to [-180, 180].  The physical rotator
    has an allowed range of approximately [-90, 90] degrees.

    Parameters
    ----------
    par_angle_deg : float or array-like
        Boresight parallactic angle in degrees
        (from ``visitInfo.getBoresightParAngle()``).
    rotpa_deg : float or array-like
        Boresight rotation angle in degrees
        (from ``visitInfo.getBoresightRotAngle()``).

    Returns
    -------
    rotator_angle : float or `numpy.ndarray`
        Physical rotator angle in degrees, wrapped to [-180, 180].
    """
    par = np.atleast_1d(np.asarray(par_angle_deg, dtype=float))
    rotpa = np.atleast_1d(np.asarray(rotpa_deg, dtype=float))

    rotator_angle = par - rotpa - 90.0

    # Wrap to [-180, 180] to handle values near +/-360
    rotator_angle = (rotator_angle + 180.0) % 360.0 - 180.0

    if rotator_angle.size == 1:
        return float(rotator_angle[0])
    return np.asarray(rotator_angle)
