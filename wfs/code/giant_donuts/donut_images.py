"""Read giant-donut stamps off a raw LSSTCam exposure.

Shared by the radial-profile notebook and the danish fitting driver, so both work
from pixels reduced the same way.

The ISR is deliberately minimal -- overscan, assembly and nominal gains only. The
giant-donut exposures predate any matched calibrations in the raw collection, and
neither the radial profile nor a donut fit is sensitive to flat-field structure at
the percent level this study works at.
"""
import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import label, find_objects

import lsst.geom
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.ip.isr import IsrTask, IsrTaskConfig

# Pixel pitch of the LSSTCam science sensors, in meters.
PIXEL_SIZE_M = 10.0e-6

__all__ = [
    'PIXEL_SIZE_M', 'minimal_isr_config', 'run_isr', 'field_angle_deg',
    'find_donut', 'cut_stamp',
]


def minimal_isr_config():
    """`lsst.ip.isr.IsrTaskConfig` doing overscan, assembly and gains only.

    Returns
    -------
    config : `lsst.ip.isr.IsrTaskConfig`
        Configuration with every calibration-product step disabled.
    """
    config = IsrTaskConfig()
    for name in ("doBias", "doDark", "doFlat", "doFringe", "doDefect",
                 "doLinearize", "doCrosstalk", "doBrighterFatter", "doSaturation",
                 "doWidenSaturationTrails", "doSaturationInterpolation",
                 "doSetBadRegions", "doInterpolate", "doNanMasking"):
        if hasattr(config, name):
            setattr(config, name, False)
    config.doOverscan = True
    config.doAssembleCcd = True
    config.doApplyGains = True
    return config


def run_isr(butler, exposure, detector_name, camera, collection, instrument="LSSTCam"):
    """Overscan and gain ISR on one raw detector.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to read the raw from.
    exposure : `int`
        Exposure id.
    detector_name : `str`
        Detector name, e.g. ``'R22_S10'``.
    camera : `lsst.afw.cameraGeom.Camera`
        Camera geometry.
    collection : `str`
        Butler collection holding the raws.
    instrument : `str`, optional
        Instrument name for the data id.

    Returns
    -------
    image : `numpy.ndarray`
        Assembled image, in electrons.
    """
    raw = butler.get("raw", collections=collection,
                     dataId={"instrument": instrument, "exposure": exposure,
                             "detector": camera[detector_name].getId()})
    task = IsrTask(config=minimal_isr_config())
    return np.asarray(task.run(raw, camera=camera).exposure.image.array, dtype=float)


def field_angle_deg(detector_name, x_pix, y_pix, camera):
    """Field angle of a pixel, in degrees.

    Parameters
    ----------
    detector_name : `str`
        Detector name.
    x_pix, y_pix : `float`
        Pixel coordinates on the assembled detector.
    camera : `lsst.afw.cameraGeom.Camera`
        Camera geometry.

    Returns
    -------
    angle_x_deg, angle_y_deg : `float`
        Field angle components, in degrees.
    """
    transform = camera[detector_name].getTransform(PIXELS, FIELD_ANGLE)
    point = transform.applyForward(lsst.geom.Point2D(x_pix, y_pix))
    return float(np.rad2deg(point.getX())), float(np.rad2deg(point.getY()))


def find_donut(image, n_sigma=5.0):
    """Centre and size of the largest connected bright region.

    A smoothed-peak finder was tried first and rejected: at a 200 pixel smoothing
    scale it landed about 80 pixels off the true centre, which truncated the stamp
    and put the fitted outer edge at 157 pixels instead of the true 343. The giant
    donut is the largest contiguous bright object on the sensor by a wide margin,
    so connected-region labelling finds it directly and returns its size as a
    by-product.

    Parameters
    ----------
    image : `numpy.ndarray`
        Detector image, in electrons.
    n_sigma : `float`, optional
        Detection threshold above sky, in units of the robust scatter.

    Returns
    -------
    x_pix, y_pix : `float`
        Region centre, in pixels.
    diameter_pix : `int`
        Larger bounding-box side, in pixels.
    area_pix : `int`
        Region area, in pixels.
    """
    sky = np.median(image)
    nmad = 1.48 * np.median(np.abs(image - sky))
    labels, _ = label(image > sky + n_sigma * nmad)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    index = int(np.argmax(sizes))
    box = find_objects(labels)[index - 1]
    return (float((box[1].start + box[1].stop) / 2),
            float((box[0].start + box[0].stop) / 2),
            int(max(box[0].stop - box[0].start, box[1].stop - box[1].start)),
            int(sizes[index]))


def cut_stamp(image, x_pix, y_pix, half=430):
    """Cut a square stamp and subtract a sigma-clipped sky level.

    Parameters
    ----------
    image : `numpy.ndarray`
        Full detector image, in electrons.
    x_pix, y_pix : `float`
        Donut centre, in pixels.
    half : `int`, optional
        Half-size of the stamp, in pixels. Must comfortably exceed the donut
        radius, about 347 pixels for these 8 mm donuts, or the profile is cut off
        before the outer edge.

    Returns
    -------
    stamp : `numpy.ndarray`
        Background-subtracted stamp, in electrons, or `None` if the requested
        stamp falls outside the detector.
    sky : `float`
        Subtracted sky level, in electrons per pixel.
    """
    x0, x1 = int(round(x_pix)) - half, int(round(x_pix)) + half + 1
    y0, y1 = int(round(y_pix)) - half, int(round(y_pix)) + half + 1
    ny, nx = image.shape
    if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
        return None, np.nan

    stamp = image[y0:y1, x0:x1].copy()
    # Sky from the stamp corners, which the donut does not reach.
    corner = min(half // 3, 60)
    corners = np.concatenate([
        stamp[:corner, :corner].ravel(), stamp[:corner, -corner:].ravel(),
        stamp[-corner:, :corner].ravel(), stamp[-corner:, -corner:].ravel(),
    ])
    _, sky, _ = sigma_clipped_stats(corners, sigma=3.0, maxiters=5)
    return stamp - sky, float(sky)
