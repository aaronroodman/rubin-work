"""Guider region-of-interest (ROI) frames, including the inverse of the ``summit_utils`` chain.

A guider ``guider_raw`` stamp is read out in its **amplifier readout frame**. ``summit_utils``
converts it to the CCD pixel frame (flip by the amplifier's raw flips) and then to the Data
Visualization Coordinate System (DVCS) frame (rotate by the detector's ``nQuarter``)::

    dvcs = rot90(ampToCcdView(raw, detector, ampName), -nQuarter)

It has no inverse. This module supplies it, so a simulated stamp drawn in the CCD pixel or
DVCS frame can be written as a ``guider_raw`` stamp that the data-side guider code reads
unchanged. The flips are their own inverse, so::

    raw = ampToCcdView(rot90(dvcs, +nQuarter), detector, ampName)

The ROI's position on the CCD comes from the stamp header (``ROISEG``, ``ROICOL``,
``ROIROW``, ``ROICOLS``, ``ROIROWS``) through ``summit_utils`` `makeRoiBbox`, and the field
angle of a CCD pixel from cameraGeom ``PIXELS`` to ``FIELD_ANGLE``.
"""

import numpy as np

__all__ = [
    "amp_name",
    "ccd_to_amp",
    "dvcs_to_amp",
    "amp_to_dvcs",
    "roi_ccd_bbox",
    "roi_ccd_pixel_grid",
    "ccd_to_field_angle",
    "field_angle_to_ccd",
    "local_jacobian",
]


def amp_name(metadata):
    """Amplifier name of a guider ROI, from its ``ROISEG`` header.

    Parameters
    ----------
    metadata : `dict` or `lsst.daf.base.PropertyList`
        ``guider_raw`` exposure-level metadata.

    Returns
    -------
    name : `str`
        Amplifier name, for example ``'C14'`` for ``ROISEG = 'Segment14'``.
    """
    return "C" + str(metadata["ROISEG"])[7:]


def ccd_to_amp(ccd_array, detector, amp):
    """Convert a ROI array from the CCD pixel frame to the amplifier readout frame.

    Parameters
    ----------
    ccd_array : `numpy.ndarray`
        ROI image in the CCD pixel frame, shape ``(ROIROWS, ROICOLS)``.
    detector : `lsst.afw.cameraGeom.Detector`
        The guide sensor.
    amp : `str`
        Amplifier name, for example ``'C14'``.

    Returns
    -------
    raw : `numpy.ndarray`
        ROI image in the amplifier readout frame, as ``guider_raw`` stores it.
    """
    amplifier = detector[amp]
    raw = ccd_array
    if amplifier.getRawFlipX():
        raw = np.fliplr(raw)
    if amplifier.getRawFlipY():
        raw = np.flipud(raw)
    return np.ascontiguousarray(raw)


def dvcs_to_amp(dvcs_array, detector, amp):
    """Convert a ROI array from the DVCS frame to the amplifier readout frame.

    The exact inverse of ``summit_utils`` `roiImageToDvcs` with ``view='dvcs'``.

    Parameters
    ----------
    dvcs_array : `numpy.ndarray`
        ROI image in the DVCS frame.
    detector : `lsst.afw.cameraGeom.Detector`
        The guide sensor.
    amp : `str`
        Amplifier name.

    Returns
    -------
    raw : `numpy.ndarray`
        ROI image in the amplifier readout frame.
    """
    ccd_array = np.rot90(dvcs_array, detector.getOrientation().getNQuarter())
    return ccd_to_amp(ccd_array, detector, amp)


def amp_to_dvcs(raw, detector, amp):
    """Convert a ROI array from the amplifier readout frame to the DVCS frame.

    The same operation as ``summit_utils`` `roiImageToDvcs`, without the stack image types;
    kept here so the round trip can be tested without ``summit_utils``.

    Parameters
    ----------
    raw : `numpy.ndarray`
        ROI image in the amplifier readout frame.
    detector : `lsst.afw.cameraGeom.Detector`
        The guide sensor.
    amp : `str`
        Amplifier name.

    Returns
    -------
    dvcs_array : `numpy.ndarray`
        ROI image in the DVCS frame.
    """
    ccd_array = ccd_to_amp(raw, detector, amp)   # the flips are their own inverse
    return np.ascontiguousarray(np.rot90(ccd_array, -detector.getOrientation().getNQuarter()))


def roi_ccd_bbox(metadata, camera):
    """Bounding box of a guider ROI in the CCD pixel frame.

    Parameters
    ----------
    metadata : `lsst.daf.base.PropertyList`
        ``guider_raw`` exposure-level metadata (``RAFTBAY``, ``CCDSLOT``, ``ROISEG``,
        ``ROICOL``, ``ROIROW``, ``ROICOLS``, ``ROIROWS``).
    camera : `lsst.afw.cameraGeom.Camera`
        The LSSTCam camera.

    Returns
    -------
    bbox : `lsst.geom.Box2I`
        ROI box in CCD pixels.
    """
    from lsst.summit.utils.guiders.transformation import makeRoiBbox

    return makeRoiBbox(metadata, camera)


def roi_ccd_pixel_grid(bbox):
    """CCD pixel coordinates of every ROI pixel centre, in CCD-frame array order.

    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
        ROI box in CCD pixels, from `roi_ccd_bbox`.

    Returns
    -------
    x, y : `numpy.ndarray`
        CCD pixel coordinates (pix), each shape ``(height, width)``; ``[i, j]`` is the
        pixel at row ``i``, column ``j`` of the CCD-frame ROI array.
    """
    x0, y0 = bbox.getMinX(), bbox.getMinY()
    w, h = bbox.getWidth(), bbox.getHeight()
    y, x = np.mgrid[y0:y0 + h, x0:x0 + w]
    return x.astype(float), y.astype(float)


def ccd_to_field_angle(detector, x, y):
    """Field angle of CCD pixel positions.

    Parameters
    ----------
    detector : `lsst.afw.cameraGeom.Detector`
        Any LSSTCam detector.
    x, y : `numpy.ndarray`
        CCD pixel coordinates (pix).

    Returns
    -------
    fa_x, fa_y : `numpy.ndarray`
        Field angle (rad), cameraGeom ``FIELD_ANGLE`` frame, same shape as `x`.
    """
    import lsst.afw.cameraGeom as cg

    shape = np.shape(x)
    mapping = detector.getTransform(cg.PIXELS, cg.FIELD_ANGLE).getMapping()
    fa = mapping.applyForward(np.vstack((np.ravel(x), np.ravel(y))).astype(float))
    return fa[0].reshape(shape), fa[1].reshape(shape)


def field_angle_to_ccd(detector, fa_x, fa_y):
    """CCD pixel positions of field angles.

    Parameters
    ----------
    detector : `lsst.afw.cameraGeom.Detector`
        Any LSSTCam detector.
    fa_x, fa_y : `numpy.ndarray`
        Field angle (rad), cameraGeom ``FIELD_ANGLE`` frame.

    Returns
    -------
    x, y : `numpy.ndarray`
        CCD pixel coordinates (pix), same shape as `fa_x`.
    """
    import lsst.afw.cameraGeom as cg

    shape = np.shape(fa_x)
    mapping = detector.getTransform(cg.PIXELS, cg.FIELD_ANGLE).getMapping()
    pix = mapping.applyInverse(np.vstack((np.ravel(fa_x), np.ravel(fa_y))).astype(float))
    return pix[0].reshape(shape), pix[1].reshape(shape)


def local_jacobian(detector, x, y, step=1.0):
    """Jacobian of field angle with respect to CCD pixel at one position.

    Used as a GalSim `JacobianWCS` so a stamp is drawn directly in the CCD pixel frame,
    with the local plate scale and distortion.

    Parameters
    ----------
    detector : `lsst.afw.cameraGeom.Detector`
        Any LSSTCam detector.
    x, y : `float`
        CCD pixel position (pix).
    step : `float`, optional
        Finite-difference step (pix).

    Returns
    -------
    jac : `numpy.ndarray`
        2x2 matrix ``[[dfa_x/dx, dfa_x/dy], [dfa_y/dx, dfa_y/dy]]`` (rad/pix).
    """
    xs = np.array([x - step, x + step, x, x])
    ys = np.array([y, y, y - step, y + step])
    fx, fy = ccd_to_field_angle(detector, xs, ys)
    return np.array([[(fx[1] - fx[0]) / (2 * step), (fx[3] - fx[2]) / (2 * step)],
                     [(fy[1] - fy[0]) / (2 * step), (fy[3] - fy[2]) / (2 * step)]])
