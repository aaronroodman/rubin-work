"""Downsampled full focal-plane mosaic of LSSTCam detector images.

The method is the one in ``~/notebooks/Camera/CCDImages/run7_imagemosaics-part2.ipynb``:
an image source hands `lsst.afw.cameraGeom.utils.makeImageFromCamera` each detector
binned and rotated by its ``nQuarter``, and cameraGeom places it on the focal plane.
The notebook's source reads butler references; this one takes in-memory arrays, so
a mosaic can be built from pre-binned ISR images.

`MosaicGeometry` maps focal-plane mm (CCS) to mosaic pixels, so overlays drawn on
the mosaic land on the right detector to within one binned pixel.
"""
import math

import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom
from lsst.afw.cameraGeom import FOCAL_PLANE
from lsst.afw.cameraGeom import utils as cgu

__all__ = ['ArrayImageSource', 'MosaicGeometry', 'make_mosaic', 'bin_mean']


def bin_mean(image, factor):
    """Mean-bin a 2-D array by an integer factor, dropping the remainder.

    Parameters
    ----------
    image : `numpy.ndarray`
        Input image.
    factor : `int`
        Binning factor in both axes.

    Returns
    -------
    binned : `numpy.ndarray`
        Binned image, shape ``(ny // factor, nx // factor)``.
    """
    if factor == 1:
        return image
    ny, nx = image.shape[0] // factor, image.shape[1] // factor
    return image[:ny * factor, :nx * factor].reshape(
        ny, factor, nx, factor).mean(axis=(1, 3))


class ArrayImageSource:
    """cameraGeom image source over arrays already binned by ``pre_bin``.

    Parameters
    ----------
    images : `dict` [`str`, `numpy.ndarray`]
        Detector name to image, binned ``pre_bin`` times from the assembled
        detector, in electrons per original pixel.
    pre_bin : `int`
        Binning already applied to ``images``.
    """
    isTrimmed = True
    background = np.nan

    def __init__(self, images, pre_bin):
        self.images = images
        self.pre_bin = pre_bin

    def getCcdImage(self, det, imageFactory, binSize=1, *args, **kwargs):
        if binSize % self.pre_bin:
            raise ValueError(f'binSize {binSize} is not a multiple of {self.pre_bin}')
        array = self.images.get(det.getName())
        if array is None:
            return None, det
        dims = det.getBBox().getDimensions()
        nx, ny = dims.getX() // binSize, dims.getY() // binSize
        binned = bin_mean(array, binSize // self.pre_bin)
        image = afwImage.ImageF(nx, ny)
        image.array[:] = np.nan
        sy, sx = min(ny, binned.shape[0]), min(nx, binned.shape[1])
        image.array[:sy, :sx] = binned[:sy, :sx]
        return afwMath.rotateImageBy90(image, det.getOrientation().getNQuarter()), det


class MosaicGeometry:
    """Focal-plane mm to mosaic pixel, matching `makeImageFromCamera`.

    Parameters
    ----------
    camera : `lsst.afw.cameraGeom.Camera`
        Camera geometry.
    detector_names : `list` [`str`]
        Detectors in the mosaic.
    bin_size : `int`
        Mosaic binning, in original pixels per mosaic pixel.
    buffer_size : `int`, optional
        Border, in mosaic pixels, as passed to `makeImageFromCamera`.
    """

    def __init__(self, camera, detector_names, bin_size, buffer_size=10):
        cam_bbox = lsst.geom.Box2D()
        for name in detector_names:
            for corner in camera[name].getCorners(FOCAL_PLANE):
                cam_bbox.include(corner)
        self.pixel_mm = camera[detector_names[0]].getPixelSize().getX()
        box = cgu.getCameraImageBBox(cam_bbox, camera[detector_names[0]].getPixelSize(),
                                     buffer_size * bin_size)
        self.origin = (box.getMinX(), box.getMinY())
        self.bin_size = bin_size

    def to_pixel(self, x_mm, y_mm):
        """Mosaic pixel of a focal-plane point.

        Parameters
        ----------
        x_mm, y_mm : `float` or `numpy.ndarray`
            Focal-plane position (CCS), in mm.

        Returns
        -------
        x_pix, y_pix : `float` or `numpy.ndarray`
            Mosaic array column and row, in mosaic pixels.
        """
        x = (np.asarray(x_mm) / self.pixel_mm - self.origin[0]) / self.bin_size
        y = (np.asarray(y_mm) / self.pixel_mm - self.origin[1]) / self.bin_size
        return x, y

    def to_mm(self, x_pix, y_pix):
        """Focal-plane position of a mosaic pixel; inverse of `to_pixel`."""
        x = (np.asarray(x_pix) * self.bin_size + self.origin[0]) * self.pixel_mm
        y = (np.asarray(y_pix) * self.bin_size + self.origin[1]) * self.pixel_mm
        return x, y


def make_mosaic(camera, images, pre_bin, bin_size, buffer_size=10):
    """Build a binned focal-plane mosaic.

    Parameters
    ----------
    camera : `lsst.afw.cameraGeom.Camera`
        Camera geometry.
    images : `dict` [`str`, `numpy.ndarray`]
        Detector name to image binned ``pre_bin`` times.
    pre_bin : `int`
        Binning already applied to ``images``.
    bin_size : `int`
        Mosaic binning in original pixels; a multiple of ``pre_bin``.
    buffer_size : `int`, optional
        Border, in mosaic pixels.

    Returns
    -------
    mosaic : `numpy.ndarray`
        Mosaic, NaN off the detectors, row 0 at the focal plane's -y edge.
    geometry : `MosaicGeometry`
        Focal-plane mm to mosaic pixel mapping.
    """
    names = sorted(images)
    source = ArrayImageSource(images, pre_bin)
    image = cgu.makeImageFromCamera(camera, detectorNameList=names, background=math.nan,
                                    bufferSize=buffer_size, imageSource=source,
                                    imageFactory=afwImage.ImageF, binSize=bin_size)
    return image.array.copy(), MosaicGeometry(camera, names, bin_size, buffer_size)
