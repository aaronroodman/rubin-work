"""ISR every detector of a giant-donut visit and cache it binned.

Runs the minimal ISR of `rubinwork.giant_donuts.donut_images` (overscan,
assembly, gains) on all 189 science and 8 corner wavefront-sensor detectors, bins
each image 4x4 (mean), and saves it as Rice-compressed FITS (``<detector>.fits.fz``, quantize
level 16) with the raw WCS header.
A binned pixel is 40 um; an 8 mm giant donut is about 174 binned pixels across,
enough for the isolation metrics and the zoom pages. The donut fits re-run ISR at
full resolution on the chosen detectors.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/isr_cache.py 2025102300337 2025102300340
"""
import json
import pathlib
import sys
import time

import numpy as np

from lsst.daf.butler import Butler
from lsst.obs.lsst import LsstCam
from lsst.afw.cameraGeom import DetectorType

from rubinwork.common import study_run  # noqa: E402

from rubinwork.giant_donuts import donut_images  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
REPO = '/repo/main'
RAW_COLLECTION = 'LSSTCam/raw/all'
CACHE_BIN = 4


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
    ny, nx = image.shape[0] // factor, image.shape[1] // factor
    return image[:ny * factor, :nx * factor].reshape(
        ny, factor, nx, factor).mean(axis=(1, 3))


QUANTIZE_LEVEL = 16  # fpack default: quantization step is the pixel noise / 16


def cache_path(cache_dir, name):
    """Path of one detector's binned image: ``<name>.fits.fz``, else a legacy ``.npy``."""
    fz = pathlib.Path(cache_dir) / f'{name}.fits.fz'
    npy = fz.with_name(f'{name}.npy')
    return npy if npy.exists() and not fz.exists() else fz


def save_binned(cache_dir, name, image):
    """Write a binned image as Rice tile-compressed FITS (lossy, ``QUANTIZE_LEVEL``)."""
    from astropy.io import fits
    hdu = fits.CompImageHDU(np.asarray(image, np.float32), name='IMAGE', compression_type='RICE_1',
                            quantize_level=QUANTIZE_LEVEL, dither_seed=-1)  # -1: seed from checksum
    path = pathlib.Path(cache_dir) / f'{name}.fits.fz'
    hdu.writeto(path, overwrite=True)
    return path


def load_binned(cache_dir, name):
    """Read one detector's binned image (electrons per binned pixel) as float64."""
    path = cache_path(cache_dir, name)
    if path.suffix == '.npy':
        return np.load(path).astype(float)
    from astropy.io import fits
    with fits.open(path) as h:
        return h['IMAGE'].data.astype(float)


def cache_visit(butler, camera, exposure, out_dir, names=None):
    """ISR, bin and save the science and corner-WFS detectors of one exposure.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler holding the raws.
    camera : `lsst.afw.cameraGeom.Camera`
        Camera geometry.
    exposure : `int`
        Exposure id.
    out_dir : `pathlib.Path`
        Directory for ``<detector>.fits.fz`` and ``wcs.json``.
    names : `list` [`str`], optional
        Detectors to cache; every science and corner-WFS detector by default.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wcs_file = out_dir / 'wcs.json'
    headers = json.loads(wcs_file.read_text()) if wcs_file.exists() else {}
    if names is None:
        names = [d.getName() for d in camera
                 if d.getType() in (DetectorType.SCIENCE, DetectorType.WAVEFRONT)]
    for i, name in enumerate(names):
        path = cache_path(out_dir, name)
        if path.exists() and name in headers:
            continue
        t0 = time.time()
        try:
            image = donut_images.run_isr(butler, exposure, name, camera, RAW_COLLECTION)
            wcs = butler.get('raw.wcs', collections=RAW_COLLECTION, instrument='LSSTCam',
                             exposure=exposure, detector=camera[name].getId())
        except Exception as exc:
            print(f'{exposure} {name}: FAILED {type(exc).__name__}: {exc}', flush=True)
            continue
        save_binned(out_dir, name, bin_mean(image, CACHE_BIN))
        headers[name] = wcs.getFitsMetadata().toDict()
        wcs_file.write_text(json.dumps(headers))
        print(f'{exposure} {i + 1}/{len(names)} {name} {time.time() - t0:.1f} s',
              flush=True)


def main():
    exposures = [int(e) for e in sys.argv[1:]]
    run = study_run(STUDY, 'isr_bin4')
    butler = Butler(REPO)
    camera = LsstCam.getCamera()
    for exposure in exposures:
        cache_visit(butler, camera, exposure, run.path / str(exposure))
    print('done', run.path)


if __name__ == '__main__':
    main()
