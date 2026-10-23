"""ISR every detector of a giant-donut visit and cache it binned.

Runs the minimal ISR of ``wfs/code/giant_donuts/donut_images.py`` (overscan,
assembly, gains) on all 189 science and 8 corner wavefront-sensor detectors, bins
each image 4x4 (mean), and saves it as float32 ``.npy`` with the raw WCS header.
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

from rubinwork.common import study_run

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]
                       / 'wfs' / 'code' / 'giant_donuts'))
import donut_images  # noqa: E402

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
        Directory for ``<detector>.npy`` and ``wcs.json``.
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
        path = out_dir / f'{name}.npy'
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
        np.save(path, bin_mean(image, CACHE_BIN).astype(np.float32))
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
