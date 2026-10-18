"""Giant-donut candidates: catalog isolation (blitz) and image isolation.

Works on the 4x4-binned ISR cache written by ``isr_cache.py``. Per visit:

1. Measure the donut outer and inner radii for each sensor class (science, SW0,
   SW1) from radial profiles of the brightest detections. The corner sensors sit
   1.5 mm either side of the science surface, so their donuts differ in size and
   SW0 and SW1 swap between the extra and intra visits.
2. Detect donuts blind, with blitz's annular template (`_buildAnnularTemplate`).
3. Project the_monster refcat onto each detector through the raw WCS, then fit one
   affine map per visit, in focal-plane mm, from the projected star to the detected
   donut centre. That absorbs the raw-WCS pointing error and the scale change of the
   defocused image.
4. (a) Catalog isolation: blitz's `DonutSourceSelectorTask`, with
   ``unblendedSeparation`` and ``minBlendedSeparation`` scaled from the nominal FAM
   donut radius to this sensor's giant radius. The edge cut is decoupled from them
   (see `_EdgeDetector`).
5. (b) Image isolation: flux in an annulus just outside the donut, and the fraction
   of the donut's ring that another donut's disk covers.

Pixel positions are full-resolution assembled-detector pixels; focal-plane
positions are CCS mm. Field angles are CCS deg; at rotator 0.012 and 0.149 deg CCS
and OCS agree to that angle.
"""
import pathlib
import sys

import numpy as np

_GIANT_CODE = pathlib.Path(__file__).resolve().parents[3] / 'wfs' / 'code' / 'giant_donuts'
sys.path.insert(0, str(_GIANT_CODE))
import fit_giant_donut  # noqa: E402  (puts the blitz checkout first on sys.path)

from scipy.ndimage import zoom  # noqa: E402
from scipy.signal import correlate  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from skimage.feature import peak_local_max  # noqa: E402

import astropy.units as u  # noqa: E402
import lsst.geom  # noqa: E402
import lsst.sphgeom as sphgeom  # noqa: E402
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS, DetectorType  # noqa: E402

__all__ = ['CACHE_BIN', 'REFCAT', 'FLUX_FIELD', 'sensor_class', 'import_blitz',
           'measure_radius', 'background_map', 'blind_detect', 'load_refcat',
           'project_refcat', 'fit_affine', 'apply_affine', 'blitz_select',
           'ring_metrics']

CACHE_BIN = 4
REFCAT = 'the_monster_20250219'
# blitz's photoRefFilterPrefix default plus the band.
FLUX_FIELD = 'monster_ComCam_r_flux'

# blitz's selector defaults (pixels) and the nominal FAM donut radius they were
# tuned for (`_INSTRUMENT.donutRadius`, pixels). Both separations scale with radius.
BLITZ_UNBLENDED_PIX = 160
BLITZ_MIN_BLENDED_PIX = 120
BLITZ_FAM_RADIUS_PIX = 66.51


def sensor_class(detector):
    """``'science'``, ``'SW0'`` or ``'SW1'`` for a detector."""
    if detector.getType() == DetectorType.WAVEFRONT:
        return detector.getName()[-3:]
    return 'science'


def import_blitz():
    """Blitz's annular template builder and donut source selector.

    Returns
    -------
    build_template : callable
        ``_buildAnnularTemplate(radius, innerFrac)``.
    selector_config_class, selector_class : `type`
        `DonutSourceSelectorTaskConfig` and `DonutSourceSelectorTask`.
    """
    fit_giant_donut._import_blitz()
    from lsst.ts.wep.blitz.blindDetectTask import _buildAnnularTemplate
    from lsst.ts.wep.task.donutSourceSelectorTask import (
        DonutSourceSelectorTask, DonutSourceSelectorTaskConfig)
    return _buildAnnularTemplate, DonutSourceSelectorTaskConfig, DonutSourceSelectorTask


def radial_profile(image, x0, y0, r_max, step=1.0):
    """Median radial profile about a centre, after centroid refinement.

    Parameters
    ----------
    image : `numpy.ndarray`
        Background-subtracted binned image.
    x0, y0 : `float`
        Starting centre, in binned pixels.
    r_max : `float`
        Outer radius of the profile, in binned pixels.
    step : `float`, optional
        Radial bin width, in binned pixels.

    Returns
    -------
    x0, y0 : `float`
        Refined centre, in binned pixels.
    radius, profile : `numpy.ndarray`
        Bin inner radii (binned pixels) and median level per bin.
    """
    ny, nx = image.shape
    ys, xs = np.mgrid[:ny, :nx]
    for _ in range(3):
        r = np.hypot(xs - x0, ys - y0)
        sel = r < r_max
        level = 0.3 * np.nanpercentile(image[sel], 99)
        w = sel & (image > level)
        if w.sum() < 10:
            break
        y0, x0 = ys[w].mean(), xs[w].mean()
    r = np.hypot(xs - x0, ys - y0)
    radius = np.arange(0, r_max, step)
    profile = np.array([np.nanmedian(image[(r >= a) & (r < a + step)]) for a in radius])
    return x0, y0, radius, profile


def measure_radius(images, backgrounds, detections, names, r_guess):
    """Outer and inner donut radius from the half-maximum of radial profiles.

    Parameters
    ----------
    images, backgrounds : `dict` [`str`, `numpy.ndarray`]
        Binned images and background maps.
    detections : `dict` [`str`, `numpy.ndarray`]
        Binned-pixel ``(x, y)`` detections per detector, shape ``(n, 2)``.
    names : `list` [`str`]
        Detectors of one sensor class to use.
    r_guess : `float`
        Approximate outer radius, in binned pixels.

    Returns
    -------
    r_out, r_in : `float`
        Median outer and inner half-maximum radius, in full-resolution pixels.
    """
    outs, ins = [], []
    for name in names:
        image = images[name] - backgrounds[name]
        for x, y in detections.get(name, []):
            if not (1.3 * r_guess < x < image.shape[1] - 1.3 * r_guess
                    and 1.3 * r_guess < y < image.shape[0] - 1.3 * r_guess):
                continue
            _, _, radius, profile = radial_profile(image, x, y, 1.3 * r_guess)
            peak = np.nanmax(profile)
            above = np.where(profile > 0.5 * peak)[0]
            if len(above) < 3 or peak <= 0:
                continue
            ins.append(radius[above[0]])
            outs.append(radius[above[-1]] + 1.0)
    return float(np.median(outs)) * CACHE_BIN, float(np.median(ins)) * CACHE_BIN


def background_map(image, block=64, percentile=20.0):
    """Smooth background from a low percentile in blocks, interpolated back.

    A low percentile rather than a median because giant donuts cover a large
    fraction of a crowded sensor and only add flux.

    Parameters
    ----------
    image : `numpy.ndarray`
        Binned image.
    block : `int`, optional
        Block size, in binned pixels.
    percentile : `float`, optional
        Percentile taken per block.

    Returns
    -------
    background : `numpy.ndarray`
        Background map, same shape as ``image``.
    """
    ny, nx = image.shape
    by, bx = max(1, ny // block), max(1, nx // block)
    grid = np.empty((by, bx))
    for j in range(by):
        for i in range(bx):
            cut = image[j * ny // by:(j + 1) * ny // by, i * nx // bx:(i + 1) * nx // bx]
            grid[j, i] = np.nanpercentile(cut, percentile)
    return zoom(grid, (ny / by, nx / bx), order=1, mode='nearest')[:ny, :nx]


def blind_detect(image, background, r_out_bin, inner_frac, build_template,
                 min_distance_factor=1.6, border_factor=0.5):
    """Detect donuts by annular-template cross-correlation, as blitz does.

    Parameters
    ----------
    image, background : `numpy.ndarray`
        Binned image and its background.
    r_out_bin : `float`
        Outer donut radius, in binned pixels.
    inner_frac : `float`
        Inner over outer radius (dimensionless).
    build_template : callable
        blitz's ``_buildAnnularTemplate``.
    min_distance_factor : `float`, optional
        Peak separation in units of ``r_out_bin`` (blitz's default 1.6).
    border_factor : `float`, optional
        Peaks closer than this many radii to the edge are dropped.

    Returns
    -------
    xy : `numpy.ndarray`
        Peak positions, binned pixels, shape ``(n, 2)``.
    """
    arr = np.nan_to_num(image - background)
    heq = np.digitize(arr, np.quantile(arr, np.linspace(0, 1, 256)))
    template = build_template(r_out_bin, innerFrac=inner_frac)
    score = correlate(heq.astype(float), template, mode='same', method='fft')
    peaks = peak_local_max(score, min_distance=int(min_distance_factor * r_out_bin),
                           exclude_border=int(border_factor * r_out_bin),
                           threshold_rel=0.3)
    return peaks[:, ::-1].astype(float)


def load_refcat(butler, ra_deg, dec_deg, radius_deg=2.3, mag_max=19.0):
    """Monster refcat stars around a pointing.

    Returns
    -------
    ids : `numpy.ndarray`
        Refcat source ids.
    ra, dec : `numpy.ndarray`
        Coordinates, in rad.
    mag : `numpy.ndarray`
        ``monster_ComCam_r`` AB magnitude.
    """
    circle = sphgeom.Circle(sphgeom.UnitVector3d(sphgeom.LonLat.fromDegrees(ra_deg, dec_deg)),
                            sphgeom.Angle.fromDegrees(radius_deg))
    shards = [i for lo, hi in sphgeom.HtmPixelization(7).envelope(circle) for i in range(lo, hi)]
    refs = butler.query_datasets(REFCAT, collections='refcats',
                                 where=f"htm7 IN ({','.join(map(str, shards))})", limit=None)
    ids, ra, dec, flux = [], [], [], []
    for ref in refs:
        cat = butler.get(ref)
        ids.append(cat['id']), ra.append(cat['coord_ra']), dec.append(cat['coord_dec'])
        flux.append(cat[FLUX_FIELD])
    ids, ra, dec, flux = map(np.concatenate, (ids, ra, dec, flux))
    with np.errstate(invalid='ignore', divide='ignore'):
        mag = (flux * u.nJy).to_value(u.ABmag)
    keep = np.isfinite(mag) & (mag < mag_max)
    return ids[keep], ra[keep], dec[keep], mag[keep]


def project_refcat(wcs, detector, ra, dec, margin_pix=0):
    """Refcat stars falling on a detector, through its raw WCS.

    Returns
    -------
    index : `numpy.ndarray`
        Indices into ``ra``/``dec`` of the stars on the detector.
    x_pix, y_pix : `numpy.ndarray`
        Their full-resolution pixel positions.
    """
    x, y = wcs.skyToPixelArray(ra, dec, degrees=False)
    bbox = lsst.geom.Box2D(detector.getBBox())
    on = ((x > bbox.getMinX() - margin_pix) & (x < bbox.getMaxX() + margin_pix)
          & (y > bbox.getMinY() - margin_pix) & (y < bbox.getMaxY() + margin_pix))
    return np.where(on)[0], x[on], y[on]


def pixels_to(detector, x_pix, y_pix, system):
    """Map pixel arrays to FOCAL_PLANE (mm) or FIELD_ANGLE (rad)."""
    mapping = detector.getTransform(PIXELS, system).getMapping()
    out = mapping.applyForward(np.vstack([np.asarray(x_pix, float), np.asarray(y_pix, float)]))
    return out[0], out[1]


def focal_to_pixels(detector, x_mm, y_mm):
    """Map focal-plane mm arrays to pixels on a detector."""
    mapping = detector.getTransform(FOCAL_PLANE, PIXELS).getMapping()
    out = mapping.applyForward(np.vstack([np.asarray(x_mm, float), np.asarray(y_mm, float)]))
    return out[0], out[1]


def fit_affine(src, dst, n_iter=5, clip=3.0):
    """Robust affine fit ``dst = A @ src + b`` with iterative clipping.

    Parameters
    ----------
    src, dst : `numpy.ndarray`
        Matched points, shape ``(n, 2)``.

    Returns
    -------
    coeffs : `numpy.ndarray`
        Shape ``(3, 2)``: rows are the x, y and constant terms.
    keep : `numpy.ndarray`
        Points kept in the last iteration.
    rms : `float`
        nMAD of the residual radius over the kept points, in ``dst`` units.
    """
    design = np.column_stack([src, np.ones(len(src))])
    keep = np.ones(len(src), bool)
    for _ in range(n_iter):
        coeffs, *_ = np.linalg.lstsq(design[keep], dst[keep], rcond=None)
        resid = np.hypot(*(design @ coeffs - dst).T)
        scale = 1.4826 * np.median(resid[keep])
        keep = resid < max(clip * scale, 1e-3)
    return coeffs, keep, float(1.4826 * np.median(resid[keep]))


def apply_affine(coeffs, x, y):
    """Apply an affine map from `fit_affine` to coordinate arrays."""
    out = np.column_stack([x, y, np.ones(len(x))]) @ coeffs
    return out[:, 0], out[:, 1]


class _EdgeDetector:
    """Detector stand-in whose bounding box makes blitz's edge cut a given margin.

    blitz erodes the detector box by ``unblendedSeparation`` before selecting. For
    a giant donut that is about 2.4 donut radii, which leaves nothing on a corner
    sensor 2000 pixels tall. Here the box is grown so the net erosion is
    ``edge_margin`` -- enough for the whole donut to be on silicon -- and the
    isolation test keeps its scaled separations.
    """

    def __init__(self, detector, grow_pix):
        self._detector = detector
        self._grow = int(round(grow_pix))

    def getBBox(self):
        box = lsst.geom.Box2I(self._detector.getBBox())
        box.grow(self._grow)
        return box

    def getTransform(self, from_sys, to_sys):
        return self._detector.getTransform(from_sys, to_sys)


def blitz_select(selector_classes, detector, x_pix, y_pix, flux_njy, r_out_pix,
                 mag_min, mag_max, edge_margin_pix, isolated_mag_diff=2.0,
                 max_field_deg=2.1):
    """Run blitz's donut source selector scaled to the giant-donut radius.

    Parameters
    ----------
    selector_classes : `tuple`
        ``(DonutSourceSelectorTaskConfig, DonutSourceSelectorTask)``.
    detector : `lsst.afw.cameraGeom.Detector`
        The detector.
    x_pix, y_pix : `numpy.ndarray`
        Star positions corrected to the donut centres, full-resolution pixels.
    flux_njy : `numpy.ndarray`
        Star fluxes, in nJy.
    r_out_pix : `float`
        Donut outer radius on this detector, full-resolution pixels.
    mag_min, mag_max : `float`
        Magnitude range of selectable stars, AB mag.
    edge_margin_pix : `float`
        Minimum distance from donut centre to the detector edge, pixels.
    isolated_mag_diff : `float`, optional
        blitz's ``isolatedMagDiff``, mag.
    max_field_deg : `float`, optional
        Field radius limit, deg; set past the corners so every sensor is kept.

    Returns
    -------
    selected : `numpy.ndarray`
        Boolean, per input star.
    unblended_pix, min_blended_pix : `int`
        The scaled separations used, full-resolution pixels.
    """
    from astropy.table import QTable
    config_class, task_class = selector_classes
    scale = r_out_pix / BLITZ_FAM_RADIUS_PIX
    config = config_class()
    config.unblendedSeparation = int(round(BLITZ_UNBLENDED_PIX * scale))
    config.minBlendedSeparation = int(round(BLITZ_MIN_BLENDED_PIX * scale))
    config.isolatedMagDiff = isolated_mag_diff
    config.useCustomMagLimit = True
    config.magMin, config.magMax = mag_min, mag_max
    config.maxFieldDist = max_field_deg
    config.maxBlended = 0
    task = task_class(config=config)
    if len(x_pix) == 0:
        return np.zeros(0, bool), config.unblendedSeparation, config.minBlendedSeparation
    cat = QTable({'centroid_x': np.asarray(x_pix, float), 'centroid_y': np.asarray(y_pix, float),
                  'r_flux': np.asarray(flux_njy, float)})
    stand_in = _EdgeDetector(detector, config.unblendedSeparation - edge_margin_pix)
    result = task.selectSources(cat, stand_in, 'r')
    return result.selected, config.unblendedSeparation, config.minBlendedSeparation


def ring_metrics(image, background, x_bin, y_bin, r_out_bin, r_in_bin, others):
    """Image isolation metrics for one donut.

    Parameters
    ----------
    image, background : `numpy.ndarray`
        Binned image and background.
    x_bin, y_bin : `float`
        Donut centre, binned pixels.
    r_out_bin, r_in_bin : `float`
        Donut outer and inner radius, binned pixels.
    others : `numpy.ndarray`
        Other donuts on this detector, rows ``(x_bin, y_bin, r_out_bin)``.

    Returns
    -------
    annulus_ratio : `float`
        Mean background-subtracted level in 1.05-1.5 ``r_out`` over that in the
        donut ring (dimensionless). About 0 for an isolated donut.
    overlap_frac : `float`
        Fraction of the donut ring's pixels that lie inside another donut's outer
        disk (dimensionless).
    ring_level : `float`
        Mean background-subtracted level in the ring, electrons per full-resolution
        pixel.
    ring_p99 : `float`
        99th percentile of the raw level in the ring, electrons per full-resolution
        pixel (a binned mean, so it understates a saturated core).
    """
    half = int(np.ceil(1.6 * r_out_bin))
    x0, y0 = int(round(x_bin)), int(round(y_bin))
    ny, nx = image.shape
    ys, xs = np.mgrid[max(0, y0 - half):min(ny, y0 + half + 1),
                      max(0, x0 - half):min(nx, x0 + half + 1)]
    cut = image[ys[:, 0][0]:ys[:, 0][-1] + 1, xs[0][0]:xs[0][-1] + 1]
    sub = cut - background[ys[:, 0][0]:ys[:, 0][-1] + 1, xs[0][0]:xs[0][-1] + 1]
    r = np.hypot(xs - x_bin, ys - y_bin)
    ring = (r >= r_in_bin) & (r <= r_out_bin)
    outer = (r >= 1.05 * r_out_bin) & (r <= 1.5 * r_out_bin)
    ring_level = float(np.nanmean(sub[ring]))
    annulus_ratio = float(np.nanmean(sub[outer]) / ring_level) if ring_level > 0 else np.nan
    covered = np.zeros_like(ring)
    for xo, yo, ro in others:
        covered |= np.hypot(xs - xo, ys - yo) <= ro
    overlap_frac = float((covered & ring).sum() / max(ring.sum(), 1))
    return annulus_ratio, overlap_frac, ring_level, float(np.nanpercentile(cut[ring], 99))
