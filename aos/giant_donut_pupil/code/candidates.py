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

# fit_giant_donut puts the blitz checkout first on sys.path.
from rubinwork.giant_donuts import fit_giant_donut  # noqa: E402

from scipy.ndimage import zoom  # noqa: E402
from scipy.signal import correlate  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from skimage.feature import peak_local_max  # noqa: E402

import astropy.units as u  # noqa: E402
import lsst.geom  # noqa: E402
import lsst.sphgeom as sphgeom  # noqa: E402
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS, DetectorType  # noqa: E402

__all__ = ['CACHE_BIN', 'REFCAT', 'FLUX_FIELD', 'sensor_class', 'import_blitz',
           'measure_radius', 'background_map', 'blind_detect', 'blind_detect_scored',
           'load_refcat', 'load_refcat_bands', 'fill_r_magnitude', 'project_refcat',
           'fit_affine', 'apply_affine', 'blitz_select', 'blitz_neighbours',
           'nearest_relevant', 'ring_metrics', 'overlap_fraction',
           'annulus_model_residual', 'bleed_ratio']

CACHE_BIN = 4
REFCAT = 'the_monster_20250219'
# blitz's photoRefFilterPrefix default plus the band.
FLUX_FIELD = 'monster_ComCam_r_flux'
# Bands read for stars with no ComCam r flux, in the order they fill r.
BAND_FIELDS = {'mag_r': 'monster_ComCam_r_flux', 'mag_des_r': 'monster_DES_r_flux',
               'mag_G': 'phot_g_mean_flux', 'mag_bp': 'phot_bp_mean_flux',
               'mag_rp': 'phot_rp_mean_flux'}

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


def blind_detect_scored(image, background, r_out_bin, inner_frac, build_template,
                        min_distance_factor=1.6, border_factor=0.5, rel=0.3, n_sigma=5.0):
    """`blind_detect` with both peak thresholds evaluated on one peak list.

    Peaks are found at the lower of the two thresholds. Non-maximum suppression in
    `peak_local_max` only lets a stronger peak remove a weaker one, so each
    threshold's peak list is exactly the subset passing it.

    Parameters
    ----------
    rel : `float`, optional
        Threshold relative to the brightest score on the detector (dimensionless).
    n_sigma : `float`, optional
        Threshold in units of the score map's robust noise: median plus
        ``n_sigma`` nMAD.

    Returns
    -------
    xy : `numpy.ndarray`
        Peak positions, binned pixels, shape ``(n, 2)``.
    snr : `numpy.ndarray`
        Peak score minus the map median, over the map nMAD (dimensionless).
    pass_rel, pass_sigma : `numpy.ndarray`
        Whether each peak passes the relative or the n-sigma threshold.
    """
    arr = np.nan_to_num(image - background)
    heq = np.digitize(arr, np.quantile(arr, np.linspace(0, 1, 256)))
    template = build_template(r_out_bin, innerFrac=inner_frac)
    score = correlate(heq.astype(float), template, mode='same', method='fft')
    med = np.median(score)
    noise = 1.4826 * np.median(np.abs(score - med))
    thr_rel, thr_sigma = rel * score.max(), med + n_sigma * noise
    peaks = peak_local_max(score, min_distance=int(min_distance_factor * r_out_bin),
                           exclude_border=int(border_factor * r_out_bin),
                           threshold_abs=min(thr_rel, thr_sigma))
    values = score[peaks[:, 0], peaks[:, 1]]
    return (peaks[:, ::-1].astype(float), (values - med) / noise,
            values >= thr_rel, values >= thr_sigma)


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


def load_refcat_bands(butler, ra_deg, dec_deg, radius_deg=2.3):
    """Monster refcat stars around a pointing, with every band in `BAND_FIELDS`.

    Unlike `load_refcat`, stars with no ComCam r flux are kept.

    Returns
    -------
    stars : `pandas.DataFrame`
        Columns ``id``, ``ra``, ``dec`` (rad) and one AB magnitude per
        `BAND_FIELDS` key, NaN where the flux is missing or not positive.
    """
    import pandas as pd
    circle = sphgeom.Circle(sphgeom.UnitVector3d(sphgeom.LonLat.fromDegrees(ra_deg, dec_deg)),
                            sphgeom.Angle.fromDegrees(radius_deg))
    shards = [i for lo, hi in sphgeom.HtmPixelization(7).envelope(circle) for i in range(lo, hi)]
    refs = butler.query_datasets(REFCAT, collections='refcats',
                                 where=f"htm7 IN ({','.join(map(str, shards))})", limit=None)
    parts = []
    for ref in refs:
        cat = butler.get(ref)
        part = {'id': cat['id'], 'ra': cat['coord_ra'], 'dec': cat['coord_dec']}
        with np.errstate(invalid='ignore', divide='ignore'):
            for key, field in BAND_FIELDS.items():
                part[key] = (np.asarray(cat[field]) * u.nJy).to_value(u.ABmag)
        parts.append(pd.DataFrame(part))
    stars = pd.concat(parts, ignore_index=True)
    ra0, dec0 = np.radians(ra_deg), np.radians(dec_deg)
    cos_sep = (np.sin(dec0) * np.sin(stars.dec) + np.cos(dec0) * np.cos(stars.dec)
               * np.cos(stars.ra - ra0))
    return stars[cos_sep > np.cos(np.radians(radius_deg))].reset_index(drop=True)


def fill_r_magnitude(stars, mag_limit=17.0):
    """Best r magnitude: ComCam r, else DES r, else Gaia G with a colour term.

    The colour term is a quadratic in BP - RP fitted to ``r - G`` on stars that have
    both, with 3-sigma clipping; stars without BP/RP get the median ``r - G``.

    Parameters
    ----------
    stars : `pandas.DataFrame`
        From `load_refcat_bands`.
    mag_limit : `float`, optional
        Only stars with G brighter than this enter the colour fit, AB mag.

    Returns
    -------
    mag_best : `numpy.ndarray`
        AB magnitude.
    source : `numpy.ndarray`
        ``'r'``, ``'des_r'``, ``'G_colour'``, ``'G_median'`` or ``'none'``.
    info : `dict`
        Colour-term coefficients, median and nMAD of the fit residual (mag), and
        the number of fit stars.
    """
    r, des, g = stars.mag_r.to_numpy(), stars.mag_des_r.to_numpy(), stars.mag_G.to_numpy()
    colour = (stars.mag_bp - stars.mag_rp).to_numpy()
    fit = np.isfinite(r) & np.isfinite(g) & np.isfinite(colour) & (g < mag_limit)
    keep = fit.copy()
    for _ in range(5):
        coeffs = np.polyfit(colour[keep], (r - g)[keep], 2)
        resid = (r - g) - np.polyval(coeffs, colour)
        scale = 1.4826 * np.median(np.abs(resid[keep]))
        keep = fit & (np.abs(resid) < 3 * scale)
    offset = float(np.median((r - g)[fit]))
    best = r.copy()
    source = np.where(np.isfinite(r), 'r', 'none').astype(object)
    use = ~np.isfinite(best) & np.isfinite(des)
    best[use], source[use] = des[use], 'des_r'
    use = ~np.isfinite(best) & np.isfinite(g) & np.isfinite(colour)
    best[use], source[use] = g[use] + np.polyval(coeffs, colour[use]), 'G_colour'
    use = ~np.isfinite(best) & np.isfinite(g)
    best[use], source[use] = g[use] + offset, 'G_median'
    info = dict(colour_coeffs=coeffs.tolist(), median_r_minus_G=offset,
                resid_nmad_mag=float(scale), n_fit=int(keep.sum()))
    return best, source, info


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


def blitz_neighbours(x_pix, y_pix, mag, unblended_pix, min_blended_pix, mag_max,
                     isolated_mag_diff=2.0):
    """The neighbour that makes blitz's selector reject each star.

    Re-implements the flux branch of `DonutSourceSelectorTask.selectSources` with
    ``maxBlended = 0``, recording which neighbour fails the star. The edge, field and
    magnitude cuts are left to the caller; compare with `blitz_select` to check.

    Parameters
    ----------
    x_pix, y_pix : `numpy.ndarray`
        Star positions, full-resolution pixels.
    mag : `numpy.ndarray`
        Star magnitudes, AB mag.
    unblended_pix, min_blended_pix : `float`
        The scaled separations, full-resolution pixels.
    mag_max : `float`
        Faint limit of selectable stars, AB mag; neighbours count to
        ``mag_max + isolated_mag_diff``.
    isolated_mag_diff : `float`, optional
        blitz's ``isolatedMagDiff``, mag.

    Returns
    -------
    isolated : `numpy.ndarray`
        True where no neighbour rejects the star.
    nbr : `numpy.ndarray`
        Index of the rejecting neighbour, -1 if none.
    nbr_dist_pix : `numpy.ndarray`
        Its distance, full-resolution pixels; NaN if none.
    """
    n = len(x_pix)
    isolated = np.ones(n, bool)
    nbr = np.full(n, -1)
    nbr_dist = np.full(n, np.nan)
    use = np.where(np.asarray(mag) < mag_max + isolated_mag_diff)[0]
    if len(use) == 0:
        return isolated, nbr, nbr_dist
    # Stable magnitude order, as blitz uses to break ties.
    order = use[np.argsort(np.asarray(mag)[use], kind='stable')]
    xy = np.column_stack([np.asarray(x_pix)[order], np.asarray(y_pix)[order]])
    m = np.asarray(mag)[order]
    tree = cKDTree(xy)
    for s, idx in enumerate(tree.query_ball_point(xy, r=unblended_pix)):
        others = np.array([j for j in idx if j != s], int)
        if len(others) == 0:
            continue
        dist = np.hypot(*(xy[others] - xy[s]).T)
        dmag = m[others] - m[s]
        brighter = others < s
        if brighter.any():
            fail = brighter
        elif dmag.min() >= isolated_mag_diff:
            continue
        elif np.any((dist < min_blended_pix) & (dmag < isolated_mag_diff)):
            fail = (dist < min_blended_pix) & (dmag < isolated_mag_diff)
        else:
            fail = dmag < isolated_mag_diff
        k = np.argmin(np.where(fail, dist, np.inf))
        isolated[order[s]] = False
        nbr[order[s]] = order[others[k]]
        nbr_dist[order[s]] = dist[k]
    return isolated, nbr, nbr_dist


def nearest_relevant(target_xy, target_mag, all_xy, all_mag, dmag_max=4.0, self_tol=1e-6):
    """Nearest neighbour that can matter: any brighter star, or one under ``dmag_max`` fainter.

    Parameters
    ----------
    target_xy : `numpy.ndarray`
        Target positions, shape ``(n, 2)``, any length unit.
    target_mag : `numpy.ndarray`
        Target magnitudes, AB mag.
    all_xy, all_mag : `numpy.ndarray`
        Every catalog star, same units; may include the targets.
    dmag_max : `float`, optional
        A neighbour counts if ``mag_nbr < mag_target + dmag_max``, mag.

    Returns
    -------
    index : `numpy.ndarray`
        Into ``all_xy``; -1 if none counts.
    dist : `numpy.ndarray`
        Distance in the units of the inputs; inf if none.
    """
    tree = cKDTree(all_xy)
    n = len(target_xy)
    index, dist = np.full(n, -1), np.full(n, np.inf)
    k = min(64, len(all_xy))
    d, j = tree.query(target_xy, k=k)
    d, j = np.atleast_2d(d), np.atleast_2d(j)
    for i in range(n):
        ok = (d[i] > self_tol) & np.isfinite(d[i])
        ok[ok] &= all_mag[j[i][ok]] < target_mag[i] + dmag_max
        if ok.any():
            first = np.argmax(ok)
            index[i], dist[i] = j[i][first], d[i][first]
    return index, dist


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


def _cutout(image, x_bin, y_bin, half):
    """Sub-image and its pixel grids about a centre, clipped to the image."""
    x0, y0 = int(round(x_bin)), int(round(y_bin))
    ny, nx = image.shape
    ylo, yhi = max(0, y0 - half), min(ny, y0 + half + 1)
    xlo, xhi = max(0, x0 - half), min(nx, x0 + half + 1)
    ys, xs = np.mgrid[ylo:yhi, xlo:xhi]
    return image[ylo:yhi, xlo:xhi], xs, ys


def overlap_fraction(shape, x_bin, y_bin, r_out_bin, r_in_bin, others):
    """Fraction of a donut's ring inside other donuts' outer disks (dimensionless).

    The same quantity as ``overlap_frac`` from `ring_metrics`, for another
    neighbour list. ``others`` rows are ``(x_bin, y_bin, r_out_bin)``.
    """
    if len(others) == 0:
        return 0.0
    half = int(np.ceil(1.05 * r_out_bin))
    _, xs, ys = _cutout(np.empty(shape, bool), x_bin, y_bin, half)
    r = np.hypot(xs - x_bin, ys - y_bin)
    ring = (r >= r_in_bin) & (r <= r_out_bin)
    near = others[np.hypot(others[:, 0] - x_bin, others[:, 1] - y_bin)
                  < r_out_bin + others[:, 2]]
    covered = np.zeros_like(ring)
    for xo, yo, ro in near:
        covered |= np.hypot(xs - xo, ys - yo) <= ro
    return float((covered & ring).sum() / max(ring.sum(), 1))


def annulus_model_residual(image, background, x_bin, y_bin, r_out_bin, r_in_bin):
    """Residual of an ideal uniform annulus fitted to one donut.

    The model is ``A`` inside ``r_in <= r <= r_out`` plus a constant, fitted by
    least squares over ``r <= 1.15 r_out``. The spider and vignetting are not in
    the model, so even a clean donut has a residual; the value ranks donuts.

    Returns
    -------
    resid_ratio : `float`
        RMS of the fit residual over ``A`` (dimensionless).
    level : `float`
        Fitted ring level ``A``, electrons per full-resolution pixel.
    """
    half = int(np.ceil(1.15 * r_out_bin))
    cut, xs, ys = _cutout(image - background, x_bin, y_bin, half)
    r = np.hypot(xs - x_bin, ys - y_bin)
    use = (r <= 1.15 * r_out_bin) & np.isfinite(cut)
    ring = ((r >= r_in_bin) & (r <= r_out_bin))[use].astype(float)
    design = np.column_stack([ring, np.ones_like(ring)])
    coeffs, *_ = np.linalg.lstsq(design, cut[use], rcond=None)
    if coeffs[0] <= 0:
        return np.nan, float(coeffs[0])
    resid = cut[use] - design @ coeffs
    return float(np.sqrt(np.mean(resid ** 2)) / coeffs[0]), float(coeffs[0])


def bleed_ratio(image, background, x_bin, y_bin, r_out_bin, ring_level):
    """Excess flux along the columns through a donut, over its ring level.

    Bleed trails run along detector columns (y) and are narrower than a binned
    pixel. Over the rows ``1.05 r_out < |y - y0| < 1.6 r_out`` above and below the
    donut, each column's median level is taken. The ratio is the largest column
    median within ``r_out`` of ``x0`` minus the median over all those columns, over
    ``ring_level``; about 0 without a trail.

    Returns
    -------
    ratio : `float`
        Dimensionless; NaN if the rows leave the image.
    column_max : `float`
        That largest column median, electrons per full-resolution pixel.
    """
    half = int(np.ceil(1.6 * r_out_bin))
    cut, xs, ys = _cutout(image - background, x_bin, y_bin, half)
    dy = np.abs(ys[:, 0] - y_bin)
    rows = (dy > 1.05 * r_out_bin) & (dy < 1.6 * r_out_bin)
    cols = np.abs(xs[0] - x_bin) < r_out_bin
    if rows.sum() < 5 or cols.sum() < 5 or not ring_level > 0:
        return np.nan, np.nan
    profile = np.nanmedian(cut[rows][:, cols], axis=0)
    top = float(np.nanmax(profile))
    return float((top - np.nanmedian(profile)) / ring_level), top
