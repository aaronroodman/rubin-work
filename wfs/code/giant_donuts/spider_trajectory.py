"""Locate the spider shadows in a giant donut and profile along them.

The azimuthal projection in `radial_profile` averages a fixed annulus over
radius, which is the wrong cut for the spiders. The vanes run from M2's support
ring out to the top-end ring, so in the pupil each shadow is a **chord offset
from the pupil centre**, not a radial spoke. Its azimuth therefore drifts with
radius: a vane whose line of centres misses the pupil centre by an impact
parameter ``b`` sits at azimuth ``phi(r) = phi_0 + arcsin(b / r)``, which moves
by several degrees across the illuminated annulus. Averaging a 0.30-wide
annulus over radius smears a 0.86 deg vane across that drift, so the measured
dip is shallower than the true one and the shallowness is an artifact of the
projection rather than a statement about the optical model.

This module takes the other route: find each shadow's centre at several narrow
radii, fit the straight line those centres lie on, and then profile *along* that
line. The vane depth measured this way is the real one, and the fitted lines
themselves are the useful output -- comparing the data's eight lines against a
model's says whether the model puts the spiders in the right place, and if not,
whether the error is a rotation (``phi_0``) or a radial offset (``b``).

Pixel convention follows `radial_profile`: azimuth is degrees counterclockwise
from the +x pixel axis, and radii are normalised by the fitted outer edge.

Validated on a synthetic annular pupil with four opaque vanes of known impact
parameter and width, by ``test_spider_trajectory.py``: the depth comes back at
1.0000 (dimensionless) exactly, the impact parameter to within 1.6 pixels of
truth, and the width to about 7 per cent. The small impact bias comes from
centroiding dips at trace radii near the annulus edges and sets the floor on
what a data-minus-model placement difference can be believed at.

Import by putting this study's ``code/`` on the path::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path.cwd().parents[1] / 'wfs' / 'code' / 'giant_donuts'))
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

# LSSTCam has four vanes, and each one crosses the pupil, so eight shadows
# appear in the donut -- the count the detector logic expects to find.
N_SPIDER_SHADOWS = 8

# Radii at which shadow centres are measured, as a fraction of the fitted outer
# edge. Spaced over the illuminated annulus but clear of both edges, where the
# roll-off would bias a dip centroid.
#
# The impact parameter is the fitted line's perpendicular distance from the
# pupil centre, so an error in the line's direction is levered by the roughly
# 250 pixel distance from the traced points to the centre: 0.4 deg of direction
# error is 1.7 pixels of impact error. On the synthetic pupil the direction
# comes back about 0.2 to 0.4 deg high in *both* vane families, which is a
# common rotation rather than a per-shadow scatter -- it comes from centroiding
# a dip over an annulus whose illumination is not flat. Because it is common it
# cancels in the data-minus-model differences this module is used for, so the
# span is kept clear of the edge roll-off rather than widened to chase it:
# pushing the outermost radius to 0.96 made the bias worse, not better.
TRACE_RADII_NORM = (0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92)

# Width of each tracing annulus, in pixels. 5 pixels is narrow enough that the
# azimuthal drift within one annulus is far below a vane width -- at r = 0.68 of
# a 347 pixel edge the drift over 5 pixels is under 0.1 deg -- while still
# leaving about 20 pixels per 0.25 deg azimuthal bin.
TRACE_ANNULUS_PIX = 5.0

# Azimuthal bin size for the tracing profiles, in degrees. Finer than the
# `radial_profile` choice because here the profile is only used to centroid a
# dip, and a narrow annulus has no radial smearing to hide behind.
TRACE_BIN_DEG = 0.20

# Half-width of the window, in degrees, that a dip centroid is computed over,
# measured about the local minimum. About 1.5 vane widths, so the window spans
# the shadow and a little of the shoulder on each side without reaching the
# neighbouring vane.
DIP_WINDOW_DEG = 1.5

# Half-width of the cross-vane profile, in pixels. The profile runs
# perpendicular to the fitted trajectory, so this sets how much unobscured pupil
# is shown on either side of the shadow. The measured shadows are about 9 pixels
# wide extra-focally, so 20 pixels leaves a clear shoulder on both sides to
# reference the depth against.
CROSS_HALF_PIX = 20.0

# Band of |perpendicular offset|, in pixels, taken as unobscured pupil and used
# as the depth reference. Starts outside the widest measured shadow.
SHOULDER_PIX = (10.0, 20.0)

__all__ = [
    'N_SPIDER_SHADOWS', 'TRACE_RADII_NORM', 'TRACE_ANNULUS_PIX',
    'TRACE_BIN_DEG', 'DIP_WINDOW_DEG', 'CROSS_HALF_PIX', 'SHOULDER_PIX',
    'trace_shadow_centres', 'fit_trajectories', 'trajectory_profile',
    'vane_depth',
]


def _annulus_azimuthal(image, center, r_mid_pix, width_pix, bin_deg):
    """Azimuthal mean flux in one narrow annulus.

    Parameters
    ----------
    image : `numpy.ndarray`
        Donut stamp, background-subtracted, in arbitrary flux units.
    center : `tuple` [`float`]
        ``(x0, y0)`` donut centre, in pixels.
    r_mid_pix : `float`
        Annulus mid-radius, in pixels.
    width_pix : `float`
        Full radial width of the annulus, in pixels.
    bin_deg : `float`
        Azimuthal bin size, in degrees.

    Returns
    -------
    angle_deg : `numpy.ndarray`
        Bin centre azimuth, in degrees counterclockwise from +x.
    flux : `numpy.ndarray`
        Mean flux per pixel in each bin, in the image's units. `numpy.nan` where
        a bin is empty.
    """
    x0, y0 = center
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - x0, yy - y0)

    use = (np.isfinite(image) & (r >= r_mid_pix - 0.5 * width_pix)
           & (r < r_mid_pix + 0.5 * width_pix))
    angle = np.degrees(np.arctan2(yy - y0, xx - x0)) % 360.0

    n_bins = int(round(360.0 / bin_deg))
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    which = np.digitize(angle[use], edges) - 1
    values = image[use]

    flux = np.full(n_bins, np.nan)
    total = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=int)
    np.add.at(total, which, values)
    np.add.at(count, which, 1)
    good = count > 0
    flux[good] = total[good] / count[good]
    return 0.5 * (edges[:-1] + edges[1:]), flux


def _dip_azimuths(angle_deg, flux, n_dips=N_SPIDER_SHADOWS,
                  window_deg=DIP_WINDOW_DEG):
    """Azimuths of the `n_dips` deepest, well-separated minima in a profile.

    Dips are found on the profile divided by a wide running median, so a slow
    illumination gradient across the annulus does not make one side's vanes look
    deeper than the other's. Each dip's azimuth is the flux-deficit-weighted
    centroid over a window about its local minimum, which is less sensitive to
    photon noise than the single deepest bin.

    Parameters
    ----------
    angle_deg : `numpy.ndarray`
        Bin centre azimuth, in degrees.
    flux : `numpy.ndarray`
        Mean flux per bin, in arbitrary units.
    n_dips : `int`, optional
        Number of dips to return.
    window_deg : `float`, optional
        Centroid half-window about each local minimum, in degrees.

    Returns
    -------
    azimuth_deg : `numpy.ndarray`
        Dip azimuths, in degrees, sorted ascending. Shorter than `n_dips` if
        fewer separated minima were found.
    depth : `numpy.ndarray`
        Fractional flux deficit at each dip, dimensionless: one minus the
        minimum of the flattened profile.
    """
    n = len(flux)
    bin_deg = 360.0 / n

    # Flatten with a wide circular running median: wide compared with a vane so
    # the vane itself does not enter its own baseline.
    half = max(1, int(round(10.0 / bin_deg)))
    padded = np.concatenate([flux[-half:], flux, flux[:half]])
    baseline = np.array([np.nanmedian(padded[i:i + 2 * half + 1])
                         for i in range(n)])
    with np.errstate(invalid='ignore', divide='ignore'):
        flat = flux / baseline

    # Exclusion radius: a vane subtends under 1 deg, and the shadows are 45 deg
    # apart, so 5 deg keeps one dip per vane without risking a merge.
    exclude = max(1, int(round(5.0 / bin_deg)))
    work = flat.copy()
    work[~np.isfinite(work)] = np.inf

    azimuths = []
    depths = []
    for _ in range(n_dips):
        i = int(np.argmin(work))
        if not np.isfinite(work[i]) or work[i] >= 1.0:
            break

        # Deficit-weighted centroid over the window, with the angles unwrapped
        # about the minimum so a dip near 0 deg is not split.
        w_half = max(1, int(round(window_deg / bin_deg)))
        idx = (np.arange(i - w_half, i + w_half + 1)) % n
        deficit = np.clip(1.0 - flat[idx], 0.0, None)
        deficit[~np.isfinite(deficit)] = 0.0
        offsets = np.arange(-w_half, w_half + 1) * bin_deg
        if deficit.sum() > 0:
            centre = angle_deg[i] + float((deficit * offsets).sum()
                                          / deficit.sum())
        else:
            centre = angle_deg[i]

        azimuths.append(centre % 360.0)
        depths.append(float(1.0 - flat[i]))
        work[(np.arange(i - exclude, i + exclude + 1)) % n] = np.inf

    order = np.argsort(azimuths)
    return np.asarray(azimuths)[order], np.asarray(depths)[order]


def trace_shadow_centres(image, center, r_edge_pix,
                         radii_norm=TRACE_RADII_NORM,
                         width_pix=TRACE_ANNULUS_PIX, bin_deg=TRACE_BIN_DEG,
                         n_shadows=N_SPIDER_SHADOWS):
    """Pixel positions of the spider shadows at several radii.

    Step 1 of the trajectory method: at each of a few narrow radii, build an
    azimuthal profile and centroid the eight deepest dips. Because the annulus
    is only a few pixels wide, the vane's azimuthal drift with radius does not
    smear the dip, so its centre is measured rather than averaged away.

    Shadows are matched across radii by azimuth, using the innermost traced
    radius as the reference. That works here because the drift between adjacent
    traced radii is a few degrees at most, far below the 45 deg shadow spacing.

    Parameters
    ----------
    image : `numpy.ndarray`
        Donut stamp, background-subtracted, in arbitrary flux units.
    center : `tuple` [`float`]
        ``(x0, y0)`` donut centre, in pixels.
    r_edge_pix : `float`
        Fitted outer edge, in pixels, as returned by
        `radial_profile.normalise_profile`.
    radii_norm : `tuple` [`float`], optional
        Trace radii, as a fraction of the outer edge, dimensionless.
    width_pix : `float`, optional
        Radial width of each tracing annulus, in pixels.
    bin_deg : `float`, optional
        Azimuthal bin size for the tracing profiles, in degrees.
    n_shadows : `int`, optional
        Number of shadows expected.

    Returns
    -------
    centres : `numpy.ndarray`
        Shape ``(n_shadows, len(radii_norm), 2)``, the ``(x, y)`` pixel position
        of each shadow at each radius. `numpy.nan` where a shadow was not found.
    azimuth_deg : `numpy.ndarray`
        Shape ``(n_shadows, len(radii_norm))``, the shadow azimuths in degrees.
    depth : `numpy.ndarray`
        Shape ``(n_shadows, len(radii_norm))``, the fractional flux deficit at
        each dip, dimensionless.
    """
    x0, y0 = center
    n_r = len(radii_norm)
    azimuth = np.full((n_shadows, n_r), np.nan)
    depth = np.full((n_shadows, n_r), np.nan)

    reference = None
    for j, r_norm in enumerate(radii_norm):
        angle_deg, flux = _annulus_azimuthal(
            image, center, r_norm * r_edge_pix, width_pix, bin_deg)
        found, found_depth = _dip_azimuths(angle_deg, flux, n_dips=n_shadows)
        if len(found) == 0:
            continue

        if reference is None:
            # First traced radius defines shadow identity and ordering.
            azimuth[:len(found), j] = found
            depth[:len(found), j] = found_depth
            reference = np.array(found, dtype=float)
            continue

        # Match to the reference by smallest circular azimuth separation.
        for a, d in zip(found, found_depth):
            separation = np.abs((a - reference + 180.0) % 360.0 - 180.0)
            k = int(np.nanargmin(separation))
            if separation[k] < 10.0 and not np.isfinite(azimuth[k, j]):
                azimuth[k, j] = a
                depth[k, j] = d

    radii_pix = np.asarray(radii_norm, dtype=float) * r_edge_pix
    centres = np.stack([
        x0 + radii_pix[None, :] * np.cos(np.deg2rad(azimuth)),
        y0 + radii_pix[None, :] * np.sin(np.deg2rad(azimuth)),
    ], axis=-1)
    return centres, azimuth, depth


def fit_trajectories(centres, center):
    """Fit a straight line through each shadow's traced centres.

    Step 2: the physical vane is straight, so its shadow is a straight chord in
    the pupil. Fitting that line gives two numbers with direct optical meaning --
    the line's direction angle, which a camera-rotation or vane-azimuth error
    moves, and its **impact parameter**, the signed perpendicular distance from
    the donut centre, which is where the vane's offset from the optical axis
    shows up. A pure spoke model would give an impact parameter of zero.

    The fit is a total-least-squares (principal-axis) line rather than a
    ``y`` on ``x`` regression, because the traced points can run in any direction
    and a near-vertical set would blow up an ordinary least-squares slope.

    Parameters
    ----------
    centres : `numpy.ndarray`
        Shape ``(n_shadows, n_radii, 2)``, traced ``(x, y)`` pixel positions, as
        returned by `trace_shadow_centres`.
    center : `tuple` [`float`]
        ``(x0, y0)`` donut centre, in pixels.

    Returns
    -------
    trajectories : `list` [`dict`]
        One entry per shadow, each holding:

        ``'point'``
            ``(x, y)`` centroid of the traced points, in pixels (`tuple`).
        ``'direction'``
            Unit vector along the fitted line (`numpy.ndarray`).
        ``'angle_deg'``
            Direction angle, in degrees counterclockwise from +x, folded onto
            ``[0, 180)`` since a line has no sense (`float`). All eight shadows
            share only two values of this, one per vane pair, so use
            ``'azimuth_deg'`` to tell the shadows apart.
        ``'azimuth_deg'``
            Azimuth of the traced points' centroid, in degrees counterclockwise
            from +x, which is the shadow's own position in the donut and is what
            labels it (`float`).
        ``'impact_pix'``
            Signed perpendicular distance from the donut centre to the line, in
            pixels. Positive when the centre lies to the left of the direction
            vector (`float`).
        ``'rms_pix'``
            Perpendicular scatter of the traced points about the line, in
            pixels; a straightness check (`float`).
        ``'n_points'``
            Number of traced points used (`int`).
    """
    x0, y0 = center
    trajectories = []
    for pts in centres:
        good = np.isfinite(pts).all(axis=1)
        if good.sum() < 2:
            trajectories.append(dict(point=(np.nan, np.nan),
                                     direction=np.array([np.nan, np.nan]),
                                     angle_deg=np.nan, azimuth_deg=np.nan,
                                     impact_pix=np.nan,
                                     rms_pix=np.nan, n_points=int(good.sum())))
            continue

        p = pts[good]
        mean = p.mean(axis=0)
        # Principal axis of the traced points: the direction of largest spread.
        _, _, vt = np.linalg.svd(p - mean, full_matrices=False)
        direction = vt[0] / np.linalg.norm(vt[0])
        normal = np.array([-direction[1], direction[0]])

        offsets = (p - mean) @ normal
        impact = float((np.array([x0, y0]) - mean) @ normal)

        trajectories.append(dict(
            point=(float(mean[0]), float(mean[1])),
            direction=direction,
            angle_deg=float(np.degrees(np.arctan2(direction[1],
                                                  direction[0])) % 180.0),
            azimuth_deg=float(np.degrees(np.arctan2(mean[1] - y0,
                                                    mean[0] - x0)) % 360.0),
            impact_pix=impact,
            rms_pix=float(np.sqrt(np.mean(offsets ** 2))),
            n_points=int(good.sum()),
        ))
    return trajectories


def trajectory_profile(image, trajectory, center, r_edge_pix,
                       r_range_norm=(0.66, 0.94), cross_half_pix=CROSS_HALF_PIX,
                       n_cross=81):
    """Flux profile across a spider shadow, perpendicular to its trajectory.

    Step 3: having the shadow's own line, the vane profile can be built without
    any radial smearing. Each pixel inside the annulus is assigned its signed
    perpendicular distance from the line, and the flux is averaged in bins of
    that distance. Because the line follows the shadow, every pixel in a bin
    sees the same part of the vane, so the dip reaches its true depth instead of
    a radius-averaged one.

    Parameters
    ----------
    image : `numpy.ndarray`
        Donut stamp, background-subtracted, in arbitrary flux units.
    trajectory : `dict`
        One entry from `fit_trajectories`.
    center : `tuple` [`float`]
        ``(x0, y0)`` donut centre, in pixels.
    r_edge_pix : `float`
        Fitted outer edge, in pixels.
    r_range_norm : `tuple` [`float`], optional
        Radial range to average over, as a fraction of the outer edge,
        dimensionless. Excludes both pupil edges.
    cross_half_pix : `float`, optional
        Half-width of the profile, in pixels.
    n_cross : `int`, optional
        Number of bins across the full width. Odd, so one bin is centred on the
        line itself.

    Returns
    -------
    offset_pix : `numpy.ndarray`
        Bin centre perpendicular offset from the fitted line, in pixels.
        Positive on the side the line's normal points to.
    flux : `numpy.ndarray`
        Mean flux per pixel in each bin, in the image's units. `numpy.nan` where
        a bin is empty.
    flux_err : `numpy.ndarray`
        Standard error on the mean, same units.
    n_pix : `numpy.ndarray`
        Pixel count per bin.
    """
    if not np.isfinite(trajectory['impact_pix']):
        empty = np.full(n_cross, np.nan)
        return empty.copy(), empty.copy(), empty.copy(), np.zeros(n_cross, int)

    x0, y0 = center
    px, py = trajectory['point']
    direction = trajectory['direction']
    normal = np.array([-direction[1], direction[0]])

    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - x0, yy - y0)
    perp = (xx - px) * normal[0] + (yy - py) * normal[1]
    along = (xx - px) * direction[0] + (yy - py) * direction[1]

    # Only the half of the chord this shadow occupies: the fitted line is
    # infinite and crosses the pupil twice, so the far crossing -- which is a
    # different vane's shadow -- has to be excluded.
    sign = np.sign(np.dot(np.array([px - x0, py - y0]), direction))
    if sign == 0:
        sign = 1.0

    use = (np.isfinite(image) & (np.abs(perp) <= cross_half_pix)
           & (r >= r_range_norm[0] * r_edge_pix)
           & (r <= r_range_norm[1] * r_edge_pix)
           & (sign * along >= -0.25 * r_edge_pix))

    edges = np.linspace(-cross_half_pix, cross_half_pix, n_cross + 1)
    which = np.digitize(perp[use], edges) - 1
    values = image[use]

    flux = np.full(n_cross, np.nan)
    flux_err = np.full(n_cross, np.nan)
    n_pix = np.zeros(n_cross, dtype=int)
    for b in range(n_cross):
        sel = which == b
        n = int(np.count_nonzero(sel))
        n_pix[b] = n
        if n:
            flux[b] = float(values[sel].mean())
        if n > 1:
            flux_err[b] = float(values[sel].std(ddof=1) / np.sqrt(n))

    return 0.5 * (edges[:-1] + edges[1:]), flux, flux_err, n_pix


def vane_depth(offset_pix, flux, shoulder_pix=SHOULDER_PIX):
    """Depth and width of a vane shadow, from a cross-vane profile.

    The depth is referenced to the unobscured shoulders on both sides rather
    than to the profile median, so it is the fraction of the local pupil
    illumination that the vane removes. A fully opaque, fully resolved vane
    would give 1.0; anything less is blur, diffraction, or a vane narrower than
    the model's.

    Parameters
    ----------
    offset_pix : `numpy.ndarray`
        Perpendicular offset from the fitted line, in pixels.
    flux : `numpy.ndarray`
        Mean flux per bin, in arbitrary units.
    shoulder_pix : `tuple` [`float`], optional
        Inner and outer edge of the shoulder band, in pixels of ``|offset|``,
        used as the unobscured reference level.

    Returns
    -------
    depth : `float`
        Fractional flux deficit at the shadow's deepest point, dimensionless:
        one minus minimum over shoulder level.
    fwhm_pix : `float`
        Full width at half the deficit, in pixels, by linear interpolation of
        the profile. `numpy.nan` if the half-deficit level is not crossed on
        both sides.
    shoulder : `float`
        Shoulder flux level, in the profile's units.
    """
    absolute = np.abs(offset_pix)
    band = (np.isfinite(flux) & (absolute >= shoulder_pix[0])
            & (absolute <= shoulder_pix[1]))
    if not band.any():
        return np.nan, np.nan, np.nan
    shoulder = float(np.nanmedian(flux[band]))
    if not np.isfinite(shoulder) or shoulder <= 0:
        return np.nan, np.nan, shoulder

    core = np.isfinite(flux) & (absolute <= shoulder_pix[0])
    if not core.any():
        return np.nan, np.nan, shoulder
    minimum = float(np.nanmin(flux[core]))
    depth = 1.0 - minimum / shoulder

    # FWHM at half the deficit, from the outermost half-level crossing on each
    # side of the minimum. Taken as a crossing of the interpolated profile
    # rather than by walking outward bin by bin: the shadow floor is flat and
    # noisy, so a walk can stop on a single bin that fluctuates above the half
    # level and report a width far too small, or never stop at all.
    half_level = shoulder - 0.5 * (shoulder - minimum)
    i_min = int(np.nanargmin(np.where(core, flux, np.inf)))

    def _crossing(indices):
        """Interpolated offset of the first half-level crossing along `indices`.

        Empty bins are dropped rather than breaking the scan, so the crossing is
        bracketed by the nearest populated bins on either side of it.
        """
        kept = [i for i in indices if np.isfinite(flux[i])]
        for i, j in zip(kept[:-1], kept[1:]):
            f0, f1 = flux[i], flux[j]
            if f0 == f1:
                continue
            if (f0 - half_level) * (f1 - half_level) <= 0:
                t = (half_level - f0) / (f1 - f0)
                return offset_pix[i] + t * (offset_pix[j] - offset_pix[i])
        return np.nan

    # Scanned inward from each end rather than outward from the minimum: the
    # shadow floor is flat and noisy, so a walk outward can stop on a single
    # bin that fluctuates above the half level and report far too small a width.
    left = _crossing(list(range(0, i_min + 1)))
    right = _crossing(list(range(len(flux) - 1, i_min - 1, -1)))
    fwhm = float(abs(right - left)) if np.isfinite([left, right]).all() else np.nan
    return float(depth), fwhm, shoulder
