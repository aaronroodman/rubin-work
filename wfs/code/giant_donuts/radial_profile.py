"""Radial flux profiles of giant donuts, intra against extra focal.

Josh's observation is that extra-focal giant donuts carry a ring of excess flux
that appears systematically. A turned-down edge on the inside of M1 is a candidate:
it is an optical path difference (OPD) error concentrated at the pupil's inner
boundary, and it moves flux in opposite radial directions on the two sides of
focus, so it shows as a ring on one side and a deficit on the other.

That hypothesis is separable from a pupil-model error by *where* and *how* the
feature sits:

* A **mask boundary** error moves the donut's edge. It is a step at the rim, its
  radius differs between the models, and it does not move flux across the
  boundary.
* An **OPD** error such as a turned-down edge redistributes flux smoothly across
  the boundary, conserving the total. It appears as a ring just inside or outside
  the edge with a compensating deficit on the other side, and it reverses sign
  between intra and extra.

So the diagnostic is the intra-minus-extra difference of the normalised radial
profile, not either profile alone.

The relevant geometry, traced through the full system on-axis: the surviving pupil
annulus runs 2.5580 to 4.1796 m in v3.14 and 2.5840 to 4.1650 m in v1000, where
the v1000 outer edge is set by the M1 baffles rather than M1's rim. The inner edge
is M1's inner radius in both. The inner edge is therefore at a normalised pupil
radius of about 0.612 (v3.14) or 0.620 (v1000), which is where a turned-down M1
inner edge would put its ring.

Import by putting this study's ``code/`` on the path, or the repository root::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path.cwd().parents[1] / 'wfs' / 'code' / 'giant_donuts'))
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

# Normalised pupil radius of the inner obscuration edge, per pupil model, taken
# from a full-system on-axis batoid trace rather than from the nominal
# `obscuration` value.
INNER_EDGE_NORM = {'v3.14': 0.6120, 'v1000': 0.6204}

__all__ = [
    'INNER_EDGE_NORM', 'RADIAL_BIN_PIX', 'AZIMUTHAL_BIN_DEG', 'SPIDER_WIDTH_M',
    'donut_centroid', 'radial_profile', 'normalise_profile',
    'profile_difference', 'ring_excess', 'azimuthal_profile',
    'bin_convergence',
]

# Bin sizes chosen by measuring where the features stop changing, not by rule of
# thumb; `bin_convergence` reproduces the scans these came from.
#
# Radial, 1.5 pixel per bin. Photon noise is irrelevant here -- even 1 pixel bins
# reach a signal-to-noise per bin of order 1000 -- so the limit is resolution.
# The sharpest radial feature is the intra-focal outer edge, whose 90 to 10 per
# cent roll-off spans 9.0 pixel, and the inner-edge ring excess converges to
# +0.136 (dimensionless) for bins of 1.5 pixel and finer. Coarser bins average
# the ring down: 2.15 pixel per bin reads +0.153, about 12 per cent high, and
# 3.58 pixel per bin collapses to +0.054 and mislocates the peak to the outer
# edge entirely.
RADIAL_BIN_PIX = 1.5

# Azimuthal, 0.25 deg per bin. The spider vanes are the sharpest azimuthal
# feature: a 0.05 m vane at 0.8 of the pupil radius subtends only 0.86 deg, so
# the 5 deg bins this study started with washed them out completely, and 1.0 deg
# resolves a vane with a single bin. 0.25 deg puts about 3.4 bins across a vane,
# which is what it takes to see its profile rather than just its presence.
# Peak-to-peak has converged by then -- 1.0150 at 0.25 deg against 1.0182 at
# 0.20 deg (dimensionless, extra-focal) -- while 122 pixels per bin still leave
# peak-to-peak over median-error at 82 (dimensionless), so the vanes are far
# above the noise.
AZIMUTHAL_BIN_DEG = 0.25

# Spider vane width, in meters, from policy/instruments/LsstCam.yaml.  Used to
# state the angular scale the azimuthal binning has to resolve.
SPIDER_WIDTH_M = 0.05


def donut_centroid(image, mask=None):
    """Flux-weighted centroid of a donut stamp.

    A donut's centroid is well defined despite the central hole, because the
    annulus is symmetric about it. Using the centroid rather than the stamp centre
    matters: an error in the centre leaks into the radial profile as a spurious
    broadening of both edges, which is exactly the signal being measured.

    Parameters
    ----------
    image : `numpy.ndarray`
        Donut stamp, background-subtracted, in arbitrary flux units.
    mask : `numpy.ndarray`, optional
        Boolean, True for pixels to use. Defaults to all finite pixels.

    Returns
    -------
    x0, y0 : `float`
        Centroid in pixel coordinates, in the stamp's own frame.
    """
    use = np.isfinite(image) if mask is None else (mask & np.isfinite(image))
    weight = np.where(use, image, 0.0)
    weight = np.clip(weight, 0.0, None)
    total = weight.sum()
    if total <= 0:
        ny, nx = image.shape
        return (nx - 1) / 2.0, (ny - 1) / 2.0

    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return float((weight * xx).sum() / total), float((weight * yy).sum() / total)


def radial_profile(image, center=None, n_bins=60, r_max_pix=None, mask=None):
    """Azimuthally averaged radial flux profile of a donut stamp.

    Parameters
    ----------
    image : `numpy.ndarray`
        Donut stamp, background-subtracted, in arbitrary flux units.
    center : `tuple` [`float`], optional
        ``(x0, y0)`` centre in pixels. Defaults to `donut_centroid`.
    n_bins : `int`, optional
        Number of radial bins.
    r_max_pix : `float`, optional
        Outer radius of the profile, in pixels. Defaults to the largest radius
        fully inside the stamp, so no bin is partially outside it.
    mask : `numpy.ndarray`, optional
        Boolean, True for pixels to use.

    Returns
    -------
    r_pix : `numpy.ndarray`
        Bin centre radii, in pixels.
    flux : `numpy.ndarray`
        Mean flux per pixel in each bin, in the image's flux units.
    flux_err : `numpy.ndarray`
        Standard error on the mean in each bin, same units. `numpy.nan` where a
        bin holds fewer than two pixels.
    n_pix : `numpy.ndarray`
        Pixel count per bin.
    """
    if center is None:
        center = donut_centroid(image, mask=mask)
    x0, y0 = center

    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - x0, yy - y0)

    if r_max_pix is None:
        r_max_pix = min(x0, y0, nx - 1 - x0, ny - 1 - y0)

    use = np.isfinite(image) & (r <= r_max_pix)
    if mask is not None:
        use &= mask

    edges = np.linspace(0.0, r_max_pix, n_bins + 1)
    which = np.digitize(r[use], edges) - 1
    values = image[use]

    flux = np.full(n_bins, np.nan)
    flux_err = np.full(n_bins, np.nan)
    n_pix = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = which == b
        n = int(np.count_nonzero(sel))
        n_pix[b] = n
        if n:
            flux[b] = float(values[sel].mean())
        if n > 1:
            flux_err[b] = float(values[sel].std(ddof=1) / np.sqrt(n))

    return 0.5 * (edges[:-1] + edges[1:]), flux, flux_err, n_pix


def normalise_profile(r_pix, flux, inner_edge_norm=None, outer_frac=0.98):
    """Put a profile on a normalised radius and unit mean flux.

    Comparing an intra against an extra donut requires removing the two trivial
    differences between them: the donuts have different sizes, and different total
    flux. Radius is scaled so the donut's outer edge is at 1.0, and flux is scaled
    to unit mean over the illuminated annulus. What survives is shape.

    The outer edge is located from the profile itself, as the steepest negative
    gradient in the outer half, rather than assumed from the defocus. That keeps
    the normalisation independent of the pupil model under test.

    Parameters
    ----------
    r_pix : `numpy.ndarray`
        Bin radii, in pixels.
    flux : `numpy.ndarray`
        Mean flux per bin, in arbitrary units.
    inner_edge_norm : `float`, optional
        Expected inner edge in normalised radius, used only to choose the annulus
        over which the flux is normalised. Defaults to the v3.14 value.
    outer_frac : `float`, optional
        Fraction of the fitted outer edge taken as the top of the normalising
        annulus, keeping the edge roll-off itself out of the normalisation.

    Returns
    -------
    r_norm : `numpy.ndarray`
        Radius normalised so the outer edge is 1.0, dimensionless.
    flux_norm : `numpy.ndarray`
        Flux normalised to unit mean over the annulus, dimensionless.
    r_edge_pix : `float`
        Fitted outer edge, in pixels.
    """
    if inner_edge_norm is None:
        inner_edge_norm = INNER_EDGE_NORM['v3.14']

    good = np.isfinite(flux)
    if good.sum() < 8:
        return np.full_like(r_pix, np.nan), np.full_like(flux, np.nan), np.nan

    half = len(r_pix) // 2
    gradient = np.gradient(np.where(good, flux, 0.0), r_pix)
    r_edge_pix = float(r_pix[half + int(np.argmin(gradient[half:]))])
    if not np.isfinite(r_edge_pix) or r_edge_pix <= 0:
        return np.full_like(r_pix, np.nan), np.full_like(flux, np.nan), np.nan

    r_norm = r_pix / r_edge_pix
    annulus = good & (r_norm > inner_edge_norm * 1.05) & (r_norm < outer_frac)
    scale = float(np.nanmean(flux[annulus])) if annulus.any() else np.nan
    if not np.isfinite(scale) or scale == 0:
        return r_norm, np.full_like(flux, np.nan), r_edge_pix
    return r_norm, flux / scale, r_edge_pix


def profile_difference(r_norm_intra, flux_intra, r_norm_extra, flux_extra,
                       n_grid=120, r_range=(0.3, 1.2)):
    """Intra minus extra normalised profile, on a common radius grid.

    This is the diagnostic. A pupil-model error shows as a narrow feature at the
    edges; an OPD error such as a turned-down M1 inner edge shows as a ring just
    inside or outside the inner boundary with a compensating deficit beside it,
    reversing sign between the two sides of focus.

    Parameters
    ----------
    r_norm_intra, r_norm_extra : `numpy.ndarray`
        Normalised radii, dimensionless.
    flux_intra, flux_extra : `numpy.ndarray`
        Normalised fluxes, dimensionless.
    n_grid : `int`, optional
        Points on the common grid.
    r_range : `tuple` [`float`], optional
        Normalised radius range of the grid.

    Returns
    -------
    r_grid : `numpy.ndarray`
        Common normalised radius, dimensionless.
    difference : `numpy.ndarray`
        Intra minus extra normalised flux, dimensionless.
    """
    r_grid = np.linspace(r_range[0], r_range[1], n_grid)

    def _interp(r, f):
        good = np.isfinite(r) & np.isfinite(f)
        if good.sum() < 4:
            return np.full(n_grid, np.nan)
        return np.interp(r_grid, r[good], f[good], left=np.nan, right=np.nan)

    return r_grid, _interp(r_norm_intra, flux_intra) - _interp(r_norm_extra, flux_extra)


def azimuthal_profile(image, center=None, n_bins=72, r_range_norm=(0.65, 0.95),
                      r_edge_pix=None, mask=None):
    """Azimuthal flux profile over an annulus, at fixed radius range.

    The radial profile averages over azimuth, so it dilutes anything localised:
    a figure error on one sector of M1, or the spider vanes, average away. This
    is the complementary cut, averaging over radius instead.

    Two features are expected and are worth telling apart. The **spiders** give
    narrow, deep, regularly spaced dips whose count and spacing are set by the
    vane geometry, and they sit at the same azimuth on both sides of focus. A
    **localised figure error** gives a broader modulation at one azimuth, and like
    any OPD term it should reverse sign between intra and extra.

    The annulus deliberately excludes both edges by default, so the profile is
    not dominated by edge roll-off or by a small centring error.

    Parameters
    ----------
    image : `numpy.ndarray`
        Donut stamp, background-subtracted, in arbitrary flux units.
    center : `tuple` [`float`], optional
        ``(x0, y0)`` centre in pixels. Defaults to `donut_centroid`.
    n_bins : `int`, optional
        Number of azimuthal bins over the full turn.
    r_range_norm : `tuple` [`float`], optional
        Radial range of the annulus, as a fraction of the outer edge,
        dimensionless.
    r_edge_pix : `float`, optional
        Outer edge in pixels, as returned by `normalise_profile`. Required to
        interpret `r_range_norm`; without it the largest radius fully inside the
        stamp is used, which is only correct if the donut fills the stamp.
    mask : `numpy.ndarray`, optional
        Boolean, True for pixels to use.

    Returns
    -------
    angle_deg : `numpy.ndarray`
        Bin centre azimuth, in degrees, measured counterclockwise from the +x
        pixel axis.
    flux : `numpy.ndarray`
        Mean flux per pixel in each bin, in the image's flux units.
    flux_err : `numpy.ndarray`
        Standard error on the mean in each bin, same units. `numpy.nan` where a
        bin holds fewer than two pixels.
    n_pix : `numpy.ndarray`
        Pixel count per bin.
    """
    if center is None:
        center = donut_centroid(image, mask=mask)
    x0, y0 = center

    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - x0, yy - y0)

    if r_edge_pix is None or not np.isfinite(r_edge_pix) or r_edge_pix <= 0:
        r_edge_pix = min(x0, y0, nx - 1 - x0, ny - 1 - y0)

    use = (np.isfinite(image)
           & (r >= r_range_norm[0] * r_edge_pix)
           & (r <= r_range_norm[1] * r_edge_pix))
    if mask is not None:
        use &= mask

    angle = np.degrees(np.arctan2(yy - y0, xx - x0)) % 360.0
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    which = np.digitize(angle[use], edges) - 1
    values = image[use]

    flux = np.full(n_bins, np.nan)
    flux_err = np.full(n_bins, np.nan)
    n_pix = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = which == b
        n = int(np.count_nonzero(sel))
        n_pix[b] = n
        if n:
            flux[b] = float(values[sel].mean())
        if n > 1:
            flux_err[b] = float(values[sel].std(ddof=1) / np.sqrt(n))

    return 0.5 * (edges[:-1] + edges[1:]), flux, flux_err, n_pix


def bin_convergence(image, kind='radial', n_bins_grid=None, r_edge_pix=None,
                    r_range_norm=(0.65, 0.95), zone=(0.62, 0.70)):
    """Scan bin count to find where a profile's features stop changing.

    Both profiles here are resolution-limited rather than noise-limited, so "more
    bins" is not automatically worse and "fewer bins" is not automatically safer.
    The right bin size is the largest one that does not yet distort the feature
    being measured, and this finds it by measuring rather than asserting.

    Two statistics do the work:

    * ``amplitude`` -- the feature's measured size. Too-coarse bins average it
      down, so it rises as bins shrink and then plateaus. The plateau is the
      answer.
    * ``autocorr`` -- lag-1 autocorrelation of the profile about its median,
      dimensionless. Negative means oversmoothed, with adjacent bins
      anti-correlated. Rising through about 0.5 to 0.7 means neighbouring bins
      are tracking the same real feature. Saturating toward 1.0 means the curve
      is already resolved and further bins only subdivide it. **Only meaningful
      for the azimuthal profile**: the radial profile is a smooth monotonic
      curve, so adjacent bins correlate at better than 0.98 regardless of bin
      width and the statistic cannot discriminate.
    * ``edge_width`` -- for the radial profile instead, the 90 to 10 per cent
      roll-off width of the outer edge in pixels. This is the sharpest radial
      feature, so it is what sets the resolution limit: as bins shrink the
      measured width falls to the true value and then stops. `numpy.nan` for the
      azimuthal profile.

    Parameters
    ----------
    image : `numpy.ndarray`
        Donut stamp, background-subtracted, in arbitrary flux units.
    kind : {'radial', 'azimuthal'}, optional
        Which profile to scan.
    n_bins_grid : `sequence` [`int`], optional
        Bin counts to try. Defaults span the useful range for each kind.
    r_edge_pix : `float`, optional
        Outer edge in pixels, required for ``'azimuthal'``.
    r_range_norm : `tuple` [`float`], optional
        Annulus for the azimuthal profile, dimensionless.
    zone : `tuple` [`float`], optional
        Normalised-radius zone whose mean flux is the radial amplitude statistic.

    Returns
    -------
    report : `list` [`dict`]
        Per bin count: ``n_bins``, ``bin_size`` (pixels for radial, degrees for
        azimuthal), ``amplitude`` (dimensionless), ``median_err``
        (dimensionless), ``autocorr`` (dimensionless), ``edge_width`` (pixels,
        radial only).
    """
    report = []
    if kind == 'radial':
        grid = n_bins_grid or (80, 120, 172, 200, 286, 430, 860)
    else:
        grid = n_bins_grid or (72, 180, 360, 720, 1440, 1800, 2880)

    for n_bins in grid:
        if kind == 'radial':
            r_pix, flux, flux_err, _ = radial_profile(image, n_bins=n_bins)
            r_norm, flux_norm, edge = normalise_profile(r_pix, flux)
            if not np.isfinite(edge):
                continue
            sel = np.isfinite(flux_norm) & (r_norm >= zone[0]) & (r_norm < zone[1])
            amplitude = float(np.mean(flux_norm[sel])) if sel.any() else np.nan
            err_norm = flux_err / np.nanmean(flux[np.isfinite(flux)])
            bin_size = (image.shape[0] / 2.0) / n_bins
            values = flux_norm

            # 90 to 10 per cent roll-off width of the outer edge, the sharpest
            # radial feature and so the one that sets the resolution limit.
            near_edge = (np.isfinite(flux_norm) & (r_norm > 0.90)
                         & (r_norm < 1.10))
            edge_width = np.nan
            if near_edge.sum() > 3:
                rr, ff = r_norm[near_edge], flux_norm[near_edge]
                r90 = rr[np.argmin(np.abs(ff - 0.9))]
                r10 = rr[np.argmin(np.abs(ff - 0.1))]
                edge_width = float(abs(r10 - r90) * edge)
        else:
            _, flux, flux_err, _ = azimuthal_profile(
                image, n_bins=n_bins, r_range_norm=r_range_norm,
                r_edge_pix=r_edge_pix)
            scale = np.nanmean(flux)
            values = flux / scale
            err_norm = flux_err / scale
            amplitude = float(np.nanmax(values) - np.nanmin(values))
            bin_size = 360.0 / n_bins
            edge_width = np.nan

        deviation = values - np.nanmedian(values)
        good = np.isfinite(deviation)
        autocorr = np.nan
        if good.sum() > 8:
            a, b = deviation[good][:-1], deviation[good][1:]
            if a.std() > 0 and b.std() > 0:
                autocorr = float(np.corrcoef(a, b)[0, 1])

        report.append({
            'n_bins': int(n_bins), 'bin_size': float(bin_size),
            'amplitude': amplitude,
            'median_err': float(np.nanmedian(err_norm)),
            'autocorr': autocorr,
            'edge_width': edge_width,
        })
    return report


def ring_excess(r_norm, flux_norm, edge_norm, width=0.06, baseline_gap=0.02,
                baseline_width=0.12):
    """Flux excess in a band just outside a pupil boundary, over a local baseline.

    Quantifies Josh's ring as one number per donut, so it can be compared between
    the two sides of focus and between pupil models.

    The baseline is taken **further out in the illuminated annulus**, not just
    inside the boundary: inside the inner edge lies the central hole, so
    differencing across the boundary would measure the hole-to-annulus step —
    which is present whether or not there is a ring — rather than the ring itself.
    A positive value therefore means flux piled up against the boundary relative
    to the flat part of the annulus.

    Parameters
    ----------
    r_norm : `numpy.ndarray`
        Normalised radius, dimensionless.
    flux_norm : `numpy.ndarray`
        Normalised flux, dimensionless.
    edge_norm : `float`
        Normalised radius of the boundary, e.g. `INNER_EDGE_NORM`.
    width : `float`, optional
        Width of the ring band outside the boundary, in normalised radius.
    baseline_gap : `float`, optional
        Gap left between the ring band and the baseline band, in normalised
        radius, so a broad ring does not contaminate its own reference.
    baseline_width : `float`, optional
        Width of the baseline band, in normalised radius.

    Returns
    -------
    excess : `float`
        Mean normalised flux in the ring band minus the local baseline,
        dimensionless.  `numpy.nan` if either band is empty.
    """
    ring_lo = edge_norm
    ring_hi = edge_norm + width
    base_lo = ring_hi + baseline_gap
    base_hi = base_lo + baseline_width

    finite = np.isfinite(flux_norm)
    ring = finite & (r_norm >= ring_lo) & (r_norm < ring_hi)
    base = finite & (r_norm >= base_lo) & (r_norm < base_hi)
    if not ring.any() or not base.any():
        return np.nan
    return float(np.mean(flux_norm[ring]) - np.mean(flux_norm[base]))
