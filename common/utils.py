"""
common/utils.py — Shared utility functions for rubin-work notebooks.

Add reusable functions here to avoid duplicating code across notebooks.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def setup_plotting(figsize=(10, 6), dpi=100, style='default'):
    """Configure matplotlib defaults for consistent plots across notebooks."""
    plt.style.use(style)
    plt.rcParams.update({
        'figure.figsize': figsize,
        'figure.dpi': dpi,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.constrained_layout.use': True,
    })


def detect_rsp_location():
    """Detect which Rubin Science Platform we are running on.

    Returns
    -------
    str
        'summit' if on the Summit RSP (/home/aroodman/),
        'usdf' if on the USDF RSP (/home/r/roodman/),
        'local' otherwise.
    """
    home = str(Path.home())
    if home.startswith('/home/aroodman'):
        return 'summit'
    elif '/roodman' in home and '/home/r/' in home:
        return 'usdf'
    else:
        return 'local'


def get_packages_dir(location=None):
    """Return the path to the user packages directory for the given RSP location.

    Parameters
    ----------
    location : str, optional
        'summit', 'usdf', or 'local'. If None, auto-detected via
        detect_rsp_location().

    Returns
    -------
    str
        Absolute path to the packages directory.

    Raises
    ------
    ValueError
        If location is 'local' (no RSP packages directory available).
    """
    if location is None:
        location = detect_rsp_location()

    packages_dirs = {
        'summit': '/home/aroodman/packages',
        'usdf': '/sdf/group/rubin/u/roodman/LSST/packages',
    }

    if location in packages_dirs:
        return packages_dirs[location]
    else:
        raise ValueError(
            f"No packages directory for location='{location}'. "
            f"Set ofc_config_dir manually. Known locations: {list(packages_dirs.keys())}"
        )


def repo_root(start=None):
    """Return the rubin-work repo root, found by walking up from `start`.

    Parameters
    ----------
    start : `str`, `pathlib.Path`, or `None`, optional
        Where to start walking up. Pass ``__file__`` from a script to get a
        location-independent answer. A directory works too. If None, falls back to
        the current working directory — the only option in a **notebook**, which has
        no ``__file__``.

    Returns
    -------
    root : `pathlib.Path`
        The repo root (the directory containing both ``CLAUDE.md`` and ``common/``).

    Raises
    ------
    `FileNotFoundError`
        If no ancestor of `start` looks like the repo root.

    Notes
    -----
    Scripts do not need this function: they cannot import it until the repo root is
    already on ``sys.path``, which is the chicken-and-egg this used to have. A script
    should use the two-line idiom instead (see the root CLAUDE.md, "Imports"):

        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    This function is for **notebooks**, and for code that already has `common` importable
    and wants the root path as a value (e.g. to build an output path).
    """
    import sys
    from pathlib import Path

    base = Path.cwd() if start is None else Path(start).resolve()
    if base.is_file():
        base = base.parent
    for parent in [base] + list(base.parents):
        # Require both markers: CLAUDE.md alone also matches topic dirs (aos/, guider/).
        if (parent / 'CLAUDE.md').exists() and (parent / 'common').is_dir():
            root = parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    raise FileNotFoundError(
        f"Could not find the rubin-work repo root above {base} "
        "(looking for a directory with both CLAUDE.md and common/)")


def add_repo_root_to_path():
    """Deprecated alias for repo_root(); returns the root as a str.

    Kept so existing callers keep working. Prefer repo_root(), which accepts a
    ``__file__`` and so does not depend on the current working directory.
    """
    return str(repo_root())


def nmad(x, min_n=3):
    """Normalized median absolute deviation — a robust standard-deviation estimate.

    ``1.4826 * median(|x - median(x)|)``, the 1.4826 factor making it consistent with
    the standard deviation for Gaussian data. Non-finite values are dropped.

    Parameters
    ----------
    x : `array_like`
        Values in any single unit; the result carries that same unit.
    min_n : `int`, optional
        Return NaN if fewer than this many finite values remain.

    Returns
    -------
    sigma : `float`
        Robust scatter, in the units of `x`, or NaN if under-determined.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < min_n:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def alt_to_deg(alt):
    """Express an altitude/elevation array in degrees, auto-detecting radian input.

    Rubin telemetry delivers altitude in radians from some sources and degrees from
    others. This applies the convention used across `rubin-work`: if the largest
    absolute value is below 2*pi, treat the input as radians and convert; otherwise
    assume it is already degrees.

    Parameters
    ----------
    alt : `array_like`
        Altitude/elevation, in radians or degrees.

    Returns
    -------
    alt_deg : `numpy.ndarray`
        Altitude, in degrees.

    Notes
    -----
    The detection is a heuristic on magnitude, and it has one real failure mode: an
    array of genuine **degrees that all happen to fall below 6.28 deg** is
    misidentified as radians and scaled by 180/pi (so 1-5 deg becomes 57-287 deg).
    Real Rubin altitudes are well above that, but do not reuse this for a quantity
    whose degree values can be small — pass explicit units instead.

    An all-NaN input returns all-NaN and emits numpy's "All-NaN slice" warning.
    """
    a = np.asarray(alt, dtype=float)
    if np.nanmax(np.abs(a)) < 2.0 * np.pi + 1e-3:
        return np.rad2deg(a)
    return a


#: Rubin Observatory on Cerro Pachon: geodetic latitude [deg], longitude [deg] east of
#: Greenwich, and height above the WGS84 ellipsoid [m]. The values `lsst.obs.lsst` carries for
#: the Simonyi Survey Telescope.
RUBIN_SITE_LAT_DEG = -30.244639
RUBIN_SITE_LON_DEG = -70.749417
RUBIN_SITE_HEIGHT_M = 2663.0


def rubin_site_location():
    """The Simonyi Survey Telescope site as an astropy location.

    Returns
    -------
    location : `astropy.coordinates.EarthLocation`
        Cerro Pachon, from `RUBIN_SITE_LAT_DEG`, `RUBIN_SITE_LON_DEG` and
        `RUBIN_SITE_HEIGHT_M`.

    Notes
    -----
    Built from stored constants rather than `EarthLocation.of_site`, which reaches the network
    for its site registry and so fails in a batch job with no outbound route.
    """
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    return EarthLocation(lat=RUBIN_SITE_LAT_DEG * u.deg, lon=RUBIN_SITE_LON_DEG * u.deg,
                         height=RUBIN_SITE_HEIGHT_M * u.m)


def sun_altitude_deg(mjd, location=None):
    """Apparent altitude of the Sun at a set of times.

    Parameters
    ----------
    mjd : `array_like`
        Modified Julian Date, UTC [d].
    location : `astropy.coordinates.EarthLocation`, optional
        Observing site. Defaults to `rubin_site_location`.

    Returns
    -------
    alt_deg : `numpy.ndarray`
        Sun altitude [deg], negative below the horizon.

    Notes
    -----
    Uses astropy's built-in low-precision solar ephemeris, good to about 0.01 deg, which is far
    inside what any twilight-referenced timing needs. No network and no JPL kernel.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, get_sun
    from astropy.time import Time
    loc = rubin_site_location() if location is None else location
    m = np.atleast_1d(np.asarray(mjd, dtype=float))
    out = np.full(m.shape, np.nan)
    ok = np.isfinite(m)
    if ok.any():
        t = Time(m[ok], format='mjd', scale='utc')
        out[ok] = get_sun(t).transform_to(AltAz(obstime=t, location=loc)).alt.to_value(u.deg)
    return out


def evening_twilight_mjd(day_obs, alt_deg=0.0, location=None):
    """When the Sun sets through a given altitude on the evening of each `day_obs`.

    Parameters
    ----------
    day_obs : `array_like`
        Observation night as the integer ``YYYYMMDD`` of its evening, the Rubin convention.
    alt_deg : `float`, optional
        Sun altitude defining the crossing [deg]. The default of 0 deg is geometric sunset,
        the "0 degree twilight" an observer refers to; -12 and -18 deg are nautical and
        astronomical twilight.
    location : `astropy.coordinates.EarthLocation`, optional
        Observing site. Defaults to `rubin_site_location`.

    Returns
    -------
    mjd : `numpy.ndarray`
        MJD of the evening descending crossing, one per input night [d]. NaN where no crossing
        is found, which at this latitude does not happen.

    Notes
    -----
    Solved by bisection on the sun altitude over the 15:00-03:00 local-clock window after the
    night's calendar date. That brackets the evening crossing year-round at Cerro Pachon with
    margin: geometric sunset runs from about 17:50 in June to about 20:15 in January, local
    standard time, and astronomical twilight at -18 deg trails it by up to about 1.6 h. The
    altitude falls monotonically from local mid-afternoon to local midnight, so 40 bisection
    steps converge to well under a second of time.

    A window that opened at 18:00 local would sit **below** geometric sunset in June and July
    and silently return the window edge, so the margin is load-bearing rather than cautious.

    Returned per night rather than per visit so a long visit list costs 147 ephemeris solves
    rather than 68000, and so every visit on a night shares one reference epoch.
    """
    import pandas as pd
    nights = np.atleast_1d(np.asarray(day_obs)).astype(int)
    uniq = np.unique(nights)
    # Local standard time at Cerro Pachon is UTC-4, so 15:00 local on the evening of `day_obs`
    # is 19:00 UTC of that same calendar date.
    start = (pd.to_datetime(uniq.astype(str), format='%Y%m%d').values.astype('datetime64[s]')
             .astype('float64') / 86400.0 + 40587.0)
    lo = start + 19.0 / 24.0
    hi = start + 31.0 / 24.0
    loc = rubin_site_location() if location is None else location
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        above = sun_altitude_deg(mid, location=loc) > alt_deg
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    solved = pd.Series(0.5 * (lo + hi), index=uniq)
    return solved.reindex(nights).to_numpy(float)


def fixed_width_edges(lo, hi, width):
    """Bin edges of fixed `width` spanning [lo, hi], aligned to multiples of width
    (bin *edges* fall on 0, width, 2*width, ...)."""
    start = np.floor(lo / width) * width
    stop = np.ceil(hi / width) * width
    return np.arange(start, stop + 0.5 * width, width)


def centered_edges(lo, hi, step):
    """Bin edges of width `step` whose bin *centers* fall on multiples of `step`
    (..., -step, 0, step, ...), covering [lo, hi].

    e.g. centered_edges(-60, 60, 15) -> centers at -60,-45,...,60 (edges at
    -67.5, -52.5, ..., 67.5).
    """
    k_lo = int(np.floor(lo / step + 0.5))
    k_hi = int(np.ceil(hi / step - 0.5))
    k_hi = max(k_hi, k_lo)
    centers = np.arange(k_lo, k_hi + 1) * step
    return np.concatenate([centers - 0.5 * step, [centers[-1] + 0.5 * step]])


def text_hist2d(x, y, *, ax=None, xbins=20, ybins=20, range=None, weights=None,
                fmt='{:.0f}', fontsize=7, text_color='black', min_count=1,
                grid=True, grid_color='0.8'):
    """ROOT 'TEXT'-style 2-D histogram: bin (x, y) and print the entry count at
    the center of each bin (no color fill), on a white background with a light
    dotted grid at the bin edges.

    Parameters
    ----------
    x, y : array-like
        Point coordinates to histogram.
    ax : matplotlib Axes, optional
        Target axes (default: current axes).
    xbins, ybins : int or sequence
        Bin count or explicit bin edges (as for numpy.histogram2d).  For fixed
        bin *width*, pass edges from :func:`fixed_width_edges`.
    range, weights : passed through to numpy.histogram2d.
    fmt : str
        Format for each printed value (default integer counts).
    min_count : float
        Only annotate bins with at least this value (default 1 = non-empty).

    Returns
    -------
    ax, H, xedges, yedges : the axes and the numpy.histogram2d result.
    """
    if ax is None:
        ax = plt.gca()
    H, xe, ye = np.histogram2d(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                               bins=[xbins, ybins], range=range, weights=weights)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    # np.argwhere (not the builtin range, which the `range` kwarg shadows here)
    for i, j in np.argwhere(H >= min_count):
        ax.text(xc[i], yc[j], fmt.format(H[i, j]), ha='center', va='center',
                fontsize=fontsize, color=text_color)
    if grid:
        ax.set_xticks(xe, minor=True)
        ax.set_yticks(ye, minor=True)
        ax.grid(which='minor', ls=':', lw=0.4, color=grid_color)
        ax.grid(which='major', ls=':', lw=0.6, color='0.6')
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    return ax, H, xe, ye
