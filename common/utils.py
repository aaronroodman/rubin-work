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
        'usdf': '/home/r/roodman/u/LSST/packages',
    }

    if location in packages_dirs:
        return packages_dirs[location]
    else:
        raise ValueError(
            f"No packages directory for location='{location}'. "
            f"Set ofc_config_dir manually. Known locations: {list(packages_dirs.keys())}"
        )


def add_repo_root_to_path():
    """Add the repository root to sys.path so 'from common import ...' works."""
    import sys
    from pathlib import Path
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / 'CLAUDE.md').exists():
            repo_root = str(parent)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return repo_root
    raise FileNotFoundError("Could not find repository root (looking for CLAUDE.md)")


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
