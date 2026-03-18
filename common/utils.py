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
