"""
common/utils.py — Shared utility functions for rubin-work notebooks.

Add reusable functions here to avoid duplicating code across notebooks.
"""

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
