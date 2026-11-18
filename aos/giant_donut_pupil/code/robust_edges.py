"""Robust donut edges against azimuth: now `rubinwork.giant_donuts.edge_profiles`.

Kept so this study's scripts keep their ``import robust_edges as rob``.
"""
from rubinwork.giant_donuts.edge_profiles import (  # noqa: F401
    NBIN,
    DR_PIX,
    MIN_TOP_PIX,
    MIN_TOP_SNR,
    WINDOW_PIX,
    M_MAX,
    CLIP_SIGMA,
    TOP_WINDOW,
    SMOOTH_STEPS,
    WIDE_WINDOW,
    profiles,
    _smooth,
    _crossings,
    bin_edges,
    harmonic_fit,
    robust_edges)
