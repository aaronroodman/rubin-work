"""Wedge geometry of the WFS mimic: four annular wedges of FAM donuts at the corner-sensor
radius, centred ``delta + [0, 90, 180, 270]`` deg, each reduced to a median deviation.

Shared by the ``aos/wfs_mimic`` study (the covariance), ``aos/psf_fp_maps`` and
``aos/closed_loop`` (the mimic correction), so all three draw the same wedges.
"""
import numpy as np

__all__ = ["DEFAULT", "CORNER_OFFSETS", "N_CORNERS", "in_ranges", "wedge_medians"]

DEFAULT = dict(wfs_inner_radius_deg=1.60, wfs_outer_radius_deg=1.725,
               wfs_azimuth_width_deg=7.5, delta_deg=0.0,
               min_donuts_per_wedge=3, rotator_keep=None)
CORNER_OFFSETS = [0.0, 90.0, 180.0, 270.0]
N_CORNERS = len(CORNER_OFFSETS)


def in_ranges(angle, ranges):
    """True if scalar ``angle`` falls in any (lo, hi) range; ranges None -> all."""
    if ranges is None:
        return True
    if not np.isfinite(angle):
        return False
    return any(lo <= angle <= hi for lo, hi in ranges)


def wedge_medians(dev, thx_deg, thy_deg, sec):
    """Median deviation in each of the four WFS wedges for one image.

    ``dev`` is (n_donut, nZk).  Returns (N_CORNERS, nZk) of medians (NaN-aware)
    or ``None`` if any wedge has fewer than ``min_donuts_per_wedge`` donuts."""
    r = np.hypot(thx_deg, thy_deg)
    az = np.degrees(np.arctan2(thy_deg, thx_deg)) % 360.0
    radial = (r >= sec['wfs_inner_radius_deg']) & (r <= sec['wfs_outer_radius_deg'])
    half = sec['wfs_azimuth_width_deg'] / 2.0
    out = np.full((N_CORNERS, dev.shape[1]), np.nan)
    for i, off in enumerate(CORNER_OFFSETS):
        c = (sec['delta_deg'] + off) % 360.0
        lo, hi = (c - half) % 360.0, (c + half) % 360.0
        az_in = (az >= lo) & (az <= hi) if lo < hi else (az >= lo) | (az <= hi)
        w = radial & az_in
        if int(w.sum()) < sec['min_donuts_per_wedge']:
            return None
        out[i] = np.nanmedian(dev[w], axis=0)
    return out
