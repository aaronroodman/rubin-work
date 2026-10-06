"""Recover known spider geometry from a synthetic pupil.

The trajectory method's whole claim is that it measures a vane's depth and
placement without the radial smearing a fixed-annulus azimuthal projection
suffers. That claim is only worth anything if the method returns the truth when
the truth is known, so this builds an annular pupil with four opaque vanes at a
chosen impact parameter and width and checks what comes back.

Run as ``python test_spider_trajectory.py``; no stack or Butler needed.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import radial_profile as rp
import spider_trajectory as st

# Synthetic pupil matching the real giant donuts: 341.5 pixel outer edge and a
# 0.612 inner obscuration, which is the v3.14 value.
STAMP_PIX = 861
OUTER_PIX = 341.5
INNER_FRAC = 0.612

# Truth, in pixels. The impact parameter is the perpendicular distance from the
# pupil centre to each vane's line of centres; a radial spoke would be zero.
TRUTH_IMPACT_PIX = 28.0
TRUTH_HALF_WIDTH_PIX = 4.5

# Tolerances, set from the measured performance rather than from a wish.
#
# The impact parameter carries a 0.9 pixel bias and a 1.7 pixel spread among the
# eight shadows, both traceable to a common +0.3 deg bias in the fitted
# direction: the dip centroid is pulled by the annulus illumination not being
# flat across the tracing annulus, and that direction error is levered by the
# 250 pixel distance to the pupil centre. The bias is common to both vane
# families, so it cancels in the data-minus-model differences this module is
# used for, but it is the floor on any absolute statement about placement.
IMPACT_TOL_PIX = 3.5
# Width comes back about 7 per cent narrow, from the same centroid pull.
WIDTH_TOL_FRAC = 0.10
# Common rotation bias of the fitted trajectory directions, in degrees.
ANGLE_BIAS_TOL_DEG = 0.6


def synthetic_pupil():
    """Annular pupil with four opaque vanes at known impact parameter.

    Returns
    -------
    image : `numpy.ndarray`
        Synthetic stamp, flux 1000 in the illuminated annulus and 0 elsewhere.
    center : `tuple` [`float`]
        ``(x0, y0)`` pupil centre, in pixels.
    """
    center = ((STAMP_PIX - 1) / 2.0, (STAMP_PIX - 1) / 2.0)
    yy, xx = np.mgrid[0:STAMP_PIX, 0:STAMP_PIX]
    radius = np.hypot(xx - center[0], yy - center[1])
    image = (((radius <= OUTER_PIX) & (radius >= INNER_FRAC * OUTER_PIX))
             .astype(float) * 1000.0)

    # Two vane directions, each crossing the pupil on both sides of centre, so
    # eight shadows appear -- the same count as LSSTCam.
    for angle_deg in (45.0, 135.0):
        direction = np.array([np.cos(np.deg2rad(angle_deg)),
                              np.sin(np.deg2rad(angle_deg))])
        normal = np.array([-direction[1], direction[0]])
        perp = (xx - center[0]) * normal[0] + (yy - center[1]) * normal[1]
        for impact in (+TRUTH_IMPACT_PIX, -TRUTH_IMPACT_PIX):
            image[np.abs(perp - impact) <= TRUTH_HALF_WIDTH_PIX] = 0.0
    return image, center


def main():
    image, _ = synthetic_pupil()
    center = rp.donut_centroid(image)
    centres, azimuth, _ = st.trace_shadow_centres(image, center, OUTER_PIX)
    trajectories = st.fit_trajectories(centres, center)

    assert len(trajectories) == st.N_SPIDER_SHADOWS, len(trajectories)

    truth_fwhm = 2.0 * TRUTH_HALF_WIDTH_PIX
    print(f"truth: |impact| {TRUTH_IMPACT_PIX:.2f} pixel, "
          f"FWHM {truth_fwhm:.2f} pixel, depth 1.0000 (dimensionless)")
    print(f"{'k':>3s} {'azimuth':>9s} {'line':>8s} {'impact':>9s} "
          f"{'depth':>8s} {'FWHM':>8s}")

    for k, trajectory in enumerate(trajectories):
        offset, flux, _, _ = st.trajectory_profile(image, trajectory, center,
                                                   OUTER_PIX)
        depth, fwhm, _ = st.vane_depth(offset, flux)
        print(f"{k:>3d} {trajectory['azimuth_deg']:>7.2f} d "
              f"{trajectory['angle_deg']:>6.2f} d "
              f"{trajectory['impact_pix']:>+8.2f} p "
              f"{depth:>8.4f} {fwhm:>6.2f} p")

        assert depth > 0.99, (k, depth)
        assert abs(abs(trajectory['impact_pix']) - TRUTH_IMPACT_PIX) \
            < IMPACT_TOL_PIX, (k, trajectory['impact_pix'])
        assert abs(fwhm - truth_fwhm) < WIDTH_TOL_FRAC * truth_fwhm, (k, fwhm)

        # The fitted line must be straight to well under a pixel, or the
        # cross-vane profile is averaging across the shadow rather than along it.
        assert trajectory['rms_pix'] < 0.5, (k, trajectory['rms_pix'])

    # Every shadow must drift in azimuth: that drift is the reason the method
    # exists, and a zero drift would mean the vanes were built as spokes.
    drift = azimuth[:, -1] - azimuth[:, 0]
    assert np.all(np.abs(drift) > 0.5), drift
    print(f"\nazimuthal drift {np.abs(drift).min():.2f} to "
          f"{np.abs(drift).max():.2f} deg over trace radii "
          f"{st.TRACE_RADII_NORM[0]} to {st.TRACE_RADII_NORM[-1]} "
          f"(dimensionless)")

    # The direction error must be a common rotation, not per-shadow scatter.
    # Only then does it cancel in a data-minus-model comparison, which is the
    # one thing this module's placement numbers are used for.
    folded = np.array([(t['angle_deg'] - 45.0) % 90.0 for t in trajectories])
    bias, scatter = float(np.mean(folded)), float(np.std(folded))
    print(f"direction bias {bias:+.3f} deg, scatter among shadows "
          f"{scatter:.3f} deg")
    assert abs(bias) < ANGLE_BIAS_TOL_DEG, bias
    assert scatter < 0.5 * ANGLE_BIAS_TOL_DEG, (bias, scatter)

    impact = np.array([abs(t['impact_pix']) for t in trajectories])
    print(f"|impact| mean {impact.mean():.2f} pixel against truth "
          f"{TRUTH_IMPACT_PIX:.2f} pixel, spread among shadows "
          f"{impact.std():.2f} pixel")
    print("all checks passed")


if __name__ == '__main__':
    main()
