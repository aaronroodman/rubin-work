"""The ``sim_donut_visits`` product: simulated CWFS visits with noise and seeing.

Each visit puts one donut on each ``batoid_intrinsic`` point of the variant's
intrinsic (8 per corner half-chip, inside 1.725 deg). The noise-free optical donut of a
point is ray-traced once through the batoid model, perfectly aligned
(``rubinwork.donut_sim.raytrace``); each visit convolves it with an elliptical von
Kármán kernel, shifts it by a random sub-pixel offset, scales it to a star's flux and
adds sky, Poisson and read noise (``rubinwork.donut_sim.observe``). The kernel's
major-axis angle is drawn per visit and corner.

A build holds:

``donuts.parquet``
    one row per visit and point: field angles, the star, sky and seeing drawn.
``stamps_<ccd>.fits``
    per donut an image HDU, ``EXTNAME`` ``v<visit>_<point id>``, shape ``(2, n, n)``:
    the sky-subtracted noisy stamp and its noise-free expectation (electrons per pixel).

Read a build::

    from rubinwork.products import sim_donut_visits
    df = sim_donut_visits.load("v4_speed_cwfs8_vk")
    obs, exp, hdr = sim_donut_visits.stamp("v4_speed_cwfs8_vk", "R00_SW0", 0, "R00_SW0_k0")

Build one, in batch (``run_build.sbatch``) or from the repository root::

    python -m rubinwork.products.sim_donut_visits.builders.build --variant v4_speed_cwfs8_vk --jobs 8
"""

from .reader import (INDEX, PRODUCT, build_dir, extname, load, stamp, stamp_file,
                     variant_config, variants)

__all__ = ["INDEX", "PRODUCT", "build_dir", "extname", "load", "stamp", "stamp_file",
           "variant_config", "variants"]
