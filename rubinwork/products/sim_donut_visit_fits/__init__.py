"""The ``sim_donut_visit_fits`` product: Danish fits of the ``sim_donut_visits`` stamps.

Each stamp, noisy or its noise-free expectation (the variant's ``plane``), is binned and
fit one donut at a time as ts_wep's ``DanishAlgorithm`` fits an unpaired donut
(``rubinwork.donut_sim.fit``): flux, shift, Kolmogorov FWHM and the variant's Noll set
free. Danish's model wavefront is the batoid intrinsicTA of the sims'
``batoid_intrinsic`` variant with, for the fitted terms, the intrinsic replaced by the
fitted OPD, as ts_wep builds it; the stored quantity is the Zernike deviation, OPD minus
intrinsic. The optics are perfectly aligned, so the expected deviation is zero.

A build holds:

``fits.parquet``
    one row per stamp: the sim's draws, fitted parameters, Zernike deviations (CCS and
    OCS) and intrinsic, chi²/dof, ``l1``, optimizer status.
``stamps_<ccd>.fits``
    per donut an image HDU, ``EXTNAME`` ``v<visit>_<point id>``, shape ``(2, n, n)``:
    the binned stamp and the best-fit model (electrons per binned pixel, block mean).

Read a build::

    from rubinwork.products import sim_donut_visit_fits
    df = sim_donut_visit_fits.load("v4_speed_cwfs8_vk_dense_prod")

Build one, in batch (``run_build.sbatch``) or from the repository root::

    python -m rubinwork.products.sim_donut_visit_fits.builders.build --variant v4_speed_cwfs8_vk_dense_prod --jobs 16
"""

from .reader import (INDEX, PRODUCT, build_dir, dz_columns, load, stamp, stamp_file,
                     variant_config, variants)

__all__ = ["INDEX", "PRODUCT", "build_dir", "dz_columns", "load", "stamp", "stamp_file",
           "variant_config", "variants"]
