"""The ``sim_donut_fits`` product: Danish fits of the ``sim_donuts`` stamps.

Each noise-free ``sim_donuts`` stamp is scaled to a flux in electrons, binned, and fit
one donut at a time as ts_wep's ``DanishAlgorithm`` fits an unpaired donut
(``rubinwork.donut_sim.fit``): flux, shift, Kolmogorov FWHM and the variant's Noll set
free. Danish's model wavefront is the batoid intrinsicTA (the defocused
transverse-aberration Zernikes, Noll 0-78) of the sim's ``batoid_intrinsic`` variant,
with the intrinsic of each fitted term replaced by the fitted OPD, as ts_wep builds it;
the stored quantity is the Zernike deviation, OPD minus intrinsic. The danish mask is
that variant's obscuration file. The sims are traced through the same batoid model, so
the expected deviation is zero. Fit variance: the variant's sky plus the model (Poisson).

A build holds:

``fits.parquet``
    one row per stamp: field angles, fitted parameters and Zernike deviations (CCS and OCS),
    chi²/dof, ``l1`` and ``pow`` residual metrics, optimizer status.
``stamps_<sim variant>_<band>_<config>_<side>.fits``
    per donut an image HDU, ``EXTNAME`` the point id, shape ``(2, n, n)``: the binned
    stamp and the best-fit model (electrons per binned pixel, block mean).

Read a build::

    from rubinwork.products import sim_donut_fits
    df = sim_donut_fits.load("v4_speed_sparse")
    data, model, hdr = sim_donut_fits.stamp("v4_speed_sparse", "v4_speed_mono", "r",
                                            "fam", "extra", "r05_a10")

Build one, in batch (``run_build.sbatch``) or from the repository root::

    python -m rubinwork.products.sim_donut_fits.builders.build --variant v4_speed_sparse --jobs 16
"""

from .reader import (INDEX, PRODUCT, build_dir, dz_columns, load, stamp, stamp_file,
                     variant_config, variants)

__all__ = ["INDEX", "PRODUCT", "build_dir", "dz_columns", "load", "stamp", "stamp_file",
           "variant_config", "variants"]
