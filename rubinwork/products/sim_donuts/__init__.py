"""The ``sim_donuts`` product: noise-free donuts ray-traced through a batoid model.

For each band and defocus configuration of a ``batoid_intrinsic`` variant (``fam``:
camera ±1.5 mm; ``cwfs``: detector ±1.5 mm on each corner half-chip's own side;
``giant``: M2 and camera ±4 mm), donuts at that variant's field points, traced through
the full model so its apertures, baffles and spider vignette them. Kolmogorov blur,
10 µm pixels, centred on the chief ray, row = CCS y and column = CCS x; pixel values are
m² of transmitted collecting area. Noise is left to the fitting.

A build holds:

``donuts.parquet``
    one row per stamp: field angles (CCS and OCS), flux, trace statistics, and where the
    stamp is.
``stamps_<band>_<config>_<side>.fits``
    one image HDU per donut, ``EXTNAME`` the ``batoid_intrinsic`` point id.

A variant with a ``dof`` scan traces the same donuts once per DOF, each with that one
DOF set (batoid_rubin, ts_ofc convention); its stamp files carry ``_dof<NN>`` and its
index the DOF and the perturbed in-focus OPD. Each DOF is built as a partition
(``--dof-part``, one batch array task each), then ``--finalize`` merges the index and
writes the manifest.

Read a build::

    from rubinwork.products import sim_donuts
    index = sim_donuts.load("v4_speed_mono")
    img, hdr = sim_donuts.stamp("v4_speed_mono", "r", "fam", "extra", "r05_a10")

Build one, in batch (``run_build.sbatch``) or from the repository root::

    python -m rubinwork.products.sim_donuts.builders.build --variant v4_speed_mono --jobs 16
"""

from .reader import (INDEX, PRODUCT, build_dir, dof_partitions, load, stamp, stamp_file,
                     variant_config, variants)

__all__ = ["INDEX", "PRODUCT", "build_dir", "dof_partitions", "load", "stamp", "stamp_file",
           "variant_config", "variants"]
