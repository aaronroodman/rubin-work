"""The ``wfs_corner_compare`` product: corner wavefront sensor (CWFS) against full-array
mode (FAM) wavefront, corner by corner.

A build holds two tables:

``wfs_corner_compare.parquet``
    one row per (FAM triplet, corner sensor, Noll j): the median in-focus CWFS Zernike
    and the FAM wavefront interpolated in azimuth to that corner, both in µm of wavefront
    in the variant's frame (OCS by default), with the corner azimuth and the visit's
    rotator angle, elevation, azimuth, MJD and science program.
``corner_stats.parquet``
    per (corner, j): Pearson r and Spearman rho, the Huber slope and offset of CWFS
    against FAM, and the nMAD of the residuals.

The computation is `rubinwork.corner_compare`. The plots are the ``aos/cwfs`` study's.
A variant is one ``cwfs_tables`` variant, compared against the ``fam_tables`` variant it
is paired to.

Read a build::

    from rubinwork.products import wfs_corner_compare
    table, stats = wfs_corner_compare.load("d12-refitWcs")

Build one, from the repository root::

    snakemake -s rubinwork/products/wfs_corner_compare/Snakefile \\
        --config variant=d12-refitWcs
"""

from .reader import (PRODUCT, TABLE_KEYS, TABLES, build_dir, cwfs_variant, dir_name,
                     fam_variant, load, load_table, settings, variant_config, variants)

__all__ = ["PRODUCT", "TABLE_KEYS", "TABLES", "build_dir", "cwfs_variant", "dir_name",
           "fam_variant", "load", "load_table", "settings", "variant_config", "variants"]
