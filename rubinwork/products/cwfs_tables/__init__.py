"""The ``cwfs_tables`` product: in-focus corner-wavefront-sensor tables, paired to the
Full Array Mode (FAM) triplets.

The corner wavefront sensors (CWFS) -- the SW0 extra-focal inner halves of rafts R00,
R04, R40 and R44 -- see defocused donuts during the *in-focus* exposure of each FAM
triplet. The triplet order is intra, extra, in-focus, and a ``fam_tables``
``visits.parquet`` row keys on the EXTRA-focal exposure, so the in-focus exposure is
``FAM_seq + seq_offset`` with ``seq_offset`` defaulting to +1.

A build walks one ``fam_tables`` variant's visits, reads each in-focus exposure's
aggregate from the variant's Butler collection, tags every donut with its paired FAM
``seq_num`` and that FAM visit's rotator angle and elevation (so the CWFS and FAM share
one rotator binning), and writes two tables:

``donuts.parquet``
    One row per corner donut: the measured and intrinsic wavefront Zernike coefficients
    (microns of wavefront), the donut's field position in both the Optical (OCS) and
    Camera (CCS) Coordinate Systems, the detector, and the paired FAM ``seq_num``.
``visits.parquet``
    One row per in-focus exposure: the FAM pairing, rotator angle, elevation, band and
    donut count.

A build also writes ``wfs_mktable_validation.pdf``, the CWFS-against-FAM check.

Variants are defined in ``variants.yaml``, which is the source of truth for the FAM/CWFS
triplet link; ``rubinwork.products.fam_tables.gen_param_sets`` reads it back through
`wfs_collections_for` to generate ``aos/param_sets.yaml``.

Read a build with `load`, which returns the two tables::

    from rubinwork.products.cwfs_tables import load
    donuts, visits = load("d12-refitWcs")

Build one with the product's Snakefile, from the repository root::

    snakemake -s rubinwork/products/cwfs_tables/Snakefile \\
        --config variant=d12-refitWcs build=20261010 fam_build=20260920
"""

from .reader import (DONUT_KEY, TABLES, VISIT_KEY, build_dir, fam_variant, load,
                     load_table, variant_config, variants, wfs_collections_for)

__all__ = ["DONUT_KEY", "TABLES", "VISIT_KEY", "build_dir", "fam_variant", "load",
           "load_table", "variant_config", "variants", "wfs_collections_for"]
