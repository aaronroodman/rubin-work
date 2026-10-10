"""The ``miw`` product: the Measured Intrinsic Wavefront.

The Measured Intrinsic Wavefront (MIW) is the intrinsic wavefront measured from Full Array
Mode (FAM) triplets, as opposed to the batoid ray-trace prediction. A build runs four
steps over one ``fam_tables`` build:

``build_intrinsic``, once per camera rotator angle bin
    iterates the U-mode-constrained recovery over the bin's visits and writes
    ``build/rot_<lo>_<hi>/intrinsic_grid.parquet``, the median intrinsic on a field grid.
``intrinsic_split``
    decomposes the per-bin grids into a telescope-fixed component in the Observatory
    Coordinate System (OCS) and a camera-fixed one in the Camera Coordinate System (CCS),
    the latter rotating with the rotator, and writes
    ``intrinsic_split_{maps,decomp,rms}.parquet`` plus ``intrinsic_split.pdf``.
``intrinsic_sidecar``
    reconstructs the MIW at every FAM donut and writes ``zk_intrinsic.parquet``,
    row-aligned to the FAM ``donuts.parquet``. The corner-wavefront-sensor form of the
    same step writes ``wfs/<cwfs variant>/zk_intrinsic.parquet``.
``refit_mi``
    refits the FAM Double Zernikes (DZ) with this MIW subtracted, into ``fits.parquet``.

The build code itself is **external**, in the LSST-TS package ``ts_intrinsic_wavefront``;
this product owns the configuration, the Snakemake rules, the readers and the manifest.
A fix to build logic belongs in that package, and ``scons`` must be re-run after editing
it.

Variants are defined in ``variants.yaml``, which is the source of truth for the
measured-intrinsic configuration; ``gen_mi_config`` generates ``aos/mi_config.yaml`` from
it, byte-identical in its body, because the external package reads that path.

The official MIW -- Guillem's, built by ``ts_intrinsic_wavefront`` and read from a Butler
as an `lsst.ip.isr.IntrinsicZernikes` calibration -- is registered as an **external**
variant: a manifest with a ``location`` and no files. Nothing the official MIW needs comes
from here.

Read a build with `load`::

    from rubinwork.products.miw import load
    maps, = load("d12-A_50_34_i_5rot", tables=("intrinsic_split_maps",))

Build one with the product's Snakefile, from the repository root::

    snakemake -s rubinwork/products/miw/Snakefile \\
        --config variant=d12-A_50_34_i_5rot build=20261010 fam_build=20260920
"""

from .reader import (DECOMP_KEY, FIT_KEY, JS_DEFAULT, MAP_KEY, PRODUCT, RMS_KEY, TABLES,
                     build_dir, build_source, dir_name, external_variants, fam_variant,
                     groups, load, load_maps, load_miw, load_table, mi_keys,
                     rotator_bins, split_rotator_bins, variant_config, variants)

__all__ = ["DECOMP_KEY", "FIT_KEY", "JS_DEFAULT", "MAP_KEY", "PRODUCT", "RMS_KEY",
           "TABLES", "build_dir", "build_source", "dir_name", "external_variants",
           "fam_variant", "groups", "load", "load_maps", "load_miw", "load_table",
           "mi_keys", "rotator_bins", "split_rotator_bins", "variant_config", "variants"]
