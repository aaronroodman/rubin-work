"""The ``coadds`` product: per-FAM-block Measured Intrinsic Wavefront remnant maps.

The ``miw`` product pools every Full Array Mode (FAM) visit in a camera-rotator bin, over
many nights, and extracts the remnant wavefront left after the iterative optical-state
correction. That mixes nights. This product runs the *same* iterative analysis but
aggregates per **contiguous FAM block** -- one block being the roughly twelve triplets
taken at a single pointing and rotator angle on one night -- so each block gets its own
Measured-Intrinsic-Wavefront-style (MIW-style) remnant map. Comparing those maps to the
MIW itself is what shows whether the anomalous wavefront seen in four of the nine
rotator-bin MIW maps is confined to one ``day_obs`` or recurs.

A build has **two upstreams**: a ``fam_tables`` build, whose donuts and visits it blocks
up, and a ``miw`` build, whose per-donut sidecar is the reference each block's remnant is
compared against. Both are recorded in the manifest.

One rule, one script. ``run_coadd_blocks_miw`` detects the blocks, runs the
three-iteration U-mode-constrained build per block, and writes:

``blocks_summary.parquet``
    every detected block: program, ``day_obs``, seq range, visit count, alt/az/rotator
    means in deg, blur full width at half maximum (FWHM) in arcsec, band, the
    in-rotator-family flag, and thirteen thermal telemetry means.
``coadd_metrics.parquet``
    one row per built coadd: that metadata plus the per-Zernike and combined residual
    root mean square (RMS) in µm of wavefront and spatial Pearson correlation against
    the MIW, over the full focal plane and over the corner-wavefront-sensor annulus
    separately, plus the 34 u-mode amplitudes in µm of wavefront.
``block_grids.npz``
    the stacked coadd and MIW Z5-Z8 maps, the u-modes, and the Double Zernike (DZ)
    primitives -- block-median raw and residual DZ vectors with the subspace bases --
    so the u-mode amplitudes and the reachable-versus-null split of the residual are
    derivable without the LSST stack.
``block_grids_ext.npz``
    the same maps for the extra pupil Zernikes. Z5-Z8 are all primary, so these are what
    reveal which radial orders a retrieval bias mixes.

Two PDFs are build artifacts: a per-block map page set and a time series.

The iterative build itself is **external**, in the LSST-TS package
``ts_intrinsic_wavefront`` (``measured_intrinsic.build_measured_intrinsic_uconstrained``);
only the visit selection is this product's. Running it needs ``lsst.ts.ofc`` and
``TS_CONFIG_MTTCS_DIR``.

Variants are defined in ``variants.yaml``, which is also where the block-selection
settings live.

Read a build with `load` and `load_grids`::

    from rubinwork.products.coadds import load, load_grids
    summary, metrics = load("d12-A_50_34_i_5rot")
    grids = load_grids(variant="d12-A_50_34_i_5rot")

Build one with the product's Snakefile, from the repository root::

    snakemake -s rubinwork/products/coadds/Snakefile \\
        --config variant=d12-A_50_34_i_5rot build=20261010 \\
                 fam_build=20260920 miw_build=20261010
"""

from .reader import (BLOCK_KEY, BUILD_FIELDS, GRIDS, PLOTS, PRODUCT, TABLES, build_dir,
                     build_settings, dir_name, fam_variant, load, load_grids, load_table,
                     miw_variant, variant_config, variants)

__all__ = ["BLOCK_KEY", "BUILD_FIELDS", "GRIDS", "PLOTS", "PRODUCT", "TABLES",
           "build_dir", "build_settings", "dir_name", "fam_variant", "load",
           "load_grids", "load_table", "miw_variant", "variant_config", "variants"]
