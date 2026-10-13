"""The ``dof_lut`` product: the averaged degree-of-freedom (DOF) look-up table (LUT).

Given a mode count ``n_dof`` and a kept-mode count ``n_keep``, a build projects the
per-visit Double Zernike (DZ) fits of one ``miw`` build onto the Optical Feedback
Control (OFC) sensitivity-matrix singular-value decomposition (SVD), recovers the
physical DOF per visit, and **collapses over every elevation and rotator angle** into one
averaged LUT. Unlike the measured-intrinsic-wavefront (MIW) build it deliberately does
not bin by rotator and by default applies no elevation window: it averages every visit
passing the band, program and good-fit cuts.

**The upstream is the post-MIW DZ table** (Aaron, 2026-10-13, before the phase 3c
rebuild). A build reads ``fits.parquet`` from a ``miw`` build -- its ``refit_mi`` output,
the DZs refitted with that build's MIW subtracted -- and ``visits.parquet`` from the
``fam_tables`` build, which is where the visits live. The recovered DOF are therefore
**deviations from the as-built optics**, not absolute positions.

The previous plain-FAM LUT stays only in the old ``aos/output/lut/`` tree: this is a
variant-definition change, not a builder knob.

**It does not fix the implausible DOF magnitudes**, which was the motivation. The plain
FAM fit already subtracts the tabulated batoid design intrinsic, and ``refit_mi`` only
substitutes the measured one for it, so ``M2_dy`` moves from -2407.11 to -2407.34 µm of
M2 hexapod decentre on the ``miw`` 3a reference. What the switch buys is fit quality: the
averaged DZ residual falls by a factor of 6, to 0.00172 µm of wavefront RMS. The
magnitudes are an ill-conditioned inversion and remain open -- see
``aos/docs/studies/lut.md``.

The same ``miw`` entry also supplies the *settings* -- the mode counts, the focal order
range ``k_min`` to ``k_max``, the coordinate system, the band and program cuts, and the
rotator bin edges -- which stay under the manifest's ``config``.

One rule, one script. ``run_build_lut`` writes:

``lut.parquet``
    one row per recovered DOF: the absolute DOF index, its label and unit, the reduced
    ``value`` (median over kept visits by default), the ``mean``, the robust
    median-absolute-deviation scatter, and the visit count. Translations and bending
    modes are in µm and hexapod tip-tilt in arcsec, per the row's ``dof_unit``.
``lut_dz.parquet``
    one row per (focal order k, pupil Noll index j): the averaged raw DZ, its
    ``n_keep``-mode SVD reconstruction and the residual, all in µm of wavefront.
``lut_by_rotbin.parquet``
    the same DOF medians per rotator bin, the diagnostic for whether the all-rotator
    median hides rotator-coupled structure. Written only when the fit table carries a
    rotator-angle column, so a build may lack it.
``lut_config.yaml``
    the builder's frozen provenance.

``lut.parquet`` and ``lut_dz.parquet`` also carry eight fields of **parquet schema
metadata** (`reader.SCHEMA_METADATA_FIELDS`) -- the mode counts and visit count are
recorded only there. Read them with `build_metadata`. A one-page four-panel DOF summary
PDF is a build artifact.

Needs ``lsst.ts.ofc`` and ``TS_CONFIG_MTTCS_DIR`` for the sensitivity-matrix SVD, so it
is RSP-only in the same sense the other moved rules are. The production build takes about
9 s of wall clock.

Variants are defined in ``variants.yaml``.

Read a build with `load`::

    from rubinwork.products.dof_lut import build_metadata, load
    lut, lut_dz, by_rotbin = load("d12-A_50_34_i_5rot")
    n_keep = build_metadata("d12-A_50_34_i_5rot")["n_keep"]

Build one with the product's Snakefile, from the repository root::

    snakemake -s rubinwork/products/dof_lut/Snakefile \\
        --config variant=d12-A_50_34_i_5rot build=20261011 fam_build=20260920
"""

from .reader import (PLOTS, PRODUCT, PROVENANCE, SCHEMA_METADATA_FIELDS, TABLE_KEYS,
                     TABLES, build_dir, build_metadata, build_provenance, dir_name,
                     fam_variant, load, load_table, mi_keys, miw_variant, variant_config,
                     variants)

__all__ = ["PLOTS", "PRODUCT", "PROVENANCE", "SCHEMA_METADATA_FIELDS", "TABLE_KEYS",
           "TABLES", "build_dir", "build_metadata", "build_provenance", "dir_name",
           "fam_variant", "load", "load_table", "mi_keys", "miw_variant",
           "variant_config", "variants"]
