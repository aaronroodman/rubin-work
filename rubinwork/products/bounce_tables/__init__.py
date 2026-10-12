"""The ``bounce_tables`` product: the optical-state change across a telescope bounce.

A **bounce** throws the telescope between a reference (A) set and a comparison (B) set
within one night -- BLOCK-T720 in elevation at fixed camera rotator angle, BLOCK-T724 in
rotator angle at fixed elevation -- so the change in optical state between the two
positions is measured as a **time-ordered paired difference**, not as a difference of
nightly means, and drift between the sets cancels. A build reduces one Double-Zernike
(DZ) fit table into the paired Δ per (focal order k, pupil Noll index j), per Optical
Feedback Control (OFC) v-mode and per physical degree of freedom (DOF), with robust
median-absolute-deviation errors, under **five** recovery schemes: the 50/34 truncated
recovery, Range-Bounded Recovery (RBR), the 22/12 reduced DOF set the Automated Optical
System is expected to operate in, the ts_ofc Optimal Integral Controller (OIC) quadratic
motion penalty, and the 5/5 camera-hexapod-only recovery on the rotator bounce.

**The product is the four tables; the nine PDFs are the study.** Both halves call
`rubinwork.bounce.analysis.run_analysis` for the same reduction -- the builder writes the
tables, and ``aos/code/bounce/run_bounce.py`` renders the figures. That is route (a) of
the phase 3a split, Aaron's call 2026-10-12: three of the nine pages plot the
**per-visit** arrays ``C_all``, ``DOF_all`` and ``DOF5_all``, which no table carries, so
the study cannot be served by reading the product back. The cost is one extra run of the
analysis (about 170 s of wall clock) when both halves are built. A later option is to
write the per-visit arrays as a fifth table, after which the study would read only the
product.

Why a product at all: ``aos/code/cwfs_lut/bounce_compare.py`` is a **different study**
that reads ``bounce_dof_stats.parquet``, which rule 3 of the plan's section 2 forbids
between studies.

One rule, one script. ``build_bounce_tables`` writes:

``bounce_kj_stats.parquet``
    long-format paired Δ per (bounce, night, comparison, k, j), in µm of wavefront, with
    the reference and comparison statistics beside it. A pooled row per leg carries the
    literal ``night = 'all'``.
``bounce_dof_stats.parquet``
    per-DOF and per-v-mode Δ, one row per (bounce, leg, night, ``kind``, index) over the
    seven ``kind`` values (``dof``, ``dof22``, ``dof5``, ``dof_oic``, ``vmode``,
    ``vmode22``, ``vmode5``). ``delta`` is in the unit the row's ``unit`` column names:
    µm for translations and bending-mode amplitudes, arcsec for hexapod rotations,
    dimensionless for v-modes. The ``dof`` rows also carry RBR as ``delta_rbr`` and
    ``delta_rbr_err`` in the same unit, plus the allowed range ``dof_range`` and the two
    dimensionless amplitude-over-range ratios. Carries the **measured** elevation and
    rotator angle of both legs, in deg, rather than the nominal config window.
``bounce_fwhm_metric.parquet``
    the correctable full-width-half-maximum (FWHM) per leg, in arcsec: ``fwhm_before``
    and one ``fwhm_after_*`` per scheme. Every "after" value is the **achieved** residual
    ``dW - S (d / w)`` of its own scheme, so the schemes are comparable -- the subspace
    projection cannot see a regularizer trading wavefront for amplitude.
``bounce_fwhm_vs_bvalue.parquet``
    the same three FWHM series in arcsec per (night, leg), against the B-set position in
    deg.

Needs ``lsst.ts.ofc`` and ``TS_CONFIG_MTTCS_DIR`` for the sensitivity-matrix
singular-value decomposition and ``lsst.ts.wep`` for the FWHM conversion, so it is
RSP-only in the same sense the other moved rules are. The production build takes about
170 s of wall clock.

Variants are defined in ``variants.yaml``.

Read a build with `load`::

    from rubinwork.products.bounce_tables import load
    kj, dof, fwhm, bval = load("d12-A_50_34_i_5rot_july")

Build one with the product's Snakefile, from the repository root::

    snakemake -s rubinwork/products/bounce_tables/Snakefile \\
        --config variant=d12-A_50_34_i_5rot_july build=20261012 miw_build=20261007
"""

from .reader import (PRODUCT, TABLE_KEYS, TABLES, build_dir, dir_name, fam_variant,
                     fits_name, load, load_table, mi_keys, miw_variant, variant_config,
                     variants)

__all__ = ["PRODUCT", "TABLE_KEYS", "TABLES", "build_dir", "dir_name", "fam_variant",
           "fits_name", "load", "load_table", "mi_keys", "miw_variant",
           "variant_config", "variants"]
