"""The ``bounce_aba`` product: the A–B–A optical-state change across a telescope bounce.

Per (night, leg) of the BLOCK-T720 elevation sweep, each B visit is differenced against
the A (70 deg reference) visits on either side of it, with A interpolated in time to the
B time; the leg's ΔDZ is the median over its B visits. ΔDZ is recovered into degrees of
freedom (DOF) and v-modes under the variant's scheme, and every quantity carries a
Monte Carlo (MC) error from a noise model fitted to the same visits. The method is
`rubinwork.bounce.aba`; it reproduces the ``aos/bounce`` notebooks
``bounce_aba_optical_state_v2``-``v5``.

This is a different method from ``bounce_tables``, which reduces the time-ordered B−A
pairs by their median, with median-absolute-deviation errors.

A variant is one (scheme, intrinsic) pair (``variants.yaml``):

- scheme ``22_12``: 22 DOF, 12 modes, linear truncated solve; v-modes of the
  ``standard_22`` StateEstimator;
- scheme ``rbr_50_50``: 50 DOF, 50 modes, Range-Bounded Recovery (RBR); v-modes of the
  ``all_50`` StateEstimator;
- intrinsic ``batoid``: the ``fam_tables`` DZ fit, batoid intrinsic subtracted;
  intrinsic ``miw``: the DZ refit of a ``miw`` build.

A build writes four tables:

``aba_legs.parquet``
    one row per (night, leg): ``b_elev_deg`` (nominal B elevation, deg), visit counts
    ``n_A``, ``n_B``, ``n_aba``, the ΔDZ power (µm², sum over terms), the fraction of it
    the fit leaves (dimensionless, power) and the residual chi2 with its dof.
``aba_dz.parquet``
    one row per (night, leg, k, j): ``delta``, ``fit`` and ``resid`` (= delta − fit),
    each with ``*_err``, in µm of wavefront.
``aba_state.parquet``
    one row per (night, leg, kind, index), ``kind`` = ``dof`` or ``vmode``: ``name``,
    ``unit`` (``um``, ``deg`` for hexapod rotations, ``dimensionless`` for v-modes),
    ``delta``, ``err``. DOF ``index`` is the position in the 50-DOF list; v-mode
    ``index`` is 1-based.
``aba_noise.parquet``
    the noise model: per (k, j) the white per-visit sigma (µm). The red-noise fit of
    DZ(1,4) and the Ledoit–Wolf shrinkage are in the manifest under
    ``config.noise_model``.

Read a build with `load`::

    from rubinwork.products.bounce_aba import load
    legs, dz, state, noise = load("d13-batoid-22_12")

Build one, from the repository root (about a minute; RSP stack needed for ts_ofc)::

    python -m rubinwork.products.bounce_aba.builders.build --variant d13-batoid-22_12
"""

from .reader import (PRODUCT, TABLE_KEYS, TABLES, build_dir, fits_input, load, load_table,
                     mi_keys, variant_config, variants)

__all__ = ["PRODUCT", "TABLE_KEYS", "TABLES", "build_dir", "fits_input", "load",
           "load_table", "mi_keys", "variant_config", "variants"]
