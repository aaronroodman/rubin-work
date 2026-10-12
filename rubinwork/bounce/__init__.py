"""Bounce-test analysis: a library, shared by the product and the study.

FAM-triplet telescope-position bounce tests throw the telescope between a
reference (A) set and a comparison (B) set within one night, so the optical-state
change between the two positions is measured as a time-ordered paired difference.

Two modules:

- `legs` — leg definitions, A/B pair formation, the per-(k, j) and per-DOF paired
  Δ with robust (MAD) errors, and the regularized-recovery solvers' Δ.
- `analysis` — `run_analysis`, which performs the whole reduction once and
  returns a `BounceAnalysis`.

`rubinwork.products.bounce_tables` writes the four product tables from a
`BounceAnalysis`; the `bounce` study in `aos/` renders the nine PDFs from the
same object.  Neither imports the other.
"""
from rubinwork.bounce.legs import (  # noqa: F401
    CAM_HEX_DOF, DEFAULT, DEFAULT_BOUNCES, DOF22, RBR_DEFAULTS, RBR_METHOD_NAME,
    ab_position_table, bounce_nights, bounce_program_mask, diff_of_deltas,
    diff_stats, filter_visits, form_pairs, leg_axis_name, leg_b_value,
    leg_night_coverage, leg_pointing, paired_delta, paired_deltas_kj,
    paired_deltas_matrix, passing_terms, rbr_deltas, rbr_dof_per_pair,
    rbr_module, resolve_config, run_bounce, solver_deltas, solver_dof_per_pair,
    stats_per_kj, to_long_df, visit_list_str,
)
