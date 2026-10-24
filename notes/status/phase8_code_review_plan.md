# Phase 8 plan: code review in the new layout

> **Status:** plan only, nothing started · **Last updated:** 2026-10-24 · **Kind:** plan

Part C of the old plan (`notes/status/reorg_review_plan_2026-09.md` at tag
`pre-products-reorg-2026-10-09`, sections C1–C4) applied to the products, libraries and
studies of `organization_plan.md` section 2. Part C's rules carry over: review by unit, not
by file; one unit is one commit; verify each finding against the source before acting;
behavior-changing fixes go in a backlog with the reasoning, not straight into the code.

## What changed since Part C was written

- **C4's premise is gone.** Part C said there were no tests. `pytest rubinwork` now runs
  432 tests (count; 10 more skip) plus the per-study tests, and every product has a `compare_builds.py`
  that checks two builds column by column at zero tolerance. The missing piece is
  narrower: a fixed, small reference build per product that a refactor can be checked
  against in minutes, without a batch rebuild.
- **The worst hotspot has moved.** `aos/code/science_lut/` (9,476 lines, count) became
  `thermal_focus`: `run_thermal_focus_analysis.py` (3,630 lines) plus the library
  `rubinwork.thermal_focus` (`fit.py` 1,300, `lib.py` 304, `sample.py` 218).
- **Code to review is mostly in `rubinwork/` now:** 39,377 lines (count) of non-test
  Python. The largest: `products/value_added/reader.py` 1,858; `thermal_focus/fit.py`
  1,300; `products/guider_moments/guiderMoments.py` 1,040;
  `products/coadds/builders/run_coadd_blocks_miw.py` 993;
  `products/fam_tables/builders/blitz_reader.py` 992; `bounce/legs.py` 959;
  `products/value_added/builders/build_optical_state.py` 958;
  `common/FocalPlaneInterpolator.py` 929; `products/catalog.py` 721.

## Regression references (from phases 2–3)

Each product was rebuilt with the moved code and matched its pre-move reference at zero
tolerance. Those references are the regression tests. They sit under
`/sdf/data/rubin/user/roodman/LSST/rubin-work/_phase2_refbuild/`: 2.30 GB for the references,
4.60 GB with the work areas.

| product | reference | result it reproduced | commits |
|---|---|---|---|
| `fam_tables` | `_phase2_refbuild/danish_1_2/` (one chunk, Snakemake path, 380 MB); `danish_1_3_v1000/` (blitz path, 164 MB) | 0 differing columns of 48/600/653 and 66/23/448 | `58a6606`, `b355446`, `0781229` |
| `cwfs_tables` | `_phase2_refbuild/cwfs_tables/` (one night, 41 MB) | 0 differing columns of 20 (donuts) and 8 (visits) | `71a345f`, `b880f2d` |
| `miw` | `_phase2_refbuild/miw/ref/` with `mi_config_ref.yaml` (1.4 GB) | 0 differing columns across 12 tables | `d81cc3d`, `b9756aa` |
| `coadds` | `_phase2_refbuild/coadds/` (139 MB) | 0 differing columns across 2 tables, 0 differing arrays across 2 npz files | `3f607ab`, `d0e4182` |
| `dof_lut` | `_phase2_refbuild/dof_lut/`; stack-move copy in `studies/phase3c_validation/stack_move/w39/` | md5 equal on all 3 parquet files (w_2026_39 against w_2026_42) | `d54e5bd` |
| `bounce_tables` | `_phase2_refbuild/bounce_tables/` (17 MB) | 0 differing columns across 4 tables | `884fb18`, `3adce0f` |
| `value_added` | `_phase2_refbuild/value_added/` (94 MB) | 0 unexpected differences | `fb14ab5` |
| `guider_moments` | `_phase2_refbuild/guider_moments/` (3.4 MB) | exact | `b4ca14d`, `8b488c9` |

The `_scratch*` and `_timing_step5` directories beside them are work areas, not references.

**Before the review starts:** write down, per product, the one command that rebuilds its
reference into a scratch directory and runs its `compare_builds.py` against the reference.
Most are in the phase 2 and 3 handoffs; collect them in one place
(`rubinwork/products/REGRESSION.md` or a `--regression` flag on each `compare_builds`).
For studies, the regression check is the study's own printed numbers: the
`thermal_focus` analysis (`--no-pdf`, 46 s of wall clock), the `trim_calculator` check
(3 s), and each study's Log values.

## Units of review, in order

Order by risk times rate of change, as in C2. Run C1 first.

**C1. One review backlog.** `aos/docs/status/code_review_findings.md` (2026-06-27, stale
anchors) and `code_review_backlog.md` (2026-09-14) still both exist. Re-anchor the
still-real findings into one backlog, retire the stale file. Known still-real items: the
silent `np.nan_to_num` in `aos/bounce/code/bounce_lib.py` (lines 405, 700–701, 890, 900)
and `aos/closed_loop/code/run_closed_loop.py:126`, and the row-count guard in
`analyze_miw_field_order.py`. The 1.2533 median-SEM factor is fixed
(`aos/bounce/bounce.md:529`).

**Tier 1, libraries that several studies or products import.** A defect here biases
every reader.

1. `rubinwork.thermal_focus` (`fit.py`, `lib.py`, `sample.py`) with
   `thermal_focus/thermal_focus/code/run_thermal_focus_analysis.py`. The former
   `science_lut` hotspot. Check: `trim_calculator/code/check_against_fit.py` (max
   difference 0.0034 µm of equivalent hexapod dz, count 68,079 visits) and the analysis's
   printed sections.
2. **The three `assign_blocks`**, same name, three behaviors:
   `rubinwork/products/coadds/builders/run_coadd_blocks_miw.py:191`,
   `rubinwork/thermal_focus/fit.py:1150`, `aos/miw_origin/code/analyze_dz_goodness_of_fit.py:100`.
   Rename or unify. Check: the `coadds` reference.
3. `rubinwork.aos_state` and `rubinwork.smatrix` (the v-modes, DOF sets, SVD). Check: the
   `miw`, `dof_lut` and `bounce_tables` references, and the smatrix study numbers.
4. `rubinwork.common` (`utils.py`, `FocalPlaneInterpolator.py`, `telemetry_clients.py`,
   `ess_telemetry.py`, `visit_telemetry.py`). Check: the full pytest.
5. `rubinwork.hsm_moments`: shared by guider and science-CCD moments on purpose. Check:
   the `guider_moments` reference.

**Tier 2, product builders and readers.** Results are pinned by the references, so these
are the safest to refactor.

6. `products/value_added` (`reader.py` 1,858 lines; `build_optical_state.py`,
   `build_efd_db.py`). Check: the `value_added` reference.
7. `products/fam_tables` (`blitz_reader.py`, `run_blitz_mktable.py`, Snakemake builders)
   and `products/cwfs_tables`. Check: both `fam_tables` references and `cwfs_tables`.
8. `products/coadds`, `products/miw`, `products/dof_lut`, `products/bounce_tables`,
   `rubinwork/bounce`. Check: their references.
9. `products/guider_moments` (`guiderMoments.py`). Check: its reference.
10. `products/catalog.py` and `rubinwork/studies`. Check: `rubinwork/products/tests`,
    `rubinwork/studies/tests`, and `/review` output unchanged.

**Tier 3, active studies.** Each in its own sitting, folded into the next real work on
it, not reviewed cold: `aos/intra_extra`, `aos/cwfs_lut`, `aos/miw_recovery_constraint`,
`guider/*` (four active), `smatrix/sparse_fit`, `smatrix/regularized_inversion`,
`blocks/image_quality_trending`. The C2 warm-up item belongs here:
`aos/cwfs/code/run_wfs_dof_compare.py` reimplements `zj_to_fwhm` and `fp_grid`
(`aos/code/aos_fwhm.py`) and `corner_matrix_at` (`aos/code/psf_maps_lib.py`).

**Not reviewed:** done, parked and dropped studies, unless they are rerun. The 82 files
(count) `/review` lists as covered by no study are mostly tooling; review one when it is
next used.

## Method per unit (C3, updated)

1. `/code-review` scoped to the unit's directory; effort `high` for Tier 1.
2. Verify every finding against the source.
3. Fix the mechanically safe items; record behavior-changing ones in the backlog with the
   reasoning.
4. Bring the touched docstrings to the Rubin DM standard (`rubin-docstrings`).
5. Run the unit's regression check above. A product whose output changes gets a new
   build, not an edit, and the readers that pin it are listed for a rerun.
6. One commit per unit; a Log line in each affected study.
