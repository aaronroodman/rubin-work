# AOS analysis studies — inventory

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (inventory)

The `aos/` topic is not one project. It is **eleven separable studies** that share the
FAM donut tables and the OFC sensitivity matrix. This file is the map: what each study
asks, which code implements it, what it reads and writes, and where it stands.

Written during the Phase 3 reorganization (`../../notes/status/memory_cleanup_plan.md`).

> **Note:** the per-study links below point at `studies/<study>.md`, which **Phase 4
> creates** — they do not resolve yet. Everything on *this* page is complete and
> current; the per-study pages will carry the detail (method, commands, findings).
> The MIW pipeline's step-by-step reference moves to `miw_pipeline.md` in Phase 4 and
> currently still lives inline in `../README.md`.

## Why this exists

`aos/` holds **56 Python files (18.5 K lines) and 13 notebooks**, of which only **16
scripts are driven by the Snakefile**. Before this reorganization, 35 of the 56 scripts
and 9 of the 13 notebooks were **not mentioned anywhere** in `README.md` — the README
documented the MIW pipeline thoroughly and was nearly silent on the other ~11 K lines.

## The studies

| study | files | lines | pipeline rules | the question it answers |
|---|---|---|---|---|
| [`miw`](studies/miw.md) | 9 | 3207 | 4 | What is the measured intrinsic wavefront, and is the build trustworthy? |
| [`coadd`](studies/coadd.md) | 9 | 3506 | 1 | Why does the per-block FAM coadd disagree with the MIW? |
| [`cwfs`](studies/cwfs.md) | 9 | 2828 | 4 | Does the corner WFS recover the same optical state as FAM? |
| [`static_optics`](studies/static_optics.md) | 8 | 1862 | 0 | Can any static optical figure (M3, lenses, gravity) explain the MIW? |
| [`telemetry`](studies/telemetry.md) | 6 | 1602 | 0 | What was the telescope's commanded/thermal state per visit? |
| [`correlations`](studies/correlations.md) | 4 | 1472 | 4 | What does the residual DZ correlate with — itself, v-modes, or temperature? |
| [`smatrix_vmode`](studies/smatrix_vmode.md) | 4 | 876 | 1 | What does the sensitivity matrix's mode structure allow us to observe? |
| [`bounce`](studies/bounce.md) | 2 | 1394 | 2 | Do elevation/rotator bounce tests show a repeatable Δ? |
| [`processing_compare`](studies/processing_compare.md) | 2 | 922 | 0 | Do two reductions of the same data agree? |
| [`psf`](studies/psf.md) | 2 | 776 | 0 | What PSF does a given wavefront actually produce? |
| [`infra`](studies/infra.md) | 1 | 126 | 0 | (support) How many usable cores does this node have? |

Every one of the 56 files belongs to exactly one study — verified by set comparison,
zero orphans, none assigned twice.

## Cross-cutting code — deliberately *not* in a study subdirectory

These stay flat at `aos/code/` because moving them breaks things:

| file | why it stays flat |
|---|---|
| `aos_trim.py` | **21 references from 4 other topics** (`blocks/`, `olr/`, `optatmo/`, `guider/`) by bare module name |
| `aos_state.py` | **15 references from other topics**, same mechanism |
| `aos_consdb_efd.py` | **3 references from `blocks/`** |
| `aos_fwhm.py` | used by `cwfs` and `correlations` |
| `combine_parquets.py` | used by the pipeline across studies |

The first three are effectively **shared infrastructure**, not aos-private: sibling
topics reach them via a hardcoded `sys.path.insert(.../aos/code)`. See the root
`CLAUDE.md` "Topic independence and shared code", and `../CLAUDE.md` in this topic.

## Notebook → study

| notebook | study |
|---|---|
| `aos_miw_ocs_ccs_maps.ipynb` | `miw` (OCS/CCS map reader) |
| `aos_miw_cwfs_intrinsic_check.ipynb` | `cwfs` |
| `wfs_corner_compare_correlations.ipynb` | `cwfs` (8 half-sensors, v3, tarts) |
| `wfs_corner_compare_correlations-aidonut.ipynb` | `cwfs` (4 corners, v2, ai_donut) |
| `wfs_mimic_covariance.ipynb` | `cwfs` (mimic covariance reader) |
| `aos_danish_tarts_compare_20260713.ipynb` | `processing_compare` (Danish vs TARTS) |
| `study_compare_donuts.ipynb` | `processing_compare` — **TODO: port to a pipeline script** |
| `smatrix_vmode_info.ipynb` | `smatrix_vmode` |
| `vmode_dof_ts_ofc.ipynb` | `smatrix_vmode` (ts_ofc StateEstimator normalization) |
| `jk_coverage_plots.ipynb` | `smatrix_vmode` (50-DOF SVD visualizations) |
| `snippets.ipynb`, `moresnippets.ipynb`, `danish_snippets.ipynb` | **untracked scratch** — not part of any study |

## Output layout

Outputs are keyed by `param_set` (a Butler collection × processing variant) and then by
`mi_name` (a measured-intrinsic build):

```
output/<param_set>/
  chunks/<dmin>_<dmax>/{donuts,fits,visits}.parquet   # per date chunk
  {donuts,fits,visits}.parquet                        # combined -> all downstream input
  <mi_name>/
    build/rot_<lo>_<hi>/intrinsic_grid.parquet
    intrinsic_split_{maps,decomp,rms}.parquet
    fits.parquet                                      # MI-refit DZ
    plots/                                            # <- flat, mixes 4 studies
```

**Known wart, to be fixed in Phase 7:** `<mi_name>/plots/` is a flat dump holding
`dz_correlations*`, `vmode_correlations*`, `thermal_correlations*`, `dz_explained*`,
`bounce_*`, and `fam_coadd_miw_maps.pdf` side by side — four studies' outputs in one
directory. Also loose at the wrong level: `vmode_dof_matrix_*.pdf` and
`visits_check.pdf` inside a param_set, and `output/miw_cwfs_intrinsic_check.parquet` at
the very top.

## Where the physics is written down

Do not re-derive these; link to them.

| doc | covers |
|---|---|
| `miw_coadd_equations.md` | the coadd-vs-MIW derivations — reference for `coadd` |
| `status/miw_investigation_handoff.md` | portable state of the MIW investigation, **with retracted claims** |
| `camera_gravity.md` | camera-gravity model — part of `static_optics` |
| `ts_wep_zernike_intrinsics.md` | how ts_wep/Danish compute the intrinsic; `zk_*` column meanings |
| `double_zernike_convention_validation.md` | DZ index/normalization conventions |
| `../../smatrix/docs/conventions.md` | sensitivity-matrix sign and unit conventions |
| `../../smatrix/docs/miw_astig_coma_investigation.md` | the high-field-order Z5–Z8 excess |

## Open questions and parked work

Carried here so they are visible in one place; detail in each study doc.

- **`coadd`** — the retrieval-bias hypothesis is the live explanation for the
  coadd↔MIW disagreement; see the handoff's **retracted claims** section before
  repeating any earlier conclusion.
- **`coadd`** — `analyze_miw_field_order.py` currently **fails on stale data**:
  `block_grids.npz` (130 umode rows) and `coadd_metrics_rebin3.parquet` (221 rows) in
  `coadd_50_34/` are from different runs. Needs a regenerate, or an explicit assert
  instead of an `IndexError`. See `status/code_review_findings.md`.
- **`cwfs`** — `run_wfs_refit_ensemble.py` and `run_wfs_fam_refit_compare.py` are
  **parked** pending Danish-1.2 FAM reprocessing.
- **`cwfs`** — the Z11/Z14 intra- vs extra-focal split is **unexplained** and is not a
  known instrumental effect.
- **`miw`** — 83 % of MIW **power** sits above the `k<=6` focal orders the build fits,
  which reframes any DZ-subspace analysis.
- **`smatrix_vmode`** — `analyze_sensitivity_sparse.py` and
  `analyze_sparse_observability.py` still hardcode a `/Users/roodman` data path and
  will fail on S3DF.
- **`processing_compare`** — `study_compare_donuts.ipynb` is still a notebook; porting
  it to a pipeline script is a standing TODO.
