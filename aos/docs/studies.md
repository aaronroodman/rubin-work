# AOS analysis studies — inventory

> **Status:** current · **Last updated:** 2026-09-06 · **Kind:** reference (inventory)

Inventory of the eleven studies in the `aos/` directory: the code implementing each one,
what it reads and writes, and its current state. All eleven draw on a common base — the
Full Array Mode (FAM) donut tables and the Optical Feedback Control (OFC) sensitivity
matrix — but are otherwise independent lines of work.

Per-study detail is in `studies/<study>.md`; the Snakemake pipeline that produces the
shared inputs is documented in [`miw_pipeline.md`](miw_pipeline.md).

## The studies

Counts are of files and lines in `aos/code/`, and of Snakemake rules driving them. Of
the 56 Python files, 16 are pipeline-driven and 40 are standalone.

| study | files | lines | pipeline rules | content |
|---|---|---|---|---|
| [`miw`](studies/miw.md) | 9 | 3207 | 4 | Construction of the Measured Intrinsic Wavefront (MIW) from FAM donut data, and validation of the build |
| [`coadd`](studies/coadd.md) | 9 | 3506 | 1 | Per-block FAM wavefront coadds compared against the MIW, and the retrieval-bias model for their disagreement |
| [`cwfs`](studies/cwfs.md) | 9 | 2828 | 4 | Optical state recovered from the Corner Wavefront Sensors (CWFS) compared with the FAM full-focal-plane measurement |
| [`static_optics`](studies/static_optics.md) | 8 | 1862 | 0 | Whether a static optical figure — mirror surface, camera lenses, or gravitational flexure — reproduces the MIW |
| [`telemetry`](studies/telemetry.md) | 6 | 1602 | 0 | Per-visit telescope state from the EFD and ConsDB: commanded degrees of freedom (DOF), hexapod look-up tables, temperatures |
| [`correlations`](studies/correlations.md) | 4 | 1472 | 4 | Correlations of the residual Double Zernikes (DZ) with each other, with v-modes, and with telemetry |
| [`smatrix_vmode`](studies/smatrix_vmode.md) | 4 | 876 | 1 | Structure of the OFC sensitivity matrix: singular value decomposition, v-mode composition, DOF observability |
| [`bounce`](studies/bounce.md) | 2 | 1394 | 2 | Elevation and rotator bounce test data, for Look-Up-Table (LUT) development |
| [`processing_compare`](studies/processing_compare.md) | 2 | 922 | 0 | Agreement between two reductions of the same donut data across code versions, binnings and fitting algorithms |
| [`psf`](studies/psf.md) | 2 | 776 | 0 | Focal-plane Point Spread Function (PSF) maps rendered from a wavefront — FWHM and ellipticity |
| [`infra`](studies/infra.md) | 1 | 126 | 0 | Node CPU and memory capability, for sizing pipeline concurrency |

Every file in `aos/code/` belongs to exactly one study.

## Cross-cutting code

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

`<mi_name>/plots/` currently holds the output of four separate studies side by side —
`dz_correlations*`, `vmode_correlations*`, `thermal_correlations*`, `dz_explained*`,
`bounce_*` and `fam_coadd_miw_maps.pdf`. Three products also sit at the wrong level:
`vmode_dof_matrix_*.pdf` and `visits_check.pdf` inside a `param_set`, and
`miw_cwfs_intrinsic_check.parquet` at the top of `output/`. Splitting these per study is
outstanding work.

## Supporting documentation

The derivations and conventions underlying these studies:

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
