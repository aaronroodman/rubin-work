# AOS analysis studies — inventory

> **Status:** current · **Last updated:** 2026-09-06 · **Kind:** reference (inventory)

Inventory of the thirteen studies in the `aos/` directory: the code implementing each one,
what it reads and writes, and its current state. All thirteen draw on a common base — the
Full Array Mode (FAM) donut tables and the Optical Feedback Control (OFC) sensitivity
matrix — but are otherwise independent lines of work.

Per-study detail is in `studies/<study>.md`; the Snakemake pipeline that produces the
shared inputs is documented in [`miw_pipeline.md`](miw_pipeline.md).

## The studies

Counts are of files and lines in `aos/code/`, including the shared modules that stay
flat there, and of Snakemake rules driving each study. Of
the 57 Python files, 16 are referenced by the Snakefile and the rest are standalone.

| study | files | lines | pipeline rules | content |
|---|---|---|---|---|
| [`miw`](studies/miw.md) | 1 | 219 | 2 | Construction of the Measured Intrinsic Wavefront (MIW) from FAM donut data; the build itself is in the external `ts_intrinsic_wavefront` package |
| [`dzfit`](studies/dzfit.md) | 6 | 1598 | 2 | Validation of the per-visit Double Zernike (DZ) fit against the batoid design intrinsic, and quality checks on the donut data |
| [`coadd`](studies/coadd.md) | 9 | 3493 | 1 | Per-block FAM wavefront coadds compared against the MIW, and the retrieval-bias model for their disagreement |
| [`cwfs`](studies/cwfs.md) | 9 | 2824 | 4 | Optical state recovered from the Corner Wavefront Sensors (CWFS) compared with the FAM full-focal-plane measurement |
| [`static_optics`](studies/static_optics.md) | 8 | 1881 | 0 | Whether a static optical figure — mirror surface, camera lenses, or gravitational flexure — reproduces the MIW |
| [`telemetry`](studies/telemetry.md) | 6 | 1602 | 0 | Per-visit telescope state from the EFD and ConsDB: commanded degrees of freedom (DOF), hexapod look-up tables, temperatures |
| [`correlations`](studies/correlations.md) | 4 | 1466 | 4 | Correlations of the residual Double Zernikes (DZ) with each other, with v-modes, and with telemetry |
| [`smatrix_vmode`](studies/smatrix_vmode.md) | 4 | 894 | 1 | Structure of the OFC sensitivity matrix: singular value decomposition, v-mode composition, DOF observability |
| [`bounce`](studies/bounce.md) | 2 | 1393 | 2 | Elevation and rotator bounce test data, for Look-Up-Table (LUT) development |
| [`processing_compare`](studies/processing_compare.md) | 2 | 914 | 0 | Agreement between two reductions of the same donut data across code versions, binnings and fitting algorithms |
| [`psf`](studies/psf.md) | 2 | 560 | 0 | Expected PSF from the optical contribution: focal-plane FWHM, ellipticity and shape maps rendered from a given wavefront |
| [`closedloop`](studies/closedloop.md) | 1 | 237 | 0 | AOS closed-loop control simulated over a FAM visit sequence, and the delivered PSF that results |
| [`infra`](studies/infra.md) | 1 | 126 | 0 | Node CPU and memory capability, for sizing pipeline concurrency |

Every file in `aos/code/` belongs to exactly one study.

## Code layout

`aos/code/` is organized by study, one subdirectory each: `dzfit/`, `miw/`, `coadd/`,
`cwfs/`, `static_optics/`, `correlations/`, `smatrix_vmode/`, `bounce/`,
`processing_compare/`, `psf/`, `closedloop/`, `infra/`.

Nine modules stay flat at `aos/code/`:

| file | why it stays flat |
|---|---|
| `aos_trim.py` | **21 references from 4 other topics** (`blocks/`, `olr/`, `optatmo/`, `guider/`) by bare module name |
| `aos_state.py` | **15 references from other topics**, same mechanism |
| `aos_consdb_efd.py` | **3 references from `blocks/`** |
| `aos_fwhm.py` | used by `cwfs` and `correlations` |
| `fam_selection.py` | FAM visit selection + DZ column helper; used by all four `correlations` scripts |
| `miw_io.py` | reads the MIW parquet field maps; used by `static_optics` |
| `dz_plotting.py` | used by `dzfit` and `correlations` |
| `psf_maps_lib.py` | star sampling, MIW lookup, DZ residuals, page layout; used by `psf` and `closedloop` |
| `run_backfill_thermal.py`, `run_backfill_camera_telemetry.py`, `test_m1m3.py` | telemetry utilities belonging to no single study |

The first three are effectively **shared infrastructure**, not aos-private: sibling
topics reach them via a hardcoded `sys.path.insert(.../aos/code)`. See the root
`CLAUDE.md` "Topic independence and shared code", and `../CLAUDE.md` in this topic.

## Notebook → study

| notebook | study |
|---|---|
| `aos_miw_ocs_ccs_maps.ipynb` | `miw` (OCS/CCS split-map reader) |
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

Outputs are keyed by `param_set` — a Butler collection paired with a processing variant
— and then, where the product depends on which Measured Intrinsic Wavefront build was
used, by `mi_name`. Within each level the products are grouped by study, so a study's
output is found by name rather than by searching a shared `plots/` directory.

```
output/
  <param_set>/
    {donuts,fits,visits}.parquet          # combined tables, input to everything
    chunks/<dmin>_<dmax>/                 # per-chunk tables
    dzfit/                                # DZ-fit validation: trio comparison, aberration pairs
    processing_compare/                   # cross-param_set comparison
    wfs/<cwfs_variant>/                   # corner-WFS ingest and corner comparison
    coadd_50_34/, coadd_50_34_v2/         # per-block coadd vs MIW
    <mi_name>/
      intrinsic_split_{maps,decomp,rms}.parquet, intrinsic_split.pdf
      study_radialbins.pdf, zk_intrinsic.parquet
      fits.parquet                        # DZ refit against the MIW
      correlations/                       # DZ, v-mode and thermal correlations
      psf/                                # focal-plane PSF maps
      closedloop/                         # closed-loop simulation pages
      bounce/                             # bounce-test Δ, PDFs and parquets together
      lut/                                # DOF look-up table
      wfs/<cwfs_variant>/, wfs_mimic/     # MIW-subtracted corner-WFS products
      plots/                              # coadd-vs-MIW maps
  smatrix_vmode/                          # OFC matrix diagnostics; no param_set dependence
  archive/                                # superseded param_sets
  camera_gravity/                         # static_optics; no param_set dependence
  danish_tarts_compare_<day_obs>/         # dated processing comparison
```

Which level a study writes to follows one rule: **if the product changes when a
different MIW build is chosen, it lives under `<mi_name>/`; otherwise under
`<param_set>/`.** So `correlations` and `bounce` are under `<mi_name>/` because they run
on the MIW-subtracted fits, while `dzfit`'s validation is under `<param_set>/` because
it precedes any MIW. `smatrix_vmode` sits at the **top level**, outside any
`param_set`: the v-mode/DOF matrix is a property of the OFC sensitivity matrix and the
DOF scheme alone, and the one data-derived input (the pupil-Zernike set) is identical
in every `param_set` built to date, so it defaults in code.

`psf/` and `closedloop/` are under `<mi_name>/`: both read the MIW split maps and the
per-visit FAM fits from a single measured-intrinsic build. They previously mixed two
builds via `--split-mi` and `--fam-mi`; that collapsed to one `--mi` on 2026-09-07.

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

## Shared helpers

Helpers used by more than one study live in one place rather than being copied:

| helper | location | used by |
|---|---|---|
| `nmad(x)` — normalized median absolute deviation, robust sigma | `common/utils.py` | `cwfs`, `processing_compare` |
| `alt_to_deg(alt)` — altitude in degrees, auto-detecting radian input | `common/utils.py` | `bounce`, `cwfs` |
| `dz_coeff_columns(df, prefix)` — DZ coefficient column names | `aos/code/dz_columns.py` | all four `correlations` scripts |
| `repo_root(start)` — repo root for notebooks | `common/utils.py` | notebooks |

`alt_to_deg` detects radians by magnitude: if the largest absolute value is below 2*pi it
converts, otherwise it assumes degrees. That misidentifies genuine degree values that all
fall below 6.28 deg, which real Rubin altitudes never do — the docstring says so.

Two similar-looking helpers are **deliberately not shared**, because the copies are not
equivalent:

- **`quality_cut`** appears in four `correlations` scripts with two different signatures
  (`max_coeff_um` versus `maxc`) and differing bodies.
- **`load_miw`** appears in four scripts across `static_optics` and `coadd` with four
  different behaviours, some taking a `stride` argument and some not.

Unifying either would change results, not just structure, so each needs its own decision
about what the single correct behaviour is.
