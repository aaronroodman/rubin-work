# AOS analysis studies — inventory

> **Status:** current · **Last updated:** 2026-09-22 · **Kind:** reference (inventory)

Inventory of the fifteen studies in the `aos/` directory: the code implementing each one,
what it reads and writes, and its current state. Most draw on a common base — the
Full Array Mode (FAM) donut tables and the Optical Feedback Control (OFC) sensitivity
matrix — but are otherwise independent lines of work. Two work instead from the repository's
value-added Engineering Facility Database (EFD) and Consolidated Database (ConsDB) store:
`science_lut`, from science exposures, and `fam_focus`, from the in-focus visit of each FAM
triplet.

Per-study detail is in `studies/<study>.md`; the Snakemake pipeline that produces the
shared inputs is documented in [`miw_pipeline.md`](miw_pipeline.md).

## The studies

Counts are of files and lines in `aos/code/`, including the shared modules that stay
flat there, and of Snakemake rules driving each study. Of
the 90 Python files, 18 are referenced by the Snakefile and the rest are standalone. The
`science_lut` count is of its two current scripts; the `(+4)` marks four superseded ones still
present on disk.

| study | files | lines | pipeline rules | content |
|---|---|---|---|---|
| [`miw`](studies/miw.md) | 1 | 219 | 2 | Construction of the Measured Intrinsic Wavefront (MIW) from FAM donut data; the build itself is in the external `ts_intrinsic_wavefront` package |
| [`fam_processing`](studies/fam_processing.md) | 9 | 3389 | 0 | Auditing the FAM chunk build: pre-flight checks, Butler provenance, coverage maps, an all-chunks status roll-up, and the recast of Danish 1.3 unpaired output into the paired Danish 1.2 table schema |
| [`dzfit`](studies/dzfit.md) | 2 | 604 | 2 | Validation of the per-visit Double Zernike (DZ) fit against the batoid design intrinsic |
| [`coadd`](studies/coadd.md) | 9 | 3493 | 1 | Per-block FAM wavefront coadds compared against the MIW, and the retrieval-bias model for their disagreement |
| [`correlations`](studies/correlations.md) | 4 | 1466 | 4 | Correlations of the residual Double Zernikes (DZ) with each other, with v-modes, and with telemetry |
| [`cwfs`](studies/cwfs.md) | 9 | 2824 | 4 | Optical state recovered from the Corner Wavefront Sensors (CWFS) compared with the FAM full-focal-plane measurement |
| [`bounce`](studies/bounce.md) | 2 | 1393 | 2 | Elevation and rotator bounce test data, for Look-Up-Table (LUT) development |
| [`lut`](studies/lut.md) | 1 | 308 | 1 | Averaged DOF look-up table from the FAM Double Zernike fits, collapsed over all elevation and rotator angles |
| [`science_lut`](studies/science_lut.md) | 2 (+4) | 4459 | 0 | Focus look-up table from science exposures: the uniform-defocus error of the CWFS optical state predicted from thermal telemetry alone, its per-band calibration, and the absence of any remaining elevation dependence |
| [`fam_focus`](studies/fam_focus.md) | 1 | 1353 | 0 | Focus drift — v-mode 1 — against exposure sequence number within contiguous FAM blocks at fixed pointing, raw and corrected by the `science_lut` thermal model, and against the DZ defocus term of each triplet's own FAM pair |
| [`psf`](studies/psf.md) | 2 | 560 | 0 | Expected PSF from the optical contribution: focal-plane FWHM, ellipticity and shape maps rendered from a given wavefront |
| [`processing_compare`](studies/processing_compare.md) | 2 | 914 | 0 | Agreement between two reductions of the same donut data across code versions, binnings and fitting algorithms |
| [`static_optics`](studies/static_optics.md) | 8 | 1881 | 0 | Whether a static optical figure — mirror surface, camera lenses, or gravitational flexure — reproduces the MIW |
| [`closed_loop`](studies/closed_loop.md) | 1 | 237 | 0 | AOS closed-loop control simulated over a FAM visit sequence, and the delivered PSF that results |
| [`infra`](studies/infra.md) | 1 | 126 | 0 | Node CPU and memory capability, for sizing pipeline concurrency |

Every file in `aos/code/` belongs to exactly one study.

## Code layout

`aos/code/` is organized by study, one subdirectory each, listed here in the same
general-to-specialized order as `../README.md`: `miw/`, `dzfit/`,
`coadd/`, `correlations/`, `cwfs/`, `bounce/`, `lut/`, `fam_processing/`, `psf/`,
`processing_compare/`,
`static_optics/`, `closed_loop/`, `infra/`.

Seven modules stay flat at `aos/code/`:

| file | why it stays flat |
|---|---|
| `aos_state.py` | **15 references from other topics** by bare module name |
| `aos_fwhm.py` | used by `cwfs` and `correlations` |
| `fam_selection.py` | FAM visit selection + DZ column helper; used by all four `correlations` scripts |
| `miw_io.py` | reads the MIW parquet field maps; used by `static_optics` |
| `dz_plotting.py` | used by `dzfit` and `correlations` |
| `psf_maps_lib.py` | star sampling, MIW lookup, DZ residuals, page layout; used by `psf` and `closed_loop` |
| `test_m1m3.py` | manual EFD probe belonging to no single study |

`aos_state.py` is effectively **shared infrastructure**, not aos-private: sibling topics
reach it via a hardcoded `sys.path.insert(.../aos/code)`. The engineering telemetry that
used to sit beside it — Trim, the LUT, the ConsDB transformed-EFD path, wind and
camera-body temperatures — is now in `common/`. See the root `CLAUDE.md` "Topic
independence and shared code", and `../CLAUDE.md` in this topic.

## Notebooks

Notebooks live in `notebooks/<study>/`, mirroring `code/<study>/`.

| notebook | content |
|---|---|
| `notebooks/miw/aos_miw_ocs_ccs_maps.ipynb` | reads the OCS and CCS MIW split maps |
| `notebooks/cwfs/aos_miw_cwfs_intrinsic_check.ipynb` | verifies the Butler-ingested `intrinsicZernikes` calibration against what ts_wep computes per detector |
| `notebooks/cwfs/wfs_corner_compare_correlations.ipynb` | corner comparison, 8 half-sensors, TARTS |
| `notebooks/cwfs/wfs_corner_compare_correlations-aidonut.ipynb` | corner comparison, 4 corners, ai_donut |
| `notebooks/cwfs/wfs_mimic_covariance.ipynb` | reads the mimic covariance product |
| `notebooks/processing_compare/aos_danish_tarts_compare_20260713.ipynb` | Danish versus TARTS on one day_obs |
| `notebooks/processing_compare/study_compare_donuts.ipynb` | cross-param_set donut comparison — **TODO: port to a pipeline script** |
| `notebooks/fam_processing/fam_telemetry_history.ipynb` | per-visit telemetry time histories and distributions, one quantity per group |
| `notebooks/fam_processing/blitz_vs_danish12_20260315.ipynb` | Danish 1.3 blitz unpaired output against Danish 1.2 on one FAM triplet: column census, table metadata and Butler input provenance, Noll basis, matched-donut Zernike and blur comparison per side of focus and averaged over the two sides |
| `notebooks/fam_processing/blitz_cwfs_vs_danish12_20260315.ipynb` | the same comparison on the corner wavefront sensors, where Danish 1.2 pairs two different stars across the SW0 and SW1 half-sensors and Danish 1.3 fits each side separately: one reference visit in detail, then all 62 visits of the night pooled for the per-Noll Zernike statistics |

`snippets.ipynb`, `moresnippets.ipynb` and `danish_snippets.ipynb` in the topic root are
untracked scratch, gitignored, and belong to no study.

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
      closed_loop/                         # closed-loop simulation pages
      bounce/                             # bounce-test Δ, PDFs and parquets together
      lut/                                # DOF look-up table
      wfs/<cwfs_variant>/, wfs_mimic/     # MIW-subtracted corner-WFS products
      plots/                              # coadd-vs-MIW maps
  archive/                                # superseded param_sets
  camera_gravity/                         # static_optics; no param_set dependence
  science_lut/                            # science-exposure focus LUT; reads the value-added DB
  danish_tarts_compare_<day_obs>/         # dated processing comparison
```

Which level a study writes to follows one rule: **if the product changes when a
different MIW build is chosen, it lives under `<mi_name>/`; otherwise under
`<param_set>/`.** So `correlations` and `bounce` are under `<mi_name>/` because they run
on the MIW-subtracted fits, while `dzfit`'s validation is under `<param_set>/` because
it precedes any MIW. The OFC sensitivity matrix's own mode structure has no `param_set`
dependence at all and is not an `aos` study — it is `vmode` in the `smatrix` topic
([`../../smatrix/docs/studies/vmode.md`](../../smatrix/docs/studies/vmode.md)), writing to
`smatrix/output/vmode/`.

`psf/` and `closed_loop/` are under `<mi_name>/`: both read the MIW split maps and the
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
  known instrumental effect. It appears in the Danish 1.3 unpaired corner output too: over
  `day_obs` 20260315 the two half-sensors straddle the Danish 1.2 joint fit on 14 of 21 Noll
  terms, most strongly on Z11 and Z14. See [`fam_processing`](studies/fam_processing.md).
- **`fam_processing`** — the Danish 1.3 blitz Double Zernike fits carry a real per-Noll mean
  offset against Danish 1.2 on the astigmatism and coma terms, largest on Z7 Coma_y and Z6
  Astig0. Whether that is a Danish version difference or a convention difference is
  unresolved.
- **`fam_processing`** — on the corner sensors, whether the mean of the two unpaired halves
  or the extra-focal half alone is the better estimator of the Danish 1.2 joint fit depends
  on the metric and on Noll order; neither is uniformly better at n = 1215 pairs.
- **`miw`** — 83 % of MIW **power** sits above the `k<=6` focal orders the build fits,
  which reframes any DZ-subspace analysis.
- **`processing_compare`** — `study_compare_donuts.ipynb` is still a notebook; porting
  it to a pipeline script is a standing TODO.

## Shared helpers

Helpers used by more than one study live in one place rather than being copied:

| helper | location | used by |
|---|---|---|
| `nmad(x)` — normalized median absolute deviation, robust sigma | `common/utils.py` | `cwfs`, `processing_compare` |
| `alt_to_deg(alt)` — altitude in degrees, auto-detecting radian input | `common/utils.py` | `bounce`, `cwfs` |
| `dz_coeff_columns(df, prefix)` — DZ coefficient column names | `aos/code/fam_selection.py` | all four `correlations` scripts |
| `fam_quality_selection(df, ...)` — which FAM visits are usable | `aos/code/fam_selection.py` | all four `correlations` scripts |
| `repo_root(start)` — repo root for notebooks | `common/utils.py` | notebooks |

`alt_to_deg` detects radians by magnitude: if the largest absolute value is below 2*pi it
converts, otherwise it assumes degrees. That misidentifies genuine degree values that all
fall below 6.28 deg, which real Rubin altitudes never do — the docstring says so.

Two helpers that were previously duplicated with non-equivalent bodies are now unified,
each in one place: `fam_quality_selection` (was four copies of `quality_cut` implementing
two different cuts) and `load_miw` (was four copies with four row cuts). The reasoning and
the verification are in
[`status/code_review_backlog.md`](status/code_review_backlog.md).
