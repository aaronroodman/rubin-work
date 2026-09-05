# AOS — Active Optics wavefront analysis

Analysis of the Rubin Active Optics System, mostly from Full Array Mode (FAM) donut
data: per-donut wavefront tables, Double-Zernike (DZ) fits, the **Measured Intrinsic
Wavefront (MIW)**, corner-wavefront-sensor comparisons, DOF look-up tables, and
operational studies. A Snakemake pipeline runs the MIW chain per `param_set`; a larger
set of standalone analyses hangs off its outputs.

This is the **largest topic in the repo** — 56 Python files, 13 notebooks — and it is
not one project. It is **eleven separable studies**.

## Start here

| doc | what it gives you |
|---|---|
| **[`docs/studies.md`](docs/studies.md)** | the map: all 11 studies, what each asks, which code implements it, what it reads and writes, current state |
| [`docs/miw_pipeline.md`](docs/miw_pipeline.md) | step-by-step reference for the Snakemake pipeline (every rule, config file, output path) |
| [`CLAUDE.md`](CLAUDE.md) | the traps — frames, units, which code lives outside this repo, open questions not to present as settled |

## The studies

Detail for each is in `docs/studies/<study>.md`.

| study | what it asks | pipeline? |
|---|---|---|
| [`miw`](docs/studies/miw.md) | What is the measured intrinsic wavefront, and is the build trustworthy? | 4 rules |
| [`coadd`](docs/studies/coadd.md) | Why does the per-block FAM coadd disagree with the MIW? | 1 rule |
| [`cwfs`](docs/studies/cwfs.md) | Does the corner WFS recover the same optical state as FAM? | 4 rules |
| [`static_optics`](docs/studies/static_optics.md) | Can any static optical figure (M3, lenses, gravity) explain the MIW? | standalone |
| [`telemetry`](docs/studies/telemetry.md) | What was the telescope's commanded and thermal state per visit? | standalone |
| [`correlations`](docs/studies/correlations.md) | What does the residual DZ correlate with — itself, v-modes, temperature? | 4 rules |
| [`smatrix_vmode`](docs/studies/smatrix_vmode.md) | What does the sensitivity matrix's mode structure let us observe? | 1 rule |
| [`bounce`](docs/studies/bounce.md) | Do elevation/rotator bounce tests show a repeatable Δ? | 2 rules |
| [`processing_compare`](docs/studies/processing_compare.md) | Do two reductions of the same data agree? | standalone |
| [`psf`](docs/studies/psf.md) | What PSF does a given wavefront actually produce? | standalone |
| [`infra`](docs/studies/infra.md) | (support) How many usable cores does this node have? | standalone |

## Running the pipeline

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh                       # detached run, survives a dropped SSH; logs to logs/
./run_snake.sh -n                    # dry run: what is stale / would run
./run_snake.sh --until combine_donuts
```

Full detail — every rule, the four config files, `mem_mb` throttling, and the
`ts_intrinsic_wavefront` package setup that must be `scons`-built first — is in
[`docs/miw_pipeline.md`](docs/miw_pipeline.md).

**Batch runs are a MUST-ASK**: never submitted from a session. See the root
`CLAUDE.md` "Batch jobs" for the submit-plus-monitor command pair. Batch goes from an
s3df node (`slacrd`), never an RSP pod.

## Data dependencies at a glance

- **Butler + ConsDB/EFD, RSP only:** `mktable`, `wfs_mktable`, and the telemetry
  backfills. The expensive Butler step (`mktable`) is deliberately *not* re-triggered
  by code edits.
- **`lsst.ts.ofc` + `lsst.ts.wep`, `$TS_CONFIG_MTTCS_DIR`, batoid height maps:** all of
  Phase 2/3 and anything projecting onto the OFC SVD. Note these are **not** in
  `lsst_distrib` — they need the AOS/CWFS environment.
- **Parquet only (no LSST stack):** `fit`, `combine_*`, `plots`, `aberration_pairs`,
  and `processing_compare` run anywhere the tables already exist.

## Output layout

Keyed by `param_set` (Butler collection × processing variant), then by `mi_name`
(a measured-intrinsic build):

```
output/<param_set>/
  chunks/<dmin>_<dmax>/{donuts,fits,visits}.parquet   # per date chunk
  {donuts,fits,visits}.parquet                        # combined -> downstream input
  <mi_name>/
    build/rot_<lo>_<hi>/intrinsic_grid.parquet
    intrinsic_split_{maps,decomp,rms}.parquet
    fits.parquet                                      # MI-refit DZ
    lut/, wfs/<cwfs>/, wfs_mimic/
    plots/                                            # flat; being split per study
```

Outputs are **gitignored** and symlinked to
`/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/` on the USDF RSP.
`calibration/` holds frozen, version-controlled calibration products — see
[`calibration/README.md`](calibration/README.md).

## Notebooks

Each notebook belongs to a study; the mapping (all 13, including untracked scratch) is
in [`docs/studies.md`](docs/studies.md#notebook--study).

**AOS control** notebooks — closed-loop performance, separate from the FAM wavefront
work — moved to the `nightlyiq/` topic: `nightly_tablemaker.ipynb`,
`aos_nightly_plots.ipynb`, `aos_openloop.ipynb`.

**Superseded and removed.** Notebooks fully replaced by pipeline steps were deleted
(recoverable from git history); the pipeline step is the reference now:
`intrinsics_fit` → `fit`; `build_measured_intrinsic` → `build_intrinsic`;
`intrinsic_camera_telescope_split` → `intrinsic_split`; `study_aberrationpairs` →
`aberration_pairs`; `study_doublezernike` → `dz_correlations` (+ LUT sections →
`build_lut`); `intrinsics_thermal_correlations` → `thermal_correlations`;
`study_bounce` → `bounce`; `intrinsics_mktable`/`intrinsics_plots` →
`mktable`/`plots`; `study_wfs_mimic` → `wfs_mimic` + `wfs_dof_compare`. The one-off
`intrinsics_checkZ4` / `intrinsic_Zj` Z4-height validation notebooks were also removed.

## Other docs

Reference docs in `docs/`; transient working state in `docs/status/`. Each carries a
status + last-updated line under its title.

| doc | what it holds |
|---|---|
| [`docs/studies.md`](docs/studies.md) | **inventory of the 11 analysis studies** — the map for this topic |
| [`docs/miw_pipeline.md`](docs/miw_pipeline.md) | Snakemake pipeline reference: every rule, config, output path |
| [`docs/miw_coadd_equations.md`](docs/miw_coadd_equations.md) | MIW notation and the coadd-vs-MIW residual, derived at equation level |
| [`docs/camera_gravity.md`](docs/camera_gravity.md) | whether camera-lens gravitational flexure can produce the MIW astig/coma excess |
| [`docs/ts_wep_zernike_intrinsics.md`](docs/ts_wep_zernike_intrinsics.md) | how ts_wep + Danish compute the off-axis intrinsic; what the `zk_*` columns mean |
| [`docs/double_zernike_convention_validation.md`](docs/double_zernike_convention_validation.md) | validation of the DZ index/normalization conventions, vs GalSim and `ts_ofc` |
| [`docs/status/miw_investigation_handoff.md`](docs/status/miw_investigation_handoff.md) | portable state of the MIW investigation, with an explicit list of retracted claims |
| [`docs/status/code_review_findings.md`](docs/status/code_review_findings.md) | review backlog — **stale line anchors**, verify before acting |

Related, in sibling topics: [`../smatrix/docs/conventions.md`](../smatrix/docs/conventions.md)
(DZ sensitivity-matrix sign/unit conventions) and
[`../smatrix/docs/miw_astig_coma_investigation.md`](../smatrix/docs/miw_astig_coma_investigation.md).
