# AOS

Analysis and Development for the Rubin Observatory Active Optics System (AOS). This
directory contains multiple studies of AOS engineering data and development of
calibrations and methods for AOS operation. These include the construction of the
Measured Intrinsic Wavefront (MIW) from Full Array Mode (FAM) data, study of
correlations between Double Zernike (DZ), v-modes and Rubin telemetry, analysis of
bounce test data for Look-Up-Tables (LUT), and comparisons between FAM and Corner
Wavefront Sensor (CWFS) data.

## Studies

The work divides into eleven studies. Each has a detail document under
`docs/studies/`; [`docs/studies.md`](docs/studies.md) is the combined inventory,
listing the code, inputs, outputs and current state of every one.

| study | content |
|---|---|
| [`miw`](docs/studies/miw.md) | Construction of the Measured Intrinsic Wavefront from FAM donut data, and validation of the build |
| [`coadd`](docs/studies/coadd.md) | Comparison of per-block FAM wavefront coadds against the MIW, and the retrieval-bias model for their disagreement |
| [`cwfs`](docs/studies/cwfs.md) | Comparison of the optical state recovered from the Corner Wavefront Sensors with the FAM full-focal-plane measurement |
| [`static_optics`](docs/studies/static_optics.md) | Whether a static optical figure — mirror surface, camera lenses, or gravitational flexure — reproduces the MIW |
| [`telemetry`](docs/studies/telemetry.md) | Per-visit telescope state from the Engineering Facility Database (EFD) and Consolidated Database (ConsDB): commanded degrees of freedom (DOF), hexapod look-up tables, temperatures |
| [`correlations`](docs/studies/correlations.md) | Correlations of the residual Double Zernikes with each other, with v-modes, and with telemetry |
| [`smatrix_vmode`](docs/studies/smatrix_vmode.md) | Structure of the Optical Feedback Control (OFC) sensitivity matrix: its singular value decomposition, v-mode composition, and DOF observability |
| [`bounce`](docs/studies/bounce.md) | Elevation and rotator bounce test data, for Look-Up-Table development |
| [`processing_compare`](docs/studies/processing_compare.md) | Agreement between two reductions of the same donut data across code versions, binnings and fitting algorithms |
| [`psf`](docs/studies/psf.md) | Focal-plane Point Spread Function (PSF) maps rendered from a wavefront — full width at half maximum (FWHM) and ellipticity |
| [`infra`](docs/studies/infra.md) | Node CPU and memory capability, for sizing pipeline concurrency |

## Pipeline

A Snakemake pipeline builds the donut tables, the Double Zernike fits and the MIW for
each `param_set` — a Butler collection paired with a processing variant. Run it from
this directory:

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh                       # detached run; logs to logs/
./run_snake.sh -n                    # dry run: report what is stale
./run_snake.sh --until combine_donuts
```

[`docs/miw_pipeline.md`](docs/miw_pipeline.md) documents every rule, the four
configuration files, the memory throttling, and the `ts_intrinsic_wavefront` package
setup the pipeline requires.

Batch submission goes from an s3df node (`slacrd`), not a Rubin Science Platform (RSP)
pod, which has no Slurm. Batch jobs are started by Aaron, not from a Claude session —
see the root `CLAUDE.md`.

## Data dependencies

- **Butler and ConsDB/EFD, RSP only:** the donut and corner-WFS table builds
  (`mktable`, `wfs_mktable`) and the telemetry backfills.
- **`lsst.ts.ofc`, `lsst.ts.wep`, `$TS_CONFIG_MTTCS_DIR`, batoid height maps:** the MIW
  build and anything projecting onto the OFC sensitivity matrix. These packages are not
  part of `lsst_distrib` and need the AOS/CWFS environment.
- **Parquet only:** the DZ fits, combines, validation plots, aberration pairs and
  processing comparisons run wherever the tables exist, including the laptop.

## Output layout

Keyed by `param_set` — a Butler collection paired with a processing variant — then by
`mi_name` for products that depend on which MIW build was used. Within each level,
output is grouped by study:

```
output/<param_set>/
  {donuts,fits,visits}.parquet     # combined tables, input to everything
  chunks/<dmin>_<dmax>/            # per-chunk tables
  miw/  psf/  smatrix_vmode/  processing_compare/  wfs/<variant>/
  coadd_50_34/  coadd_50_34_v2/
  <mi_name>/
    intrinsic_split_{maps,decomp,rms}.parquet    # the MIW itself
    fits.parquet                                 # DZ refit against the MIW
    correlations/  bounce/  lut/  wfs/<variant>/  wfs_mimic/
```

A study writes under `<mi_name>/` when its result depends on which MIW build was used,
and under `<param_set>/` otherwise. Superseded param_sets are in `output/archive/`.

Outputs are gitignored, and symlinked to
`/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/` on the USDF RSP.
`calibration/` holds frozen, version-controlled calibration products — see
[`calibration/README.md`](calibration/README.md).

## Notebooks

Each notebook belongs to a study; the mapping is in
[`docs/studies.md`](docs/studies.md#notebook--study).

Closed-loop AOS performance notebooks live in the `nightlyiq/` topic, not here.

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
| [`docs/status/code_review_backlog.md`](docs/status/code_review_backlog.md) | open review items: non-equivalent duplicate helpers, confirmed live defects, `common/` candidates |
| [`docs/status/code_review_findings.md`](docs/status/code_review_findings.md) | the earlier full review — **stale line anchors**, verify before acting |
| [`CLAUDE.md`](CLAUDE.md) | conventions and known pitfalls for this directory: frames, units, and which code lives in the external `ts_intrinsic_wavefront` package |

Related, in sibling topics: [`../smatrix/docs/conventions.md`](../smatrix/docs/conventions.md)
(DZ sensitivity-matrix sign/unit conventions) and
[`../smatrix/docs/miw_astig_coma_investigation.md`](../smatrix/docs/miw_astig_coma_investigation.md).
