# AOS

Analysis and Development for the Rubin Observatory Active Optics System (AOS). This
directory contains multiple studies of AOS engineering data and development of
calibrations and methods for AOS operation. These include the construction of the
Measured Intrinsic Wavefront (MIW) from Full Array Mode (FAM) data, study of
correlations between Double Zernike (DZ), v-modes and Rubin telemetry, analysis of
bounce test data for Look-Up-Tables (LUT), and comparisons between FAM and Corner
Wavefront Sensor (CWFS) data.

## Processing

Every study draws on a common set of tables built from the Full Array Mode (FAM)
observations, in date-range **chunks** keyed by a `param_set` — a Butler collection paired
with a processing variant. Because the studies all consume these tables and none of them
owns the code that makes them, the processing is described here rather than as a study of
its own.

### The code lives in an external package

The build steps run from **`ts_intrinsic_wavefront`** (`lsst.ts.intrinsic.wavefront`), a
separate LSST-TS repository, not from `aos/code/`. The Snakefile locates it through the
environment:

```python
_WF_DIR = os.environ["TS_INTRINSIC_WAVEFRONT_DIR"]
WF_BIN  = f"{_WF_DIR}/bin"            # scons-built shims of bin.src/
WF_LIB  = f"{_WF_DIR}/python/lsst/ts/intrinsic/wavefront"
```

So a fix to build, fit or split *logic* usually belongs in that package. After editing it,
`scons` must be re-run: it generates the gitignored `version.py` and the `bin/` shims.

### The chain

| rule | script | produces |
|---|---|---|
| `mktable` | `{WF_BIN}/run_mktable.py` | `output/<ps>/chunks/<dmin>_<dmax>/{donuts,visits}.parquet` |
| `fit` | `{WF_BIN}/run_dz_fit.py` | `chunks/<dmin>_<dmax>/fits.parquet` |
| `combine_donuts` / `combine_fits` / `combine_visits` | `{WF_BIN}/combine_parquets.py` | `output/<ps>/{donuts,fits,visits}.parquet` |

`mktable` queries the Butler for donut Zernikes per visit and **fetches the telemetry in
the same step** — it takes `--no-thermal`, `--temp-time-window` and `--consdb-url`, and
passes `include_thermal` down to `intrinsics_lib.run_mktable`. It is the expensive step and
is *deliberately* not re-triggered by code edits; see the Snakefile comments.

`combine_parquets.py` unifies chunk schemas as their intersection, so a 0-row sentinel
chunk would silently drop columns. It now skips empty inputs instead.

### Telemetry

Per-visit telescope state comes from the Engineering Facility Database (EFD) and the
Consolidated Database (ConsDB): air and structural temperatures, wind and airflow,
commanded degrees of freedom (DOF), and hexapod and mirror look-up-table (LUT) values.

`mktable` merges the thermal columns as it builds a chunk.
`code/fam_processing/run_attach_telemetry.py` attaches the full set in one pass, reading
ConsDB where it carries a quantity for Full Array Mode (FAM) exposures and the EFD
otherwise:

| group | columns | source |
|---|---|---|
| thermal | ESS air temperatures, their differences, TMA truss | ConsDB |
| gradients | M1M3 spatial temperature gradients | EFD |
| wind | inside and outside wind speed and direction, sonic temperature | ConsDB |
| camera | 24 camera-body, housing, lens and shutter temperatures | EFD |
| lut | M1M3 elevation and M2 gravity axial forces | ConsDB |
| trim | the 50 accumulated DOF offsets | EFD |
| tweak | the per-iteration DOF correction, differenced from Trim | derived |

Each chunk gets a `telemetry.parquet` holding every fetched column; `--merge` joins it
into the per-chunk and combined `visits.parquet`.

Supporting modules at `code/`: `aos_trim.py` (Trim and the LUT fetchers), `aos_state.py`
(per-visit optical-state helpers), `aos_consdb_efd.py` (bulk ConsDB telemetry). EFD and
ConsDB clients come from [`../common/telemetry_clients.py`](../common/telemetry_clients.py).

[`docs/telemetry.md`](docs/telemetry.md) inventories every quantity with its ConsDB or
EFD name, measured coverage on FAM exposures, and units. Trim is absent from ConsDB for
FAM exposures and is read from the EFD by time;
[`docs/status/dof_telemetry_availability.md`](docs/status/dof_telemetry_availability.md)
holds that measurement.

### Reviewing the processed chunks

The [`fam_processing`](docs/studies/fam_processing.md) study holds the tools for checking a
chunk before and after it is built — pre-flight surveys, Butler provenance consistency,
coverage maps, and an all-chunks status roll-up.

### Terminology — do not interchange

`optical_state`, **Tweak** and **Trim** are different quantities:
`optical_state` is recovered from the measured wavefront, `Tweak = PID(optical_state)` is
the per-iteration correction, and `Trim` is the accumulated offset the EFD reports as
`aggregatedDoF0..49`. See `../notes/claude-memory/aos-dof-terminology.md`. The 22-DOF
reduced set has **specific indices** and is not the first 22 — use `aos_state.DOF22`.

## Studies

The work divides into fourteen studies, ordered here from the most general to the most
specialized. Each has a detailed document under `docs/studies/`;
[`docs/studies.md`](docs/studies.md) is the combined inventory, listing the code, inputs,
outputs and current state of every one.

| study | content |
|---|---|
| [`smatrix_vmode`](docs/studies/smatrix_vmode.md) | Structure of the Optical Feedback Control (OFC) sensitivity matrix: its singular value decomposition, v-mode composition, and degree-of-freedom (DOF) observability |
| [`miw`](docs/studies/miw.md) | Construction of the Measured Intrinsic Wavefront (MIW) from Full Array Mode (FAM) donut data |
| [`fam_processing`](docs/studies/fam_processing.md) | Auditing the FAM chunk build: pre-flight checks, Butler provenance consistency, coverage, and an all-chunks status roll-up |
| [`dzfit`](docs/studies/dzfit.md) | Validation of the per-visit Double Zernike (DZ) fit against the batoid design intrinsic |
| [`coadd`](docs/studies/coadd.md) | Comparison of per-block FAM wavefront coadds against the MIW, and the retrieval-bias model for their disagreement |
| [`correlations`](docs/studies/correlations.md) | Correlations of the residual Double Zernikes with each other, with v-modes, and with telemetry |
| [`cwfs`](docs/studies/cwfs.md) | Comparison of the optical state recovered from the Corner Wavefront Sensors (CWFS) with the FAM full-focal-plane measurement |
| [`bounce`](docs/studies/bounce.md) | Elevation and rotator bounce test data, for Look-Up-Table (LUT) development |
| [`lut`](docs/studies/lut.md) | Averaged degree-of-freedom (DOF) look-up table built from the FAM Double Zernike fits, collapsed over all elevation and rotator angles |
| [`psf`](docs/studies/psf.md) | Expected Point Spread Function (PSF) from the optical contribution: focal-plane full width at half maximum (FWHM), ellipticity and shape maps |
| [`processing_compare`](docs/studies/processing_compare.md) | Agreement between two reductions of the same donut data across code versions, binnings and fitting algorithms |
| [`static_optics`](docs/studies/static_optics.md) | Whether a static optical figure — mirror surface, camera lenses, or gravitational flexure — reproduces the MIW |
| [`closed_loop`](docs/studies/closed_loop.md) | AOS closed-loop control simulated over a FAM visit sequence, and the delivered PSF that results |
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

## Data dependencies

- **Butler and ConsDB/EFD, RSP only:** the donut and corner-WFS table builds
  (`mktable`, `wfs_mktable`) and the telemetry attachment.
- **`lsst.ts.ofc`, `lsst.ts.wep`, `$TS_CONFIG_MTTCS_DIR`, batoid height maps:** the MIW
  build and anything projecting onto the OFC sensitivity matrix. These packages are not
  part of `lsst_distrib` and need the AOS/CWFS environment.
- **Parquet only:** the DZ fits, combines, validation plots, aberration pairs and
  processing comparisons need no Butler or AOS packages — they run wherever the tables
  exist.

## Output layout

Keyed by `param_set` — a Butler collection paired with a processing variant — then by
`mi_name` for products that depend on which MIW build was used. Within each level,
output is grouped by study:

```
output/
  <param_set>/
    {donuts,fits,visits}.parquet   # combined tables, input to everything
    chunks/<dmin>_<dmax>/          # per-chunk tables
    dzfit/  processing_compare/  wfs/<variant>/
    coadd_50_34/  coadd_50_34_v2/
    <mi_name>/
      intrinsic_split_{maps,decomp,rms}.parquet  # the MIW itself
      fits.parquet                               # DZ refit against the MIW
      correlations/  bounce/  psf/  closed_loop/  lut/  wfs/<variant>/  wfs_mimic/
  smatrix_vmode/                   # OFC matrix diagnostics, no param_set dependence
  camera_gravity/                  # static_optics, no param_set dependence
  archive/                         # superseded param_sets
```

A study writes under `<mi_name>/` when its result depends on which MIW build was used,
under `<param_set>/` when it does not, and at the top level when it depends on neither.

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
| [`docs/studies.md`](docs/studies.md) | **inventory of the 13 analysis studies** — the map for this topic |
| [`docs/miw_pipeline.md`](docs/miw_pipeline.md) | Snakemake pipeline reference: every rule, config, output path |
| [`docs/telemetry.md`](docs/telemetry.md) | **telemetry inventory** — every quantity, its ConsDB/EFD name, measured coverage on FAM exposures, units, and which source to prefer |
| [`docs/miw_coadd_equations.md`](docs/miw_coadd_equations.md) | MIW notation and the coadd-vs-MIW residual, derived at equation level |
| [`docs/camera_gravity.md`](docs/camera_gravity.md) | whether camera-lens gravitational flexure can produce the MIW astig/coma excess |
| [`docs/ts_wep_zernike_intrinsics.md`](docs/ts_wep_zernike_intrinsics.md) | how ts_wep + Danish compute the off-axis intrinsic; what the `zk_*` columns mean |
| [`docs/double_zernike_convention_validation.md`](docs/double_zernike_convention_validation.md) | validation of the DZ index/normalization conventions, vs GalSim and `ts_ofc` |
| [`docs/status/miw_investigation_handoff.md`](docs/status/miw_investigation_handoff.md) | portable state of the MIW investigation, with an explicit list of retracted claims |
| [`docs/status/rerun_needed.md`](docs/status/rerun_needed.md) | outputs that predate a code change and need regenerating |
| [`docs/status/code_review_backlog.md`](docs/status/code_review_backlog.md) | open review items: non-equivalent duplicate helpers, confirmed live defects, `common/` candidates |
| [`docs/status/code_review_findings.md`](docs/status/code_review_findings.md) | the earlier full review — **stale line anchors**, verify before acting |
| [`CLAUDE.md`](CLAUDE.md) | conventions and known pitfalls for this directory: frames, units, and which code lives in the external `ts_intrinsic_wavefront` package |

Related, in sibling topics: [`../smatrix/docs/conventions.md`](../smatrix/docs/conventions.md)
(DZ sensitivity-matrix sign/unit conventions) and
[`../smatrix/docs/miw_astig_coma_investigation.md`](../smatrix/docs/miw_astig_coma_investigation.md).
