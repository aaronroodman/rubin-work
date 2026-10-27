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
| `mktable` | `{WF_BIN}/run_mktable.py` | `output/fam_processing/<P>/chunks/<dmin>_<dmax>/{donuts,visits}.parquet` |
| `fit` | `{WF_BIN}/run_dz_fit.py` | `chunks/<dmin>_<dmax>/fits.parquet` |
| `combine_donuts` / `combine_fits` / `combine_visits` | `{WF_BIN}/combine_parquets.py` | `output/fam_processing/<P>/{donuts,fits,visits}.parquet` |

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
`rubinwork.products.fam_tables.builders.run_attach_telemetry` attaches the full set in one pass, reading
ConsDB where it carries a quantity for Full Array Mode (FAM) exposures and the EFD
otherwise:

| group | columns | source |
|---|---|---|
| thermal | ESS air temperatures, their differences, TMA truss | ConsDB |
| gradients | M1M3 spatial temperature gradients | EFD |
| wind | inside and outside wind speed and direction, sonic temperature | ConsDB |
| camera | 24 camera-body, housing, lens and shutter temperatures | EFD |
| lut | M1M3 elevation and M2 gravity axial forces | ConsDB |
| hexlut | the camera and M2 hexapod look-up-table baseline | EFD |
| trim | the 50 accumulated DOF offsets | EFD |
| tweak | the per-iteration DOF correction, differenced from Trim | derived |

Each chunk gets a `telemetry.parquet` holding every fetched column; `--merge` joins it
into the per-chunk and combined `visits.parquet`.

The optical-state helpers are the library `rubinwork.aos_state`. The telemetry fetchers
are shared across topics and live in `rubinwork/common/` —
[`dof_telemetry.py`](../rubinwork/common/dof_telemetry.py) for Trim and the LUT,
[`consdb_efd.py`](../rubinwork/common/consdb_efd.py) for the bulk ConsDB path, and
[`visit_telemetry.py`](../rubinwork/common/visit_telemetry.py) for wind and camera-body
temperatures. EFD and ConsDB clients come from
[`../common/telemetry_clients.py`](../common/telemetry_clients.py).

[`docs/telemetry.md`](docs/telemetry.md) inventories every quantity with its ConsDB or
EFD name, measured coverage on FAM exposures, and units. Trim and the hexapod LUT are read
from the EFD by time — the ConsDB copy of Trim is absent on FAM exposures, and its hexapod
LUT reaches only a few percent — while the mirror LUT comes from ConsDB;
[`docs/status/dof_telemetry_availability.md`](docs/status/dof_telemetry_availability.md)
holds that measurement.

### Reviewing the processed chunks

The [`fam_processing`](../rubinwork/products/fam_tables/qa/fam_processing.md) study holds the tools for checking a
chunk before and after it is built — pre-flight surveys, Butler provenance consistency,
coverage maps, and an all-chunks status roll-up.

### Terminology — do not interchange

`optical_state`, **Tweak** and **Trim** are different quantities:
`optical_state` is recovered from the measured wavefront, `Tweak = PID(optical_state)` is
the per-iteration correction, and `Trim` is the accumulated offset the EFD reports as
`aggregatedDoF0..49`. See `../notes/claude-memory/aos-dof-terminology.md`. The 22-DOF
reduced set has **specific indices** and is not the first 22 — use `aos_state.DOF22`.

<!-- BEGIN generated: python -m rubinwork.studies index -->

## Studies

Generated from each `<study>/study.md`; edit those, not this table.

| study | status | last log | answer |
|---|---|---|---|
| [bending_mode_test](bending_mode_test/study.md) | parked | 2026-09-18 | In the bending-mode test blocks T377-T380, which hexapod LUT and Trim values were commanded at each corner-sensor exposure? |
| [blitz_vs_danish12](blitz_vs_danish12/study.md) | done | 2026-09-22 | Same convention; the DZ offsets Z7 +0.1388 and Z6 -0.1116 um of wavefront (mean offsets) come from unpairing, not from the code version. |
| [bounce](bounce/study.md) | done | 2026-10-12 | Yes, and the default recovery puts DOF out of range: 106 of 450 per-(leg, night) rows exceed their range, against 7 under Range-Bounded Recovery, at a median cost of +0.0027 arcsec FWHM. |
| [closed_loop](closed_loop/study.md) | parked | 2026-09-07 | What delivered PSF does AOS closed-loop control give over a FAM visit sequence, given the corner-sensor recovery and a proportional controller? |
| [correlations](correlations/study.md) | parked | 2026-09-23 | What do the residual Double Zernike coefficients left after the MIW is subtracted correlate with: each other, the OFC v-modes, or telescope telemetry? |
| [cwfs](cwfs/study.md) | active | 2026-10-24 | Danish 1.3 blitz unpaired agrees well (median Pearson r 0.69-0.86 per corner sensor, dimensionless), against 0.26-0.42 for Danish 1.2 refitWcs. Z11 and Z14 still carry equal and opposite intra/extra offsets (Z14 about +0.26 um SW0, -0.26 um SW1); that split belongs to aos/intra_extra. |
| [cwfs_blur_intra_extra](cwfs_blur_intra_extra/study.md) | active | 2026-10-26 | Does the intra - extra donut blur difference in the corner wavefront sensors, over the science visits, match the FAM difference, and which thermal telemetry drives it? |
| [cwfs_lut](cwfs_lut/study.md) | active | 2026-10-12 | One term dominates: M2 hexapod dx against rotator angle, +20.39 um/deg (Pearson r = +0.718, Spearman rho = +0.759, n = 96,278 visits). Agreement with the bounce tests is cosine +0.829 (dimensionless) on the six hexapod translations. |
| [dzfit](dzfit/study.md) | done | 2026-09-22 | Yes: the recomputed residual nMAD matches the stored Huber scale, Pearson r = +0.999665 and median ratio 0.9974 (both dimensionless), n = 1050 visits. |
| [giant_donut_pupil](giant_donut_pupil/study.md) | parked | 2026-10-23 | v4.02 fits giant donuts better than v1000 (median residual RMS ratio 0.94, dimensionless), and the outer-band Z5-Z7 is pupil bias, not wavefront; the camera silhouette and vane shadows are measured against the model. Open: a common image-frame translation of the spider shadows (dy -1.65 / -2.07 pixels OCS image, extra / intra) and the vane shadow widths (102.8 / 61.3 mm against CAD 50 mm); parked for the CAD review and a Batoid rebuild. |
| [infra](infra/study.md) | dropped | 2026-10-10 | Dropped: tooling, not a study. The scripts moved to rubinwork/common/scripts/. |
| [intra_extra](intra_extra/study.md) | active | 2026-10-23 | Not a static pupil or model mismatch: the blur difference (median +0.3440 arcsec, n 873 FAM triplets) tracks camera-body minus M1M3 air temperature (Spearman rho -0.7828), v1000 cuts the giant-donut Z11 split from 0.81 to 0.37 um of wavefront, and diffraction gives about one third of it. A phase that flips sign with focus (a thermal strip or a turned-down M1 edge) is the open candidate. |
| [lut](lut/study.md) | dropped | 2026-10-13 | Dropped as a study: the LUT is the dof_lut product. The open question, plain against post-MIW input, stays with the product; the switch to post-MIW did not make the recovered DOF plausible. |
| [miw_hull](miw_hull/study.md) | answered | 2026-10-14 | No MIW value inside the 1.725 deg of field angle selection limit depends on a continued ring — the builder continues exactly ONE ring, at r = 1.7500 deg, the last fitted ring is r = 1.7286 deg, and 0 of the 3,833 map cells at r <= 1.725 deg lie outside it by either consumer route. Whether that one rim ring carries signal or artifact was NOT pursued: no consumer inside 1.725 deg depends on it, and both builds continue the same ring so analysis 3's two-build contrast is empty. |
| [miw_origin](miw_origin/study.md) | parked | 2026-10-11 | Retrieval bias leads: the per-visit L test gives leave-night-out CV R2 = +0.408 (dimensionless) against a null of +0.014, n = 1126 visits. No physically plausible static figure explains more than about 0.11 of the MIW (dimensionless, explained variance), and 83 % of MIW power sits at focal orders above the k <= 6 the fit uses. |
| [miw_compare_rbr](miw_compare_rbr/study.md) | active | 2026-10-08 | No: unconstrained 50/34 puts 33 of 50 DOF out of range on at least 5 % of visits. RBR brings the worst range ratio from 60.25 to 1.713 (dimensionless) but grows the MIW by 0.2764 um RMS of wavefront. |
| [miw_version_comparisons](miw_version_comparisons/study.md) | done | 2026-10-05 | Danish 1.2 and 1.3 builds agree in Z5-Z8 to about 9 % of amplitude RMS. v1000 moves the MIW by 0.1027 of amplitude (dimensionless, difference RMS over build-A RMS) with no pupil-rim signature. |
| [processing_compare](processing_compare/study.md) | dropped | 2026-10-12 | Dropped as a study: it is a regression harness. compare_donuts.py became the library rubinwork.compare_donuts; the TARTS notebook belongs to todo item 1. |
| [psf_fp_maps](psf_fp_maps/study.md) | parked | 2026-09-20 | What focal-plane PSF FWHM, ellipticity and shape does a given optical wavefront deliver, measured the way the survey measures them? |
| [unpaired_intra_vs_extra](unpaired_intra_vs_extra/study.md) | active | 2026-10-23 | Yes to both, with a split. Z14 carries the largest difference (-0.5221 um of wavefront, 8.6x its own two-side mean) but only Z11 tracks the blur difference within a night and only Z11 responds to the pupil model. Z4 is the surprise: a modest median difference with the second-largest robust scatter over visit pairs (0.1710 um of wavefront), 25.3 per cent of which the thermal model removes. |
| [wavefront_outliers](wavefront_outliers/study.md) | parked | 2026-09-22 | A robust z-score above 20 (dimensionless) against the per-CCD median flags 1.535e-02 of donuts on danish_1_3_test; which threshold to adopt, and whether it belongs in the fam_tables build, is a product decision. |
| [wfs_mimic](wfs_mimic/study.md) | parked | 2026-09-20 | A covariance product only (84 by 84 cross-corner, 21 by 21 single-corner); no conclusion of its own. |

<!-- END generated -->

Several studies project onto the Optical Feedback Control (OFC) sensitivity matrix; its
singular value decomposition and v-mode structure are the `vmode` study in `smatrix`
([`../smatrix/vmode/vmode.md`](../smatrix/vmode/vmode.md)). The focus prediction from
thermal telemetry is the `thermal_focus` topic. The MIW build is the `miw` product; its
write-up is [`docs/studies/miw.md`](docs/studies/miw.md). The FAM chunk QA is with the
`fam_tables` product
([`../rubinwork/products/fam_tables/qa/fam_processing.md`](../rubinwork/products/fam_tables/qa/fam_processing.md)).

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

Study outermost, then one directory naming the data the study was run against. The two
data axes are `param_set` — a Butler collection paired with a processing variant — and
`mi_name`, which MIW build was used; a product depending on both carries the two joined
into a single name:

```
output/
  fam_processing/<P>/               # phase-1 tables, input to every study
    {donuts,fits,visits}.parquet  chunk_status.parquet  visits_check.pdf
    chunks/<dmin>_<dmax>/           # per-chunk tables
  miw/<P>_<M>/                      # the MIW build itself
    intrinsic_split_{maps,decomp,rms}.parquet  intrinsic_split.pdf
    fits.parquet                    # DZ refit against the MIW
    zk_intrinsic.parquet            # per-donut intrinsic, row-aligned to donuts
    study_radialbins.pdf  build/rot_<lo>_<hi>/intrinsic_grid.parquet
  dzfit/<P>/  processing_compare/<P>/  coadd/<P>/{50_34,50_34_v2}/
  correlations/<P>/                 # aberration pairs, truss: no MIW dependence
  correlations/<P>_<M>/             # DZ, thermal and v-mode correlations
  bounce/<P>_<M>/  lut/<P>_<M>/  psf/<P>_<M>/  closed_loop/<P>_<M>/
  wfs_mimic/<P>_<M>/
  wfs_ingest/<P>/<cwfs>/            # CWFS tables, one directory per variant
  wfs_corner_compare/<P>/<cwfs>/    # CWFS against FAM; now the wfs_corner_compare product
  wfs_fam_compare/<P>/
  wfs_dof_compare/<P>_<M>/<cwfs>/   # recovered optical state, needs the MIW
  camera_gravity/                   # static_optics, no data dependence
  archive/                          # superseded param_sets, old layout
```

`<P>` and `<M>` are short directory names — `danish_1_2`, `A_50_34_i_5rot` — set by
`dir_name` in `param_sets.yaml` and `mi_config.yaml`. The long keys stay the identity that
`--param-set`, the value-added database rows and the frozen provenance resolve against; the
short forms appear only in paths. A study directory therefore lists exactly the data sets it
was run against, which is what distinguishes "not run" from "not applicable".

The corner wavefront sensor (CWFS) variant nests one level below the data directory rather
than joining as a third axis, which would make the name unreadable.

`fits.parquet` appears twice by design: the phase-1 per-visit DZ fit under
`fam_processing/`, and the DZ refit referenced to the MIW under `miw/`. They are different
products and the directory is what distinguishes them.

Outputs are gitignored, and symlinked to
`/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/` on the USDF RSP.
`calibration/` holds frozen, version-controlled calibration products — see
[`calibration/README.md`](calibration/README.md).

## Notebooks

Each notebook is in its study's `notebooks/`.

Closed-loop AOS performance notebooks live in the `nightlyiq/` topic, not here.

## Other docs

Reference docs in `docs/`; transient working state in `docs/status/`. Each carries a
status + last-updated line under its title.

| doc | what it holds |
|---|---|
| [`docs/miw_pipeline.md`](docs/miw_pipeline.md) | Snakemake pipeline reference: every rule, config, output path |
| [`docs/frame_conventions.md`](docs/frame_conventions.md) | the CCS, OCS and DVCS frames, which camera rotator angle to read (`rotTelPos`, not ConsDB alone, never `boresightRotAngle`), and why static optics follow OCS |
| [`docs/telemetry.md`](docs/telemetry.md) | **telemetry inventory** — every quantity, its ConsDB/EFD name, measured coverage on FAM exposures, units, and which source to prefer |
| [`docs/miw_coadd_equations.md`](docs/miw_coadd_equations.md) | MIW notation and the coadd-vs-MIW residual, derived at equation level |
| [`miw_origin/camera_gravity.md`](miw_origin/camera_gravity.md) | whether camera-lens gravitational flexure can produce the MIW astig/coma excess |
| [`docs/ts_wep_zernike_intrinsics.md`](docs/ts_wep_zernike_intrinsics.md) | how ts_wep + Danish compute the off-axis intrinsic; what the `zk_*` columns mean |
| [`docs/double_zernike_convention_validation.md`](docs/double_zernike_convention_validation.md) | validation of the DZ index/normalization conventions, vs GalSim and `ts_ofc` |
| [`miw_origin/miw_investigation_handoff.md`](miw_origin/miw_investigation_handoff.md) | portable state of the MIW investigation, with an explicit list of retracted claims |
| [`docs/status/miw_danish_1_3_proposal.md`](docs/status/miw_danish_1_3_proposal.md) | what the Danish 1.3 MIW build needs, which columns its narrow visits table does and does not carry, and the rotator-bin visit counts against Danish 1.2 |
| [`docs/status/rerun_needed.md`](docs/status/rerun_needed.md) | outputs that predate a code change and need regenerating |
| [`docs/status/corner_recovery_route_comparison.md`](docs/status/corner_recovery_route_comparison.md) | the three routes from measured corner Zernikes to DOF, the measurements between them, and the hybrid that was adopted |
| [`docs/status/code_review_backlog.md`](docs/status/code_review_backlog.md) | the one review backlog: open defects with current anchors, closed items, resolved helpers |
| [`docs/status/code_review_findings.md`](docs/status/code_review_findings.md) | the 2026-06-27 review, superseded by `code_review_backlog.md` |
| [`CLAUDE.md`](CLAUDE.md) | conventions and known pitfalls for this directory: frames, units, and which code lives in the external `ts_intrinsic_wavefront` package |

Related, in sibling topics: [`../smatrix/docs/conventions.md`](../smatrix/docs/conventions.md)
(DZ sensitivity-matrix sign/unit conventions) and
[`miw_origin/miw_astig_coma_investigation.md`](miw_origin/miw_astig_coma_investigation.md).
