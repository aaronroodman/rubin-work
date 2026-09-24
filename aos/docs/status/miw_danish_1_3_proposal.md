# Proposal — a Danish 1.3 MIW comparable to `danish_1_2_A_50_34_i_5rot`

> **Status:** current · **Last updated:** 2026-09-23 · **Kind:** status (accepted; build not yet run)

Plan for building a Measured Intrinsic Wavefront (MIW) from the Danish 1.3 "blitz"
unpaired wavefronts, named `danish_1_3_test_A_50_34_i_5rot`, over the same five camera
rotation data sets used by `danish_1_2_A_50_34_i_5rot`, so the two MIW builds can be
compared directly. The approach below is accepted and the two `mi_config.yaml` entries
are in place; the build itself has not yet been run. See
[Validated job list](#validated-job-list) for what a run will do.

## Verdict on feasibility

The Danish 1.3 parquets carry everything the MIW chain reads. No new extraction is needed
and no code change is required. Two findings drive that conclusion.

**The donut table is complete.** `output/fam_processing/danish_1_3_test/donuts.parquet`
has the same 48 columns as `danish_1_2`, with zero missing and zero extra, including
`zk_OCS`, `zk_intrinsic_OCS`, `zk_deviation_OCS`, `thx_OCS`/`thy_OCS` and `detector`.

**The visits table is narrow but sufficient.** It has 19 columns against Danish 1.2's 405.
The 386 absent columns are all engineering telemetry — `lut_dof0`–`lut_dof9`,
`m1m3elev_*`, the wind and camera-body temperatures, `sonic_temperature`, Trim and Tweak.
The build chain reads only eleven visits columns, and all eleven are present:

| column | role in the chain |
|---|---|
| `day_obs`, `seq_num` | join keys, and the `day_obs_max` cut |
| `alt` | the elevation window, `alt_min_deg` to `alt_max_deg` |
| `rotator_angle` | the rotator binning |
| `band` | the `filter` whitelist |
| `science_program` | the `programs` whitelist |
| `nollIndices` | the Noll index list |
| `visit_quality_pass` | the refit quality cut |
| `n_donuts`, `n_detectors_with_min_donuts`, `median_blur_arcsec` | the `quality_visit_mask` cuts |

Only `day_obs`, `seq_num`, `alt` and `rotator_angle` are hard requirements; the rest sit
behind `in colnames` or `_col()` guards, so no absent column can raise a `KeyError`. The
telemetry-producing code in `intrinsics_lib.py` is reached only by `mktable`, never by the
build: of the seven build-chain files, only `dz_fitting.py` imports `intrinsics_lib`, and
only for `quality_visit_mask`, which reads the three quality columns listed above.

`alt` is stored in **radians** in both param_sets (Danish 1.3 spans 0.524 to 1.226 rad,
30.0 to 70.2 deg). `apply_visit_filters` in `measured_intrinsic.py` converts when the
maximum absolute value is below 2*pi, so the `alt_min_deg`/`alt_max_deg` window works
unchanged. Rotator bins are half-open, `[lo, hi)`.

## The five rotation data sets are present

Applying the `mi_config.yaml` default cuts — band `i`, program `BLOCK-T614_triplets`,
elevation 65.0 to 75.0 deg, `visit_quality_pass` — gives 275 visits for Danish 1.3
against 271 for Danish 1.2, spread over all nine rotator bins:

| rotator bin (deg) | Danish 1.2 visits | Danish 1.3 visits | in the 5-bin split |
|---|---|---|---|
| [-65, -55] | 36 | 36 | yes |
| [-50, -40] | 23 | 15 | |
| [-35, -25] | 24 | 24 | |
| [-20, -10] | 45 | 48 | yes |
| [-3, 3] | 37 | 49 | yes |
| [10, 20] | 36 | 36 | yes |
| [25, 35] | 19 | 17 | |
| [40, 50] | 16 | 22 | |
| [55, 65] | 35 | 28 | yes |
| **total in the five** | **189** | **197** | |

Both draw on the same five nights, `day_obs` 20260315, 20260316, 20260317, 20260324 and
20260409, and in both the five selected bins are populated only by the four March nights —
20260409 sits solely in the +-30 and +-45 deg bins and so is excluded by angle, which is
what the `_5rot` entry is for. **182 (day_obs, seq_num) visits are common to both
processings** inside the five bins, with 7 only in Danish 1.2 and 15 only in Danish 1.3, so
the comparison rests on a largely shared visit set rather than on two different samples.

## What has to be built

`A_50_34_i_5rot` is not a standalone build. In `mi_config.yaml` it carries
`build_from: pathA_50_34_i`, reusing that entry's per-rotator-bin grids and re-running only
the split over five of the nine bins, with `split_js: [4]`. So the parent entry is
required, and `danish_1_2_A_50_34_i/build/` accordingly holds all nine `rot_*` grids while
`danish_1_2_A_50_34_i_5rot/` holds none.

Two `mi_config.yaml` entries are therefore needed under a new
`danish_1_3_test:` key, mirroring the Danish 1.2 pair:

- `pathA_50_34_i` with `dir_name: A_50_34_i`, `n_dof: 50`, `n_keep: 34`, and **no
  `day_obs_max`** — that key exists on the Danish 1.2 parent only to freeze the MIW to its
  chunks 1-3 and omit the later `wep_17_7_0` reprocessing, a situation Danish 1.3 does not
  have. Its own `day_obs` range already ends at 20260619 and the elevation, band, program
  and rotator cuts select the same five nights regardless.
- `pathA_50_34_i_5rot` with `dir_name: A_50_34_i_5rot`, `build_from: pathA_50_34_i`, and
  the identical five-bin `rotator_select` plus `split_js: [4]`.

Every other knob comes from the shared `defaults:` block, so the two builds differ only in
the wavefronts themselves — which is the point of the comparison.

Because the joined directory name is the param_set `dir_name` plus the MI `dir_name`, this
yields exactly the requested `output/miw/danish_1_3_test_A_50_34_i_5rot/`, alongside
`output/miw/danish_1_3_test_A_50_34_i/` for the parent.

## Products, matching the existing build

`output/miw/danish_1_3_test_A_50_34_i_5rot/` would hold the same items as the Danish 1.2
directory:

| product | rule | notes |
|---|---|---|
| `intrinsic_split_maps.parquet` | `intrinsic_split` | the MIW field maps |
| `intrinsic_split_decomp.parquet` | `intrinsic_split` | |
| `intrinsic_split_rms.parquet` | `intrinsic_split` | |
| `intrinsic_split.pdf` | `intrinsic_split` | |
| `study_radialbins.pdf` | `study_radialbins` | OCS MIW at the WFS radius, per bin |
| `zk_intrinsic.parquet` | `intrinsic_sidecar` | per-donut sidecar, about 0.64 GB |
| `fits.parquet` | `refit_mi` | DZ fits against the measured intrinsic |

The parent `danish_1_3_test_A_50_34_i/` additionally holds `build/rot_*/intrinsic_grid.parquet`
for the nine bins. `fits_july.parquet` in the Danish 1.2 directory is a hand-made extra
produced by no rule and is not reproduced.

Estimated disk is about 1 GB for the pair, scaling the Danish 1.2 sidecar's 201.8 bytes per
donut by the 3155734 Danish 1.3 donuts; 756 GB are free.

## Consequences to decide before running

**`rule all` would grow.** The Snakemake `MI_DTAGS` list is built from every
`(param_set, mi_name)` pair in `mi_config.yaml`, and `rule all` expands several targets over
it. Adding these two entries therefore also requests, for each, `output/lut/<d>/lut.parquet`,
six `output/correlations/<d>/*.pdf`, `output/bounce/<d>/bounce_kj_stats.parquet`,
`output/wfs_dof_compare/<d>/...` and `output/wfs_mimic/<d>/wfs_mimic_cov84.parquet` — none of
which was asked for here.

The `correlations` and `bounce` targets are the real problem: they correlate the DZ fits
against exactly the thermal and Trim telemetry that Danish 1.3's visits table lacks. The
narrow visits table propagates into the MI `fits.parquet` through the wholesale left join in
`dz_fitting.py`, so those consumers would find their inputs missing. The 13 columns absent
from the Danish 1.3 phase-1 `fits.parquet` relative to Danish 1.2 — `cam_air_temp`,
`m1m3_air_temp`, `outside_temp`, `x_gradient` and the rest — are the same shortfall seen one
step earlier.

Requesting the MIW targets by explicit path rather than through `rule all` avoids this, and
is the recommended course:

```bash
cd ~/notebooks/rubin-work/aos && ./run_snake.sh -n \
  output/miw/danish_1_3_test_A_50_34_i_5rot/intrinsic_split_maps.parquet \
  output/miw/danish_1_3_test_A_50_34_i_5rot/study_radialbins.pdf \
  output/miw/danish_1_3_test_A_50_34_i_5rot/zk_intrinsic.parquet \
  output/miw/danish_1_3_test_A_50_34_i_5rot/fits.parquet
```

The alternative — attaching the telemetry to the Danish 1.3 visits table so the whole
`rule all` chain works — is a separate and larger piece of work. It is not needed for the
MIW comparison and is not proposed here.

**The convex hull defect is carried deliberately.** It is present in both builds, so it
affects the comparison equally, and leaving it in place keeps the two MIW versions
comparable. It should not be described as fixed.

**`danish_1_3_test` is absent from `snake_config.yaml`.** Its tables came from the blitz
recast rather than the chunked Butler build, so it has no `chunks:` entry and no
`output/fam_processing/danish_1_3_test/chunks/` directory. The MIW rules read the combined
tables directly and do not need chunks, but any rule that fans out over chunks cannot be
requested for this param_set — the same limitation already met when making the dzfit movie.

## Validated job list

Requesting the four MIW targets by explicit path plans **13 jobs**, and nothing else:

| rule | jobs | memory each (MB) | writes |
|---|---|---|---|
| `build_intrinsic` | 9 | 8000 | `danish_1_3_test_A_50_34_i/build/rot_*/intrinsic_grid.parquet`, one per rotator bin |
| `intrinsic_split` | 1 | 4000 | the four `intrinsic_split*` products in `_5rot/` |
| `study_radialbins` | 1 | 4000 | `study_radialbins.pdf` |
| `intrinsic_sidecar` | 1 | 8000 | `zk_intrinsic.parquet` |
| `refit_mi` | 1 | 4000 | `fits.parquet` |

No `correlations`, `bounce`, `lut` or `wfs_*` job appears, which is the point of requesting
by path. The nine builds are independent, so under the batch default of 32 CPUs and
`--resources mem_mb=90000` they run concurrently. Snakemake reports the three combined
phase-1 tables as having "missing provenance/metadata" — it did not produce them itself,
which is expected for tables written by the blitz recast.

Planning the full `rule all` alongside these entries still returns zero errors, at 109 jobs.

### Two Snakefile guards this needed

`danish_1_3_test` is absent from `snake_config.yaml`, so it is absent from `PSETS` and from
`PS_BY_DIR`. Two places raised `KeyError` while the DAG was merely being *planned*, before
any job could be selected:

- `_chunk_entries` read `CFG[ps]["chunks"]`, now `(CFG.get(ps) or {}).get("chunks", [])`.
- `chunk_files` read `PS_BY_DIR[psd]`, now `PS_BY_DIR.get(psd, psd)`.

Both now resolve to an empty chunk list for a param_set built outside the pipeline, whose
combined tables are terminal inputs. Requesting a chunk-fanout target for such a param_set
still fails, correctly, on the empty input list. `coord()` is guarded the same way and
falls back to `OCS`.

## Verification performed

- Column diff of `donuts`, `visits` and `fits` between the two param_sets.
- Every visits-column access in `run_build_intrinsic.py`, `measured_intrinsic.py`,
  `run_intrinsic_split.py`, `intrinsic_split.py`, `run_make_intrinsic_sidecar.py`,
  `run_dz_fit.py`, `dz_fitting.py`, `intrinsic_build_plots.py` and
  `code/miw/run_study_radialbins.py`, checked for guards.
- The selection cuts reproduced independently on both tables, giving the rotator-bin table
  above and the 182-visit overlap.
- `lsst.ts.ofc` and `lsst.ts.wep` import; `$TS_CONFIG_MTTCS_DIR` resolves;
  `ccd_height_map.fits.gz` is present in `batoid_rubin_data/ccd_height_map/`.
