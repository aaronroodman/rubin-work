# Study: `miw` — the Measured Intrinsic Wavefront

> **Status:** current · **Last updated:** 2026-09-23 · **Kind:** reference (study)

> **Code:** `code/miw/` · **Notebooks:** `notebooks/miw/`
> **Output:** `output/miw/<P>_<M>/intrinsic_split_{maps,decomp,rms}.parquet`, `output/miw/<P>_<M>/intrinsic_split.pdf`, `output/miw/<P>_<M>/study_radialbins.pdf`, `output/miw/<P>_<M>/fits.parquet`

Construction and validation of the Measured Intrinsic Wavefront (MIW) — the static
wavefront of the telescope and camera, measured on sky.

The MIW is the empirical, on-sky replacement for the batoid design intrinsic: the static
wavefront that remains after the reachable (OFC-controllable) part of the optical state
has been removed. It splits into a telescope-fixed component **O** (OCS frame) and a
camera-fixed component **C** (CCS, rotating with the rotator).

This study covers the pipeline's own products and the QA around them. The *library* that
builds the MIW is **not here** — it is in the external `ts_intrinsic_wavefront` package
(`lsst.ts.intrinsic.wavefront`). `aos/code/` holds only the analysis and QA scripts.

## Code

| file | role |
|---|---|
| `run_study_radialbins.py` | pipeline `study_radialbins` rule — OCS measured intrinsic in four WFS radial shells, overlaid by rotator bin |
| `compare_miw_versions.py` | two MIW builds term by term as a PDF — one page per Noll term holding both field maps on a shared colour scale and their difference on its own |

The build itself is in the external `ts_intrinsic_wavefront` package
(`measured_intrinsic.build_measured_intrinsic_uconstrained`, driven by the
`build_intrinsic` and `intrinsic_split` rules), not here.

Validation of the per-visit DZ fit that feeds the build, and quality checks on the donut
data, are the [`dzfit`](dzfit.md) study.

## Inputs and outputs

Reads the combined `output/fam_processing/<P>/{donuts,fits,visits}.parquet`; the per-rotator-bin
grids come from the package's `build_intrinsic`. Writes `study_radialbins.pdf` and the
`intrinsic_split_{maps,decomp,rms}.parquet` products that the other studies consume.

The **canonical MIW product** for downstream use is the `_5rot` `intrinsic_split_maps`
(OCS columns) — see `../../../notes/claude-memory/miw-products-and-m3-backprojection.md`.

## Which builds exist

Each MIW build is one `(param_set, mi_name)` pair from `mi_config.yaml`, written to the
joined directory `output/miw/<P>_<M>/`. Two wavefront versions are configured, with
identical knobs so that they differ only in the wavefronts they were built from:

| param_set | wavefront version | entries |
|---|---|---|
| `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` | Danish 1.2.0_alpha0, paired | `A_50_34_i`, `A_50_34_i_5rot` |
| `danish_1_3_test` | Danish 1.3 "blitz", unpaired | `A_50_34_i`, `A_50_34_i_5rot` |

In each pair the `_5rot` entry carries `build_from`, reusing the parent's nine
per-rotator-bin grids and re-running only the OCS/CCS split over the five in-family
rotator bins — so it has no `build/` directory of its own and the parent is a required
input.

The Danish 1.3 pair is requested **by explicit target path, not through `rule all`**: its
`visits.parquet` carries only 19 columns and none of the engineering telemetry, so the
`correlations` and `bounce` targets that `rule all` expands over every pair have no
thermal or Trim columns to read. The MIW chain itself reads eleven visits columns, all
present. The comparison of the two builds rests on 182 `(day_obs, seq_num)` visits common
to both processings inside the five rotator bins. Detail, including the per-bin visit
counts and the column audit, is in
[`../status/miw_danish_1_3_proposal.md`](../status/miw_danish_1_3_proposal.md).

### Comparing two builds

`compare_miw_versions.py` writes one page per pupil Zernike Noll term with three field
maps: build A, build B on the **same** colour scale, and B minus A on its own scale set
from the 2nd to 98th percentile of the difference. Both builds must sit on the same field
grid, which two builds sharing a `rotator_select` do; the script checks that row-for-row
rather than interpolating.

```bash
cd ~/notebooks/rubin-work/aos
python code/miw/compare_miw_versions.py \
  --miw-a output/miw/danish_1_2_A_50_34_i_5rot/intrinsic_split_maps.parquet \
  --miw-b output/miw/danish_1_3_test_A_50_34_i_5rot/intrinsic_split_maps.parquet \
  --label-a "Danish 1.2 (paired)" \
  --label-b "Danish 1.3 blitz (unpaired)" \
  --out-dir output/miw/danish_1_2_vs_1_3 \
  --out-name miw_danish_1_2_vs_1_3_OCS
```

Colour scales are computed inside a field radius of 1.70 deg, set by
`--scale-r-max-deg`. The outermost ring carries the convex-hull edge defect — across the
1.70 to 1.75 deg step the Z5 OCS difference root-mean-square rises from 0.0499 to 0.1109
µm of wavefront and reaches 0.6441 µm — and would otherwise set the range and flatten the
real structure. Those 240 of 3985 field points are still plotted, and saturate.

Alongside the PDF the script writes a `_summary.parquet` carrying, per Noll term, the
root-mean-square of each build and of the difference in µm of wavefront, the difference
normalized median absolute deviation, and both colour limits.

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh -n                                    # what is stale
./run_snake.sh --until plots
python code/miw/check_chunk.py --help                    # pre-flight, before mktable
```

Phase 2 needs `lsst.ts.ofc`/`lsst.ts.wep`, `$TS_CONFIG_MTTCS_DIR`, and batoid height
maps — RSP only. `mktable` is the expensive Butler step and is deliberately **not**
re-triggered by code edits.

## State and open questions

- **83 % of MIW power sits above the `k<=6` focal orders the build actually fits.** This
  reframes every DZ-subspace analysis: the fit constrains 34 v-modes from k≤6, then
  subtracts only the k≤6 part of that state's wavefront. Quantified in
  `../../code/coadd/analyze_miw_dz_full_k.py` and `analyze_miw_field_order.py` (`coadd` study).
- The MIW's high-field-order astigmatism/coma excess (Z5–Z8, OCS, ~0.1 µm) is **not**
  predicted by the batoid design model. Whether static optics can explain it is the
  [`static_optics`](static_optics.md) study; the write-up is
  [`../../../smatrix/docs/miw_astig_coma_investigation.md`](../../../smatrix/docs/miw_astig_coma_investigation.md).

## Notebooks

`notebooks/miw/aos_miw_ocs_ccs_maps.ipynb` reads the MIW OCS/CCS split maps — a viewer for
`<mi>/intrinsic_split_maps.parquet`, no repo-module imports.

## See also

- [`../miw_pipeline.md`](../miw_pipeline.md) — every rule, config file, output path
- [`../miw_coadd_equations.md`](../miw_coadd_equations.md) — the notation and derivations
- [`../status/miw_investigation_handoff.md`](../status/miw_investigation_handoff.md) —
  portable state, **with an explicit list of retracted claims**
- [`../double_zernike_convention_validation.md`](../double_zernike_convention_validation.md)
