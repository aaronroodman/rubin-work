# Study: `miw` — the Measured Intrinsic Wavefront

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

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

The build itself is in the external `ts_intrinsic_wavefront` package
(`measured_intrinsic.build_measured_intrinsic_uconstrained`, driven by the
`build_intrinsic` and `intrinsic_split` rules), not here.

Validation of the per-visit DZ fit that feeds the build, and quality checks on the donut
data, are the [`dzfit`](dzfit.md) study.

## Inputs and outputs

Reads the combined `output/<ps>/{donuts,fits,visits}.parquet`; the per-rotator-bin
grids come from the package's `build_intrinsic`. Writes `plots/trio_comparison_all.pdf`,
`aberration_pairs.{pdf,parquet}`, `study_radialbins.pdf`, and the
`intrinsic_split_{maps,decomp,rms}.parquet` products that the other studies consume.

The **canonical MIW product** for downstream use is the `_5rot` `intrinsic_split_maps`
(OCS columns) — see `../../../notes/claude-memory/miw-products-and-m3-backprojection.md`.

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
