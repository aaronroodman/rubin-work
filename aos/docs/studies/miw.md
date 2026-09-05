# Study: `miw` — the Measured Intrinsic Wavefront

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** What is the measured intrinsic wavefront (MIW) of the telescope + camera,
and is the build trustworthy?

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
| `dz_plotting.py` | plotting library: fit parameters, residual maps, trio comparisons (1243 lines, the largest file in `aos/code/`) |
| `run_dz_plots.py` | pipeline `plots` rule — data/model/residual trio validation on the combined tables |
| `run_aberration_pairs.py` | pipeline `aberration_pairs` — per-donut primary→secondary aberration-pair correlations |
| `run_study_radialbins.py` | pipeline `study_radialbins` — OCS MIW at the WFS radius, overlaid per rotator bin |
| `combine_parquets.py` | streaming row-group concatenation of per-chunk tables (**stays flat**, cross-study) |
| `plot_visits_summary.py` | per-param_set visit summary: elevation vs rotator, one panel per band |
| `check_chunk.py` | pre-flight: visits in ConsDB but missing from the Butler collection, before `mktable` runs |
| `inspect_visit_provenance.py` | Butler provenance consistency across a param_set's date chunks |
| `compare_to_archive.py` | new combined outputs vs the archived pre-reorg ones, after a from-scratch run |

## Inputs and outputs

Reads the combined `output/<ps>/{donuts,fits,visits}.parquet`; the per-rotator-bin
grids come from the package's `build_intrinsic`. Writes `plots/trio_comparison_all.pdf`,
`aberration_pairs.{pdf,parquet}`, `study_radialbins.pdf`, and the
`intrinsic_split_{maps,decomp,rms}.parquet` products the rest of the topic consumes.

The **canonical MIW product** for downstream use is the `_5rot` `intrinsic_split_maps`
(OCS columns) — see `../../../notes/claude-memory/miw-products-and-m3-backprojection.md`.

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh -n                                    # what is stale
./run_snake.sh --until plots
python code/check_chunk.py --help                    # pre-flight, before mktable
```

Phase 2 needs `lsst.ts.ofc`/`lsst.ts.wep`, `$TS_CONFIG_MTTCS_DIR`, and batoid height
maps — RSP only. `mktable` is the expensive Butler step and is deliberately **not**
re-triggered by code edits.

## State and open questions

- **83 % of MIW power sits above the `k<=6` focal orders the build actually fits.** This
  reframes every DZ-subspace analysis: the fit constrains 34 v-modes from k≤6, then
  subtracts only the k≤6 part of that state's wavefront. Quantified in
  `../../code/analyze_miw_dz_full_k.py` and `analyze_miw_field_order.py` (`coadd` study).
- The MIW's high-field-order astigmatism/coma excess (Z5–Z8, OCS, ~0.1 µm) is **not**
  predicted by the batoid design model. Whether static optics can explain it is the
  [`static_optics`](static_optics.md) study; the write-up is
  [`../../../smatrix/docs/miw_astig_coma_investigation.md`](../../../smatrix/docs/miw_astig_coma_investigation.md).

## See also

- [`../miw_pipeline.md`](../miw_pipeline.md) — every rule, config file, output path
- [`../miw_coadd_equations.md`](../miw_coadd_equations.md) — the notation and derivations
- [`../status/miw_investigation_handoff.md`](../status/miw_investigation_handoff.md) —
  portable state, **with an explicit list of retracted claims**
- [`../double_zernike_convention_validation.md`](../double_zernike_convention_validation.md)
