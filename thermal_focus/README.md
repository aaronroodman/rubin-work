# thermal_focus

> **Status:** current · **Last updated:** 2026-09-23 · **Kind:** topic scope and index

<!-- BEGIN generated: python -m rubinwork.studies index -->

## Studies

Generated from each `<study>/study.md`; edit those, not this table.

| study | status | last log | answer |
|---|---|---|---|
| [thermal_focus](thermal_focus/study.md) | active | 2026-10-24 | Yes, between nights: 5 thermal channels in one band-independent Huber fit cut the focus error from 336.8 to 58.2 µm of equivalent hexapod dz nMAD (in sample; 59.9 µm night-grouped) over 68,079 science visits and 147 nights; the truss carries +125.09 µm per deg C. The start of the night is worse: 84.9 against 54.9 µm nMAD before and after 3 h past sunset. |
| [thermal_vmodes](thermal_vmodes/study.md) | done | 2026-10-06 | None: only v1 passes the false-discovery-rate cut at q = 0.05 (1 of 34 modes) over 72,835 science visits and 175 nights, with skill +0.808 against +0.243 for the next mode, v10 (dimensionless, fractional reduction in out-of-fold residual nMAD). |
| [trim_calculator](trim_calculator/study.md) | active | 2026-10-24 | Yes: trim_calculator.py imports only numpy, pyyaml and argparse and reads every fitted coefficient, with units and provenance, from trim_coefficients.yaml; the thermal_focus analysis checks it against the pipeline it was fitted with. |

<!-- END generated -->

Prediction of the Rubin Observatory telescope's uniform-defocus error from thermal telemetry, so
that focus can be set open-loop from a table rather than being driven by the wavefront sensors.

The Active Optics System (AOS) holds the telescope in focus by measuring the wavefront and
commanding the camera and M2 hexapods and the mirror bending modes. The focus error it has to
remove is largely thermal: as the Telescope Mount Assembly (TMA) truss and the M1M3 mirror change
temperature, the spacing along the optical axis changes with them. This topic measures that
relation on ordinary science exposures — whose four Corner Wavefront Sensors (CWFS) the
Consolidated Database (ConsDB) records for every visit — and turns it into a feed-forward
correction.

The focus error is measured as v-mode 1, the amplitude of the first singular vector of the AOS
sensitivity matrix, which is essentially uniform defocus, and is reported in µm of equivalent
hexapod dz: the total defocus travel, shared as 0.5 µm on each hexapod. Five thermal channels —
the TMA truss temperature and the four M1M3 bulk thermal gradients — fitted with one
band-independent Huber robust linear model predict it to 59.9 µm of equivalent hexapod dz from an
uncorrected 336.8 µm, over 68,079 science visits across 147 nights. The truss temperature carries
most of it, at +125.09 µm of equivalent hexapod dz per °C.

Two limits on where that correction applies are part of the result. It is fitted and scored
**between whole nights**, holding nights out, because within a night the thermal telemetry barely
moves and a visit-level split lets a model recall the night instead of predicting it. Inside a single Full
Array Mode (FAM) observing block the AOS does not re-command, so the Trim is frozen there; the
within-block comparison that once argued the correction fails inside a block measured that frozen
Trim rather than a focus prediction, and was dropped from the study (2026-10-08). Applied to the
FAM in-focus acquisition visits as an independent sample, the same coefficients cut the scatter
by 3.39 times (dimensionless; see `thermal_focus/thermal_focus.md`).

## Studies

- [`thermal_focus`](thermal_focus/thermal_focus.md) — the response definition, the fitted thermal
  model and its night-grouped evaluation, the FAM blocks as an independent sample,
  the Double Zernike (DZ) cross-check, the v-mode-1 conversion across projection schemes, the
  correction expressed as degrees of freedom (DOF), and the comparison against the Trim the
  observatory's initial alignment block settles on at the start of each night.
- [`thermal_vmodes`](thermal_vmodes/thermal_vmodes.md) — the same question asked of all 34 v-modes
  rather than v-mode 1 alone: per-mode skill against an intercept-only null, a
  false-discovery-rate cut over the 34 simultaneous tests, where the four-corner measurement
  noise floor sits, and whether the two intrinsic routes agree on which modes are thermal.

## Code

| file | role |
|---|---|
| `rubinwork/thermal_focus/lib.py` | the response definition, the conversions and the feature groups |
| `thermal_focus/thermal_focus/code/run_thermal_focus.py` | build: the value-added database plus live ConsDB, writing the cached tables |
| `rubinwork/thermal_focus/fit.py` | the fitting core: models, night-grouped evaluation, FAM block assignment |
| `thermal_focus/thermal_focus/code/run_thermal_focus_analysis.py` | the analysis: sixteen sections and one document, no network |
| `thermal_vmodes/code/thermal_vmodes.py` | the all-34-v-mode study: per-mode fits, the null, the FDR cut, the noise floor |
| `thermal_vmodes/code/run_thermal_vmodes.py` | the all-34-v-mode run: the per-mode table, the noise floor, both intrinsic routes, the figures |
| `thermal_vmodes/code/thermal_vmodes_figures.py` | the all-34-v-mode figures: five pages, one function each |
| `thermal_vmodes/code/thermal_vmodes_channels.py` | the per-channel thermal screen: every v-mode against every telemetry line, and the v-mode Zernike content |
| `thermal_vmodes/code/thermal_vmodes_channel_figures.py` | the screen's figures: the correlation heatmaps and one page per responding mode |
| `trim_calculator/code/trim_calculator.py` | the standalone online calculator: numpy and pyyaml only, no repository imports |
| `trim_calculator/code/trim_coefficients.yaml` | the fitted coefficients the calculator reads, with units and provenance |
| `trim_calculator/code/test_trim_calculator.py` | the calculator's tests, data in `trim_calculator/code/trim_test_cases.yaml` |

The build stage is the only one that needs the network, because the mean TMA truss temperature is
derived on a ConsDB join rather than stored. It caches to parquet, so the analysis runs offline:

```bash
python thermal_focus/thermal_focus/code/run_thermal_focus.py
python thermal_focus/thermal_focus/code/run_thermal_focus_analysis.py
python thermal_focus/trim_calculator/code/test_trim_calculator.py
```

The calculator imports numpy, pyyaml and argparse and nothing else, so it can be copied to a summit
machine and run there; every coefficient lives in `trim_calculator/code/trim_coefficients.yaml` with its units and
provenance, so copy both files together. The analysis checks the calculator against the pipeline it
fitted.

This topic imports the library `rubinwork.aos_state` for the v-modes and the degree-of-freedom
sets, and reads the value-added DuckDB through the `rubinwork.products.value_added` reader.

## Notebooks

- `notebooks/corner_z4_vs_temperature_science.ipynb` — the four-corner mean Z4 of the
  total optical state against truss, outside-air and camera-body temperature, an independent route
  to the same physical question from a different and noisier estimator of the measured wavefront.
  Needs ConsDB and the Engineering Facility Database (EFD).

## Output

`output/` holds `thermal_focus.parquet` (one row per science visit),
`thermal_focus_t539.parquet` (one row per night of the initial alignment block),
`thermal_focus.pdf` (the analysis document) and, under the FAM variant's short directory name,
`thermal_focus_fam.parquet` (one row per FAM triplet).
