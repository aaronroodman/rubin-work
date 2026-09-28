# thermal_focus

> **Status:** current · **Last updated:** 2026-09-23 · **Kind:** topic scope and index

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
moves and a visit-level split lets a model recall the night instead of predicting it. And it does
**not** work inside a single Full Array Mode (FAM) observing block. The focus error being modelled
is the difference between what the AOS has commanded and what the wavefront sensors measure, and
between nights the commanded part dominates it; inside a block the AOS does not re-command, so the
part the model predicts is frozen and only the measured part is left, with the opposite sign.
Applying the correction there makes within-block scatter worse rather than better.

## Studies

- [`thermal_focus`](docs/thermal_focus.md) — the response definition, the fitted thermal
  model and its night-grouped evaluation, the elevation null result, the FAM within-block drift,
  the Double Zernike (DZ) cross-check, the v-mode-1 conversion across projection schemes, the
  correction expressed as degrees of freedom (DOF), and the comparison against the Trim the
  observatory's initial alignment block settles on at the start of each night.

## Code

| file | role |
|---|---|
| `code/thermal_focus_lib.py` | the response definition, the conversions and the feature groups |
| `code/run_thermal_focus.py` | build: the value-added database plus live ConsDB, writing the cached tables |
| `code/thermal_focus_fit.py` | the fitting core: models, night-grouped evaluation, FAM block assignment |
| `code/run_thermal_focus_analysis.py` | the analysis: sixteen sections and one document, no network |
| `code/trim_calculator.py` | the standalone online calculator: numpy only, no repository imports |

The build stage is the only one that needs the network, because the mean TMA truss temperature is
derived on a ConsDB join rather than stored. It caches to parquet, so the analysis runs offline:

```bash
python code/run_thermal_focus.py
python code/run_thermal_focus_analysis.py
python code/trim_calculator.py --self-test
```

The calculator imports numpy and argparse and nothing else, so it can be copied to a summit machine
and run there; every coefficient is inlined with its units and provenance, and the analysis checks
it against the pipeline it fitted.

This topic imports `aos_state` from `aos/code` for the v-modes and the degree-of-freedom sets,
through `sys.path.insert`, as `blocks/`, `olr/`, `optatmo/`, `smatrix/` and `value_added/` do. It
reads the value-added DuckDB through `value_added/code/efd_db.py`.

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
