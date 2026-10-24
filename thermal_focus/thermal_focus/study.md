---
study: thermal_focus
topic: thermal_focus
status: active
question: "Can the uniform-defocus error of the telescope be predicted from thermal telemetry alone, so focus can be set open-loop?"
answer: "Yes, between nights: 5 thermal channels in one band-independent Huber fit cut the focus error from 336.8 to 58.2 µm of equivalent hexapod dz nMAD (in sample; 59.9 µm night-grouped) over 68,079 science visits and 147 nights; the truss carries +125.09 µm per deg C. The start of the night is worse: 84.9 against 54.9 µm nMAD before and after 3 h past sunset."
uses: {}  # reads the value-added DuckDB through value_added/code/efd_db.py plus live ConsDB; value_added not yet pinned
started: 2026-09-09
---

# `thermal_focus`: the focus error as a function of temperature

> **Write-up:** [`thermal_focus.md`](thermal_focus.md) · **Code:** `code/` (build and analysis); the shared modules are the library `rubinwork.thermal_focus` · **Notebook:** `notebooks/corner_z4_vs_temperature_science.ipynb` · **Output:** `thermal_focus/output/`

## Question

Can the uniform-defocus error of the telescope be predicted from thermal telemetry alone, so focus can be set open-loop?

## Current answer

Yes, between whole nights. Five thermal channels — the TMA truss temperature and the four M1M3 bulk thermal gradients — fitted jointly with one band-independent Huber robust linear model:

- cut the focus error from **336.8 to 58.2 µm of equivalent hexapod dz** nMAD over 68,079 science visits and 147 nights (in sample; 59.9 µm with 5-fold `day_obs`-grouped evaluation, an optimism of 1.04, dimensionless);
- the truss carries most of it, at **+125.09 µm of equivalent hexapod dz per °C**;
- applied unchanged to 1,346 FAM in-focus acquisition visits over 43 nights (Danish 1.2 retrieval), they cut the scatter by **3.39 times** (dimensionless; 282.2 to 83.3 µm nMAD);
- at the first science visit after the initial alignment block, predicted against settled focus differ by **103.4 µm** nMAD over 145 nights.

**The one systematic not captured is the start of the night:** residual nMAD **84.9 µm before 3 h after sunset against 54.9 µm after** (ratio 1.55, dimensionless), with a +23.3 µm median bias early. It is a step, not a drift, and reads as thermal lag.

The elevation-hysteresis test and the within-FAM-block argument were dropped from the study (2026-10-08). Details in [`thermal_focus.md`](thermal_focus.md).

## Log

### 2026-10-24 — code split by study

`thermal_focus/code/` is gone (phase 5 decision 10). `thermal_focus_lib` and
`thermal_focus_fit` are the library `rubinwork.thermal_focus` (`lib`, `fit`); the build and
analysis are in this study's `code/`, the v-mode scripts in `thermal_vmodes`, the calculator in
`trim_calculator`. Two cross-study imports remain, allow-listed: the analysis's section 11
check of the calculator, and `thermal_vmodes` taking its sample from `load_science` here.
Default output stays `thermal_focus/output/`.

### 2026-10-08 — backfilled from docs and history

Backfilled 2026-10-18 from [`thermal_focus.md`](thermal_focus.md) (formerly `thermal_focus/docs/`) and the session handoff, now [`intranight_and_twilight_handoff.md`](intranight_and_twilight_handoff.md). Converted as active in the phase 5 triage.

Last session (2026-10-08, `49b3b99`, `36fcc38`): the intra-night step above; the start-of-night comparison moved to the first science visit (Huber slope +0.814 ± 0.016, dimensionless, predicted per actual); long alignment blocks are an operations delay between exposures, not the alignment struggling. Report at 26 PDF pages; 23 tests pass.

The code stays in `thermal_focus/code/` for now: `thermal_vmodes` imports `thermal_focus_lib` and `thermal_focus_fit`, and the analysis imports `trim_calculator`, so splitting the code by study would create cross-study imports. The shared modules belong in a library (phase 5, section 2 step). The notebook moved here; it finds the topic by `code/thermal_focus_lib.py` and still writes to `thermal_focus/output/`.

Next: model the start-of-night thermal lag.
