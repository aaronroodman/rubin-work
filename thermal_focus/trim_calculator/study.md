---
study: trim_calculator
topic: thermal_focus
status: active
question: "Can the thermal-focus correction be computed on a summit machine with no repository code, as the camera hexapod Trim to command?"
answer: "Yes: trim_calculator.py imports only numpy, pyyaml and argparse and reads every fitted coefficient, with units and provenance, from trim_coefficients.yaml; the thermal_focus analysis checks it against the pipeline it was fitted with."
uses: {}  # none; its version is trim_coefficients.yaml
started: 2026-09-23
---

# `trim_calculator`: the standalone summit Trim calculator

> **Code:** `code/trim_calculator.py`, `trim_coefficients.yaml` (copy both together), tests `test_trim_calculator.py` with `trim_test_cases.yaml` · **Model:** [`../thermal_focus/thermal_focus.md`](../thermal_focus/thermal_focus.md), "DoF Trim from the thermal focus prediction"

## Question

Can the thermal-focus correction be computed on a summit machine with no repository code, as the camera hexapod Trim to command?

## Current answer

Yes: trim_calculator.py imports only numpy, pyyaml and argparse and reads every fitted coefficient, with units and provenance, from trim_coefficients.yaml; the thermal_focus analysis checks it against the pipeline it was fitted with.

## Log


### 2026-10-24 — moved into the study

The calculator, its coefficients, tests and test cases moved from `thermal_focus/code/` to
`code/` here (phase 5 decision 10). `run_thermal_focus_analysis.py` still imports
`trim_calculator` and `test_trim_calculator` for its section 11 check, by a path insert of this
`code/`; that check is allow-listed in `/review` until Aaron decides where it lives.

### 2026-10-08 — backfilled from docs and history

Backfilled 2026-10-18 from `thermal_focus/README.md` and the commits `41ce749` (coefficients to `trim_coefficients.yaml`, tests to `test_trim_calculator.py`) and earlier. Converted as a study, not a product (Aaron, 2026-10-18, triage decision 5); the coefficient file is its version.

The files stay in `thermal_focus/code/`: `run_thermal_focus_analysis.py` imports `trim_calculator` and `test_trim_calculator` to check the calculator against the fitted pipeline, so moving them here would be a cross-study import. They move when the shared thermal-focus code becomes a library.
