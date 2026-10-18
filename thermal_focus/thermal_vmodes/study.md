---
study: thermal_vmodes
topic: thermal_focus
status: done
question: "Which v-modes other than v1 carry a component predictable from telescope thermal telemetry?"
answer: "None: only v1 passes the false-discovery-rate cut at q = 0.05 (1 of 34 modes) over 72,835 science visits and 175 nights, with skill +0.808 against +0.243 for the next mode, v10 (dimensionless, fractional reduction in out-of-fold residual nMAD)."
uses: {}  # reads the value-added DuckDB optical_state, variant v50_34_rbr__batoid__consdb_v1
started: 2026-09-30
---

# `thermal_vmodes`: thermal response of all 34 v-modes

> **Write-up:** [`thermal_vmodes.md`](thermal_vmodes.md) · **Code:** `thermal_focus/code/thermal_vmodes*.py`, `run_thermal_vmodes.py` · **Output:** `thermal_focus/output/thermal_vmodes/`

## Question

Which v-modes other than v1 carry a component predictable from telescope thermal telemetry?

## Current answer

**V-mode 1 is the only thermal mode.** Over 72,835 science visits and 175 nights on `v50_34_rbr__batoid__consdb_v1`, one mode of 34 survives the false-discovery-rate cut at `q = 0.05` (dimensionless): v1, skill **+0.808**, against **+0.243** for the next mode, v10 (dimensionless, fractional reduction in out-of-fold residual nMAD).

A per-channel screen finds seven modes with |Spearman rho| ≥ 0.4 (dimensionless) against at least one thermal channel — v-mode 18 against the M1M3 z-gradient at rho −0.511 is real — but only v1 behaves like a mode with a thermal origin. The M1M3 prediction holds for one pair of fifteen. Details in [`thermal_vmodes.md`](thermal_vmodes.md).

## Log

### 2026-10-06 — backfilled from docs and history

Backfilled 2026-10-18 from [`thermal_vmodes.md`](thermal_vmodes.md) (formerly `thermal_focus/docs/`) and `notes/status/vmode_thermal_and_lut_handoff.md`. Done in the phase 5 triage: a clear negative. Code stays in `thermal_focus/code/` with the shared thermal-focus modules.
