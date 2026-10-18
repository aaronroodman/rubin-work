---
study: t539_closedloop
topic: blocks
status: parked
question: "What AOS state (DOF, v-modes, per-corner Zernikes) and telemetry go with the converged closed-loop BLOCK-T539 in-focus alignment images?"
answer: ''
uses: {}  # no rubin-work products; ConsDB, EFD and the OFC v13 sensitivity matrix
started: 2026-07-08
---

# `t539_closedloop`: closed-loop AOS state of the BLOCK-T539 alignment images

> **Pipeline:** `Snakefile`, `run_snake.sh`, `config.yaml` (run from this directory; `./run_snake.sh -n` for a dry run) · **Code:** `code/` · **Notebook:** `notebooks/t539_closedloop_aos.ipynb` · **Output:** `output/`

## Question

What AOS state — degrees of freedom (DOF), v-modes and per-corner Zernikes — and what thermal and wind telemetry go with the converged closed-loop BLOCK-T539 in-focus alignment images? The same code also builds a per-night table of all science visits (`night_table`) for studying the mirror look-up-table bending modes over the full elevation range.

## Current answer

None written. The pipeline builds the table and validation PDFs (day_obs 20260420 to 20260720 in the old tree), with no written conclusion.

## Log

### 2026-09-22 — backfilled from docs and history

Backfilled 2026-10-18 from the `Snakefile` and `config.yaml` headers and the commits `a3a0d83` to `7961075`. Parked in the phase 5 triage: validation PDFs to day_obs 20260720, no written conclusion, plumbing only since August.

The pipeline moved here from `blocks/` with its code and notebook. Its outputs now go to this study's `output/`; the old tables and PDFs stay in `blocks/output/`, so a dry run plans 4 jobs (count). The Snakefile's library inputs now point at `rubinwork/common/` (the `../common/*.py` paths were stale since the phase 1 move). It still lists `../../aos/code/aos_state.py`, a known dependency the rule checker does not scan.

Owed from organization plan section 10: rerun `build_table`, `night_table` and `plots` after the ESS and DOF telemetry moves.
