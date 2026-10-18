---
study: olr_pipeline
topic: olr
status: dropped
question: "Can the open-loop wavefront be reproduced per night by removing the applied Trim from the measured corner Zernikes?"
answer: "Dropped: broken by design, since olr.py is a guard that raises; its sign was wrong (it added the Trim contribution). Superseded by rubinwork.open_loop and the value-added optical_state OLR columns."
uses: {}  # none
started: 2026-06-24
---

# `olr_pipeline`: the superseded Zernike-space OLR pipeline

> **Pipeline (do not run):** `Snakefile`, `run_snake.sh`, `config.yaml`, `code/` · **Notebook:** `notebooks/olr_quicklook.ipynb`

## Question

Can the open-loop wavefront be reproduced per night by removing the applied Trim from the measured corner Zernikes?

## Current answer

Dropped: broken by design, since olr.py is a guard that raises; its sign was wrong (it added the Trim contribution). Superseded by rubinwork.open_loop and the value-added optical_state OLR columns.

## Log

### 2026-09-29 — backfilled from docs and history

Backfilled 2026-10-18 from `olr/README.md` and `rubinwork/open_loop.py`'s module docstring. Dropped in the phase 5 triage. The superseded calculation added `z_change = S @ trim` where it must subtract it; the identity `olr_deviation == olr_opd − zk_intrinsic` holds under either sign, so it never caught the error. The original is archived at `scratch/archive/item2/olr_superseded.py`.

The pipeline files moved here from `olr/` as a record. `code/telemetry.py` (a backward-compatible alias for `rubinwork.common.ess_telemetry`) stays in `olr/code/`.
