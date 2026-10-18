---
study: aos_dof_audit
topic: olr
status: parked
question: "Are the commanded AOS DOF (LUT, Trim, Tweak, applied) recorded in the EFD self-consistent visit by visit?"
answer: ''
uses: {}  # none; EFD on the Summit
started: 2026-07-17
---

# `aos_dof_audit`: LUT, Trim, Tweak and applied DOF per visit

> **Code:** `code/aos_dof_audit.py` (Summit RSP only; writes `output/<day_obs>/dof_audit.parquet`, run from this study directory) · **Notebook:** `notebooks/aos_dof_audit.ipynb`

## Question

Are the commanded AOS DOF (LUT, Trim, Tweak, applied) recorded in the EFD self-consistent visit by visit?

## Current answer

None written. The tool pulls one row per visit from the EFD (Summit RSP only) and the notebook checks Trim and Tweak history and their accumulation; no conclusion recorded.

## Log

### 2026-07-17 — backfilled from docs and history

Backfilled 2026-10-18 from `olr/README.md` ("AOS DOF audit") and commit history. Parked in the phase 5 triage: the tool exists, no conclusion. Both files moved here; the notebook reads `../output/<day_obs>/dof_audit.parquet`.
