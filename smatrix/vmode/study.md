---
study: vmode
topic: smatrix
status: done
question: "What is the v-mode structure of the OFC sensitivity matrix: DOF composition, singular-value spectrum, observability, and do the two sanctioned SVD engines agree?"
answer: "Reference diagnostics: StateEstimator and build_ofc_svd agree exactly on the matched full slab (max |delta| 0 for weights, singular values and N·V; |cos| = 1.0000000000 per mode, dimensionless) for both the 22/12 and 50/34 schemes."
uses: {}  # no rubin-work products; ts_ofc v13 and the old aos/output param_sets for the Zernike set
started: 2026-09-01
---

# `vmode`: SVD diagnostics of the sensitivity matrix

> **Write-up:** [`vmode.md`](vmode.md) · **Code:** `code/plot_vmode_dof_matrix.py` · **Notebooks:** `notebooks/` · **Output:** `smatrix/output/vmode/` (topic data)

## Question

What is the v-mode structure of the OFC sensitivity matrix: DOF composition, singular-value spectrum, observability, and do the two sanctioned SVD engines agree?

## Current answer

Reference diagnostics: StateEstimator and build_ofc_svd agree exactly on the matched full slab (max |delta| 0 for weights, singular values and N·V; |cos| = 1.0000000000 per mode, dimensionless) for both the 22/12 and 50/34 schemes.

## Log

### 2026-10-11 — backfilled from docs and history

Backfilled 2026-10-18 from [`vmode.md`](vmode.md) (formerly `smatrix/docs/studies/`). The triage split the old study in two: this reference half is done; the sparse-fit half is the study `sparse_fit`. The script and the five notebooks moved here. The script now imports `rubinwork.aos_state` and `rubinwork.smatrix.normalization_weights` directly; the notebooks find the topic by name (`smatrix`) rather than by the first parent holding `code/`, which is now the study directory. Outputs stay in `smatrix/output/vmode/`.
