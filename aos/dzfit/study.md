---
study: dzfit
topic: aos
status: done
question: "Does the per-visit Double Zernike fit of the FAM donut wavefront, made against the batoid design intrinsic, reproduce the stored fit and its Huber scale?"
answer: "Yes: the recomputed residual nMAD matches the stored Huber scale, Pearson r = +0.999665 and median ratio 0.9974 (both dimensionless), n = 1050 visits."
uses: {}  # old aos/output tree
started: 2026-09-06
---

# `dzfit`

> **Code:** `code/` (`run_dz_plots.py`, `run_dz_fit_check.py`; the `plots`, `dz_fit_check` and `residual_movie_chunk` rules) · **Write-up:** [`dzfit.md`](dzfit.md) · **Output:** `output/<P>/` (the study `output@`)

## Question

Does the per-visit Double Zernike fit of the FAM donut wavefront, made against the batoid design intrinsic, reproduce the stored fit and its Huber scale?

## Current answer

The per-visit DZ fit is trustworthy: recomputed nMAD of the residuals against the stored Huber `_scale`, Pearson r = +0.999665 and median ratio 0.9974 (both dimensionless), n = 1050 visits. The validation rules belong with the `fam_tables` product QA (triage).

## Log

### 2026-09-22 — backfilled from docs and history

Moved from `aos/code/dzfit/` and `aos/docs/studies/dzfit.md`. Snakefile rules now run `dzfit/code/`. Done in the triage (answered). Converted 2026-10-23 in phase 5.
