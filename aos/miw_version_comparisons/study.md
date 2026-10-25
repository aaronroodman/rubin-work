---
study: miw_version_comparisons
topic: aos
status: done
question: "How much do MIW builds differ between Danish 1.2 and 1.3 wavefronts and between the legacy and v1000 pupil models, term by term and as image quality?"
answer: "Danish 1.2 and 1.3 builds agree in Z5-Z8 to about 9 % of amplitude RMS. v1000 moves the MIW by 0.1027 of amplitude (dimensionless, difference RMS over build-A RMS) with no pupil-rim signature."
uses:  # its Danish 1.2 side was the old-tree d12-A_50_34_i_5rot build of 2026-07-30, not 20261014
  miw: d13t-A_50_34_i_5rot@old-tree-20260928
started: 2026-09-28
---

# `miw_version_comparisons`

> **Code:** `code/` (`compare_miw_versions.py`, `compare_pupil_models.py`; `compare_build_dof.py` is in `miw_recovery_constraint`, beside the `check_dof_ranges` it imports) · **Notebook:** `notebooks/aos_miw_ocs_ccs_maps.ipynb` · **Write-up:** [`../docs/studies/miw.md`](../docs/studies/miw.md), "Comparing two builds" and "The pupil model"

## Question

How much do MIW builds differ between Danish 1.2 and 1.3 wavefronts and between the legacy and v1000 pupil models, term by term and as image quality?

## Current answer

- Danish 1.2 against 1.3 builds: Z5–Z8 agree to about 9 % of amplitude RMS; Z9 and Z10 are larger in the unpaired build by 1.76 and 1.50 of amplitude (dimensionless).
- Legacy against v1000 pupil: difference RMS over build-A RMS 0.1027 (dimensionless, both amplitudes), with no pupil-rim signature.

## Log

### 2026-10-05 — backfilled from docs and history

Split from aos `miw`. The three comparison scripts and the OCS/CCS map viewer moved from `aos/code/miw/` and `aos/notebooks/miw/` and import `rubinwork.products.miw` in place of the `miw_io` shim. Converted 2026-10-23 in phase 5.
