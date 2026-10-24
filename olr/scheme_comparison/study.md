---
study: scheme_comparison
topic: olr
status: done
question: "On science visits, what image quality and DOF do the 22/12, 50/34 and 50/34+RBR correction schemes recover from the same corner wavefront measurements?"
answer: "Median CWFS FWHM contribution 0.4050, 0.2673 and 0.2905 arcsec for 22/12, 50/34 and 50/34+RBR over 96,278 paired visits; 50/34 improves on 22/12 by a median 0.1270 arcsec on 96.6% of visits, and RBR gives back 0.0245 arcsec of that."
uses: {}  # value-added optical_state, built before the value_added product was registered
started: 2026-10-03
---

# `scheme_comparison`: the three correction schemes on the stored OLR

> **Write-up:** [`scheme_comparison.md`](scheme_comparison.md) · **Data:** the value-added `optical_state` columns, built by `value_added/code/build_optical_state.py`

## Question

On science visits, what image quality and DOF do the 22/12, 50/34 and 50/34+RBR correction schemes recover from the same corner wavefront measurements?

## Current answer

Over 96,278 science and acquisition visits recovered in all three schemes (`day_obs` 20250724 to 20260714, 214 nights), the median CWFS residual FWHM contribution is:

| scheme | median (arcsec) |
|---|---|
| `22_12` | 0.4050 |
| `50_34` | 0.2673 |
| `50_34_rbr` | 0.2905 |

Paired: 50/34 minus 22/12 is −0.1270 arcsec, better on 96.6% of visits (fraction of visits); adding RBR gives back +0.0245 arcsec, about 19% of the gain (fraction of the FWHM gain), while improving on 22/12 more reliably (98.7% of visits). The schemes differ in scatter more than in median. Details, v-modes and sign conventions in [`scheme_comparison.md`](scheme_comparison.md).

## Log

### 2026-10-04 — backfilled from docs and history

Backfilled 2026-10-18 from [`scheme_comparison.md`](scheme_comparison.md) (formerly `olr/docs/`). Done in the phase 5 triage. `olr/` is now a study topic for analyses of the stored OLR, not a product (triage). `aos/miw_recovery_constraint/code/compare_schemes.py` is a different comparison, inside the MIW build on FAM visits, and stays with the aos MIW studies.
