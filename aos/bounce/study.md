---
study: bounce
topic: aos
status: done
question: "Do the elevation and rotator bounce tests show repeatable, gravity-driven changes in the optical state, for the look-up table?"
answer: "Yes, and the default recovery puts DOF out of range: 106 of 450 per-(leg, night) rows exceed their range, against 7 under Range-Bounded Recovery, at a median cost of +0.0027 arcsec FWHM."
uses: {}  # old aos/output tree
started: 2026-09-06
---

# `bounce`

> **Code:** `code/` (`run_bounce.py`, `bounce_lib.py`; the `bounce` rule); the reduction is `rubinwork.bounce`, the tables the `bounce_tables` product · **Write-up:** [`bounce.md`](bounce.md) · **Output:** `output/<P>_<M>/` (the study `output@`) · **Note:** `notes/aos-bounce-test-summary/`

## Question

Do the elevation and rotator bounce tests show repeatable, gravity-driven changes in the optical state, for the look-up table?

## Current answer

Results note delivered (`notes/aos-bounce-test-summary/`). Counting `kind = dof` rows, 106 of 450 per-(leg, night) rows exceed their range under the default recovery against 7 under RBR, at a median cost of +0.0027 arcsec FWHM (range −0.0002 to +0.0158 arcsec over nine (leg, night) points).

## Log

### 2026-10-12 — backfilled from docs and history

Moved from `aos/code/bounce/` and the study doc. `rule bounce` runs `bounce/code/run_bounce.py`. The bending-mode notebook is its own study, `bending_mode_test`. Converted 2026-10-23 in phase 5.
