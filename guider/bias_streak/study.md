---
study: bias_streak
topic: guider
status: parked
question: "Does a column bias from the two blank pre-shutter stamps, plus a streak correction keyed on the ROI row, remove the hatch and streak residuals of the 10th-percentile guider bias?"
answer: "The blank-2-stamp bias adds negligible read noise against the 10th-percentile bias; the streak model is not validated over several nights, so nothing is folded into processStamps."
uses: {}  # none; raw guider stamps
started: 2026-09-01
---

# `bias_streak`: blank-stamp column bias and ROI-row streaks

> **Plan:** [`bias_streak_plan.md`](bias_streak_plan.md)

## Question

Does a column bias from the two blank pre-shutter stamps, plus a streak correction keyed on the ROI row, remove the hatch and streak residuals of the 10th-percentile guider bias?

## Current answer

The blank-2-stamp bias adds negligible read noise against the 10th-percentile bias; the streak model is not validated over several nights, so nothing is folded into processStamps.

## Log

### 2026-10-11 — backfilled from docs and history

Backfilled 2026-10-18 from [`bias_streak_plan.md`](bias_streak_plan.md) (formerly `guider/docs/status/`). Parked in the phase 5 triage: Aaron chose to study it over several nights and evaluate the effective read noise before changing `summit_utils` `processStamps`. Do not implement it there without asking.

Not validated: the streak subtraction as a correction (multi-night test), and the effective read noise of the full scheme.
