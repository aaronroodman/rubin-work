---
study: centroid_timescale
topic: guider
status: done
question: "Is there a natural timescale that splits guide-star centroid motion into slow wander and fast jitter?"
answer: "No: the motion is scale-free, with no knee in the variance against smoothing timescale and a featureless PSD. The split is fixed at a 1.6 s FWHM Gaussian (8 stamps at 5 Hz); the fast fraction is 0.20 to 0.33 (dimensionless, of variance)."
uses: {}  # guider moments from the old guider/output tree
started: 2026-09-01
---

# `centroid_timescale`: the timescale of guide-star centroid motion

> **Write-up:** [`centroid_motion_timescale.md`](centroid_motion_timescale.md)

## Question

Is there a natural timescale that splits guide-star centroid motion into slow wander and fast jitter?

## Current answer

No: the motion is scale-free, with no knee in the variance against smoothing timescale and a featureless PSD. The split is fixed at a 1.6 s FWHM Gaussian (8 stamps at 5 Hz); the fast fraction is 0.20 to 0.33 (dimensionless, of variance).

## Log

### 2026-10-11 — backfilled from docs and history

Backfilled 2026-10-18 from [`centroid_motion_timescale.md`](centroid_motion_timescale.md) (formerly `guider/docs/`). Done in the phase 5 triage. Data: Gaussian smoothing FWHM scan of 0.4, 0.8, 1.6 and 3.2 s on day_obs 20260706, seqNum 734 to 794. The 1.6 s split is implemented in `summit_utils` `stripPlot(smoothFwhmSec=1.6)` and in `guiderMoments` schema v6 (`cen_var_slow`, `cen_var_fast`, arcsec²).
