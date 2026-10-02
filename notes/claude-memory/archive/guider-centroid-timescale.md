---
name: guider-centroid-timescale
description: Guider centroid motion is scale-free (no knee); 1.6s Gaussian split; cen_var in guiderMoments schema v6
metadata: 
  node_type: memory
  type: project
  originSessionId: 086f02ad-edae-43f2-b884-836dc3cd9f2c
  modified: 2026-09-04T18:21:31.468Z
---

Guider centroid-motion (image-motion) timescale finding: the motion is **scale-free** — no knee in the fast/slow variance vs smoothing timescale, and a featureless PSD (`tau~0.5s` is just the integral time, not a break). Consistent with Kolmogorov turbulence. Verified via a Gaussian-FWHM scan (2/4/8/16 stamps at 5 Hz) over 20260706 seq 734–794.

**Chosen fixed timescale: 1.6 s FWHM (= 8 stamps at 5 Hz).** At 1.6 s only ~20–33% of the variance is "fast" (below the timescale) — motion is dominated by slow wander.

Implemented:
- `summit_utils` `plotting.py` `stripPlot(smoothFwhmSec=1.6)` overlays a per-detector Gaussian slow curve on the centroid-motion strip plots (dalt/daz/dx/dy). See [[guider-fixes-branch-state]].
- `rubin-work/guider/code/guiderMoments.py` **schema v6**: `summarizeExposure(cenSmoothSec=1.6)` adds per-detector `cen_var_slow`, `cen_var_fast` (arcsec^2, noise-subtracted via xerr/yerr) and `cen_smooth_sec`, from the (dalt,daz) series. Helper `centroidVarianceSplit`. Populating these needs a moments rerun (v6).
- Ensemble notebook `guider_moments_ensemble.ipynb` has a "Centroid motion: slow wander vs fast jitter" section (distributions, slow fraction, vs seeing, same/cross-corner correlations via `_corner_pairs`).

Prior related finding: dynamic ellipticity is strongly correlated same-corner, nearly uncorrelated cross-corner → turbulence decorrelates on raft scales (not tracking/wind-shake). A centroid-error sim proved the ~0.1" per-stamp scatter is turbulence, not measurement noise (measurement floor ~0.01" at 1e4 e-).
