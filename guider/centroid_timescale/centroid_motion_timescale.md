# Guider centroid-motion timescale

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** investigation

Guide-star centroid (image) motion has no characteristic timescale: it is scale-free,
consistent with Kolmogorov turbulence. The 1.6 s full width at half maximum (FWHM)
split between "slow" and "fast" motion used in the plots and the moments tables is a
fixed choice, not a measured break.

## Result

- No knee in the fast and slow variance as a function of smoothing timescale, and the
  power spectral density (PSD) is featureless. The integral time `tau` ≈ 0.5 s is not a
  break frequency.
- Data: Gaussian smoothing FWHM scan of 2, 4, 8 and 16 stamps at the 5 Hz guider rate
  (0.4, 0.8, 1.6 and 3.2 s), day_obs 20260706, seqNum 734–794. Plots:
  `output/guider_centroid_smooth_20260706_734_794.pdf`.
- Chosen split: Gaussian FWHM = 1.6 s (8 stamps at 5 Hz). At this timescale the fast
  fraction (variance below 1.6 s divided by total variance, dimensionless) is about
  0.20–0.33; the motion is dominated by slow wander.

## Related findings

- Dynamic ellipticity is strongly correlated between the two guider sensors of the same
  corner raft and nearly uncorrelated across corners. Turbulence decorrelates on raft
  scales; the signal is not tracking error or wind shake.
- A centroid-error simulation shows the per-stamp centroid scatter of about 0.1 arcsec
  is turbulence, not measurement noise. The measurement floor is about 0.01 arcsec at a
  star flux of 1e4 electrons.

## Where it is implemented

- `summit_utils` `guiders/plotting.py`: `stripPlot(smoothFwhmSec=1.6)` overlays a
  per-detector Gaussian-smoothed slow curve on the centroid-motion strip plots
  (`dalt`, `daz`, `dx`, `dy`).
- `guider/code/guiderMoments.py`, schema v6: `summarizeExposure(cenSmoothSec=1.6)` adds
  per-detector `cen_var_slow` and `cen_var_fast` (arcsec², noise-subtracted using the
  per-stamp centroid errors `xerr`, `yerr`) and `cen_smooth_sec` (s), computed from the
  (`dalt`, `daz`) series by `centroidVarianceSplit`. Nights processed before v6 have
  these columns empty until the moments are rerun.
- `guider/guider_moments_ensemble.ipynb`, section "Centroid motion: slow wander vs fast
  jitter (1.6 s split)": distributions, slow fraction, dependence on seeing, and
  same-corner vs cross-corner correlations (via `_corner_pairs`).
