# Guider bias and streak correction plan

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** working state (plan)

Planned replacement for the guider column-bias subtraction in `summit_utils`
`guiders/reading.py` `processStamps`, plus a streak correction keyed on the region of
interest (ROI) row. This is a plan. It is not in `processStamps`: Aaron's decision is to
study it over several nights and evaluate the effective read noise first.

## Current correction and its residuals

`processStamps` subtracts a per-column bias equal to the 10th percentile of each column
(`biasPercentile=10.0`, `np.nanpercentile(data, biasPercentile, axis=0)`). The 10th
percentile replaced the median because the median is pulled up by bright stars and
trails and then subtracts a dark dip through them. It still leaves two residuals:

- a "hatch" pattern: correlated diagonal bias in a block of rows at the ROI edges;
- streaks along the star's column.

## Planned bias

- The first 2 stamps of every guider ROI are pre-shutter blanks. Shown by the star-flux
  turn-on: 2 empty stamps for R00 and R04. R40 and R44 show 3, but the 3rd is unreliable
  because the shutter moves in different directions from image to image. Use only
  stamps 0 and 1.
- Bias per column = median down each column of stamps 0 and 1 concatenated along rows.
  The structure is columnar, so this is a column median, not a pixel-by-pixel average of
  the 2 stamps.
- `guider/guide_star_saturation/code/run_guider_saturation.py` already uses this bias (`--nblank 2`).

## Planned streak model

Reading out the ROI while the shutter is open smears charge along the star's column. The
streak's position and type follow the header keyword `ROIROW` (ROI starting row). The
ITL CCDs have 2000 rows; the ROI height `nR` is always 200 rows. Five exclusive
categories, by `ROIROW` in rows:

| `ROIROW` | streak type |
|---|---|
| < 2 | connected, high |
| > 1798 | connected, low |
| 2 to 300 | disconnected, high |
| 1500 to 1798 | disconnected, low |
| 300 to 1500 | small |

Connected: the streak runs continuously from the star to the readout edge.
Disconnected: a separate high segment. Header keywords used: `ROIROW`, `ROICOL`,
`ROISEG` (from `raw[det].metadata`).

## Validation status

Validated:
- The blank-2-stamp bias adds negligible read noise relative to the 10th-percentile
  bias (2.5σ-clipped RMS of the corrected stamps).
- The streak categories were derived from day_obs 20260705–20260712, mainly seqNum
  734–794 on 20260706.

Not validated:
- The streak subtraction as a correction (multi-night test outstanding).
- The effective read noise of the full scheme.
- Any change to `processStamps`. Do not implement it there without asking Aaron.

Test cases: 20260706 seqNum 779 R44_SG0 (connected streak); 20260706 seqNum 784 R04_SG0
(disconnected, heavy streak). Study plots in `guider/output/`:
`guider_bias_compare_20260706.pdf`, `guider_streak_study_20260706_734_794.pdf`,
`streak_categories_20260706_734_794.pdf`, and the notebook
`output/night_20260706/guider_streak_examples.ipynb`.
