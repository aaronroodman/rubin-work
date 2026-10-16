---
study: itl_dip
topic: camera
status: active
question: How deep is the ITL column dip beside a saturated star, how wide, and how far along the column does it reach, measured against the sky level on on-sky images?
answer: About 3 to 4% of sky deep (50 to 72 counts against 1630 to 1735 counts, i band, 30 s) and 10 to 19 pixels wide on ITL detectors; an e2v control shows 1.3 counts. Measured on one exposure only.
uses: {}  # no rubin-work products; raw frames and calibrations from the /repo/main Butler
started: 2026-10-15
---

# `itl_dip`: the sky deficit in columns of saturated stars on ITL CCDs

> **Notebook:** `notebooks/itl_column_dip_v1.ipynb` · **Method:** [`method.md`](method.md)
> · **Output:** `output/itl_dip_<day_obs>_<seq_num>.pdf` (all figures, one multi-page PDF)

## Question

On Imaging Technology Laboratory (ITL) Charge-Coupled Devices (CCDs) in LSSTCam, the sky
background is depressed in the columns that contain a saturated star, and the e2v CCDs
do not show it. How deep is this dip, how wide, and how far along the column does it
reach, measured against the sky level on real on-sky images?

## Current answer

**The dip is about 3 to 4% of sky deep and 10 to 19 pixels wide on ITL detectors.** All
numbers are from one exposure, 2026070800022 (`day_obs` 20260708, `seq_num` 22; a
`lowdust` field, i band, 30 s), so they are indicative, not characterised.

| measurement | depth (counts) | sky (counts) | depth / sky |
|---|---|---|---|
| R02_S12, three dips with partially filled bleed trails | 72.3, 68.8, 67.6 | 1733 | 0.039 to 0.042 |
| one dip each on three detectors (R43_S11, R02_S12, R01_S12) | 50.2, 71.8, 61.7 | 1630 to 1735 | 0.031 to 0.041 |
| e2v control, R11_S10 | 1.3 | — | about 0.001 |

`depth / sky` is dimensionless: deficit in counts over sky level in counts. The three
R02_S12 dips agree with one another to 3% (dimensionless); the e2v control is a factor
of 53 (dimensionless) smaller. On detector R02_S12 the pipeline's own `ITL_DIP` mask
finds 9 dips, 10 to 19 pixels wide.

Two findings decide whether the measurement works at all (see [`method.md`](method.md)):

- **Use a sparse field.** In crowded `bulgy` fields the same measurement gives 4 to 43
  counts with no consistency between stars; the confusion noise is as large as the dip.
- **Run ISR with the dip and saturation masking off.** The production configuration
  masks the dip (`doITLDipMask`, ticket DM-50343, rules on 25 ITL detectors and no e2v
  detector), so the stored pipeline products cannot show it.

Not yet measured: how far along the column the dip reaches, the stack over many dips,
and the dependence on star magnitude and on position within the amp. See the Log.

## Log

### 2026-10-16 — tidied to the study layout

Split the method guide into [`method.md`](method.md) and added front matter and this
Log (organization plan section 7). No new measurement. The figures stay in `output/`;
`figures/` is not used. The exploratory scripts under `scratch/itldip/`, including the
throwaway generator for the notebook, were removed.

### 2026-10-15 — first measurement, one exposure

Commits `9969bcc` (study and notebook), `d7a883f` (moved to the phase 5 study layout)
and `f185da4` (placeholder directories dropped). Output:
`output/itl_dip_20260708_22.pdf`.

Results as in Current answer. Method decisions, all recorded in
[`method.md`](method.md):

- dips located from the pipeline's `ITL_DIP` mask with the production per-detector
  rule, not from star positions: the bleed trail is itself a bright column, and the dip
  sits beside it;
- the profile is taken from rows beyond the end of a partially filled bleed trail, and
  dips whose trail leaves no clean rows are rejected;
- turning off brighter-fatter and deferred-charge correction takes ISR on one detector
  from several minutes to about 7 seconds, with no effect on the sky level.

**Next:**

1. stack many dips, aligned on the dip columns, across the 25 affected detectors;
2. depth and width against star magnitude and against each detector's
   `itlDipBackgroundFraction`;
3. dependence on where in the 509-pixel-wide amp the star sits;
4. the dip's extent along the column.
