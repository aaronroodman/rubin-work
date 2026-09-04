---
name: guider-bias-streak-plan
description: Planned guider blank-2-stamp column bias + ROIROW-informed streak subtraction (not yet in processStamps)
metadata: 
  node_type: memory
  type: project
  originSessionId: 086f02ad-edae-43f2-b884-836dc3cd9f2c
  modified: 2026-09-04T18:21:18.400Z
---

Next-step guider bias/streak correction, being validated before folding into summit_utils `processStamps` (Aaron: "too soon to modify processStamps" — study over multiple nights, evaluate effective read noise first).

**Bias:** the first **2 stamps** of every guider ROI are pre-shutter blanks (confirmed via star-flux turn-on: 2 empty for R00/R04; R40/R44 show 3 but the 3rd is unreliable because the shutter moves in different directions image-to-image — so use ONLY stamps 0,1). Build a per-column bias = **median over the columns of the concatenated stamps 0,1** (structure is columnar; median of the 2 blanks, not a 2-stamp pixel average). Adds negligible read noise vs the current 10th-pct default (validated with 2.5σ-clipped RMS).

**Streak:** ROI readout with the shutter open smears charge along the star's column. The streak position/type tracks the metadata **ROIROW** (ITL CCDs are 2000 rows, `nR`=ROI height=200 always). 5 exclusive categories by ROIROW: (1) `<2` connected high; (2) `>1798` connected low; (3) `2..300` disconnected high; (4) `1500..1798` disconnected low; (5) `300..1500` small streak. Connected = streak continuous from star to readout edge; disconnected = separate high segment. Metadata keys: ROIROW, ROICOL, ROISEG.

Current status: studied across nights (day_obs 20260705–20260712, seq focus 734–794 on 0706); the current 10th-pct correction leaves a "hatch" pattern (correlated diagonal bias in a row-block at ROI edges) + streaks. Example cases for testing: 20260706/779 R44_SG0 (connected streak), 20260706/784 R04_SG0 (disconnected heavy streak). Related: [[guider-fixes-branch-state]].
