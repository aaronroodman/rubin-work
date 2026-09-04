---
name: guider-fixes-branch-state
description: State of the summit_utils u/aaronroodman/guider-fixes branch (guider RubinTV fixes) as of 2026-09
metadata: 
  node_type: memory
  type: project
  originSessionId: 086f02ad-edae-43f2-b884-836dc3cd9f2c
  modified: 2026-09-04T18:20:56.118Z
---

Working on `summit_utils` branch **`u/aaronroodman/guider-fixes`** (repo: `~/Astrophysics/Claude/packages/summit_utils`, remote pushes to `origin/u/aaronroodman/guider-fixes`). Forked from `main` at `0959ea2`.

**Already incorporated (verified 2026-09):** all 8 `deploy-summit` guider commits by Johnny Esteves (the summit-deployed guider code that was never merged to `main`) are already on this branch by patch-equivalence — including `0f9701d` "percentile instead of median for column bias" (`biasPercentile=10.0`, `np.nanpercentile(data, biasPercentile, axis=0)` in `reading.py` `processStamps`), the `isBlankImage` MAD/flux changes, source-detection robustness, single-stamp fallback. Branch also has an extra `0a211d2` STDEVCLIP/sigma68 fallback. So `guider-fixes` is **at or ahead of** the summit's deployed guider code. `deploy-summit`'s remaining non-guider commits are irrelevant. No cherry-pick needed. See [[deploy-summit-guider-branch]].

**My changes on top (committed):**
- `plotting.py`: corner-paired `MosaicLayout` (labels outside circles); `residualScale` in `stripPlot`; 1.6s Gaussian slow-curve overlay on the centroid strip plots; independent star vs full-frame mosaic geometry + circular NaN pixel mask — see [[guider-mosaic-layout]].
- `reading.py`: missing-guider tolerance in `getGuiderRawStamps`/`get` (skip missing detector, raise only if none). NOTE: `deploy-summit`/RubinTV still has the all-or-nothing bug.
- `detection.py`: `galsim.Image(imageArray, xmin=0, ymin=0)` 0-based fix.
- `metrics.py`: `DEFAULT_RESIDUAL_SCALE=5.0`, `residualScale` param.

**Not yet done:** processStamps still uses the 10th-pct per-column bias; the blank-2-stamp bias + ROIROW streak rework is deliberately NOT folded in yet ("too soon", validate first) — see [[guider-bias-streak-plan]]. A PR `guider-fixes` -> `main` would let main catch up with both the summit fixes and these improvements.
