---
name: deploy-summit-guider-branch
description: summit_utils origin/deploy-summit carried 8 guider fixes never merged to main
metadata: 
  node_type: memory
  type: reference
  originSessionId: 086f02ad-edae-43f2-b884-836dc3cd9f2c
  modified: 2026-09-04T18:21:05.648Z
---

The guider code actually running at the summit lives on `summit_utils` branch **`origin/deploy-summit`**, not `main`. As of 2026-09 it carried 8 guider commits by Johnny Esteves (2–4 Mar 2026) in `python/lsst/summit/utils/guiders/` (reading.py, detection.py, tracking.py) that were never merged to `main`:

- `0f9701d` reading.py — **10th percentile instead of median for column-bias** (the key one; Robert's suggestion: median gets pulled by bright stars/trails and subtracts a dark dip through them; 10th pct avoids the dark bands). Added `biasPercentile=10.0` to `GuiderReader.get()` and `processStamps()`.
- `6f7a79b` per-candidate rejection diagnostics; `d7362ff`+`45ac07a` flux-weighted centroid fallback (added then reverted); `928437f` flux threshold in `isBlankImage`; `d4c0813` MAD std in `isBlankImage`; `0e2af24` source-detection robustness; `f213728` single-stamp fallback when coadd detection fails.

These are all already incorporated into [[guider-fixes-branch-state]]. To inspect: `git log main..origin/deploy-summit -- python/lsst/summit/utils/guiders/`. `deploy-summit` also has many unrelated non-guider deploy commits.
