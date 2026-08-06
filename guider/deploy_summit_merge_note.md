# Guider fixes on `deploy-summit` are not in `main`/weeklies

**TL;DR:** RubinTV's guider star detection is substantially more robust than the
`summit_utils` weeklies because RubinTV builds from the `deploy-summit` branch,
which carries ~9 guider commits that were never merged to `main`. Anyone running
a weekly stack (RSP analyses, etc.) gets the old, buggier guider detection and
silently loses good guide stars that RubinTV finds. Request: merge the
`deploy-summit` guider fixes into `main` so they reach the weeklies.

## What I found

- The RubinTV container builds `summit_utils` from `ac8fc39` (Dockerfile
  `ARG summit_utils_ref`), which is the tip of **`origin/deploy-summit`** =
  `main` (`0959ea2`, 2026-06-10) **+ 9 commits**.
- `main` HEAD is `0959ea2`, and weekly tags `w.2026.27/28/29` all point to that
  same commit — so **no weekly contains these fixes**.
- The 9 deploy-summit commits are almost entirely guider robustness work:
  - `6aaffe7` / `279f707` — `isBlankImage` uses a MAD-based peak-SNR test
    (`peakSnr > 5` OR `peakFlux > 300`) instead of an absolute 300-ADU cut.
  - `ead1d37` — single-stamp fallback when coadd detection fails.
  - `78b54da` — source-detection robustness.
  - edge handling — `measureStarOnStamp` fills border-clipped NaNs with the
    median so GalSim can measure near-edge stars.
  - `9a6559a` — percentile (not median) column-bias subtraction.
  - `c7cbf46` — per-candidate rejection diagnostics.

## Concrete impact (weekly vs deploy-summit)

Exposure **20260709 seqNum 808**, sensor **R44_SG0**: a clean, well-centred,
~40σ guide star. On a weekly stack (`main`) it is **dropped** — the coadd peak
is 279 ADU, under `isBlankImage`'s absolute 300-ADU threshold, so detection is
skipped entirely and the sensor reports "no sources". On `deploy-summit` the
MAD-based `isBlankImage` (and the single-stamp fallback) find it normally — which
is why it appears in the RubinTV movie but not in a weekly-based notebook. This
is not specific to one star: the absolute threshold drops any faint-but-
significant or lower-amplitude guide star on the median coadd.

## Requests

1. **Merge the `deploy-summit` guider fixes into `main`** (or cut them as a PR)
   so they land in the weeklies. Right now RubinTV and every weekly-based
   analysis are running divergent guider code.
2. **Pre-existing typo, present on both branches:** the ellipticity quality cut
   in `applyQualityCuts` (and the copy in `_diagnoseQualityCutRejections` on
   deploy-summit) is written `(|e1| <= max) & (|e1| <= max)` — it tests `e1`
   twice; the second term should be `e2`. Harmless today because the separate
   `|e| = hypot(e1,e2) <= max` cut covers it, but the per-component `e2` cut
   never fires. Fix for both instances is on branch
   `u/aaronroodman/guider-fixes` (based on `deploy-summit`) — happy to open a PR.

Happy to help with the merge or review.
