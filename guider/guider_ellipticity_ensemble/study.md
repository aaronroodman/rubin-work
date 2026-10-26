---
study: guider_ellipticity_ensemble
topic: guider
status: active
question: "What do the guide-star moments show across many exposures: how are static (optics) and dynamic (image-motion) ellipticity split, and how do they correlate between guider sensors?"
answer: "Dynamic ellipticity is strongly correlated between the two guider sensors of one corner raft and nearly uncorrelated across corners, so turbulence decorrelates on raft scales; it is not tracking error or wind shake."
uses:  # every night of each variant (incremental, no single build); v6 when built
  guider_moments: v4
started: 2026-07-16
---

# `guider_ellipticity_ensemble`: guide-star ellipticity and centroid motion over many exposures

> **Notebooks:** `notebooks/guider_moments_ensemble.ipynb` (ensemble), `notebooks/guider_star_ellipticity.ipynb` (per exposure) · **Derivation:** [`moment_weighting_analysis.md`](moment_weighting_analysis.md) · **Input:** the `guider_moments` product, every night (v6, then v4); the adjacent-CCD moments from `../guider_vs_ccd_moments/output/`

## Question

What do the guide-star moments show across many exposures: how are static (optics) and dynamic (image-motion) ellipticity split, and how do they correlate between guider sensors?

## Current answer

Dynamic ellipticity is strongly correlated between the two guider sensors of one corner raft and nearly uncorrelated across corners, so turbulence decorrelates on raft scales; it is not tracking error or wind shake.

## Log

### 2026-09-01 — backfilled from docs and history

Backfilled 2026-10-18 from the notebook headers, `centroid_motion_timescale.md` ("Related findings") and `moment_weighting_analysis.md` (formerly `guider/docs/`); commits from 2026-07-16 to 2026-09-01. Converted as active in the phase 5 triage; the parent study of `guider_vs_ccd_moments` and `centroid_timescale`.

The notebook decomposes the coadd second moments into static per-stamp plus image motion (the additivity check), compares optics and motion terms, and shows ensemble PSDs and 3rd-order moments. A centroid-error simulation shows the per-stamp centroid scatter of about 0.1 arcsec is turbulence; the measurement floor is about 0.01 arcsec at a star flux of 1e4 electrons.

Both notebooks moved here. Imports now come from `rubinwork.products.guider_moments.guiderMoments` and `rubinwork.guider_edge_recovery` instead of the `guider/code` shims. The ensemble notebook still reads the old `guider/output/night_*` tree and writes plots to this study's `output/`.
