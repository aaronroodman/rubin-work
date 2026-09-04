# Guider work — session handoff (2026-09)

Consolidated state of the LSSTCam guider-sensor analysis + summit_utils guider-code
work, for handing off to another Claude account. Repos:

- **summit_utils**: `~/Astrophysics/Claude/packages/summit_utils`, branch
  **`u/aaronroodman/guider-fixes`** (pushes to `origin/u/aaronroodman/guider-fixes`).
- **rubin-work**: `~/Astrophysics/Claude/rubin-work` (laptop) / `~/notebooks/rubin-work`
  (USDF RSP). Guider work in `rubin-work/guider/`.
- Laptop python: `/opt/local/bin/python3` (MacPorts, no stack). RSP has the LSST stack.
- `git pull` both repos before running anything.

---

## 1. summit_utils `guider-fixes` branch state

Forked from `main` at `0959ea2`.

**Already incorporated (verified 2026-09):** all 8 `deploy-summit` guider commits
(the summit-deployed guider code, never merged to `main`) are on this branch by
patch-equivalence — including `0f9701d` "10th percentile instead of median for
column bias" (`biasPercentile=10.0`, `np.nanpercentile(data, biasPercentile, axis=0)`
in `reading.py` `processStamps`), the `isBlankImage` MAD/flux changes,
source-detection robustness, and single-stamp fallback. Plus an extra `0a211d2`
STDEVCLIP/sigma68 fallback. So `guider-fixes` is **at or ahead of** the summit's
deployed guider code. **No cherry-pick needed.**

**My changes on top (committed & pushed):**
- `plotting.py`: corner-paired `MosaicLayout` (labels outside circles); `residualScale`
  in `stripPlot`; 1.6 s Gaussian slow-curve overlay on centroid strip plots;
  independent star vs full-frame mosaic geometry + circular NaN pixel mask (see §5).
- `reading.py`: missing-guider tolerance in `getGuiderRawStamps`/`get` (skip a missing
  detector, raise only if none present). NOTE: `deploy-summit`/RubinTV still has the
  all-or-nothing bug.
- `detection.py`: `galsim.Image(imageArray, xmin=0, ymin=0)` 0-based fix.
- `metrics.py`: `DEFAULT_RESIDUAL_SCALE=5.0`, `residualScale` param.

**Not yet done:** `processStamps` still uses the 10th-pct per-column bias; the
blank-2-stamp bias + ROIROW streak rework is deliberately NOT folded in yet (§3).
A PR `guider-fixes` -> `main` would let main catch up with both the summit fixes and
these improvements.

### deploy-summit reference
The summit runs `summit_utils` branch **`origin/deploy-summit`**, not `main`. The 8
Esteves guider commits (2–4 Mar 2026) touch `reading.py`, `detection.py`,
`tracking.py`. Inspect with:
`git log main..origin/deploy-summit -- python/lsst/summit/utils/guiders/`.
The key one is `0f9701d` (percentile bias; Robert's suggestion — median gets pulled by
bright stars/trails and subtracts a dark dip through them; 10th pct avoids the dark
bands). `deploy-summit` also has many unrelated non-guider deploy commits.

---

## 2. Guider-vs-adjacent-CCD 2nd-moment comparison (matched HSM)

Ensemble-notebook plot merges guider moments (`df`, `hsm_*`) with adjacent-CCD PSF
moments (`psf`, `*_ccd`) on `(expId, detector==guider)`.

- **Same estimator both sides:** optatmo galsim-HSM (`measure_hsm_moments`,
  PIFF naming e0=T, e1=Q1, e2=Q2). No unweighted-vs-weighted confound.
- **Guider side (`hsm_Q1/Q2/T`):** Aaron's own `guiderMoments` pipeline (NOT
  summit_utils moments — summit_utils only supplies raw stamps + the star tracker).
  HSM applied to the guide-star **coadd** in the **guider pixel frame, `view='dvcs'`**,
  unrotated. `run_guider_moments.py` injects `hsmFunc=measure_hsm_moments`.
- **Science-CCD side (`Q1_ccd/Q2_ccd/T_ccd`):** NOT ConsDB. From
  `extract_adjacent_psf_moments.py` (Snakefile `psfmoments` rule, `/repo/main`):
  `single_visit_star` (SNR-selected, cleanStars) list, stamps from
  **`preliminary_visit_image`** on each guider's adjacent science CCD
  (`guider_adjacent_ccds.yaml`, `guider_science_collections.yaml`); same HSM per star,
  then **median (+RMS) over that CCD's stars**. In the **science-CCD DVCS pixel frame**,
  unrotated.
- **Frames caveat:** both sides are in their own detector's DVCS pixel frame, NEITHER
  rotated to OCS. **T** (trace) is a genuine 1:1 test; **Q1/Q2 are frame-dependent** and
  only line up after an OCS rotation (each side by its recorded `rot_deg` = physical
  rotator = `parallacticAngle - boresightRotAngle - 90`, + cameraGeom detector yaw).
  The plot shows Q1/Q2 unrotated "for scale/pattern only." Open follow-up: add an
  OCS-rotated Q1/Q2 panel to make it a true 1:1.

---

## 3. Bias/streak correction plan (not yet in processStamps)

Aaron: "too soon to modify processStamps" — validate over multiple nights + evaluate
effective read noise first.

**Bias:** the first **2 stamps** of every guider ROI are pre-shutter blanks (confirmed
via star-flux turn-on: 2 empty for R00/R04; R40/R44 show 3 but the 3rd is unreliable
because the shutter moves in different directions image-to-image — use ONLY stamps 0,1).
Build a per-column bias = **median over the columns of the concatenated stamps 0,1**
(structure is columnar; median of the 2 blanks, not a 2-stamp pixel average). Adds
negligible read noise vs the current 10th-pct default (2.5σ-clipped RMS check).

**Streak:** ROI readout with the shutter open smears charge along the star's column.
Streak position/type tracks metadata **ROIROW** (ITL CCDs are 2000 rows; `nR`=ROI
height=200 always). 5 exclusive ROIROW categories: (1) `<2` connected high;
(2) `>1798` connected low; (3) `2..300` disconnected high; (4) `1500..1798`
disconnected low; (5) `300..1500` small streak. Connected = continuous from star to
readout edge; disconnected = separate high segment. Metadata keys: ROIROW, ROICOL,
ROISEG (from `raw[det].metadata`).

**Status:** studied across nights 20260705–20260712 (seq focus 734–794 on 0706). The
current 10th-pct correction leaves a "hatch" pattern (correlated diagonal bias in a
row-block at ROI edges) + streaks. Example test cases: 20260706/779 R44_SG0 (connected
streak), 20260706/784 R04_SG0 (disconnected heavy streak).

---

## 4. Centroid-motion timescale + cen_var (schema v6)

Centroid (image) motion is **scale-free** — no knee in fast/slow variance vs smoothing
timescale, featureless PSD (`tau~0.5 s` is the integral time, not a break). Consistent
with Kolmogorov. Verified via a Gaussian-FWHM scan (2/4/8/16 stamps at 5 Hz) over
20260706 seq 734–794.

**Chosen fixed timescale: 1.6 s FWHM (= 8 stamps at 5 Hz).** At 1.6 s only ~20–33% of
the variance is "fast" (below the timescale); motion is dominated by slow wander.

Implemented:
- `plotting.py` `stripPlot(smoothFwhmSec=1.6)` — per-detector Gaussian slow curve on the
  centroid-motion strip plots.
- `guiderMoments.py` **schema v6**: `summarizeExposure(cenSmoothSec=1.6)` adds per-detector
  `cen_var_slow`, `cen_var_fast` (arcsec^2, noise-subtracted via xerr/yerr) and
  `cen_smooth_sec`, from (dalt,daz). Helper `centroidVarianceSplit`. **Populating these
  needs a moments rerun (v6).**
- `guider_moments_ensemble.ipynb`: "Centroid motion: slow wander vs fast jitter" section
  (distributions, slow fraction, vs seeing, same/cross-corner correlations via
  `_corner_pairs`).

Related findings: dynamic ellipticity is strongly correlated same-corner, nearly
uncorrelated cross-corner -> turbulence decorrelates on raft scales (not tracking/wind).
A centroid-error sim proved the ~0.1" per-stamp scatter is turbulence, not measurement
noise (floor ~0.01" at 1e4 e-).

---

## 5. Mosaic layout (`MosaicLayout`)

Corner-paired: each corner raft's 2 SG panels sit together in that raft's focal-plane
corner (R00 LL, R04 LR, R40 UL, R44 UR), symmetric under x->-x, y->-y.

**Star and full-frame views use INDEPENDENT geometry** (`positions` vs `positionsFull`):
- Star (2" circles): `radius=0.13`, `pairDist=0.25`, `pairGap=0.004` (perp offset =
  `radius/sqrt2 + pairGap`). Small gap (~0.011) between the pair, margin (~0.024) to the
  edge. `renderStampPanel` masks pixels outside the 2" circle (`maskRadius=10` px about
  viewCenter = the drawn circle) to NaN, transparent via Greys `set_bad(alpha=0)`, so a
  pair's square cutouts show only circular content and never overlap.
- Full-frame (squares): `fullFrameHalf=0.105`, `fullPairDist=0.25`, perpendicular offset
  = `fullFrameHalf` so a corner pair's boxes meet corner-to-corner (touching, no overlap).

Both views clear the center metadata box + arrow box. 2" circle = 10 px (0.2"/px);
`addReferenceOverlays` draws radii [10,5] = 2"/1".

---

## 6. Pipeline + review commands

Guider moment pipeline in `rubin-work/guider`: `Snakefile`, `snake_config.yaml`,
launcher `run_snake.sh`.

**Whole night(s):** `./run_snake.sh --day-obs 20260706` (local, nohup) or `--mode batch`
(one sbatch/night, from s3df e.g. `slacrd`). `--limit N` smoke test. Per-night targets:
`guider_moments`, `guider_psfmoments`, `guider_plots`, `guider_movies`, `guider_rtvplots`.

**Single visit** (no single-seq option in run_snake — target the per-seq files):
```
snakemake -j 4 --resources mem_mb=14000 --keep-going \
  output/night_20260706/movies/787_full.mp4 \
  output/night_20260706/movies/787_star.mp4 \
  output/night_20260706/rtvplots/787_rtvplots.pdf \
  output/night_20260706/seq/787_moments.parquet
```

**Per-visit runners directly:**
- `python code/run_guider_rubintv_plots.py --day-obs D --seq-num S --outdir DIR`
  -> `S_rtvplots.pdf` (6 strip plots + both mosaics).
- `python code/run_guider_movie.py --day-obs D --seq-num S --outdir DIR --which both --format mp4`
  -> full + star movies.

**Saturation study:** `python code/run_guider_saturation.py --day-obs D`
-> `output/guider_saturation_<D>.parquet` (one row per expId,detector: peak_top3,
peak_max, roirow, star_row/col, roicol, roiseg, ...). Ran 0705,0706,0708–0712.
Saturation level set ~40000 ADU. Notebook summary snippet: per-CCD histogram + top-per-CCD
candidate list.

**Config:** `movie_stride=50`, `residual_scale=10.0` (RANSAC inlier band),
`guider_hz=5.0`, `image_types=[science]`. Moments emit schema v6 (`cen_var_*`).

---

## 7. Open follow-ups

- Rerun moments over the processed nights to populate `cen_var_*` (schema v6).
- Regenerate movies/plots to eyeball the new star + full-frame mosaic geometry
  (tune `radius`/`pairDist`/`pairGap`; `fullFrameHalf`/`fullPairDist`).
- Validate the blank-2-stamp bias + ROIROW streak scheme over multiple nights, then fold
  into `processStamps` (§3).
- Add per-CCD saturation flag once the level is finalized.
- Optional: OCS-rotated Q1/Q2 panel in the guider-vs-CCD comparison (§2).
- Optional: PR `guider-fixes` -> `main`.
