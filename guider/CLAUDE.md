# CLAUDE.md — guider/

Scoping notes for the guider topic. Loads when files in `guider/` are touched, on top of
the root `CLAUDE.md` (read that first — the "Working with Aaron" rules apply here).

**`docs/status/handoff_guider_session_2026-09.md` is the reference for the state of this work** —
the `summit_utils` branch contents, the bias/streak plan, the centroid-timescale result,
the mosaic geometry, and the open follow-ups. Read it before starting. `README.md` is
thin (it documents one notebook) and does not reflect the current scope. This file
carries only the things that are easy to get wrong.

Note that the HANDOFF was written from the laptop and gives laptop paths
(`~/Astrophysics/Claude/...`). On S3DF the repo is `~/notebooks/rubin-work`.

## Most of the code being changed is in summit_utils, not here

The guider *pipeline* is in `guider/code/`, but much of the active work edits the
**`summit_utils`** package — a local clone on branch **`u/aaronroodman/guider-fixes`**
(pushes to `origin/u/aaronroodman/guider-fixes`). `plotting.py`, `reading.py`,
`detection.py`, and `metrics.py` changes all live there.

Two facts that matter before touching it:

- **The summit does not run `main`.** It runs `origin/deploy-summit`. The 8 Esteves
  guider commits (2–4 Mar 2026) were never merged to `main`. Inspect with
  `git log main..origin/deploy-summit -- python/lsst/summit/utils/guiders/`.
- `guider-fixes` is **at or ahead of** the summit's deployed guider code — all 8
  `deploy-summit` commits are on it by patch-equivalence, so **no cherry-pick is
  needed**. Do not "recover" them.

`processStamps` still uses the 10th-percentile per-column bias. The blank-2-stamp bias +
ROIROW streak rework is **deliberately not folded in** — Aaron's call was "too soon to
modify processStamps", pending multi-night validation and a read-noise evaluation. Do not
implement it into `processStamps` without asking.

## This topic imports from other topics

Unlike most of the repo, `guider/code/` genuinely depends on two sibling topics, via
`sys.path.insert` rather than real packages:

| import | from | used by |
|---|---|---|
| `moments_hsm.measure_hsm_moments` | `optatmo/code` | `run_guider_moments.py`, `extract_adjacent_psf_moments.py`, `guiderMoments.py` |
| `aos_trim.make_consdb_client` | `aos/code` | `check_rotator_field.py` |

So a change to `optatmo/code/moments_hsm.py` can silently change guider moments. The
matched-estimator property is the *point* — both sides of the guider-vs-CCD comparison
must use the same galsim-HSM estimator, or the comparison is confounded. Preserve that
if refactoring.

## Frames — the recurring trap

Guider moments and adjacent-CCD moments are each in **their own detector's DVCS pixel
frame, neither rotated to OCS**. Therefore:

- **T** (trace) is a genuine 1:1 comparison between the two sides.
- **Q1/Q2 are frame-dependent** and only agree after an OCS rotation (each side by its
  own recorded `rot_deg` = physical rotator = `parallacticAngle - boresightRotAngle - 90`,
  plus the cameraGeom detector yaw).

Existing plots show Q1/Q2 unrotated "for scale/pattern only" — do not read them as a 1:1
test, and do not describe them as one. Adding an OCS-rotated Q1/Q2 panel is a known open
follow-up. See `guider-vs-ccd-hsm-frames` in `../notes/claude-memory/`.

Also: the guider moments are Aaron's own `guiderMoments` pipeline, **not** summit_utils
moments — summit_utils supplies only the raw stamps and the star tracker. The science-CCD
side is **not** ConsDB; it comes from `extract_adjacent_psf_moments.py` reading
`preliminary_visit_image` stamps.

## Running the pipeline

From `guider/`. Whole nights:

```bash
./run_snake.sh --day-obs 20260706                    # local, detached via nohup
./run_snake.sh --day-obs 20260706 --limit 5 -n       # dry-run, 5-exposure smoke test
```

`--day-obs` expands to per-night **output files**
(`output/night_<D>/guider_{moments,psfmoments,plots,movies,rtvplots}_<D>.{parquet,pdf,txt}`),
not to rule names — the Snakefile's rules are `moments`, `combine`, `psfmoments`,
`movie(s)`, `rtvplot(s)`, `plots`. Ask for a target by its file path.

There is no single-seq option in `run_snake.sh` — target the per-seq output files directly
with `snakemake`, or use the per-visit runners (`run_guider_rubintv_plots.py`,
`run_guider_movie.py`) with `--day-obs` / `--seq-num`. Full command forms are in
`docs/status/handoff_guider_session_2026-09.md` §6.

**Batch submission is a hard MUST-ASK** — hand Aaron the command instead. Note the
difference from `aos/`: here `--mode batch` submits **one sbatch job per night**, so
`--day-obs A,B,C` is three independent submissions, not one. Submit from an s3df node
(`slacrd`), never an RSP pod.

## Schema version

`guiderMoments.py` is at **schema v6**, which added per-detector `cen_var_slow`,
`cen_var_fast` (arcsec², noise-subtracted) and `cen_smooth_sec`. **These columns are
unpopulated in already-processed nights** — they need a moments rerun. If an analysis
reads them and gets nulls, that is why; do not treat it as a bug in the analysis.

The centroid timescale is a *fixed chosen* 1.6 s FWHM (= 8 stamps at 5 Hz), not a fitted
break. Centroid motion is **scale-free** — no knee, featureless PSD, consistent with
Kolmogorov; the `tau ~ 0.5 s` figure is an integral time, **not** a break frequency. Do
not describe it as a measured knee (`guider-centroid-timescale`).

## Mosaic geometry

Star and full-frame views use **independent** geometry (`positions` vs `positionsFull`)
with separate tunables — changing one does not change the other. The star view masks
pixels outside the 2″ circle to NaN so pairs never visually overlap. Numbers are in the
HANDOFF §5 and `guider-mosaic-layout`; do not re-derive them by eye.
