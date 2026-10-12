# CLAUDE.md — guider/

Scoping notes for the guider topic. Loads when files in `guider/` are touched, on top of
the root `CLAUDE.md` (read that first — the "Working with Aaron" rules apply here).

Where the facts live (grep the doc the task needs rather than reading it whole):

| doc | holds |
|---|---|
| `docs/status/summit_utils_branch_state.md` | `summit_utils` `guider-fixes` vs `main` vs `deploy-summit`, git-verified |
| `docs/status/bias_streak_plan.md` | blank-2-stamp bias + ROIROW streak plan, what is validated |
| `docs/centroid_motion_timescale.md` | scale-free centroid motion, the 1.6 s split, schema v6 |
| `docs/guider_vs_ccd_moments.md` | matched-HSM guider vs adjacent-CCD moments, frames |
| `docs/mosaic_layout.md` | `MosaicLayout` geometry numbers |
| `docs/pipeline_runbook.md` | full command forms |
| `docs/status/handoff_guider_session_2026-09.md` | 2026-09 snapshot of all of the above + open follow-ups |

The handoff gives laptop paths (`~/Astrophysics/Claude/...`). On S3DF the repo is
`~/notebooks/rubin-work`. This file carries only the things that are easy to get wrong.

## Most of the code being changed is in summit_utils, not here

The guider *pipeline* is in `guider/code/`, but much of the active work edits the
**`summit_utils`** package — a local clone on branch **`u/aaronroodman/guider-fixes`**
(pushes to `origin/u/aaronroodman/guider-fixes`). `plotting.py`, `reading.py`,
`detection.py`, and `metrics.py` changes all live there.

Facts that matter before touching it:

- **The summit does not run `main`.** It runs `origin/deploy-summit`. The 8 Esteves
  guider commits (2–4 Mar 2026) were never merged to `main`. Inspect with
  `git log main..origin/deploy-summit -- python/lsst/summit/utils/guiders/`.
- `guider-fixes` is **at or ahead of** the summit's deployed guider code — all 8
  `deploy-summit` commits are on it by patch-equivalence, so **no cherry-pick is
  needed**. Do not "recover" them. Verified with `git cherry` on 2026-10-11 (remote refs
  as of the 2026-09-03 fetch); `deploy-summit` has since been rebased, so its guider
  commits carry different hashes from the copies on `guider-fixes`.
- Local `main` is stale at the fork point `0959ea2`; `origin/main` is at `f50eea2` and
  still lacks the `deploy-summit` guider commits. `0a211d2` (STDEVCLIP/sigma68 source
  detection) is on `main`, not an extra commit on `guider-fixes`.

`processStamps` still uses the 10th-percentile per-column bias. The blank-2-stamp bias +
ROIROW streak rework is **deliberately not folded in** — Aaron's call was "too soon to
modify processStamps", pending multi-night validation and a read-noise evaluation. Do not
implement it into `processStamps` without asking. Plan and validation status:
`docs/status/bias_streak_plan.md`.

## The HSM estimator is now a library, not a cross-topic reach

It used to be: `guider/code/` reached `moments_hsm` in `optatmo/code` through
`sys.path.insert(__file__/../../optatmo/code)`. Phase 3a moved it to
**`rubinwork.hsm_moments`**, so all three importers —
`rubinwork/products/guider_moments/builders/run_guider_moments.py`,
`extract_adjacent_psf_moments.py` and `optatmo/code/extract_psf_moments.py` — now do
`from rubinwork.hsm_moments import measure_hsm_moments`. A shim remains at
`optatmo/code/moments_hsm.py` and gives the same module object.

A change to `rubinwork/hsm_moments.py` still silently changes guider moments, and that
matched-estimator property is the *point*: both sides of the guider-vs-CCD comparison must
use the same galsim-HSM estimator, or the comparison is confounded. Preserve it if
refactoring.

## Frames — the recurring trap

Guider moments and adjacent-CCD moments are each in **their own detector's DVCS pixel
frame, neither rotated to OCS**. Therefore:

- **T** (trace) is a genuine 1:1 comparison between the two sides.
- **Q1/Q2 are frame-dependent** and only agree after an OCS rotation (each side by its
  own recorded `rot_deg` = physical rotator = `parallacticAngle - boresightRotAngle - 90`,
  plus the cameraGeom detector yaw).

Existing plots show Q1/Q2 unrotated "for scale/pattern only" — do not read them as a 1:1
test, and do not describe them as one. Adding an OCS-rotated Q1/Q2 panel is a known open
follow-up.

Also: the guider moments are Aaron's own `guiderMoments` pipeline, **not** summit_utils
moments — summit_utils supplies only the raw stamps and the star tracker. The science-CCD
side is **not** ConsDB; it comes from `extract_adjacent_psf_moments.py` reading
`preliminary_visit_image` stamps.

## Running the pipeline: product first, then the studies

Phase 3a split the pipeline in two. The three guider moment tables are the
**`guider_moments` product**, built from the **repository root**:

```bash
cd ~/notebooks/rubin-work
snakemake -s rubinwork/products/guider_moments/Snakefile --config night=20260706
snakemake -s rubinwork/products/guider_moments/Snakefile -n --config night=20260706 limit=3
```

What remains in `guider/Snakefile` are the **study** rules that read it: `psfmoments`,
`movie`/`movies`, `rtvplot`/`rtvplots` and `plots`. `run_snake.sh` still does a whole
night end to end — it runs the product build for each night first, then the study rules:

```bash
./run_snake.sh --day-obs 20260706                    # local, detached via nohup
./run_snake.sh --day-obs 20260706 --limit 5 -n       # dry-run, 5-exposure smoke test
```

`--day-obs` expands to per-night **study output files**
(`output/night_<D>/guider_{psfmoments,plots,movies,rtvplots}_<D>.{parquet,pdf,txt}`), not
to rule names. Ask for a target by its file path. The product's own per-night tables are
**not** targets here any more; they come from the product Snakefile.

The study rules resolve the product **old tree first, then the catalog** — `output/night_<D>/`
if `guider_moments_<D>.parquet` is there, else the catalog's `v6/nights/<D>/` partition —
and print which source each night resolved to at DAG time, the same way `aos/Snakefile`
does for its three moved products.

There is no single-seq option in `run_snake.sh` — target the per-exposure product files
directly with the product Snakefile, or use the per-visit runners
(`run_guider_rubintv_plots.py`, `run_guider_movie.py`) with `--day-obs` / `--seq-num`.
Full command forms are in `docs/pipeline_runbook.md`.

**Batch submission needs Aaron's explicit OK first** — show him the command, then submit
it yourself (see the root `CLAUDE.md`). Note the difference from `aos/`: here
`--mode batch` submits **one sbatch job per night**, so `--day-obs A,B,C` is three
independent submissions, not one, and needs an OK covering all three. Submit from an
s3df node (`slacrd`), never an RSP pod.

## Schema version

`guiderMoments.py` — now
`rubinwork/products/guider_moments/guiderMoments.py`, with a shim at the old path — is at
**schema v6**, which is also the product's **variant** (`v6`), and which added per-detector `cen_var_slow`,
`cen_var_fast` (arcsec², noise-subtracted) and `cen_smooth_sec`. **These columns are
unpopulated in already-processed nights** — they need a moments rerun. If an analysis
reads them and gets nulls, that is why; do not treat it as a bug in the analysis.

The centroid timescale is a *fixed chosen* 1.6 s FWHM (= 8 stamps at 5 Hz), not a fitted
break. Centroid motion is **scale-free** — no knee, featureless PSD, consistent with
Kolmogorov; the `tau ~ 0.5 s` figure is an integral time, **not** a break frequency. Do
not describe it as a measured knee.

## Mosaic geometry

Star and full-frame views use **independent** geometry (`positions` vs `positionsFull`)
with separate tunables — changing one does not change the other. The star view masks
pixels outside the 2″ circle to NaN so pairs never visually overlap. Numbers are in
`docs/mosaic_layout.md`; do not re-derive them by eye.
