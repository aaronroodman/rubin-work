---
name: guider-pipeline-commands
description: Guider Snakemake pipeline + per-visit review commands (rubin-work/guider)
metadata: 
  node_type: memory
  type: reference
  originSessionId: 086f02ad-edae-43f2-b884-836dc3cd9f2c
  modified: 2026-09-04T18:22:11.415Z
---

Guider moment pipeline in `~/notebooks/rubin-work/guider` (on RSP; laptop copy `~/Astrophysics/Claude/rubin-work/guider`). Snakemake: `Snakefile`, `snake_config.yaml`, launcher `run_snake.sh`. Requires `git pull` on both `rubin-work` and `summit_utils` (guider-fixes) first.

**Whole night(s)** (discovers all science guider exposures, builds `output/night_<dayObs>/`):
`./run_snake.sh --day-obs 20260706` (local, nohup) or `--mode batch` (one sbatch/night, from s3df e.g. slacrd). `--limit N` smoke test. Per-night targets: `guider_moments`, `guider_psfmoments`, `guider_plots`, `guider_movies`, `guider_rtvplots`.

**Single visit** (no run_snake single-seq option — target the per-seq output files directly):
`snakemake -j 4 --resources mem_mb=14000 --keep-going output/night_20260706/movies/787_full.mp4 output/night_20260706/movies/787_star.mp4 output/night_20260706/rtvplots/787_rtvplots.pdf output/night_20260706/seq/787_moments.parquet`

**Per-visit runners directly** (bypass snakemake):
- `python code/run_guider_rubintv_plots.py --day-obs D --seq-num S --outdir DIR` -> `S_rtvplots.pdf` (6 strip plots + both mosaics).
- `python code/run_guider_movie.py --day-obs D --seq-num S --outdir DIR --which both --format mp4` -> full+star movies.

**Saturation study:** `code/run_guider_saturation.py --day-obs D` -> `output/guider_saturation_<D>.parquet` (one row per expId,detector; peak_top3, peak_max, roirow, star_row/col, etc.). Ran nights 0705,0706,0708–0712. Saturation level set ~40000 ADU. Summary snippet in notebook does per-CCD histogram + top-per-CCD candidate list.

Config: `movie_stride=50` (movie every Nth visit), `residual_scale=10.0` (RANSAC inlier band), `guider_hz=5.0`, `image_types=[science]`. Moments emit schema v6 (cen_var_*) — see [[guider-centroid-timescale]]. Use MacPorts python on laptop (`/opt/local/bin/python3`); RSP has the LSST stack.
