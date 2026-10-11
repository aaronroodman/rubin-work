# Guider pipeline runbook

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** reference (runbook)

Commands for the guider moments pipeline (Snakemake: `guider/Snakefile`,
`guider/snake_config.yaml`, launcher `guider/run_snake.sh`) and the per-visit review
scripts in `guider/code/`. Run from `guider/`. The repo is `~/notebooks/rubin-work` on
the Rubin Science Platform (RSP) and S3DF, `~/Astrophysics/Claude/rubin-work` on the
laptop. The pipeline needs the LSST stack (RSP or S3DF); on the laptop use
`/opt/local/bin/python3`.

Before running, sync both `rubin-work` and the `summit_utils` clone on branch
`u/aaronroodman/guider-fixes` (Aaron uses his `gitpull` script).

## Whole nights

```bash
./run_snake.sh --day-obs 20260706                 # local, detached via nohup
./run_snake.sh --day-obs 20260706 --limit 5 -n    # dry run, first 5 exposures per night
./run_snake.sh --day-obs 20260706 --mode batch    # one sbatch job per night; needs Aaron's OK
```

`--day-obs` takes one day_obs or a comma list. It discovers all science guider
exposures and builds `output/night_<dayObs>/`. The per-night targets are files:
`output/night_<dayObs>/guider_{moments,psfmoments}_<dayObs>.parquet`,
`guider_plots_<dayObs>.pdf`, `guider_{movies,rtvplots}_<dayObs>.txt`. `--mode batch`
runs only from an S3DF node (`slacrd`).

## Single visit

`run_snake.sh` has no single-visit option. Ask `snakemake` for the per-visit files:

```bash
snakemake -j 4 --resources mem_mb=14000 --keep-going \
  output/night_20260706/movies/787_full.mp4 \
  output/night_20260706/movies/787_star.mp4 \
  output/night_20260706/rtvplots/787_rtvplots.pdf \
  output/night_20260706/seq/787_moments.parquet
```

Or run the per-visit scripts directly, bypassing Snakemake:

```bash
python code/run_guider_rubintv_plots.py --day-obs 20260706 --seq-num 787 --outdir output/review_787
python code/run_guider_movie.py --day-obs 20260706 --seq-num 787 --outdir output/review_787 --which both --format mp4
```

The first writes `787_rtvplots.pdf` (6 strip plots and both mosaics); the second writes
the full-frame and star movies.

## Saturation study

```bash
python code/run_guider_saturation.py --day-obs 20260706
```

Writes `output/guider_saturation_<dayObs>.parquet`, one row per (`expId`, detector),
with `peak_top3`, `peak_max` (ADU, after subtracting the stamp-0,1 column-median bias),
`roirow`, `roicol`, `roiseg`, star row and column. Run for day_obs 20260705, 20260706 and
20260708–20260712. The working saturation level is about 40000 ADU; a per-CCD
saturation flag is not yet implemented.

## Configuration (`snake_config.yaml`)

| key | value | meaning |
|---|---|---|
| `guider_hz` | 5.0 Hz | guider readout rate |
| `image_types` | `[science]` | exposure types processed |
| `tracker.residual_scale` | 10.0 (dimensionless) | random sample consensus (RANSAC) inlier band for the trend fits, in units of the residual `mad_std`; `summit_utils` default 5.0 |
| `movie_stride` | 50 (visits) | make a movie for every 50th visit; 0 disables |

Moments are written at schema v6 (adds `cen_var_slow`, `cen_var_fast`,
`cen_smooth_sec`; see `centroid_motion_timescale.md`).
