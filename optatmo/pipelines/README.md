# optatmo fit pipeline (Snakemake)

Reproduces the standalone Optics+Atmosphere PSF-moment fit end-to-end on the
USDF, from Butler data to the data-vs-model + corner-comparison plots.

Chain (per in-focus visit `seq`):

```
export_miw ──┐
extract_psf ─┼─> fit ─> plots
extract_cwfs ┘
```

- **export_miw** (once) — dump the official ip_isr `intrinsicZernikes` calib
  (per-detector CCS with CCD-height in Z4) to `data/intrinsic_official_{ocs,ccs}.parquet`
  + `data/detector_names.parquet`.
- **extract_psf** (per visit) — clean PSF stars → recomputed HSM moments →
  `data/psfmoments_<visit>.parquet`.
- **extract_cwfs** (per visit) — corner-WFS aggregate Zernikes →
  `data/cwfs_<visit>.parquet`.
- **fit** (per seq) — v-mode fit against the official MIW baseline →
  `data/vmodefit_<seq>.npz`.
- **plots** (per seq) — FWHM / ellipticity / coma / trefoil data-vs-model and
  the corner CWFS-vs-PSF comparison → `output/dm_*_<seq>.png`.

## Setup

Everything needed (Butler, `ip_isr`, `obs_lsst`, `jax`) is in the `lsst_distrib`
weekly — no AOS env:

```bash
cd rubin-work/optatmo        # run from the optatmo ROOT (data/, config.yaml, ../aos/ resolve here)
source /sdf/group/rubin/sw/tag/w_2026_27/loadLSST.bash && setup lsst_distrib
```

## Run

```bash
snakemake -s pipelines/Snakefile -n         # dry-run: show the plan
snakemake -s pipelines/Snakefile -j4        # build (4 parallel)
snakemake -s pipelines/Snakefile --dag | dot -Tpng > pipelines/dag.png
./pipelines/run_snake.sh                    # detached run (nohup)
```

Per-rule logs land in `pipelines/logs/`.

## Configure

Edit `pipelines/config.yaml`: `day`, `seqs`, Butler `repo`/collections,
`miw_collection` + `filter` (the certified MIW calib), `svd_npz` (AOS
sensitivity SVD / v-mode basis), and `reg` (Tikhonov λ).

To fit a different night, change `day` + `seqs` and the collections; the visit
ids are `int(f"{day}{seq:05d}")`.
