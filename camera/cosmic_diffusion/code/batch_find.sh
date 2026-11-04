#!/bin/bash
# Slurm array job: find_cosmics.py on one night of darks, one interleaved slice
# per array task. Submit from the repo root, e.g.:
#   DAY_OBS=20261103 RUN_DATE=20261104 sbatch --array=0-14 --export=ALL \
#       camera/cosmic_diffusion/code/batch_find.sh
# Needs code/darks_<DAY_OBS>.txt (seq_num, one per line).
#SBATCH --partition=milano
#SBATCH --account=rubin:developers@milano
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=cosmic_find
#SBATCH --output=/sdf/group/rubin/u/roodman/LSST/rubin-work/studies/camera/cosmic_diffusion/logs/find_%A_%a.out
NTASK=${NTASK:-15}
DAY_OBS=${DAY_OBS:-20261102}
LABEL=${LABEL:-find_${DAY_OBS}}
cd /sdf/home/r/roodman/notebooks/rubin-work
python camera/cosmic_diffusion/code/find_cosmics.py --day-obs "$DAY_OBS" \
    --seq-file "camera/cosmic_diffusion/code/darks_${DAY_OBS}.txt" \
    --chunk "$SLURM_ARRAY_TASK_ID" "$NTASK" \
    --label "$LABEL" --run-date "$RUN_DATE"
