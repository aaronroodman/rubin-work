#!/bin/bash
# Slurm array job: find_cosmics.py on the 20261102 darks, one interleaved slice
# per array task. Submit from the repo root:
#   sbatch --array=0-14 camera/cosmic_diffusion/code/batch_find.sh
#SBATCH --partition=milano
#SBATCH --account=rubin:developers@milano
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=cosmic_find
#SBATCH --output=/sdf/group/rubin/u/roodman/LSST/rubin-work/studies/camera/cosmic_diffusion/logs/find_%A_%a.out
NTASK=${NTASK:-15}
cd /sdf/home/r/roodman/notebooks/rubin-work
python camera/cosmic_diffusion/code/find_cosmics.py --day-obs 20261102 \
    --seq-file camera/cosmic_diffusion/code/darks_20261102.txt \
    --chunk "$SLURM_ARRAY_TASK_ID" "$NTASK" \
    --label find_20261102_all --run-date 20261103
