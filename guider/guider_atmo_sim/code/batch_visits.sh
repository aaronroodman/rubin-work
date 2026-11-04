#!/bin/bash
# Slurm array: simulate, ingest and measure one visit per task.
#   sbatch --array=0-5 guider/guider_atmo_sim/code/batch_visits.sh v2 v2 20260706 195 372 603 688 726 908
# Arguments: atmo_sim variant, atmo_sim_moments variant, day_obs, then the seq_nums.
#SBATCH --partition=roma
#SBATCH --account=rubin:developers@roma
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=atmo_sim
#SBATCH --output=guider/logs/atmo_sim_%A_%a.out
set -euo pipefail
sim_variant=$1; mom_variant=$2; night=$3; shift 3
seqs=("$@")
seq=${seqs[$SLURM_ARRAY_TASK_ID]}
cd ~/notebooks/rubin-work
echo "=== $night seq $seq atmo_sim $sim_variant $(date +%T)"
python -m rubinwork.products.atmo_sim.simulate --day-obs $night --seq-num $seq \
    --variant $sim_variant
# Registry writes and the chain redefinition are serialized across tasks.
flock guider/logs/atmo_sim_ingest.lock \
    python -m rubinwork.products.atmo_sim.write_manifest --day-obs $night --seq-num $seq \
    --variant $sim_variant
python -m rubinwork.products.atmo_sim_moments.build --day-obs $night --seq-num $seq \
    --variant $mom_variant
echo "=== done $(date +%T)"
