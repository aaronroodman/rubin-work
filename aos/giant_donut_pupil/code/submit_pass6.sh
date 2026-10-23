#!/bin/bash
# Submit one pupil-pass-6 driver as a Slurm job on milano, from the repo root.
# Usage: submit_pass6.sh <name> <cpus> <time> <script> [args...]
# Prints the job id and the log path. Extra sbatch flags (e.g. --dependency=afterok:<id>) go in SB_EXTRA.
# The job inherits this shell's environment, so the LSST stack must be set up here.
set -euo pipefail
name=$1; cpus=$2; tlim=$3; shift 3
repo=$(cd "$(dirname "$0")/../../.." && pwd)
log="$repo/aos/giant_donut_pupil/logs/batch_${name}_$(date +%Y%m%d_%H%M%S).out"
jid=$(sbatch --parsable --partition=milano --account=rubin:developers@milano --qos=normal \
    --cpus-per-task="$cpus" --mem-per-cpu=3800M --time="$tlim" --job-name="gdp_${name}" \
    --output="$log" ${SB_EXTRA:-} \
    --wrap "cd '$repo' && OMP_NUM_THREADS=1 python $*")
echo "$jid $log"
