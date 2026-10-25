#!/bin/bash
#SBATCH --job-name=fam_refit
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fam_refit_%j.log
#
# New fam_tables builds with the loose visit_quality_pass cut
# (rubinwork/products/fam_tables/quality.py), one per variant, run in parallel. Each
# reuses the current build's donut table and refits; no Butler reads. Builds are not
# set current.
#
# Submit from an s3df login node, from the aos/ dir:
#   cd ~/notebooks/rubin-work/aos && sbatch run_fam_refit_quality.sbatch
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
B=${REFIT_BUILD:-20261025}
cd ..
pids=()
for vs in danish_1_2:20261014 danish_1_3:20261024 danish_1_3_test:20261013 danish_1_3_v1000:20261013; do
  v=${vs%%:*}; s=${vs##*:}
  python -m rubinwork.products.fam_tables.builders.refit_build \
      --variant $v --source $s --build $B > aos/logs/fam_refit_${v}_${B}.log 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
tail -n 3 aos/logs/fam_refit_*_${B}.log
exit $rc
