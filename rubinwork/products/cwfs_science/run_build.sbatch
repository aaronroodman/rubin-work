#!/bin/bash
#SBATCH --job-name=cwfs_science
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=aos/logs/cwfs_science_%j.log
#
# Build one cwfs_science variant: one Snakemake job per night, 16 nights at a time.
#
# Submit from an s3df login node, from the repository root:
#   cd ~/notebooks/rubin-work && sbatch rubinwork/products/cwfs_science/run_build.sbatch
# Variant and build: CWFS_VARIANT (default d13-blitz_rso1071), CWFS_BUILD (default today).
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
V=${CWFS_VARIANT:-d13-blitz_rso1071}
B=${CWFS_BUILD:-$(date -u +%Y%m%d)}
echo "== $(date)  cwfs_science variant=$V build=$B"
python -m snakemake -s rubinwork/products/cwfs_science/Snakefile --config variant="$V" build="$B" \
    -c"$N" --resources mem_mb=60000 --keep-going -q
echo "== $(date)  done"
