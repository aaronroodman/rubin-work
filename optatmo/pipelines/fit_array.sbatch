#!/bin/bash
# SLURM array job: fit one in-focus image (seq_num) per array task.
# Each task extracts that visit's PSF moments (if needed) and runs the v-mode
# fit -> data/vmodefit_<seq>.npz + output/fitmon_<seq>.png.  Tasks are fully
# independent (distinct output files), so no Snakemake lock / no races.
#
# Build the SVD basis ONCE before submitting (avoids a 20-way race):
#     python code/build_svd_local.py
#
# Submit (set SEQS to your list; N = number of seqs):
#     SEQS="25 28 30 31 34 ..."          # your in-focus seq_nums
#     N=$(echo $SEQS | wc -w)
#     sbatch --array=0-$((N-1)) --export=ALL,SEQS="$SEQS" pipelines/fit_array.sbatch
#
# Adjust --partition/--account below to your S3DF allocation if these defaults
# are wrong (check: sacctmgr show assoc user=$USER format=account,partition).
#
#SBATCH --job-name=optatmo_fit
#SBATCH --partition=milano
#SBATCH --account=rubin:developers
#SBATCH --output=pipelines/logs/fit_%A_%a.log
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

cd /sdf/home/r/roodman/notebooks/rubin-work/optatmo

# non-interactive shell: set up the stack explicitly (.bashrc is not sourced here)
source /sdf/group/rubin/sw/tag/w_2026_27/loadLSST.bash >/dev/null 2>&1
setup lsst_distrib

DAY=20260513
SVD=data/ofc_svd_50_34_k6.npz
COLL=u/gmegias/calib/DM-55048/intrinsicZernikes.v3
FILT=i_39
REG=10
PSF_COLL=LSSTCam/runs/nightlyValidation

if [ ! -f "$SVD" ]; then
    echo "ERROR: $SVD missing -- run 'python code/build_svd_local.py' once first." >&2
    exit 1
fi

read -ra ARR <<< "${SEQS:?set SEQS via --export=ALL,SEQS=\"25 28 ...\"}"
SEQ=${ARR[$SLURM_ARRAY_TASK_ID]:?array index out of range for SEQS}
VISIT=$(printf '%s%05d' "$DAY" "$SEQ")
echo "task $SLURM_ARRAY_TASK_ID -> seq $SEQ (visit $VISIT)"

# extract this visit's PSF-star moments if not already present
if [ ! -f "data/psfmoments_${VISIT}.parquet" ]; then
    python code/extract_psf_moments.py --visits "$VISIT" \
        --collection "$PSF_COLL" --out-dir data
fi

# fit just this one image
python code/run_vmode_fit.py 1 "$SVD" reg=$REG seqs="$SEQ" day=$DAY \
    coll="$COLL" filt="$FILT"

echo "done seq $SEQ"
