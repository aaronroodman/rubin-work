#!/bin/bash
# SLURM array job: fully process one in-focus image (seq_num) per array task --
# extract PSF-star moments + corner-WFS Zernikes (if needed), run the v-mode fit
# (-> data/vmodefit_<seq>.npz + output/fitmon_<seq>.png), and make the
# data-vs-model + corner-comparison plots (-> output/dm_*_<seq>.png).  Tasks are
# fully independent (distinct output files), so no Snakemake lock / no races.
# The fit prints a chi2 line per function evaluation and checkpoints its trace to
# data/fitprog_<seq>.npz every iteration, so a time-limit kill still leaves a
# reviewable partial result.
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
CWFS_COLL='*danish_1_2_0*'

if [ ! -f "$SVD" ]; then
    echo "ERROR: $SVD missing -- run 'python code/build_svd_local.py' once first." >&2
    exit 1
fi

read -ra ARR <<< "${SEQS:?set SEQS via --export=ALL,SEQS=\"25 28 ...\"}"
SEQ=${ARR[$SLURM_ARRAY_TASK_ID]:?array index out of range for SEQS}
VISIT=$(printf '%s%05d' "$DAY" "$SEQ")
echo "task $SLURM_ARRAY_TASK_ID -> seq $SEQ (visit $VISIT)"

# extract this visit's PSF-star moments + corner-WFS Zernikes if not present
if [ ! -f "data/psfmoments_${VISIT}.parquet" ]; then
    python code/extract_psf_moments.py --visits "$VISIT" \
        --collection "$PSF_COLL" --out-dir data
fi
if [ ! -f "data/cwfs_${VISIT}.parquet" ]; then
    python code/extract_cwfs.py --visits "$VISIT" \
        --collection "$CWFS_COLL" --out-dir data
fi

# fit just this one image (prints cost/eval; checkpoints data/fitprog_<seq>.npz)
python code/run_vmode_fit.py 1 "$SVD" reg=$REG seqs="$SEQ" day=$DAY \
    coll="$COLL" filt="$FILT"

# data-vs-model + corner-comparison plots for this image
python code/plot_data_model.py "$SVD" seqs="$SEQ" day=$DAY \
    coll="$COLL" filt="$FILT"

echo "done seq $SEQ"
