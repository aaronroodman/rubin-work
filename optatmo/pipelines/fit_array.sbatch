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
# Before submitting, ONCE (from the optatmo dir): pull the code you want the
# whole array to run, and build the SVD basis (avoids a 20-way race):
#     git pull
#     python code/build_svd_local.py
# Do NOT git pull inside the task -- array tasks would race and could run
# different code versions.
#
# Submit (set SEQS to your list; N = number of seqs):
#     SEQS="25 28 30 31 34 ..."          # your in-focus seq_nums
#     N=$(echo $SEQS | wc -w)
#     sbatch --array=0-$((N-1)) --export=ALL,SEQS="$SEQS" pipelines/fit_array.sbatch
#
# Override the night / CWFS source / fit options via --export, e.g. 20260706
# (no danish aggregate table there -> corners must come from ConsDB):
#     sbatch --array=0-$((N-1)) \
#            --export=ALL,SEQS="$SEQS",DAY=20260706,CWFS_SRC=consdb \
#            pipelines/fit_array.sbatch
# ConsDB on S3DF batch reads the token FILE ~/.lsst/consdb_token (shared home is
# mounted on the milano nodes), so no token needs to be exported.
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

# print a line FIRST so the log is never empty even if setup fails below
echo "=== task ${SLURM_ARRAY_TASK_ID:-?} on $(hostname) $(date) ==="
cd "${SLURM_SUBMIT_DIR:-.}"
echo "cwd: $(pwd)"

# Set up the stack BEFORE 'set -e' and WITHOUT silencing stderr: the LSST env
# scripts are not nounset-clean, so a non-zero return here under set -euo would
# otherwise kill the job silently (empty log).
source /sdf/group/rubin/sw/tag/w_2026_27/loadLSST.bash >/dev/null
setup lsst_distrib
echo "python: $(which python)"

set -eo pipefail          # note: no -u (loadLSST/eups reference unset vars)

DAY=${DAY:-20260513}        # night (override via --export for 20260706 etc.)
SVD=data/ofc_svd_50_34_k6.npz
COLL=u/gmegias/calib/DM-55048/intrinsicZernikes.v3
FILT=i_39
REG=${REG:-10}
PSF_COLL=${PSF_COLL:-LSSTCam/runs/nightlyValidation}
CWFS_SRC=${CWFS_SRC:-consdb}   # consdb (any night, writes visitmeta) | danish (20260513 only)
CWFS_COLL=${CWFS_COLL:-'*danish_1_2_0*'}   # only used when CWFS_SRC=danish
INIT=${INIT:-zero}          # v-mode start: zero | cwfs (pass via --export)
OPTICS=${OPTICS:-free}      # free | fixed (freeze v-modes, fit only atmosphere)
MOFFSETS=${MOFFSETS:-off}   # off | higher | all | csv (constant per-moment offsets)

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
    if [ "$CWFS_SRC" = consdb ]; then
        # ConsDB corners (works on any night; also writes visitmeta_<visit> that
        # run_vmode_fit.rot_for needs off-20260513).  url='auto' + token file.
        python code/consdb_cwfs.py --visits "$VISIT" --out-dir data
    else
        # danish aggregateAOSVisitTableRaw (20260513 only)
        python code/extract_cwfs.py --visits "$VISIT" \
            --collection "$CWFS_COLL" --out-dir data
    fi
fi

# fit just this one image (prints cost/eval; checkpoints data/fitprog_<seq>*.npz)
python code/run_vmode_fit.py 1 "$SVD" reg=$REG seqs="$SEQ" day=$DAY \
    coll="$COLL" filt="$FILT" init="$INIT" optics="$OPTICS" moffsets="$MOFFSETS"

# multi-page PDF report (moments, corners, fit progress) for this image
python code/plot_data_model.py "$SVD" seqs="$SEQ" day=$DAY \
    coll="$COLL" filt="$FILT" init="$INIT" optics="$OPTICS" moffsets="$MOFFSETS"

echo "done seq $SEQ"
