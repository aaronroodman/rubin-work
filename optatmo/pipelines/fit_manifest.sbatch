#!/bin/bash
# Large-production SLURM array driven by a MANIFEST FILE of "day_obs seq_num"
# lines (one per in-focus visit), so thousands of visits need no giant command
# line.  Array task i processes line i+1 of the manifest.  Each task is a
# DISTINCT visit, so extract-if-missing is race-free (no shared output files) and
# no pre-extraction is required.
#
# Build a manifest (one visit per line):
#     20260513 22
#     20260513 25
#     20260706 40
#     ...
# then submit in CHUNKS (each array index must be < the cluster MaxArraySize;
# check: scontrol show config | grep MaxArraySize).  OFFSET selects the manifest
# block, so every array stays 0..CHUNK-1:
#     MANIFEST=pipelines/prod_famcwfs.txt; N=$(wc -l < $MANIFEST); CHUNK=500
#     for OFF in $(seq 0 $CHUNK $((N-1))); do
#       L=$((N-OFF)); [ $L -gt $CHUNK ] && L=$CHUNK
#       sbatch --array=0-$((L-1))%50 \
#         --export=ALL,OFFSET=$OFF,MANIFEST=$MANIFEST,CAMPAIGN=famcwfs_prod,\
#PLAN=focus_then_full,INIT=cwfs,MINIMIZER=lbfgsb,CWFS_SRC=danish,SKIP=1 \
#         pipelines/fit_manifest.sbatch
#     done
# (set CHUNK <= MaxArraySize.)  SKIP=1 makes it resumable: re-submitting only
# fits visits missing from the campaign.  Outputs -> output/runs/<CAMPAIGN>/<day>.
#
#SBATCH --job-name=optatmo_prod
#SBATCH --partition=milano
#SBATCH --account=rubin:developers
#SBATCH --output=pipelines/logs/prod_%A_%a.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "=== task ${SLURM_ARRAY_TASK_ID:-?} on $(hostname) $(date) ==="
cd "${SLURM_SUBMIT_DIR:-.}"

source /sdf/group/rubin/sw/tag/w_2026_27/loadLSST.bash >/dev/null
setup lsst_distrib
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
set -eo pipefail

MANIFEST=${MANIFEST:?set MANIFEST=pipelines/<file>.txt via --export}
CAMPAIGN=${CAMPAIGN:?set CAMPAIGN via --export}
PLAN=${PLAN:-focus_then_full}
INIT=${INIT:-cwfs}
MINIMIZER=${MINIMIZER:-lbfgsb}
SKIP=${SKIP:-1}                    # resumable by default for big runs
SVD=data/svd/ofc_svd_50_34_k6.npz
COLL=u/gmegias/calib/DM-55048/intrinsicZernikes.v3
FILT=i_39
PSF_COLL=${PSF_COLL:-LSSTCam/runs/nightlyValidation}
CWFS_SRC=${CWFS_SRC:-danish}
CWFS_COLL=${CWFS_COLL:-'*danish_1_2_0*'}

[ -f "$SVD" ] || { echo "ERROR: $SVD missing" >&2; exit 1; }

# manifest line = OFFSET + task id (OFFSET lets you chunk a manifest longer than
# the cluster MaxArraySize into several submissions).
OFFSET=${OFFSET:-0}
LINE_NO=$((OFFSET + SLURM_ARRAY_TASK_ID + 1))
LINE=$(sed -n "${LINE_NO}p" "$MANIFEST")
if [ -z "$LINE" ]; then
    echo "no manifest line $LINE_NO (past end of $MANIFEST); nothing to do"; exit 0
fi
DAY=$(echo "$LINE" | awk '{print $1}')
SEQ=$(echo "$LINE" | awk '{print $2}')
[ -n "$DAY" ] && [ -n "$SEQ" ] || { echo "ERROR: bad manifest line '$LINE'" >&2; exit 1; }
VISIT=$(printf '%s%05d' "$DAY" "$SEQ")
echo "task $SLURM_ARRAY_TASK_ID -> day $DAY seq $SEQ (visit $VISIT) campaign=$CAMPAIGN"

# extract this visit's inputs if missing (distinct per task -> race-free)
if [ ! -f "data/psfmoments_${VISIT}.parquet" ]; then
    python code/extract_psf_moments.py --visits "$VISIT" \
        --collection "$PSF_COLL" --out-dir data
fi
if [ ! -f "data/cwfs_${VISIT}.parquet" ]; then
    if [ "$CWFS_SRC" = consdb ]; then
        python code/consdb_cwfs.py --visits "$VISIT" --out-dir data
    else
        python code/extract_cwfs.py --visits "$VISIT" \
            --collection "$CWFS_COLL" --out-dir data
    fi
fi

SKIPFLAG=""; [ "$SKIP" = 1 ] && SKIPFLAG="--skip-existing"
python code/fit_optatmo.py --day "$DAY" --seqs "$SEQ" --campaign "$CAMPAIGN" \
    --plan "$PLAN" --init "$INIT" --minimizer "$MINIMIZER" $SKIPFLAG \
    --svd "$SVD" --coll "$COLL" --filt "$FILT"
python code/plot_data_model.py --campaign "$CAMPAIGN" --day "$DAY" --seqs "$SEQ" \
    --stage all --coll "$COLL" --filt "$FILT"

echo "done day $DAY seq $SEQ"
