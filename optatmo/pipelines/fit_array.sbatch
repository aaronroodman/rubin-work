#!/bin/bash
# SLURM array: fully process one in-focus image (seq_num) per array task with
# fit_optatmo -- extract PSF-star moments + corner-WFS Zernikes (if needed), run
# the staged v-mode fit, and make the per-stage report PDFs.  Outputs are
# organized per campaign: output/runs/<CAMPAIGN>/<DAY>/{fits,reports,ensemble}.
# Tasks are independent (distinct output files), so no lock / no races.
#
# Before submitting, ONCE (from the optatmo dir): pull the code, build the SVD,
# and PRE-EXTRACT the visits so tasks don't race the same visit's parquets:
#     git pull
#     [ -f data/svd/ofc_svd_50_34_k6.npz ] || python code/build_svd_local.py
#     SEQS="25 28 ..."; DAY=20260513
#     for S in $SEQS; do V=$(printf '%s%05d' $DAY $S); \
#         python code/extract_psf_moments.py --visits $V --collection LSSTCam/runs/nightlyValidation --out-dir data; \
#         python code/extract_cwfs.py --visits $V --collection '*danish_1_2_0*' --out-dir data; done
#
# Submit (N = number of seqs).  Pick a fit plan + minimizer via --export:
#     SEQS="25 28 ..."; N=$(echo $SEQS | wc -w)
#   stage-1-then-full, L-BFGS-B:
#     sbatch --array=0-$((N-1))%12 --export=ALL,SEQS="$SEQS",DAY=20260513,\
#CAMPAIGN=focus_then_full,PLAN=focus_then_full,INIT=cwfs,MINIMIZER=lbfgsb,CWFS_SRC=danish \
#            pipelines/fit_array.sbatch
#   full fit with iminuit MIGRAD:
#     ...,CAMPAIGN=full_migrad,PLAN=full,INIT=cwfs,MINIMIZER=migrad,...
#   then the ensemble summary for the campaign:
#     python code/ensemble_vmodes.py  --campaign <CAMPAIGN> --day 20260513
#     python code/ensemble_corners.py --campaign <CAMPAIGN> --day 20260513 --coll <COLL> --filt i_39
#
# ConsDB on S3DF batch reads ~/.lsst/consdb_token (shared home is mounted); no
# token export needed.  Adjust --partition/--account to your S3DF allocation.
#
#SBATCH --job-name=optatmo_fit
#SBATCH --partition=milano
#SBATCH --account=rubin:developers
#SBATCH --output=pipelines/logs/fit_%A_%a.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "=== task ${SLURM_ARRAY_TASK_ID:-?} on $(hostname) $(date) ==="
cd "${SLURM_SUBMIT_DIR:-.}"
echo "cwd: $(pwd)"

source /sdf/group/rubin/sw/tag/w_2026_27/loadLSST.bash >/dev/null
setup lsst_distrib
echo "python: $(which python)"
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"

set -eo pipefail

DAY=${DAY:-20260513}
CAMPAIGN=${CAMPAIGN:?set CAMPAIGN via --export=ALL,CAMPAIGN=<label>}
PLAN=${PLAN:-full}                 # full | focus | focus_then_full
INIT=${INIT:-cwfs}                 # zero | cwfs
MINIMIZER=${MINIMIZER:-lbfgsb}     # lbfgsb | migrad
SKIP=${SKIP:-0}                    # 1 => skip visits already fit in this campaign
SVD=data/svd/ofc_svd_50_34_k6.npz
COLL=u/gmegias/calib/DM-55048/intrinsicZernikes.v3
FILT=i_39
PSF_COLL=${PSF_COLL:-LSSTCam/runs/nightlyValidation}
CWFS_SRC=${CWFS_SRC:-danish}       # danish (20260513) | consdb (any night)
CWFS_COLL=${CWFS_COLL:-'*danish_1_2_0*'}

if [ ! -f "$SVD" ]; then
    echo "ERROR: $SVD missing -- run 'python code/build_svd_local.py' once first." >&2
    exit 1
fi

read -ra ARR <<< "${SEQS:?set SEQS via --export=ALL,SEQS=\"25 28 ...\"}"
SEQ=${ARR[$SLURM_ARRAY_TASK_ID]:?array index out of range for SEQS}
VISIT=$(printf '%s%05d' "$DAY" "$SEQ")
echo "task $SLURM_ARRAY_TASK_ID -> seq $SEQ (visit $VISIT) campaign=$CAMPAIGN plan=$PLAN"

# extract this visit's inputs to data/ if not present (should be pre-extracted)
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

# staged v-mode fit -> output/runs/<CAMPAIGN>/<DAY>/fits/vmodefit_<seq>.npz
SKIPFLAG=""; [ "$SKIP" = 1 ] && SKIPFLAG="--skip-existing"
python code/fit_optatmo.py --day "$DAY" --seqs "$SEQ" --campaign "$CAMPAIGN" \
    --plan "$PLAN" --init "$INIT" --minimizer "$MINIMIZER" $SKIPFLAG \
    --svd "$SVD" --coll "$COLL" --filt "$FILT"

# per-stage report PDFs -> .../reports/
python code/plot_data_model.py --campaign "$CAMPAIGN" --day "$DAY" --seqs "$SEQ" \
    --stage all --coll "$COLL" --filt "$FILT"

echo "done seq $SEQ (campaign=$CAMPAIGN plan=$PLAN minimizer=$MINIMIZER)"
