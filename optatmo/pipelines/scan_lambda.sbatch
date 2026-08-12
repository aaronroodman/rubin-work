#!/bin/bash
# SLURM array: one task per wavefront-regularization lambda.  Each task fits the
# same seq(s) at its lambda, in wavefront mode started at the CWFS state
# (regmode=wavefront init=cwfs reg=<lambda>), writing
# data/vmodefit_<seq>_cwfs_wreg_lam<lambda>.npz.  After the array finishes,
# aggregate + plot with scan_wavefront_lambda.py --no-fit (see bottom).
#
# PRE-EXTRACT ONCE before submitting, so the tasks don't race extracting the
# same visit's psfmoments/cwfs simultaneously (run a single lambda first, or the
# extract scripts):
#     SEQS="25"; DAY=20260513
#     for S in $SEQS; do V=$(printf '%s%05d' $DAY $S); \
#         python code/extract_psf_moments.py --visits $V --collection LSSTCam/runs/nightlyValidation --out-dir data; \
#         python code/extract_cwfs.py --visits $V --collection '*danish_1_2_0*' --out-dir data; done
#
# Submit (N = number of lambdas):
#     git pull
#     [ -f data/ofc_svd_50_34_k6.npz ] || python code/build_svd_local.py
#     SEQS="25"                         # seq(s) to fit at every lambda
#     LAMS="0.1 0.3 1 3 10"             # the lambda grid
#     N=$(echo $LAMS | wc -w)
#     sbatch --array=0-$((N-1)) --export=ALL,SEQS="$SEQS",LAMS="$LAMS",DAY=20260513,CWFS_SRC=danish \
#            pipelines/scan_lambda.sbatch
#
#SBATCH --job-name=optatmo_lamscan
#SBATCH --partition=milano
#SBATCH --account=rubin:developers
#SBATCH --output=pipelines/logs/lamscan_%A_%a.log
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
SVD=data/ofc_svd_50_34_k6.npz
COLL=u/gmegias/calib/DM-55048/intrinsicZernikes.v3
FILT=i_39
PSF_COLL=${PSF_COLL:-LSSTCam/runs/nightlyValidation}
CWFS_SRC=${CWFS_SRC:-danish}                 # danish (20260513) | consdb (any night)
CWFS_COLL=${CWFS_COLL:-'*danish_1_2_0*'}

if [ ! -f "$SVD" ]; then
    echo "ERROR: $SVD missing -- run 'python code/build_svd_local.py' once first." >&2
    exit 1
fi

read -ra LARR <<< "${LAMS:?set LAMS via --export=ALL,LAMS=\"0.1 0.3 1 3 10\"}"
LAM=${LARR[$SLURM_ARRAY_TASK_ID]:?array index out of range for LAMS}
# compute the tag with the SAME rule as scan_wavefront_lambda.lam_tag (%g form)
# so the aggregation step finds these npz files
LAMTAG=$(python -c "import sys;print(('%g'%float(sys.argv[1])).replace('.','p').replace('-','m'))" "$LAM")
# penalty form (matches scan_wavefront_lambda.form_suffix); quadratic -> no suffix
REGFORM=${REGFORM:-quadratic}
REGPOW=${REGPOW:-2}
if [ "$REGFORM" = quadratic ]; then
    OUTTAG="_lam${LAMTAG}"
else
    PW=$(python -c "print(('%g'%float('$REGPOW')).replace('.','p'))")
    OUTTAG="_${REGFORM}p${PW}_lam${LAMTAG}"
fi
: "${SEQS:?set SEQS via --export=ALL,SEQS=\"25\"}"
echo "task $SLURM_ARRAY_TASK_ID -> lambda $LAM (tag _lam$LAMTAG), seqs $SEQS"

# ensure inputs exist (should be pre-extracted; extract here only as a fallback)
for SEQ in $SEQS; do
    VISIT=$(printf '%s%05d' "$DAY" "$SEQ")
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
done

SEQ_CSV=$(echo $SEQS | tr ' ' ',')
python code/run_vmode_fit.py 1 "$SVD" reg=$LAM regmode=wavefront \
    regform=$REGFORM regpow=$REGPOW init=cwfs \
    seqs="$SEQ_CSV" day=$DAY coll="$COLL" filt="$FILT" outtag=$OUTTAG

# per-lambda multi-page report incl. the corner-comparison page
python code/plot_data_model.py "$SVD" seqs="$SEQ_CSV" day=$DAY \
    coll="$COLL" filt="$FILT" init=cwfs regmode=wavefront outtag=$OUTTAG

echo "done lambda $LAM (form $REGFORM^$REGPOW)"

# After the whole array finishes, aggregate + plot (no re-fit):
#     python code/scan_wavefront_lambda.py --no-fit --seqs $SEQS --lams $LAMS \
#         --day $DAY --svd $SVD --coll $COLL --filt $FILT
