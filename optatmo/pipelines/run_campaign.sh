#!/bin/bash
# Orchestrate one optatmo fitting campaign (the "option 1" wrapper -- no Snakemake):
#   1. pre-extract each visit's inputs (serial, so the array tasks don't race),
#   2. submit the fit array (one task per seq) -> output/runs/<CAMPAIGN>/<DAY>/fits,
#   3. submit the ensemble summaries as a job DEPENDENT on the fits (afterok).
# Per-task work is in fit_array.sbatch.
#
# Run from the optatmo dir:
#   pipelines/run_campaign.sh CAMPAIGN PLAN MINIMIZER DAY "SEQS" [CWFS_SRC]
# e.g.
#   pipelines/run_campaign.sh focus_then_full focus_then_full lbfgsb 20260513 "25 28 31" danish
#   pipelines/run_campaign.sh full_migrad     full            migrad 20260513 "25 28 31"
# Only fit visits missing from the campaign (incremental):
#   SKIP=1 pipelines/run_campaign.sh ...           (SKIP flows through --export=ALL)
set -eo pipefail

CAMPAIGN=${1:?CAMPAIGN}; PLAN=${2:?PLAN}; MINIMIZER=${3:?MINIMIZER}
DAY=${4:?DAY}; SEQS=${5:?"SEQS (quoted, space-separated)"}; CWFS_SRC=${6:-danish}

COLL=u/gmegias/calib/DM-55048/intrinsicZernikes.v3
FILT=i_39
PSF_COLL=LSSTCam/runs/nightlyValidation
STACK=/sdf/group/rubin/sw/tag/w_2026_27/loadLSST.bash

echo "campaign=$CAMPAIGN plan=$PLAN minimizer=$MINIMIZER day=$DAY seqs=[$SEQS] cwfs=$CWFS_SRC"
source "$STACK" >/dev/null; setup lsst_distrib
export CARGO_HOME="$HOME/.cargo" RUSTUP_HOME="$HOME/.rustup"

[ -f data/svd/ofc_svd_50_34_k6.npz ] || python code/build_svd_local.py

# 1) pre-extract inputs once (serial)
for S in $SEQS; do
    V=$(printf '%s%05d' "$DAY" "$S")
    [ -f data/psfmoments_$V.parquet ] || \
        python code/extract_psf_moments.py --visits $V --collection $PSF_COLL --out-dir data
    if [ ! -f data/cwfs_$V.parquet ]; then
        if [ "$CWFS_SRC" = consdb ]; then
            python code/consdb_cwfs.py --visits $V --out-dir data
        else
            python code/extract_cwfs.py --visits $V --collection '*danish_1_2_0*' --out-dir data
        fi
    fi
done

# 2) fit array (one task per seq)
N=$(echo $SEQS | wc -w)
JID=$(sbatch --parsable --array=0-$((N-1))%12 \
    --export=ALL,SEQS="$SEQS",DAY=$DAY,CAMPAIGN=$CAMPAIGN,PLAN=$PLAN,INIT=cwfs,MINIMIZER=$MINIMIZER,CWFS_SRC=$CWFS_SRC \
    pipelines/fit_array.sbatch)
echo "submitted fit array job $JID ($N tasks)"

# 3) ensemble summaries, dependent on all fit tasks succeeding
EID=$(sbatch --parsable --dependency=afterok:$JID \
    -p milano -A rubin:developers -t 00:30:00 -c 2 --mem 8G \
    -o pipelines/logs/ensemble_%j.log \
    --wrap "source $STACK >/dev/null; setup lsst_distrib; export CARGO_HOME=\$HOME/.cargo RUSTUP_HOME=\$HOME/.rustup; python code/ensemble_vmodes.py --campaign $CAMPAIGN --day $DAY; python code/ensemble_corners.py --campaign $CAMPAIGN --day $DAY --coll $COLL --filt $FILT")
echo "submitted ensemble job $EID (after $JID)"
echo "outputs -> output/runs/$CAMPAIGN/$DAY/{fits,reports,ensemble}"
