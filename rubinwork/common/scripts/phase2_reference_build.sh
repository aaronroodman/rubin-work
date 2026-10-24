#!/bin/bash
# Phase 2 reference build: one danish_1_2 chunk, end to end, into a scratch tree.
#
# Regression baseline for the fam_tables code move (notes/status/organization_plan.md
# phase 2).  Run once with the unmoved code and once with the moved code; the two
# outputs must agree.  The commands below are copied from the aos/Snakefile rules
# mktable, fit, combine_donuts, combine_fits, combine_visits and attach_telemetry,
# with the chunk's per-chunk overrides from snake_config.yaml spelled out, so the
# reference does not depend on the Snakefile DAG.
#
# Must run on an interactive node: attach_telemetry needs the EFD and ConsDB, which
# do not resolve on a Slurm compute node.
#
# Usage:  ./phase2_reference_build.sh <output-root>
set -euo pipefail

OUT_ROOT="${1:?usage: phase2_reference_build.sh <output-root>}"

PS=fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x
DMIN=20251116
DMAX=20251130
PHRASE=aos_fam_danish_1_2_0_wep17_7_0_bin2x_2025
COLLECTION=LSSTCam/runs/aos/fam/danish_1_2_0/wep_17_7_0/dv_4_7_0/bin_x2/paired/refitWcs
COORD=OCS
WORKERS=8

WF_BIN="${TS_INTRINSIC_WAVEFRONT_DIR:?run from a shell with ts_intrinsic_wavefront set up}/bin"
CDIR="$OUT_ROOT/danish_1_2/chunks/${DMIN}_${DMAX}"
PSDIR="$OUT_ROOT/danish_1_2"
mkdir -p "$CDIR"

echo "=== reference build into $OUT_ROOT"
echo "=== git commit: $(git -C "$(dirname "$0")/../.." rev-parse HEAD)"
echo "=== started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== [1/6] mktable"
python "$WF_BIN/run_mktable.py" --param-set "$PS" \
    --day-obs-min "$DMIN" --day-obs-max "$DMAX" \
    --workers "$WORKERS" --overwrite \
    --collections "$COLLECTION" --collection-phrase "$PHRASE" \
    --programs BLOCK CWFS AOSSEQUENCE \
    --output-dir "$CDIR"
mv "$CDIR/${PHRASE}_${DMIN}_${DMAX}.parquet"        "$CDIR/donuts.parquet"
mv "$CDIR/${PHRASE}_${DMIN}_${DMAX}_visits.parquet" "$CDIR/visits.parquet"

echo "=== [2/6] fit"
python "$WF_BIN/run_dz_fit.py" "$CDIR/donuts.parquet" --coord-sys "$COORD" \
    --visits "$CDIR/visits.parquet" --output "$CDIR/fits.parquet"

echo "=== [3/6] combine_donuts"
python "$WF_BIN/combine_parquets.py" "$CDIR/donuts.parquet" --output "$PSDIR/donuts.parquet"

echo "=== [4/6] combine_fits"
python "$WF_BIN/combine_parquets.py" "$CDIR/fits.parquet" --output "$PSDIR/fits.parquet"

echo "=== [5/6] combine_visits"
python "$WF_BIN/combine_parquets.py" "$CDIR/visits.parquet" --output "$PSDIR/visits.parquet"

echo "=== [6/6] attach_telemetry"
# --all-chunks walks <out-dir>/chunks/*, so only the one chunk above is fetched.
python code/fam_processing/run_attach_telemetry.py \
    --param-set "$PS" --out-dir "$PSDIR" --all-chunks --merge
date -u +'%Y-%m-%dT%H:%M:%SZ attached' > "$PSDIR/telemetry_attached.txt"

echo "=== finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
