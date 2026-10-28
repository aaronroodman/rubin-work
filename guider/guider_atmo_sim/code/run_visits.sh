#!/bin/bash
# Simulate, ingest and measure a list of visits of one night, then compare all of them.
#   guider/guider_atmo_sim/code/run_visits.sh 20260706 195 372 603 726 908
set -u
night=$1; shift
for seq in "$@"; do
    echo "=== $night seq $seq $(date +%T)"
    python -m rubinwork.products.atmo_sim.simulate --day-obs $night --seq-num $seq 2>&1 \
        | grep -v -i "warn\|pytest" || { echo "FAILED simulate $seq"; continue; }
    python -m rubinwork.products.atmo_sim.write_manifest --day-obs $night --seq-num $seq 2>&1 \
        | grep -v -i "warn\|pytest" || { echo "FAILED ingest $seq"; continue; }
    python -m rubinwork.products.atmo_sim_moments.build --day-obs $night --seq-num $seq 2>&1 \
        | grep -v -i "warn\|pytest" | tail -2
done
echo "=== done $(date +%T)"
