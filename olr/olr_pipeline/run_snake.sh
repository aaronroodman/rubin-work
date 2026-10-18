#!/usr/bin/env bash
# Launch the OLR Snakemake pipeline locally on the Summit RSP terminal.
#
# Summit RSP has NO Slurm/batch system, so this is local-only: run snakemake on
# THIS node, detached via nohup so a dropped SSH/network connection won't kill
# it.  Tuned for the RSP terminal allocation (~4 cores, 16 GiB): -j4 + mem_mb.
#
# Usage:
#   ./run_snake.sh                      # standard run (all nights -> combined)
#   ./run_snake.sh -n                   # dry-run (extra args pass through)
#   ./run_snake.sh --until olr          # stop after per-night OLR
#   ./run_snake.sh --until nightly_table
#
# Remember to `git pull` (or sync.sh) first to pick up code changes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ts=$(date +%Y%m%d_%H%M%S)
log="logs/run_${ts}.log"
args=(-j 4 --resources mem_mb=14000 --keep-going "$@")

nohup snakemake "${args[@]}" > "$log" 2>&1 &
pid=$!
echo "snakemake (local) launched: pid $pid"
echo "  args:   ${args[*]}"
echo "  log:    $log"
echo "  follow: tail -f $log"
echo "  check:  pgrep -af snakemake"
