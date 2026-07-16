#!/usr/bin/env bash
# Launch the guider moment-decomposition Snakemake pipeline.
#
# Runs snakemake on THIS node, detached via nohup so a dropped SSH/network
# connection won't kill it.  Tuned for the RSP terminal allocation (~4 cores):
# -j4 + a mem_mb budget.  Extra args pass through to snakemake.
#
# Usage:
#   ./run_snake.sh                  # standard run
#   ./run_snake.sh -n               # dry-run (show the plan)
#   ./run_snake.sh --until combine  # partial build
#   ./run_snake.sh -j2              # override job slots
#
# Remember to `git pull` first to pick up code changes.  Requires the LSST
# stack + snakemake on PATH (an RSP terminal with the stack set up).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ts=$(date +%Y%m%d_%H%M%S)
log="logs/run_${ts}.log"
args=(-j 4 --resources mem_mb=14000 --keep-going "$@")

nohup snakemake "${args[@]}" > "$log" 2>&1 &
pid=$!
echo "snakemake launched: pid $pid"
echo "  args:   ${args[*]}"
echo "  log:    $log"
echo "  follow: tail -f $log"
echo "  check:  pgrep -af snakemake"
