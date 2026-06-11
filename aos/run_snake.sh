#!/usr/bin/env bash
# Launch the AOS Snakemake pipeline detached so a dropped SSH/network
# connection (SIGHUP) won't kill it, logging to logs/run_<timestamp>.log.
#
# Tuned for the RSP terminal allocation (~4 usable cores, 16 GiB): -j 4 with
# single-threaded jobs + mem_mb throttling.  Run from the RSP node where the
# data mounts resolve (your notebook / devl node).
#
#   ./run_snake.sh                     # standard run
#   ./run_snake.sh -n                  # dry-run (extra args are passed through)
#   ./run_snake.sh --until combine_donuts
#
# Remember to `git pull` (or sync.sh) first to pick up code changes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

log="logs/run_$(date +%Y%m%d_%H%M%S).log"
args=(-j 4 --resources mem_mb=14000 --keep-going "$@")

nohup snakemake "${args[@]}" > "$log" 2>&1 &
pid=$!
echo "snakemake launched: pid $pid"
echo "  args:   ${args[*]}"
echo "  log:    $log"
echo "  follow: tail -f $log"
echo "  check:  pgrep -af snakemake"
