#!/usr/bin/env bash
# Launch the BLOCK-T539 Snakemake pipeline locally on the RSP terminal.
#
# The RSP has no batch system, so this runs snakemake on THIS node, detached via
# nohup so a dropped SSH/network connection won't kill it.  The build_table rule
# hits ConsDB + EFD and can take many minutes (per-visit ESS queries + per-night
# M1M3-gradient / TMA-truss loads), so watch the log.
#
# IMPORTANT: launch this from a terminal INSIDE the RSP JupyterLab (Nublado) pod
# -- the same environment as the notebook (File -> New -> Terminal).  ConsDB
# ('consdb-pq.consdb') is an in-cluster service that only resolves inside that
# pod; an S3DF SSH login node (e.g. sdfianaNNN) cannot resolve it and build_table
# will fail with "Failed to resolve 'consdb-pq.consdb'".
#
# Usage:
#   ./run_snake.sh                 # build table + validation PDF
#   ./run_snake.sh -n              # dry-run (extra args pass through)
#   ./run_snake.sh --until build_table
#
# Remember to `git pull` first to pick up code changes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs output

ts=$(date +%Y%m%d_%H%M%S)
log="logs/run_${ts}.log"
args=(-j 2 --resources mem_mb=14000 --keep-going "$@")

nohup snakemake "${args[@]}" > "$log" 2>&1 &
pid=$!
echo "snakemake (local) launched: pid $pid"
echo "  args:   ${args[*]}"
echo "  log:    $log"
echo "  follow: tail -f $log"
echo "  check:  pgrep -af snakemake"
