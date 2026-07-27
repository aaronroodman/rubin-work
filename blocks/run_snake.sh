#!/usr/bin/env bash
# Launch the BLOCK-T539 Snakemake pipeline locally on the RSP terminal.
#
# The RSP has no batch system, so this runs snakemake on THIS node, detached via
# nohup so a dropped SSH/network connection won't kill it.  The build_table rule
# hits ConsDB + EFD and can take many minutes (per-visit ESS queries + per-night
# M1M3-gradient / TMA-truss loads), so watch the log.
#
# WHERE TO RUN:
#  * In-pod (default): launch from a terminal INSIDE the RSP JupyterLab (Nublado)
#    pod (File -> New -> Terminal).  The default ConsDB URL 'consdb-pq.consdb' is
#    an in-cluster service that ONLY resolves inside that pod.
#  * S3DF batch (sdfiana / slacrd, Slurm): set config.yaml consdb_url to
#    https://usdf-rsp.slac.stanford.edu/consdb -- make_consdb_client then injects
#    the RSP token ($ACCESS_TOKEN, else ~/.lsst/consdb_token).  CAVEAT: this
#    pipeline ALSO queries the EFD (usdf_efd) for every DOF/LUT/thermal/wind
#    value (unlike the AOS-chunk build, which is ConsDB-only), so confirm the
#    EFD client resolves from your batch node before submitting a long job.
#
# Usage:
#   ./run_snake.sh                 # build T539 table + validation PDF + nights
#   ./run_snake.sh -n              # dry-run (extra args pass through)
#   ./run_snake.sh --until build_table
#   ./run_snake.sh output/night_table_20260629.parquet   # one night only
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
