#!/usr/bin/env bash
# Detached optatmo pipeline run (nohup).  Run from the optatmo/ root:
#   ./pipelines/run_snake.sh [extra snakemake args...]
# Logs to pipelines/logs/snake_run.out; per-rule logs in pipelines/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."          # -> optatmo/ root
mkdir -p pipelines/logs
JOBS="${JOBS:-4}"
nohup snakemake -s pipelines/Snakefile -j "${JOBS}" --rerun-incomplete "$@" \
    > pipelines/logs/snake_run.out 2>&1 &
echo "snakemake started (pid $!, -j ${JOBS}); tail -f pipelines/logs/snake_run.out"
