#!/usr/bin/env bash
# Launch the BLOCK-T539 / per-night AOS Snakemake pipeline, locally or on Slurm.
#
# Two modes (select with --mode; default local):
#
#   local  — run snakemake on THIS node, detached via nohup so a dropped
#            SSH/network connection won't kill it.  Use this from the USDF
#            JupyterLab RSP (Nublado) pod (which cannot submit to Slurm).  The
#            default ConsDB URL 'consdb-pq.consdb' only resolves inside that pod.
#
#   batch  — submit ONE sbatch job to the s3df cluster that runs snakemake on an
#            allocated compute node.  Submit from an s3df interactive node (e.g.
#            via the `slacrd` ssh alias) where sbatch + the LSST stack live; the
#            job inherits the submitting shell's env (--export=ALL), so
#            snakemake/python resolve to the same stack, and $ACCESS_TOKEN is
#            carried through for the ConsDB token.
#
# REQUIREMENTS off-pod (batch / any s3df node): set config.yaml consdb_url to
#   https://usdf-rsp.slac.stanford.edu/consdb   (make_consdb_client injects the
#   RSP token from $ACCESS_TOKEN, else ~/.lsst/consdb_token).  The EFD (usdf_efd)
#   is reachable from sdfiana as-is.
#
# Usage:
#   ./run_snake.sh                                   # local, build everything
#   ./run_snake.sh -n                                # local dry-run (args pass through)
#   ./run_snake.sh --mode batch                      # batch: T539 table+pdf + all nights
#   ./run_snake.sh --mode batch output/night_table_20260629.parquet   # just one night
#   ./run_snake.sh --mode batch output/t539_closedloop_aos_20260420_20260708.parquet
#   ./run_snake.sh --mode batch -n                   # batch dry-run (validates submission)
#
# Batch tunables (env vars; defaults in parens):
#   SB_PARTITION (roma)  SB_CPUS (8)  SB_MEM (32G)  SB_TIME (08:00:00)
#   SB_RESMEM (28000)    # snakemake --resources mem_mb budget on the node
#   SB_ACCOUNT (rubin:developers@roma)  SB_QOS (normal)   # non-preemptable
#
# Remember to `git pull` first to pick up code changes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs output

# ---- parse a leading --mode; everything else passes through to snakemake ----
mode=local
passthru=()
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)   mode="$2"; shift 2;;
        --mode=*) mode="${1#*=}"; shift;;
        *)        passthru+=("$1"); shift;;
    esac
done

ts=$(date +%Y%m%d_%H%M%S)

case "$mode" in
    local)
        log="logs/run_${ts}.log"
        args=(-j 2 --resources mem_mb=14000 --keep-going "${passthru[@]}")
        nohup snakemake "${args[@]}" > "$log" 2>&1 &
        pid=$!
        echo "snakemake (local) launched: pid $pid"
        echo "  args:   ${args[*]}"
        echo "  log:    $log"
        echo "  follow: tail -f $log"
        echo "  check:  pgrep -af snakemake"
        ;;
    batch)
        command -v sbatch >/dev/null || {
            echo "error: sbatch not found — submit batch mode from an s3df" \
                 "interactive node (e.g. slacrd), not the RSP pod." >&2
            exit 2; }
        part=${SB_PARTITION:-roma}
        cpus=${SB_CPUS:-8}
        mem=${SB_MEM:-32G}
        tlim=${SB_TIME:-08:00:00}
        resmem=${SB_RESMEM:-28000}
        acct=${SB_ACCOUNT:-rubin:developers@roma}
        qos=${SB_QOS:-normal}
        jlog="logs/batch_${ts}_%j.out"   # %j = Slurm job id -> unique per job
        sb=(sbatch --partition="$part" --account="$acct" --qos="$qos"
            --cpus-per-task="$cpus" --mem="$mem" --time="$tlim"
            --export=ALL --job-name=blocks_snake --output="$jlog")
        smk="snakemake -j ${cpus} --resources mem_mb=${resmem} --keep-going ${passthru[*]}"
        "${sb[@]}" --wrap "cd '$PWD' && ${smk}"
        echo "submitted batch job -> '$part' acct=$acct qos=$qos (${cpus} cpus, ${mem}, ${tlim})"
        echo "  snakemake: ${smk}"
        echo "  job log:   $jlog"
        echo "  watch:     squeue --me   |   tail -f $jlog"
        ;;
    *)
        echo "error: unknown --mode '$mode' (use 'local' or 'batch')" >&2
        exit 2;;
esac
