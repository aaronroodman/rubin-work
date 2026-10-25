#!/bin/bash
#SBATCH --job-name=backup_aos_output
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/backup_aos_output_%j.log
#
# tar + zstd backup of the old aos/output tree (phase 7), then a file-count check.
# Reads only; removes nothing.  Submit from aos/:  sbatch backup_aos_output.sbatch
set -eo pipefail
B=/sdf/data/rubin/user/roodman/LSST/backup/pre_reorg_outputs_20261024
T=/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work
ZSTD=/cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_42/conda/envs/lsst-scipipe-14.0.0-exact/bin/zstd
mkdir -p "$B"
echo "== $(date) tar"
tar -C "$T" -cf - aos/output | "$ZSTD" -T8 -q -o "$B/aos_output.tar.zst"
echo "== $(date) check"
n_tree=$(find "$T/aos/output" -type f | wc -l)
n_tar=$("$ZSTD" -dc "$B/aos_output.tar.zst" | tar -tvf - | grep -c '^-')
ls -la "$B/aos_output.tar.zst"
echo "files: tree $n_tree, archive $n_tar (count)"
[ "$n_tree" = "$n_tar" ] && echo "MATCH" || { echo "MISMATCH"; exit 1; }
