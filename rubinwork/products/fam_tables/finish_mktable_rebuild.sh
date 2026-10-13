#!/bin/bash
# Finish the phase 3c `danish_1_2` rebuild after the per-night mktable jobs land.
#
# `submit_mktable_nights.sh` runs only the expensive Butler stage, as one Slurm job per
# night.  Everything after it -- the Double-Zernike fit, the three combines, the
# telemetry attachment and the manifest -- runs here, interactively, because
# `run_attach_telemetry` still needs the Engineering Facility Database (EFD) for its
# `gradients`, `camera`, `hexlut` and `trim` groups and the EFD does not resolve on a
# batch node.
#
# It is one `snakemake` call on `rule all`: the per-night donuts and visits tables are
# already on disk, so the only pending jobs are the downstream ones.  The manifest is the
# last rule, so an interrupted run leaves the build invisible to the catalog.
#
# `catalog set-current` is NOT run here.  Moving `current@` is the step that makes a
# build the default every study reads, and phase 3c does it only after the rebuilt
# product has been compared with its old-tree build and every difference explained.
#
# Usage:
#   ./finish_mktable_rebuild.sh --dry-run          # plan the jobs, run none
#   ./finish_mktable_rebuild.sh                    # run the sequence
#   FAM_BUILD=20261013 ./finish_mktable_rebuild.sh # a build other than today's
#
# Run from an interactive s3df node (slacrd / sdfiana), where the EFD resolves.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VARIANT=danish_1_2
# NOT named BUILD: the conda environment exports BUILD=x86_64-conda-linux-gnu, which a
# ${BUILD:-default} would silently pick up as the build name.
FAM_BUILD="${FAM_BUILD:-$(date -u +%Y%m%d)}"
CORES="${CORES:-8}"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# The build directory comes from the product reader, not a path built here.
FAM_BUILD_DIR="$(python -c "from rubinwork.products import fam_tables; \
print(fam_tables.build_dir('$VARIANT', '$FAM_BUILD'))")"

echo "repo      $REPO"
echo "variant   $VARIANT   build $FAM_BUILD"
echo "build dir $FAM_BUILD_DIR"
echo

# The per-night jobs write into _work/chunks/<night>_<night>/.  Without any, every
# downstream rule would plan an mktable of its own and silently re-run the expensive
# stage interactively, so stop instead.
n_nights=$(find "$FAM_BUILD_DIR/_work/chunks" -mindepth 1 -maxdepth 1 -type d \
           2>/dev/null | wc -l)
n_donuts=$(find "$FAM_BUILD_DIR/_work/chunks" -mindepth 2 -maxdepth 2 \
           -name donuts.parquet 2>/dev/null | wc -l)
echo "per-night chunk directories (dimensionless count): $n_nights"
echo "with donuts.parquet (dimensionless count):         $n_donuts"
if [[ "$n_donuts" -eq 0 ]]; then
    echo "ERROR: no per-night donuts.parquet under $FAM_BUILD_DIR/_work/chunks --" >&2
    echo "       run submit_mktable_nights.sh first, or set FAM_BUILD" >&2
    exit 1
fi
if [[ "$n_donuts" -ne "$n_nights" ]]; then
    echo "ERROR: $((n_nights - n_donuts)) night(s) have no donuts.parquet -- a job" >&2
    echo "       failed or is still running.  Resubmit those nights before finishing," >&2
    echo "       or the combines will run on an incomplete set." >&2
    exit 1
fi
echo

# `nights=` must list exactly the nights on disk, so the chunk set Snakemake plans over
# matches what the batch jobs produced -- without it the Snakefile plans the ten parent
# chunks and treats every per-night table as unrelated.
nights=$(find "$FAM_BUILD_DIR/_work/chunks" -mindepth 1 -maxdepth 1 -type d \
         -printf '%f\n' | sed 's/_.*//' | sort -u | paste -sd,)

# --rerun-triggers mtime: modification time ONLY, not Snakemake's default set, which
# includes the params and resources provenance triggers.  Raising rule mktable's mem_mb to
# the measured 11 GB/worker (2026-10-13) counts as a params change, so the default triggers
# would re-run mktable on every per-night chunk and redo the whole expensive Butler stage
# that the batch jobs just produced.  The donuts on disk are current: mktable's own code
# path did not change, only the memory declaration.  Aaron's call, 2026-10-13.
cmd=(snakemake -s "$REPO/rubinwork/products/fam_tables/Snakefile"
     --cores "$CORES" --resources mem_mb=$((4000 * CORES))
     --rerun-triggers mtime
     --config "variant=$VARIANT" "build=$FAM_BUILD" "nights=$nights")
[[ $DRY_RUN -eq 1 ]] && cmd+=(-n -p)

printf 'running: '; printf '%q ' "${cmd[@]}"; echo; echo
cd "$REPO"
"${cmd[@]}"

echo
if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY RUN: nothing was run."
else
    echo "done.  Tables and manifest: $FAM_BUILD_DIR"
    echo
    echo "NOT done, and deliberately so -- compare against the old-tree build first,"
    echo "explain every difference, then make it the default:"
    echo "  python -m rubinwork.products.fam_tables.compare_builds \\"
    echo "    <old-tree build> $FAM_BUILD_DIR"
    echo "  python -m rubinwork.products.catalog set-current fam_tables \\"
    echo "    $VARIANT $FAM_BUILD"
fi
