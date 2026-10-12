#!/usr/bin/env bash
# Compatibility shim: run_build.sh moved into the value_added product.
#
# Phase 3a of the products/libraries/studies reorganization moved this script to
# rubinwork/products/value_added/builders/run_build.sh, with the builders it launches.
# This shim forwards every argument unchanged, so an existing invocation
#
#   cd ~/notebooks/rubin-work/value_added && ./code/run_build.sh --what telemetry ...
#
# keeps working and produces the same shard plan.  The moved script still runs with the
# value_added TOPIC directory as its working directory, so output/, logs/ and the shard
# directory are where they always were.
#
# New invocations call the moved script directly:
#
#   ~/notebooks/rubin-work/rubinwork/products/value_added/builders/run_build.sh \
#       --what telemetry --day-obs 20251103-20260418 --chunk 24
#
# The shim is removed in phase 5 (notes/status/organization_plan.md).
set -euo pipefail
exec "$(cd "$(dirname "$0")/../.." && pwd)/rubinwork/products/value_added/builders/run_build.sh" "$@"
