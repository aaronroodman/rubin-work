#!/usr/bin/env bash
# value_added catch-up, second try: m1m3_thermal_r2 for 20261026-20261110 (failed on EFD
# auth in job 45550188), then the nightly update (20261111 on, redo 3, snapshot).
# optical_state for 20261026-20261107 was built by job 45550188.
set -uo pipefail
export EFDAUTH=$HOME/.lsst/efdauth.json
cd ~/notebooks/rubin-work/value_added
live=/sdf/group/rubin/u/roodman/LSST/rubin-work/products/value_added/v1/live/aos_efd.duckdb
m=rubinwork.products.value_added.builders
echo "=== m1m3_thermal_r2 20261026-20261110"
python -u -m $m.build_m1m3_thermal_r2 --day-obs 20261026-20261110 --resume --db $live \
    || { echo "THERMAL FAILED: nightly update not run"; exit 1; }
echo "=== update_nightly"
python -u -m $m.update_nightly
