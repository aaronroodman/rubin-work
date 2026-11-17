#!/usr/bin/env bash
# value_added catch-up after the failed 2026-11-11 update, then the normal nightly update.
# Telemetry for 20261026-20261110 is already in live; m1m3_thermal_r2 and optical_state
# are not. Steps run one after another: DuckDB allows one writer.
set -uo pipefail
cd ~/notebooks/rubin-work/value_added
live=/sdf/group/rubin/u/roodman/LSST/rubin-work/products/value_added/v1/live/aos_efd.duckdb
m=rubinwork.products.value_added.builders
nights=20261026,20261027,20261028,20261029,20261030,20261031,20261101,20261102,20261103,20261104,20261105,20261106,20261107,20261108,20261109,20261110
state_nights=20261026,20261027,20261028,20261029,20261030,20261031,20261101,20261102,20261103,20261104,20261105,20261106,20261107
fail=0
echo "=== m1m3_thermal_r2 $nights"
python -u -m $m.build_m1m3_thermal_r2 --day-obs $nights --resume --db $live || fail=1
for v in v22_12__batoid__consdb_v1 v50_34__batoid__consdb_v1 v50_34__miw__consdb_v1 \
         v50_34_rbr__batoid__consdb_v1; do
    echo "=== optical_state $v $state_nights"
    python -u -m $m.build_optical_state --variant $v --day-obs $state_nights --db $live || fail=1
done
if [ $fail -ne 0 ]; then
    echo "CATCH-UP FAILED: nightly update not run"
    exit 1
fi
echo "=== update_nightly"
python -u -m $m.update_nightly
