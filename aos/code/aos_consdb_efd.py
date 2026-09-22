"""Backwards-compatible alias for :mod:`common.consdb_efd`.

The Consolidated Database (ConsDB) transformed Engineering Facility Database (EFD)
telemetry path is shared across topics and lives in ``common/consdb_efd.py``. This module
re-exports it under the original names so that existing callers importing
``aos_consdb_efd`` by bare module name after a ``sys.path.insert`` of ``aos/code`` keep
working. New code should import ``common.consdb_efd`` directly.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

from common.consdb_efd import (  # noqa: E402,F401
    HEX_AXES,
    HEX_LUT_COLS,
    HEX_TRIM_COLS,
    STRESS_COLS,
    TEMP_COLS,
    UNPIVOT_PROPS,
    WIND_COLS,
    collect_consdb_telemetry,
    fetch_arrays_unpivoted,
    fetch_scalars_pivoted,
)
