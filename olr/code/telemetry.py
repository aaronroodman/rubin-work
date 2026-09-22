"""Backwards-compatible alias for :mod:`common.ess_telemetry`.

The per-visit Environmental Sensor System (ESS) telemetry helpers are shared across
topics and live in ``common/ess_telemetry.py``. This module re-exports them under their
original names so that existing callers importing ``telemetry`` by bare module name after
a ``sys.path.insert`` of ``olr/code`` keep working. New code should import
``common.ess_telemetry`` directly.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

from common.ess_telemetry import (  # noqa: E402,F401
    DEFAULT_TEMP_WINDOW,
    ESS_TEMP_INDEX,
    GRAD_COLS,
    INSIDE_AIRTURB_INDICES,
    OUTSIDE_AIRFLOW_INDEX,
    TRUSS_ESS_INDEX,
    TRUSS_ITEMS,
    ThermocoupleAnalysis,
    fetch_dome_wind,
    fetch_dome_wind_sync,
    fetch_thermal_telemetry,
    fetch_thermal_telemetry_sync,
    get_m1m3_gradients,
    get_m1m3_gradients_sync,
)
