"""Backwards-compatible alias for :mod:`common.dof_telemetry`.

The per-visit degree-of-freedom (DOF) telemetry helpers -- look-up table (LUT), Trim and
Tweak -- are shared across topics and live in ``common/dof_telemetry.py``. This module
re-exports them under their original names so that existing callers importing
``aos_trim`` by bare module name after a ``sys.path.insert`` of ``aos/code`` keep working.
New code should import ``common.dof_telemetry`` directly.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

from common.dof_telemetry import (  # noqa: E402,F401
    _run_coro,                      # used as aos_trim._run_coro by a tracked notebook
    DEFAULT_CONSDB_URL,
    DEFAULT_EXPOSURE_TABLE,
    DOF_TOPIC,
    EXTERNAL_CONSDB_URL,
    HEX_LUT_TOPIC,
    IN_POD_CONSDB_URL,
    LUT_PROPS,
    M1M3_ELEV_TOPIC,
    M2_AXIAL_TOPIC,
    N_DOF,
    PAD_SEC,
    bending_modes_from_forces,
    derive_tweak,
    efd_window,
    fetch_aggregated_dof,
    fetch_aggregated_dof_for_visits,
    fetch_hexapod_lut_for_visits,
    fetch_lut_forces,
    fetch_mirror_lut_for_visits,
    fetch_obs_start,
    in_rsp,
    make_consdb_client,
    make_efd_client,
)
