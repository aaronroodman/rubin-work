"""Tests that EFD reads anchor exposures in UTC against the raw (UTC) EFD index.

ConsDB ``obs_start`` / ``obs_start_mjd`` are TAI; the raw EFD index is UTC. The fake EFD
client below returns the raw index when ``convert_influx_index`` is False, and imitates
the legacy flag by shifting it 37 s early when True. Each test places telemetry so that
the correct anchor and the old wrong one give different answers
(``aos/docs/telemetry.md`` section 7).

Nothing here touches a live EFD.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

from rubinwork.common import dof_telemetry, visit_telemetry

LEAP = pd.Timedelta(seconds=37)          # TAI - UTC
T_UTC = pd.Timestamp('2026-04-22T03:00:00', tz='UTC')
T_TAI_MJD = Time(T_UTC.to_pydatetime(), scale='utc').tai.mjd


class FakeEfd:
    """`select_time_series` over a fixed frame with a raw UTC index."""

    def __init__(self, frame, *args, **kwargs):
        self.frame = frame
        self.calls = []

    async def select_time_series(self, topic, fields, start, end,
                                 convert_influx_index=False, index=None, **kwargs):
        self.calls.append({'topic': topic, 'convert_influx_index': convert_influx_index,
                           'index': index})
        df = self.frame.copy()
        if convert_influx_index:
            df.index = df.index - LEAP
        t0 = pd.Timestamp(Time(start).utc.isot, tz='UTC')
        t1 = pd.Timestamp(Time(end).utc.isot, tz='UTC')
        return df[(df.index >= t0) & (df.index <= t1)]


def series(times, **cols):
    return pd.DataFrame(cols, index=pd.DatetimeIndex(times))


def test_asof_rows_by_night_ignores_events_after_obs_start():
    # Event 10 s after obs_start must not be picked; the old 37 s shift picked it.
    frame = series([T_UTC - pd.Timedelta(seconds=60), T_UTC + pd.Timedelta(seconds=10)],
                   z=[1.0, 2.0])
    efd = FakeEfd(frame)
    rows = dof_telemetry._asof_rows_by_night(
        efd, 'lsst.sal.MTHexapod.logevent_compensationOffset', ['z'],
        [Time(T_UTC.to_pydatetime(), scale='utc')], [20260421], index=1)
    assert rows[0]['z'] == 1.0
    assert efd.calls[0]['convert_influx_index'] is False


@pytest.fixture
def camera_efd(monkeypatch):
    # 1 deg C inside +-120 s of obs_start, 100 deg C outside it; 10 s cadence.
    times = T_UTC + pd.to_timedelta(np.arange(-300, 301, 10), unit='s')
    inside = np.abs((times - T_UTC).total_seconds()) <= 120
    frame = series(times, **{f: np.where(inside, 1.0, 50.0)
                             for f in visit_telemetry.CAM_FIELDS})
    fake = types.ModuleType('lsst_efd_client')
    fake.EfdClient = lambda *a, **k: FakeEfd(frame)
    monkeypatch.setitem(sys.modules, 'lsst_efd_client', fake)


def test_fetch_camera_centres_window_on_utc_obs_start(camera_efd):
    keys = pd.DataFrame({'day_obs': [20260421], 'seq_num': [1], 'mjd': [T_TAI_MJD]})
    out = visit_telemetry.fetch_camera(keys, verbose=False)
    assert out['cam_AverageTemp'].iloc[0] == pytest.approx(1.0)
    assert keys['mjd'].iloc[0] == T_TAI_MJD        # caller's frame untouched


def test_fetch_turbulence_centres_window_on_utc_obs_start(monkeypatch):
    builders = pytest.importorskip(
        'rubinwork.products.value_added.builders.build_efd_db')
    reader = builders.efd_db
    # 1 m/s inside +-15 s of obs_start, 100 m/s outside; 5 s cadence, every salIndex.
    times = T_UTC + pd.to_timedelta(np.arange(-150, 151, 5), unit='s')
    inside = np.abs((times - T_UTC).total_seconds()) <= 15
    parts = []
    for idx in reader.TURB_SALINDEX:
        cols = {f: np.where(inside, 1.0, 100.0) for f in reader.TURB_FIELDS}
        parts.append(series(times, salIndex=idx, **cols))
    frame = pd.concat(parts).sort_index()
    monkeypatch.setattr(builders, 'make_efd_client', lambda *a, **k: FakeEfd(frame))
    spine = pd.DataFrame({'visit_id': [2026042100001], 'day_obs': [20260421],
                          'seq_num': [1], 'obs_start': [Time(T_UTC.to_pydatetime(),
                                                              scale='utc').tai.isot],
                          'mjd': [T_TAI_MJD]})
    out = builders.fetch_turbulence(spine, {})
    idx = reader.TURB_SALINDEX[0]
    suffix = reader.TURB_FIELDS['speed0'][0]
    assert out[f'turb{idx}_{suffix}'].iloc[0] == pytest.approx(1.0)
    assert spine['mjd'].iloc[0] == T_TAI_MJD       # spine shared with later groups
