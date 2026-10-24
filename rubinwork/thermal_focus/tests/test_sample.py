"""`rubinwork.thermal_focus.sample.load_science` on a synthetic reader: each funnel cut."""
import numpy as np
import pandas as pd
import pytest

from rubinwork.thermal_focus import lib as L
from rubinwork.thermal_focus import sample as S

V1_PER_UM_DZ = 0.01  # dimensionless v-mode 1 per µm of hexapod dz

# One visit per cut, then two that survive. Columns: day_obs, img_type, band, truss [°C].
VISITS = [
    (20260101, 'science', 'r', 10.0),                         # kept
    (20260101, 'science', 'i', 12.0),                         # kept
    (20260101, 'acq', 'r', 10.0),                             # not science
    (20260101, 'science', 'N', 10.0),                         # not a fitted band
    (L.LUT_EPOCH_OFFSET_NIGHTS[0], 'science', 'r', 10.0),     # LUT-epoch night
    (20260101, 'science', 'r', L.TRUSS_TEMP_MAX_C + 5.0),     # too warm
]


class FakeReader:
    """Stands in for `rubinwork.products.value_added`."""

    def __init__(self):
        n = len(VISITS)
        self.ids = np.arange(1, n + 1)
        self.state = pd.DataFrame({
            'visit_id': self.ids, 'day_obs': [v[0] for v in VISITS],
            'seq_num': self.ids, 'resid_rms_um': 0.1, 'v1': 0.2, 'v1_lut': 0.0,
            'v1_trim': 0.5, 'v2_olr': 7.0, 'unused': 1.0})
        tel = {c: 0.1 for c in S.TELEMETRY_COLS}
        self.tel = pd.DataFrame({'visit_id': self.ids, 'day_obs': 0, **tel})
        self.consdb = pd.DataFrame({
            'visit_id': self.ids, 'img_type': [v[1] for v in VISITS],
            'band': [v[2] for v in VISITS], 'truss_temp_mean_c': [v[3] for v in VISITS]})

    def optical_state(self, variant, day_obs_range=None, wide=True, ok_only=True):
        return self.state

    def visits(self, day_obs_range=None, columns=None):
        return self.tel

    def m1m3_thermal_r2(self, day_obs_range=None, columns=None):
        raise FileNotFoundError('no quadratic table yet')

    def join_consdb(self, df, groups=()):
        assert tuple(groups) == S.CONSDB_GROUPS
        return df.merge(self.consdb, on='visit_id', how='left')


def test_load_science_funnel():
    df = S.load_science(FakeReader(), 'v', None, V1_PER_UM_DZ, verbose=False,
                        keep_extra=['v2_olr', 'absent'])
    assert list(df['visit_id']) == [1, 2]
    # Telemetry merges on visit_id alone, so day_obs is not duplicated.
    assert 'day_obs_x' not in df.columns
    assert (df['v2_olr'] == 7.0).all() and 'unused' not in df.columns
    expected = L.attach_response(df.drop(columns='y'), V1_PER_UM_DZ)['y']
    np.testing.assert_allclose(df['y'], expected)
    # A missing quadratic table leaves its columns empty rather than failing.
    assert df['m1m3_r2_coeff_c'].isna().all()


def test_load_science_needs_features():
    reader = FakeReader()
    reader.consdb = reader.consdb.drop(columns='truss_temp_mean_c')
    with pytest.raises((SystemExit, KeyError)):
        S.load_science(reader, 'v', None, V1_PER_UM_DZ, verbose=False)
