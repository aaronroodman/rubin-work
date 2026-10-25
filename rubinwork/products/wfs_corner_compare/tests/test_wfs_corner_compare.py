"""Tests for the ``wfs_corner_compare`` product and `rubinwork.corner_compare`.

The computation tests run on synthetic donuts with a known azimuthal pattern; the
`load` tests point ``RUBINWORK_DATA`` at a pytest ``tmp_path``.
"""

import numpy as np
import pandas as pd
import pytest

from rubinwork import corner_compare as cc
from rubinwork.products import cwfs_tables, wfs_corner_compare as wcc
from rubinwork.products.wfs_corner_compare import write_manifest

NOLL = [4, 5, 6]


def _synthetic(n_triplets=3, seed=0):
    """FAM ring donuts with Zj = a_j cos(az) + b_j, and CWFS donuts at the four corners
    carrying the same value plus 0.01 µm, so CWFS = FAM + 0.01 exactly."""
    rng = np.random.default_rng(seed)
    r_rad = np.radians(1.6)
    fam, cw, vis = [], [], []
    for t in range(n_triplets):
        seq = 100 + 3 * t
        az = rng.uniform(0, 360, 400)
        zk = np.stack([(j / 10) * np.cos(np.radians(az)) + 0.1 * t for j in NOLL], axis=1)
        fam.append(pd.DataFrame({"day_obs": 20260101, "seq_num": seq,
                                 "zk_OCS": list(zk),
                                 "thx_OCS": r_rad * np.cos(np.radians(az)),
                                 "thy_OCS": r_rad * np.sin(np.radians(az))}))
        for raft, caz in zip(cc.CORNER_RAFTS, (225.0, 135.0, 315.0, 45.0)):
            czk = np.array([(j / 10) * np.cos(np.radians(caz)) + 0.1 * t + 0.01
                            for j in NOLL])
            cw.append(pd.DataFrame({"detector": f"{raft}_SW0", "day_obs": 20260101,
                                    "seq_num": seq + 1, "fam_seq_num": seq,
                                    "zk_OCS": [czk] * 3,
                                    "thx_OCS": [np.radians(1.7) * np.cos(np.radians(caz))] * 3,
                                    "thy_OCS": [np.radians(1.7) * np.sin(np.radians(caz))] * 3}))
        vis.append(dict(day_obs=20260101, seq_num=seq, rotator_angle=1.0,
                        alt=np.radians(60.0), az=np.radians(10.0), mjd=61000.0 + t,
                        science_program="BLOCK-T1"))
    return (pd.concat(fam, ignore_index=True), pd.concat(cw, ignore_index=True),
            pd.DataFrame(vis))


@pytest.mark.parametrize("interp", ["fourier", "wedge-median"])
def test_match_corners_recovers_the_pattern(interp):
    fam, cw, vis = _synthetic()
    table = cc.match_corners(fam, cw, vis, NOLL, settings={"interp": interp},
                             log=lambda *a: None)
    assert list(table.columns) == list(cc.TABLE_COLUMNS)
    assert len(table) == 3 * 4 * len(NOLL)
    diff = table.cwfs_median - table.fam_interp
    if interp == "fourier":
        assert np.allclose(diff, 0.01, atol=1e-9)
    else:
        # A +/-30 deg wedge median of (j/10) cos(az) is off by up to 1 - cos(30 deg)
        # of the amplitude, 0.08 µm at j = 6.
        assert np.all(np.abs(diff - 0.01) <= 0.6 * (1 - np.cos(np.radians(30))))
    assert np.allclose(table.alt_deg, 60.0) and np.allclose(table.az_deg, 10.0)
    assert (table.defocus == "").all()


def test_gp_interpolation_is_reproducible():
    fam, cw, vis = _synthetic(n_triplets=1)
    a = cc.match_corners(fam, cw, vis, NOLL, log=lambda *a: None)
    b = cc.match_corners(fam, cw, vis, NOLL, log=lambda *a: None)
    assert a.fam_interp.to_numpy().tobytes() == b.fam_interp.to_numpy().tobytes()


def test_unpaired_corners_are_labelled_by_side():
    fam, cw, vis = _synthetic(n_triplets=1)
    sw1 = cw.assign(detector=cw.detector.str.replace("SW0", "SW1"))
    cw = pd.concat([cw.assign(wfs_offset=0), sw1.assign(wfs_offset=-1)], ignore_index=True)
    table = cc.match_corners(fam, cw, vis, NOLL, settings={"interp": "fourier"},
                             log=lambda *a: None)
    side = dict(zip(table.corner, table.defocus))
    assert side["R00_SW0"] == "extra" and side["R00_SW1"] == "intra"
    assert len(side) == 8


def test_corner_stats_on_a_perfect_line():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    table = pd.DataFrame({"corner": "R00_SW0", "defocus": "", "j": 5, "fam_interp": x,
                          "cwfs_median": 2 * x + 0.3})
    st = cc.corner_stats(table).iloc[0]
    assert st.n == 50 and st.r == pytest.approx(1) and st.rho == pytest.approx(1)
    assert st.slope == pytest.approx(2) and st.off == pytest.approx(0.3)
    assert st.rms == pytest.approx(0, abs=1e-12)


def test_pivot_round_trips():
    fam, cw, vis = _synthetic()
    table = cc.match_corners(fam, cw, vis, NOLL, settings={"interp": "fourier"},
                             log=lambda *a: None)
    arr, trip, corners, js = cc.pivot(table, "cwfs_median")
    assert arr.shape == (3, 4, len(NOLL)) and js == NOLL
    row = table.iloc[7]
    got = arr[trip.index((row.day_obs, row.seq_num)), corners.index(row.corner),
              js.index(row.j)]
    assert got == row.cwfs_median


def test_every_variant_names_a_real_cwfs_variant():
    for v in wcc.variants(registered_only=False):
        assert wcc.cwfs_variant(v) in cwfs_tables.variants(registered_only=False)
        assert wcc.fam_variant(v) == cwfs_tables.fam_variant(wcc.cwfs_variant(v))
        assert isinstance(wcc.dir_name(v), str)


def test_settings_cover_the_defaults():
    for v in wcc.variants():
        assert set(cc.DEFAULTS) <= set(wcc.settings(v))


def test_unknown_variant_raises():
    with pytest.raises(KeyError, match="not a wfs_corner_compare variant"):
        wcc.variant_config("nope")


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RUBINWORK_DATA", str(tmp_path))
    (tmp_path / "products").mkdir()
    (tmp_path / "studies").mkdir()
    return tmp_path


def test_load_and_manifest(data_root):
    fam, cw, vis = _synthetic()
    table = cc.match_corners(fam, cw, vis, NOLL, settings={"interp": "fourier"},
                             log=lambda *a: None)
    d = wcc.build_dir("d12-refitWcs", "20260101")
    d.mkdir(parents=True)
    table.to_parquet(d / "wfs_corner_compare.parquet")
    cc.corner_stats(table).to_parquet(d / "corner_stats.parquet")
    write_manifest.write("d12-refitWcs", "20260101",
                         inputs={"cwfs_tables": "d12-refitWcs@x",
                                 "fam_tables": "danish_1_2@y"})
    t, s = wcc.load("d12-refitWcs", "20260101")
    assert len(t) == len(table) and len(s) == 4 * len(NOLL)
    with pytest.raises(ValueError):
        wcc.load_table("nope", "d12-refitWcs", "20260101")
