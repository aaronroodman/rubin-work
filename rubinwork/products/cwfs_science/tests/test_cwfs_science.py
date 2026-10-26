"""Tests for ``cwfs_science``: the quality cut, the MTAOS selection and the per-CCD and
per-raft summaries, on synthetic corner tables."""

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.table import QTable, Table, vstack

from rubinwork.products.cwfs_science import reader, write_manifest
from rubinwork.products.cwfs_science.builders import combine
from rubinwork.products.cwfs_science.builders.summarize import (
    mtaos_selection, quality_mask, summarize_visit)

NOLL = [4, 5, 6, 7]


def _blitz(n=10, seed=0, det="R00_SW1", offset=-0.0015, det_id=192):
    rng = np.random.default_rng(seed)
    dev = np.zeros((n, 27))
    dev[:, 4] = 0.5 + 0.2 * rng.standard_normal(n)
    dev[:, 5:8] = 0.2 * rng.standard_normal((n, 3))
    intr = np.zeros((n, 67))
    intr[:, 4] = 0.1
    off = np.zeros((n, 3))
    off[:, 0] = offset
    return Table({
        "det_id": np.full(n, det_id), "det_name": np.full(n, det),
        "donut_id": np.array([f"{det}-{k}" for k in range(n)]),
        "candidate": np.ones(n, bool), "group_fit_success": np.ones(n, bool),
        "group_fit_outcome": np.full(n, "ok"), "snr": np.full(n, 500.0),
        "group_fwhm": np.linspace(0.6, 0.8, n), "defocal_offsets": off,
        "thx_ocs": np.full(n, 0.02), "thy_ocs": np.full(n, -0.02),
        "thx_ccs": np.full(n, 0.02), "thy_ccs": np.full(n, -0.02),
        "zk_deviation_ocs": dev, "zk_intrinsic_ocs": intr,
        "zk_deviation_ccs": dev, "zk_intrinsic_ccs": intr})


def _zernikes(blitz, used):
    """An online ``zernikes`` table for one raft: an average row, then one per donut."""
    n = len(blitz)
    off = np.asarray(blitz["defocal_offsets"])[:, 0]
    nan2 = np.full(2, np.nan)
    rows = dict(label=["average"] + [f"donut{k + 1}" for k in range(n)],
                used=[True] + list(used))
    intra_f, extra_f, intra_id, extra_id = [nan2], [nan2], [""], [""]
    for k in range(n):
        f = np.array([1.0, 1.0])
        if off[k] < 0:
            intra_f.append(f), extra_f.append(nan2)
            intra_id.append(str(blitz["donut_id"][k])), extra_id.append("")
        else:
            intra_f.append(nan2), extra_f.append(f)
            intra_id.append(""), extra_id.append(str(blitz["donut_id"][k]))
    t = QTable(dict(rows, intra_field=np.array(intra_f) * u.deg,
                    extra_field=np.array(extra_f) * u.deg))
    for side, ids in (("intra", intra_id), ("extra", extra_id)):
        t[f"{side}_donut_id"] = np.ma.MaskedArray(ids, mask=[i == "" for i in ids])
    dev = np.asarray(blitz["zk_deviation_ccs"])
    for j in NOLL:
        vals = dev[:, j]
        avg = np.mean(vals[np.asarray(used, bool)])
        t[f"Z{j}_deviation"] = np.concatenate([[avg], vals]) * 1000 * u.nm
    raft = str(blitz["det_name"][0])[:3]
    t.meta = {"intra": {"det_name": f"{raft}_SW1"}, "extra": {"det_name": f"{raft}_SW0"}}
    return t


def test_quality_mask_drops_nan_snr_failed_and_noncandidates():
    t = _blitz()
    t["snr"][0] = np.nan
    t["group_fit_success"][1] = False
    t["candidate"][2] = False
    t["group_fit_outcome"][3] = ""
    t["zk_deviation_ccs"][4, 6] = np.nan
    t["group_fwhm"][5] = np.nan
    keep = quality_mask(t, NOLL)
    assert not keep[:6].any() and keep[6:].all()


def test_mtaos_selection_matches_on_detector_and_id():
    t = vstack([_blitz(det="R00_SW1", offset=-0.0015),
                _blitz(det="R00_SW0", offset=0.0015, seed=1, det_id=191)])
    used = np.zeros(20, bool)
    used[[0, 3, 12, 15]] = True
    z = _zernikes(t, used)
    sel, rafts = mtaos_selection(t, {191: z}, NOLL)
    assert np.array_equal(sel, used)
    assert rafts["R00"]["n_mtaos_rows"] == 20 and rafts["R00"]["n_mtaos_unmatched"] == 0


def test_summarize_visit_ccd_and_raft_rows():
    t = vstack([_blitz(det="R00_SW1", offset=-0.0015),
                _blitz(det="R00_SW0", offset=0.0015, seed=1, det_id=191)])
    t["snr"][0] = np.nan
    used = np.zeros(20, bool)
    used[[1, 3, 12, 15]] = True
    ccd, raft = summarize_visit(t, NOLL, {191: _zernikes(t, used)},
                                dict(visit=2026031500052))
    by = {r["detector"]: r for r in ccd}
    assert set(by) == {"R00_SW0", "R00_SW1"}
    assert by["R00_SW1"]["wfs_offset"] == -1 and by["R00_SW0"]["wfs_offset"] == 0
    assert by["R00_SW1"]["n_quality"] == 9 and by["R00_SW1"]["n_mtaos"] == 2
    tot = np.asarray(t["zk_deviation_ocs"])[1:10, 4] + 0.1
    assert by["R00_SW1"]["zk_OCS_median_z4"] == pytest.approx(np.median(tot))
    mt = np.asarray(t["zk_deviation_ocs"])[[1, 3], 4] + 0.1
    assert by["R00_SW1"]["zk_OCS_mtaos_z4"] == pytest.approx(np.mean(mt))
    assert by["R00_SW1"]["thx_OCS_median"] == pytest.approx(np.degrees(0.02))
    (r,) = raft
    assert r["raft"] == "R00" and r["n_mtaos"] == 4
    assert r["n_mtaos_intra"] == 2 and r["n_mtaos_extra"] == 2
    assert r["mtaos_check_um"] == pytest.approx(0.0, abs=1e-9)
    assert r["visit"] == 2026031500052


def test_summarize_visit_without_zernikes():
    ccd, raft = summarize_visit(_blitz(), NOLL)
    assert raft == [] and ccd[0]["n_mtaos"] == 0 and np.isnan(ccd[0]["zk_OCS_mtaos_z4"])


def test_combine_refuses_duplicates(tmp_path):
    df = pd.DataFrame(dict(day_obs=[1, 1], seq_num=[2, 2], detector=["a", "a"]))
    df.to_parquet(tmp_path / "n.parquet")
    with pytest.raises(ValueError):
        combine.combine([tmp_path / "n.parquet"])


def test_manifest_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("RUBINWORK_DATA", str(tmp_path))
    variant = reader.variants()[0]
    out = reader.build_dir(variant, "20260101")
    out.mkdir(parents=True)
    pd.DataFrame(dict(day_obs=[1], seq_num=[2], detector=["R00_SW0"])).to_parquet(
        out / "ccd_visits.parquet")
    pd.DataFrame(dict(day_obs=[1], seq_num=[2], raft=["R00"])).to_parquet(
        out / "raft_visits.parquet")
    write_manifest.write(variant, "20260101")
    ccd, raft = reader.load(variant, build="20260101")
    assert len(ccd) == 1 and len(raft) == 1
