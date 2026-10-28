"""Tests of `atmo_sim`: variants, and a built visit read back like real data.

The visit tests need the LSST stack, ``/repo/main`` and the 20260706 seq 688 partition;
they are skipped otherwise.
"""

import numpy as np
import pytest

from rubinwork.products import atmo_sim

DAY_OBS, SEQ_NUM = 20260706, 688


def test_variant_config():
    cfg = atmo_sim.variant_config()
    assert atmo_sim.PRODUCT_VARIANT in atmo_sim.variants()
    assert cfg["butler"]["chain"] == atmo_sim.collection()
    assert atmo_sim.run_collection(DAY_OBS).startswith(cfg["butler"]["run_prefix"])
    assert isinstance(cfg["science"]["flux_adu"], float)


@pytest.fixture(scope="module")
def built():
    pytest.importorskip("lsst.summit.utils.guiders.reading")
    try:
        return atmo_sim.partition(DAY_OBS, SEQ_NUM)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"visit not built: {exc}")


def test_guider_files_match_real_headers(built):
    """Each simulated file has the real ROI, stamp count and per-stamp times, and its
    star peaks within 2 pix of the truth CCD position after the amplifier inverse."""
    from lsst.daf.butler import Butler
    from lsst.meas.algorithms.stamps import Stamps
    from lsst.obs.lsst import LsstCam

    from rubinwork import guider_roi

    camera = LsstCam.getCamera()
    butler = Butler("/repo/main", collections=["LSSTCam/raw/guider"])
    truth = atmo_sim.load_truth(DAY_OBS, SEQ_NUM).set_index("det_name")
    files = sorted((built / "guider_raw").glob("*_guider.fits"))
    assert len(files) == 8
    for path in files:
        sim = Stamps.readFits(str(path))
        md = sim.metadata
        name = f"{md['RAFTBAY']}_{md['CCDSLOT']}"
        real = butler.get("guider_raw", instrument="LSSTCam", day_obs=DAY_OBS,
                          seq_num=SEQ_NUM, detector=camera[name].getId())
        for key in ("ROICOL", "ROIROW", "ROISEG", "ROTPA", "N_STAMPS"):
            assert md.get(key) == real.metadata.get(key), (name, key)
        for s, r in zip(sim, real):
            assert s.metadata.get("STMPTMJD") == r.metadata.get("STMPTMJD")
        bbox = guider_roi.roi_ccd_bbox(md, camera)
        ccd = guider_roi.ccd_to_amp(sim[0].stamp_im.image.array, camera[name],
                                    guider_roi.amp_name(md.toDict()))
        iy, ix = np.unravel_index(np.argmax(ccd), ccd.shape)
        assert abs(bbox.getMinX() + ix - truth.loc[name, "xccd"]) < 2.0, name
        assert abs(bbox.getMinY() + iy - truth.loc[name, "yccd"]) < 2.0, name
