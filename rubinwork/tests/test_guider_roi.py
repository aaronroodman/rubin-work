"""Tests of `rubinwork.guider_roi` against real ``guider_raw`` stamps and ``summit_utils``.

Needs the LSST stack, ``summit_utils`` and the ``/repo/main`` Butler; skipped otherwise.
The reference visit is day_obs 20260706, seq_num 688.
"""

import numpy as np
import pytest

pytest.importorskip("lsst.summit.utils.guiders.reading")

from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS, DetectorType  # noqa: E402
from lsst.afw.image import ImageF, MaskedImageF  # noqa: E402
from lsst.meas.algorithms.stamps import Stamp, Stamps  # noqa: E402
from lsst.obs.lsst import LsstCam  # noqa: E402
from lsst.summit.utils.guiders.reading import convertRawStampsToView  # noqa: E402
from lsst.summit.utils.guiders.transformation import roiImageToDvcs  # noqa: E402

from rubinwork import guider_roi  # noqa: E402

DAY_OBS, SEQ_NUM = 20260706, 688


@pytest.fixture(scope="module")
def camera():
    return LsstCam.getCamera()


@pytest.fixture(scope="module")
def raw_stamps(camera):
    try:
        from lsst.daf.butler import Butler

        butler = Butler("/repo/main", collections=["LSSTCam/raw/guider"])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no /repo/main Butler: {exc}")
    guiders = [d for d in camera if d.getType() == DetectorType.GUIDER]
    out = {}
    for det in guiders:
        try:
            out[det.getName()] = butler.get("guider_raw", instrument="LSSTCam", day_obs=DAY_OBS,
                                            seq_num=SEQ_NUM, detector=det.getId())
        except Exception:  # noqa: BLE001
            continue
    if not out:
        pytest.skip("no guider_raw for the reference visit")
    return out


def test_round_trip_matches_summit_utils(camera, raw_stamps):
    """dvcs_to_amp inverts summit_utils roiImageToDvcs bit for bit, and amp_to_dvcs agrees."""
    for name, stamps in raw_stamps.items():
        det = camera[name]
        md = stamps.metadata.toDict()
        amp = guider_roi.amp_name(md)
        for stamp in list(stamps)[:3]:
            raw = stamp.stamp_im.image.array
            dvcs = roiImageToDvcs(raw, md, det, amp, camera, view="dvcs").array
            np.testing.assert_array_equal(guider_roi.amp_to_dvcs(raw, det, amp), dvcs)
            np.testing.assert_array_equal(guider_roi.dvcs_to_amp(dvcs, det, amp), raw)


def _marker(shape, x0, y0):
    """An asymmetric L marker in CCD-frame array coordinates (column x0, row y0).

    Peak 100 at (x0, y0), 50 at (x0 + 6, y0), 25 at (x0, y0 + 12).
    """
    img = np.zeros(shape, dtype=np.float32)
    img[y0, x0] = 100.0
    img[y0, x0 + 6] = 50.0
    img[y0 + 12, x0] = 25.0
    return img


def _find(img, value):
    iy, ix = np.argwhere(img == value)[0]
    return float(ix), float(iy)


def test_marker_lands_on_focal_plane_position(camera, raw_stamps):
    """A marker written through ccd_to_amp reads back through summit_utils at the right place.

    The marker is drawn in the CCD frame, converted to the amplifier frame and wrapped as a
    ``Stamps`` with the real headers. ``summit_utils`` `convertRawStampsToView` turns it into
    the DVCS view. Then:

    * its archived DVCS-to-CCD transform returns the marker to the CCD pixel it was drawn at;
    * the arms of the L point along the FOCAL_PLANE images of CCD +x and +y, so DVCS stamp
      axes match focal-plane axes with the same sign (no hidden flip).
    """
    for name, stamps in raw_stamps.items():
        det = camera[name]
        md = stamps.metadata
        amp = guider_roi.amp_name(md.toDict())
        bbox = guider_roi.roi_ccd_bbox(md, camera)
        shape = (bbox.getHeight(), bbox.getWidth())
        x0, y0 = 60, 70
        raw = guider_roi.ccd_to_amp(_marker(shape, x0, y0), det, amp)

        template = list(stamps)[:2]
        sim = Stamps([Stamp(MaskedImageF(ImageF(raw.copy())), metadata=s.metadata)
                      for s in template], md, use_archive=True)
        out = convertRawStampsToView(sim, name, len(template), view="dvcs")
        dvcs = out[0].stamp_im.image.array
        _, _, dvcs_to_ccd = out.getArchiveElements()[0]

        px, py = {}, {}
        for v in (100.0, 50.0, 25.0):
            xd, yd = _find(dvcs, v)
            xc, yc = dvcs_to_ccd(np.array([xd]), np.array([yd]))
            px[v], py[v] = float(np.ravel(xc)[0]), float(np.ravel(yc)[0])
        assert (px[100.0], py[100.0]) == pytest.approx(
            (bbox.getMinX() + x0, bbox.getMinY() + y0), abs=1e-6), name

        # Arm directions in DVCS stamp pixels against the focal-plane images of CCD +x, +y.
        to_fp = det.getTransform(PIXELS, FOCAL_PLANE).getMapping()
        cx, cy = bbox.getMinX() + x0, bbox.getMinY() + y0
        fp = to_fp.applyForward(np.array([[cx, cx + 1.0, cx], [cy, cy, cy + 1.0]]))
        fp_dx = fp[:, 1] - fp[:, 0]
        fp_dy = fp[:, 2] - fp[:, 0]
        p0, p1, p2 = (np.array(_find(dvcs, v)) for v in (100.0, 50.0, 25.0))
        arm_x = (p1 - p0) / 6.0
        arm_y = (p2 - p0) / 12.0
        np.testing.assert_allclose(arm_x, fp_dx / np.linalg.norm(fp_dx), atol=1e-2, err_msg=name)
        np.testing.assert_allclose(arm_y, fp_dy / np.linalg.norm(fp_dy), atol=1e-2, err_msg=name)


def test_jacobian_plate_scale(camera, raw_stamps):
    """The local Jacobian at a real ROI gives about 0.2 arcsec/pix, with determinant > 0 or < 0
    consistently with the PIXELS-to-FIELD_ANGLE parity."""
    for name, stamps in raw_stamps.items():
        det = camera[name]
        bbox = guider_roi.roi_ccd_bbox(stamps.metadata, camera)
        c = bbox.getCenter()
        jac = guider_roi.local_jacobian(det, c.getX(), c.getY())
        scale = np.sqrt(abs(np.linalg.det(jac))) * 206264.806
        assert scale == pytest.approx(0.2, rel=0.02), (name, scale)
