import numpy as np
import galsim
import batoid

from rubinwork.common import frames
from rubinwork.donut_sim import optics


def test_dvcs_to_ccs_swaps():
    assert frames.dvcs_to_ccs_field(1.0, 2.0) == (2.0, 1.0)


def test_zk_rotation_matches_field_rotation():
    # A wavefront W_CCS(p) is W_OCS(R p) with R = R(+rot): evaluate both at matched points.
    rng = np.random.default_rng(3)
    zk = np.zeros(23)
    zk[4:] = rng.normal(size=19)
    rot = 27.0
    zk_ocs = frames.ccs_to_ocs_zk(zk, rot)
    u, v = rng.uniform(-0.6, 0.6, (2, 20))
    uo, vo = frames.ccs_to_ocs_field(u, v, rot)
    np.testing.assert_allclose(galsim.zernike.Zernike(zk_ocs)(uo, vo),
                               galsim.zernike.Zernike(zk)(u, v), atol=1e-12)


def test_rotate_telescope_sign():
    # Telescope-only aberration (M2 decentered, nominal camera axisymmetric): OCS Zernikes
    # at rotator angle rot equal the rotator-0 Zernikes at the same OCS field point.
    tel = batoid.Optic.fromYaml("LSST_r.yaml").withGloballyShiftedOptic("M2", [2e-4, 0, 0])
    wl, rot = 620e-9, 40.0
    x_ccs, y_ccs = 0.9, -0.4
    x_ocs, y_ocs = frames.ccs_to_ocs_field(x_ccs, y_ccs, rot)
    zk_rot = optics.zk_gq(optics.rotate_telescope(tel, rot), x_ccs, y_ccs, wl, 22)
    zk_ref = optics.zk_gq(tel, x_ocs, y_ocs, wl, 22)
    np.testing.assert_allclose(frames.ccs_to_ocs_zk(zk_rot, rot)[4:], zk_ref[4:],
                               atol=1e-3 * np.abs(zk_ref[4:]).max())
    # The wrong sign fails.
    assert not np.allclose(frames.ccs_to_ocs_zk(zk_rot, -rot)[4:], zk_ref[4:],
                           atol=1e-3 * np.abs(zk_ref[4:]).max())


def test_instrument_uses_obscuration_file():
    import danish
    from rubinwork.products import batoid_intrinsic
    cfg = batoid_intrinsic.variant_config("v4_speed")
    instr = optics.instrument(cfg["model"], cfg["obscuration"], cfg["configs"]["fam"])
    assert instr.maskParams == danish.load_mask_params(
        str(optics.obscuration_path(cfg["obscuration"])))
    assert "M1Baffle1" in instr.maskParams
