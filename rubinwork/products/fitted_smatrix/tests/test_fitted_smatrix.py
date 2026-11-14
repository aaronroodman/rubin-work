import numpy as np
import pytest

from rubinwork.products.fitted_smatrix import reader
from rubinwork.smatrix.field_fit import fam_smatrix, field_basis


def test_variants_resolve():
    for v in reader.variants():
        cfg = reader.variant_config(v)
        assert {"fit_set", "responses", "kmax", "ofc_version"} <= set(cfg)


def test_fam_smatrix_recovers_field_coefficients():
    rng = np.random.default_rng(1)
    n, kmax = 60, 6
    thx, thy = rng.uniform(-1.2, 1.2, (2, n))
    noll = np.array([4, 5, 6])
    coef = rng.normal(size=(kmax, 50, 3))                    # (field k, DOF, Zernike)
    slope = np.einsum("pk,kdz->pdz", field_basis(kmax, thx, thy), coef)
    resp = {"kind": np.array(["fam"] * 2 * n), "side": np.array(["intra"] * n + ["extra"] * n),
            "thx": np.tile(thx, 2), "thy": np.tile(thy, 2),
            "x__slope": np.concatenate([slope, slope]), "x__noll": noll}
    S = fam_smatrix(resp, "x", kmax=kmax)
    assert S.shape == (31, 29, 50)
    np.testing.assert_allclose(S[1:kmax + 1][:, noll], coef.transpose(0, 2, 1), atol=1e-10)
    assert not S[0].any() and not S[:, [0, 1, 2, 3, 7]].any()


def test_fam_smatrix_rejects_mismatched_sides():
    resp = {"kind": np.array(["fam"] * 2), "side": np.array(["intra", "extra"]),
            "thx": np.array([0.0, 0.1]), "thy": np.zeros(2),
            "x__slope": np.zeros((2, 50, 1)), "x__noll": np.array([4])}
    with pytest.raises(ValueError):
        fam_smatrix(resp, "x", kmax=1)
