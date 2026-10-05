"""Cross-check the `ts_ofc` Range-Bounded Recovery against the prototype.

Range-Bounded Recovery (RBR) was developed as
`smatrix/code/regularized_inversion.py` and reimplemented in `ts_ofc` as
`lsst.ts.ofc.range_bounded_recovery`, against that package's own internals
rather than the prototype's `OFCSvd`. The two must agree.

This test lives here rather than in `ts_ofc` because it imports both sides, and
`ts_ofc` may not depend on this repository or on `ts_intrinsic_wavefront`.

Two independent routes to the allowed range `r_j` are also compared: the
hexapod strokes and mirror force ranges that `OFCData.dof_ranges` uses, and the
prototype's back-derivation `r_j = w_j**2 * f_j` from the shipped normalization
weights. These agree only under `w_j = r_j**0.5 * f_j**-0.5`, so the comparison
doubles as a check on that convention -- which the shipped weights file names
`range0.5_fwhm-0.15.yaml`, a typo for `fwhm-0.5`.

Run it directly:

    python code/miw/test_rbr_against_prototype.py
"""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "smatrix" / "code"))

import regularized_inversion as ri  # noqa: E402

from lsst.ts.intrinsic.wavefront import ofc_svd as osv  # noqa: E402
from lsst.ts.ofc import (  # noqa: E402
    DoubleZernikeStateEstimator,
    OFCData,
    invert_range_penalty,
    invert_truncated,
)

# The measured-intrinsic wavefront build's configuration.
I_ZS = [j for j in range(4, 27) if j not in (20, 21)]
K_MIN = 1
K_MAX = 6
N_KEEP = 34
N_DOF = 50

KAPPA = 4.0
POWER = 3


class TestRbrAgainstPrototype(unittest.TestCase):
    """Compare the ts_ofc RBR implementation with the smatrix prototype."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.weights = osv.load_normalization_weights(None, osv.DEFAULT_NORM_YAML)
        cls.svd = osv.build_ofc_svd(I_ZS, K_MIN, K_MAX, N_KEEP, n_dof=N_DOF)
        cls.estimator = DoubleZernikeStateEstimator(
            OFCData("lsst"),
            I_ZS,
            K_MIN,
            K_MAX,
            N_KEEP,
            n_dof=N_DOF,
            normalization_weights=cls.weights,
        )
        cls.v_retained = cls.estimator.V[:, cls.estimator.keep_idx]
        cls.sigma = cls.estimator.Sigma[cls.estimator.keep_idx]

        cls.ranges_ofc = cls.estimator.dof_ranges()
        cls.ranges_prototype = ri.dof_range_vector(cls.svd)

    def wavefronts(self, n: int = 5, seed: int = 29) -> np.ndarray:
        """Random DZ wavefronts, µm of wavefront.

        Scaled so the unregularized solution lands well outside the allowed
        range on the stiff bending modes, which is the regime RBR exists for.
        """
        rng = np.random.default_rng(seed)
        return rng.normal(scale=0.05, size=(n, len(self.estimator.kj_grid)))

    def test_allowed_range_agrees_between_routes(self) -> None:
        """`dof_ranges` matches the prototype's back-derivation from weights.

        The routes are independent: one reads the configured strokes and force
        ranges, the other inverts `w_j = r_j**0.5 * f_j**-0.5`.
        """
        relative = np.max(
            np.abs(self.ranges_prototype - self.ranges_ofc) / self.ranges_ofc
        )
        self.assertLess(float(relative), 1e-12)

    def test_truncated_recovery_agrees(self) -> None:
        """The unregularized solutions agree, which the penalty builds on."""
        for wavefront in self.wavefronts():
            mine = invert_truncated(
                wavefront,
                self.estimator.U_eff,
                self.sigma,
                self.v_retained,
                self.estimator.normalization_weights,
            )
            theirs = ri.invert_truncated(wavefront, self.svd)
            np.testing.assert_allclose(mine, theirs, rtol=1e-12, atol=1e-14)

    def test_range_penalty_agrees(self) -> None:
        """RBR agrees at the default knee and exponent.

        Compared relative to the largest recovered amplitude rather than per
        element: the state spans many orders of magnitude across DOF, so a
        per-element relative tolerance would be dominated by DOF recovered near
        zero.
        """
        for wavefront in self.wavefronts():
            mine = invert_range_penalty(
                wavefront,
                self.estimator.U_eff,
                self.sigma,
                self.v_retained,
                self.estimator.normalization_weights,
                self.ranges_ofc,
                kappa=KAPPA,
                power=POWER,
            )
            theirs = ri.invert_range_penalty(
                wavefront, self.svd, self.ranges_prototype, kappa=KAPPA, power=POWER
            )
            scale = np.max(np.abs(theirs))
            self.assertLess(float(np.max(np.abs(mine - theirs)) / scale), 1e-10)

    def test_range_penalty_agrees_across_settings(self) -> None:
        """Agreement holds away from the defaults, including the easy power 1.

        At ``power=1`` the penalty weight is constant and the iteration is exact
        in one step, so this separates a disagreement in the penalty itself from
        one in the iteration.

        The tolerance is looser than `test_range_penalty_agrees` because the two
        solvers stop independently: each tests its own iterate against the
        convergence tolerance, so they can halt an iteration or two apart and
        differ by the steps not taken. Measured at ``kappa=8, power=2``, they
        stop at 16 and 19 iterations and agree on the largest ``|d_j| / r_j`` to
        six decimal places while the states differ by 6e-07 relative. That is
        the iteration's stopping rule, not the penalty.
        """
        wavefront = self.wavefronts(n=1)[0]
        for kappa, power in ((1.0, 1), (2.0, 2), (4.0, 3), (8.0, 2)):
            mine = invert_range_penalty(
                wavefront,
                self.estimator.U_eff,
                self.sigma,
                self.v_retained,
                self.estimator.normalization_weights,
                self.ranges_ofc,
                kappa=kappa,
                power=power,
            )
            theirs = ri.invert_range_penalty(
                wavefront, self.svd, self.ranges_prototype, kappa=kappa, power=power
            )
            scale = np.max(np.abs(theirs))
            setting = f"kappa={kappa}, power={power}"
            self.assertLess(
                float(np.max(np.abs(mine - theirs)) / scale), 1e-5, msg=setting
            )

            # The quantity the penalty exists to control, which is insensitive
            # to where each solver stopped.
            self.assertAlmostEqual(
                float(np.max(np.abs(mine) / self.ranges_ofc)),
                float(np.max(np.abs(theirs) / self.ranges_prototype)),
                places=5,
                msg=setting,
            )

    def test_decomposition_agrees(self) -> None:
        """The two SVDs are the same decomposition, bit for bit.

        `build_ofc_svd` now delegates to the ts_ofc estimator, so this guards
        the adapter rather than the arithmetic.
        """
        np.testing.assert_array_equal(self.svd.U_eff, self.estimator.U_eff)
        np.testing.assert_array_equal(self.svd.Sigma, self.estimator.Sigma)
        np.testing.assert_array_equal(self.svd.V, self.estimator.V)
        self.assertEqual(self.svd.kj_grid, self.estimator.kj_grid)
        self.assertEqual(self.svd._keep(), self.estimator.keep_idx)


if __name__ == "__main__":
    unittest.main()
