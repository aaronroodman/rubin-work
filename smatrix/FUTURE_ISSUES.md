# smatrix — future issues / to revisit

Running list of known issues and follow-ups for the sensitivity-matrix,
mode-set, and normalization work. Newest context at the bottom of each item.

---

## 1. FWHM-normalization Zernike set: Z4–Z22 vs the nominal Z4–Z26 (excl Z20,Z21)

**What:** The `f` (FWHM) part of the geometric normalization was validated
against the official `ts_config_mttcs` v13 weights and matched to <0.1% **only
when using pupil Noll Z4–Z22** (`jmax=22`) in `convertZernikesToPsfWidth`
(see `validate_normalization.py`, `normalization_weights.compute_f_quadrature`).

**Why it matters:** In practice our nominal Zernike set is **Z4–Z26 excluding
Z20, Z21** (21 terms; `ZK_NOLL` in `rubin-work/aos/code/aos_state.py`), which is
what `ts_wep` actually produces and what the v-mode / StateEstimator work uses.
So the v13 normalization weights were generated with a **narrower** Zernike set
(Z4–Z22) than our operational one. This is a small oversight in the existing
v13 weights: the extra terms Z23–Z26 change `f` for exactly the **high-order
mirror modes** (they are the modes that drive Z23, Z24, Z25, Z26). We saw this
directly — with Z4–Z28 the high modes (M1B19, M2B16/17) were the normalization
outliers; restricting to Z4–Z22 removed them.

**To do:** Decide the intended Zernike set for the normalization and, if it
should be Z4–Z26(excl 20,21), regenerate the v13-style weights on that set and
re-validate. Our sensitivity matrix already carries pupil Zernikes up to **Z28**,
so `f` on any subset ≤ Z28 is computable directly from it.

---

## 2. Sensitivity matrix pupil-Zernike cutoff (jmax=28) is too low for high-order mirror modes

**What:** Our DZ sensitivity matrix is computed with pupil `jmax=28` (Noll
Z1–Z28), matching the OFC convention. Many **high-order mirror bending modes**
produce wavefront structure dominated by pupil Zernikes **beyond Z28**.

**Why it matters:** For those modes the current matrix does **not** capture the
full wavefront impact — the `RMS(DZ, Noll≤28)` "controllability" and any FWHM /
normalization derived from it will **under-count** the high modes' true optical
effect. This is fine for the AOS-controlled low modes (well within Z28) but is a
real limitation for the full 153 M1M3 / 69 M2 mode analysis.

**To do:** Recompute the sensitivity matrix (and/or a dedicated high-mode
variant) with a larger pupil `jmax` (e.g. 55 or higher) to capture the high
modes' wavefront, then reassess their controllability and FWHM contribution.
Cost scales with the number of Zernike terms; a targeted high-jmax run over just
the bending-mode DOF may be sufficient.

---
