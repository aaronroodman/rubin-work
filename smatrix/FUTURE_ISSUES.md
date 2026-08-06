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

## 4. MIW high-field-order astigmatism/coma — see MIW_astig_coma_investigation.md

Full analysis is in `MIW_astig_coma_investigation.md`. Corrected summary:

**Observation:** MIW Z5-Z8 show large (~±0.3µm) HIGH-field-order (many-lobe,
n=3-5) structure. The rotator-based OCS/CCS split assigns astig/coma almost
entirely to **OCS** (telescope-fixed); CCS is significant only for Z4.

**Key correction to the earlier `demo_field_order.py` conclusion:** the
high-field-order-from-near-focus-optics argument is right about *where* high
field-order can arise, but the frame assignment was wrong. A camera-optic effect
is CCS **only if it is fixed to the camera body** (lens figure, striae, mount
stress → rotate with the rotator). A **gravity-induced** camera-lens deformation
is fixed to the *telescope/elevation* frame (gravity tracks the mirrors), so it
counter-rotates in the detector frame and is assigned to **OCS** — both its
axial and lateral parts. So static camera figure/striae are excluded by CCS≈0,
but camera *gravity* is not.

**Current leading lead:** camera-lens gravity (`LSSTCam_gravity`) is all-OCS and
produces astig ~0.09µm at z=60° (comparable to the observed ~0.11µm), and is
elevation/time-dependent. Gaps: modeled field-order is low (n≈1.5-2) and coma is
~5× too small. Every other physical optic is either CCS (excluded) or
force/figure-prohibitive (mirrors), and the MIW changes night-to-night (not a
static artifact).

**To do:** (1) elevation test — MIW OCS astig/coma vs mean elevation (gravity
predicts ~sin z / cos z−1 scaling); (2) cylindrical/field-azimuthal
decomposition of the OCS maps; (3) if gravity is supported, check whether the
real lens gravitational response carries more coma / higher field-order than the
low-order `fea_legacy/L?S?zer.fits.gz` Zernike tables.

## 3. Range force source: IM vs ZEMAX basis for the full mode set

**What:** The range `r` uses force-per-micron. Two force sources exist and are
both generated (`make_normalization.py`, `normalization_all.npz`):
- **IM** (Bo's IM force = ts_config v13 files) — what the **official OFC v13**
  range used. Self-consistent with the IM-basis modes (`bend_full`, 156+72).
- **ZEMAX** (`match_forces`, in `bend_zemax`) — self-consistent with the
  ZEMAX/OFC-basis modes that reproduce the OFC *sensitivity* exactly (153+69).

**Findings (resolved):**
- The M2 force differs between the two by an exact **lbf→N** factor (4.4482);
  the raw M2 FEA / `match_forces` output is in **pounds-force**. After that
  conversion, **IM and ZEMAX M2 force are identical** for all modes.
- **M1M3** IM vs ZEMAX force differ by up to ~1.75x for a handful of modes
  (B3, B6, B7, B15, B16, B20) — genuine SVD-basis differences between Bo's IM
  decomposition and the `match_forces` reconstruction. Median ratio ~1.06.

**To decide:** which basis is canonical going forward. `50dof_im` reproduces
OFC v13 exactly (baseline). For the *full* set, `full_zemax` is the natural
self-consistent choice (matches the OFC sensitivity basis); `full_im` is the
IM-basis analogue (bend_full). Pin down whether the ~1.75x M1M3 force
differences are a `match_forces` artifact or a real basis difference before
adopting one as the reference for tracking changes.

---
