# MIW investigation — portable handoff

Self-contained state of the Measured Intrinsic Wavefront (MIW) investigation as of
**2026-09-04**, written to be handed to a fresh assistant with no prior context.
Everything needed to continue is here or in the files referenced. Companion
document: [`MIW_COADD_EQUATIONS.md`](MIW_COADD_EQUATIONS.md) (the derivations).

---

## 0. Working agreement on numbers — read this first

**Every numerical value must carry the quantity name and its units**, or an explicit
"dimensionless" with the ratio's numerator and denominator named. This is a standing
requirement from Aaron, not a style preference — he had to guess microns from a bare
`1.7e-3`, and could not decode `inflation 7.9 vs 8.5` at all.

| kind | required form |
|---|---|
| physical | `Z5 coefficient sigma = 1.7e-3 µm` |
| ratio of like things | `sigma_emp/sigma_RLM = 8.5 (dimensionless; empirical over formal coefficient error)` |
| correlation | statistic + both variables with units + n |
| chi² | always chi²/dof, with dof stated |
| R² | which model, which target, which CV scheme |
| counts | "N of M" |
| fraction | of *what*, and **power vs amplitude** stated explicitly |

Also: say power-vs-amplitude explicitly (they differ by a square); put units in code
output and axis labels too, not just prose.

---

## 1. What is being investigated

The MIW is the wavefront remaining after the FAM (Full Array Mode) measured
wavefront has the fitted optical state removed — nominally the wavefront of an
ideally-aligned telescope. Two open problems:

1. **The MIW has ~0.11 µm RMS of astigmatism and coma (Z5–Z8) whose origin is
   unknown**, at high focal-plane order, which no plausible static optics explains.
2. **Per-block FAM coadds agree with the calibration MIW to a spatial Pearson r that
   varies a lot** (median r = 0.739 dimensionless at 3×3 rebinning, but with a broad
   spread). The cause of that variation is unknown and matters for AOS performance.

Aaron's leading hypothesis, which the recent work has been testing: **a small bias in
the Danish donut-fit redistributes power among the primary/secondary/tertiary radial
orders of each aberration; the 50/34 software correction, computed from that biased
wavefront, then subtracts a pattern corresponding to no real optical state —
exacerbating rather than removing the error — and block-to-block variation in the
bias drives the coadd scatter.**

---

## 2. Environment and conventions

- Python: **`/opt/local/bin/python3`** (MacPorts). Never bare `python3`.
- Repo: `~/Astrophysics/Claude/rubin-work` (github.com:aaronroodman/rubin-work).
  Topic dirs each have `code/` and `output/`; `output/` is gitignored and synced from
  the RSP.
- Packages: `~/Astrophysics/Claude/packages/` — `ts_ofc`, `ts_intrinsic_wavefront`,
  `ts_config_mttcs`, `ts_wep`.
- Local runs need `TS_CONFIG_MTTCS_DIR=~/Astrophysics/Claude/packages/ts_config_mttcs`.
- `ts_intrinsic_wavefront`'s package `__init__` fails locally (missing scons-generated
  `version.py`); load `ofc_svd.py` directly with `importlib` and register it in
  `sys.modules` before `exec_module` (dataclass resolves by module name).
- macOS has no `timeout` command.
- Current param set: **`fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x`**.
- MIW product to use: **`pathA_50_34_i_5rot/intrinsic_split_maps.parquet`**, columns
  `thx_deg, thy_deg, Z{j}_OCS, Z{j}_CCS`. **Use the OCS columns.** 3985 field cells
  to 1.8 deg. The MIW OCS frame is the batoid frame, identity mapping.
- `fits.parquet` in the same directory has 1126 visits × 653 columns, including the
  per-visit DZ fit `z1toz6_z{j}_c{k}` and `_err` (126 each) plus `_scale` (21).

---

## 3. Notation (matches MIW_COADD_EQUATIONS.md)

| symbol | meaning | shape / units |
|---|---|---|
| j | pupil (annular) Zernike Noll index | 21 values: 4..26 excluding 20, 21 |
| k | focal-plane (circular) Zernike Noll index | **fit uses k=1..6 only** |
| w | Double-Zernike coefficient vector | 126 = 21 × 6, µm |
| S_hat = S·N | OFC sensitivity, DOF-normalized | 126 × 50 |
| U, Sigma, V | SVD of S_hat | U is 126 × 50 |
| U_eff | kept left-singular vectors | 126 × 34 (n_keep = 34) |
| P = U_eff U_effᵀ | projector onto the corrected subspace | rank 34 of 126 |
| a = U_effᵀw | **u-mode amplitudes** | 34, µm of wavefront |
| c = a/sigma | v-mode amplitudes | 34, dimensionless |
| d = U_discᵀw | reachable but discarded (u_35..50) | 16, µm |
| n_null = U_nullᵀw | **not reachable by any of the 50 DOF** | 76, µm |
| I_V | the MIW built from visit set V | 73×73 grid |

DZ row order is `(k - k_min)·n_j + j_idx`, k outer / j inner (`OFCSvd.kj_grid`).
The 21 packed measured Zernikes are `zn = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26]`.

**Subspace bookkeeping (k≤6):** S_hat is 126×50, so the DZ space splits into
span(U_eff) = 34 (reachable and removed), span(u_35..50) = 16 (reachable but
discarded by the truncation), and null(S_hatᵀ) = 76 (unreachable by any DOF).

---

## 4. Established results

### 4.1 The dominant structural fact — focal-order truncation

The per-visit fit uses only **k=1..6**, but the MIW's structure is overwhelmingly
above that. Fitting each MIW map to focal Noll k=1..45
(`code/analyze_miw_dz_full_k.py`), fractions of MIW **power**, dimensionless,
power-weighted over the 21 pupil Zernikes:

| k ≤ 6 (fitted) | 7..30 (in ts_ofc, unfitted) | k > 30 (outside ts_ofc) | unfit at k≤45 |
|---|---|---|---|
| **0.169** | **0.675** | 0.076 | 0.079 |

**83% of MIW power is at focal orders the fit never touches.** Per pupil Zernike,
fraction of that Z_j's power captured by k≤6: **Z5 0.041, Z7 0.051, Z8 0.080,
Z6 0.186**, versus Z11 0.856 and Z4 0.629. MIW RMS [µm]: Z5 0.113, Z6 0.114,
Z7 0.118, Z8 0.112, Z11 0.031, Z4 0.039.

Consequence: **every subspace analysis below lives inside k≤6 and therefore sees at
most 17% of the MIW.** The right rebuild is a k=1..30 refit (630 coefficients,
50-dim reachable, 580-dim null, 84.4% of MIW power). Above k=30 ts_ofc tabulates
nothing (field axis length 31, index 0 unused), so no reachable/null split is even
definable there. Capture must be measured as 1 − residual_power/total_power, since
the focal basis is not orthonormal on the masked grid.

### 4.2 Static optics do not explain the MIW

No physically plausible static surface figure reproduces the MIW. Single-surface
back-projection round-trip explained variance ≤ 0.11 (dimensionless); a joint
4-surface fit reaches 0.43 only with ~20–24 µm RMS figures (real figures are
≲0.1 µm). Constrained to believable figures (~0.1 µm) it explains ~0.11. **But** the
coma field *shape* correlates well: Pearson r = +0.86 (Z7) / +0.82 (Z8) at physical
amplitudes, with a ~7× amplitude deficit. So the MIW coma pattern is optics-like in
structure but not in magnitude.

### 4.3 Coadd-vs-MIW variation is thermal-associated

Predicting per-block spatial r from environmental telemetry, leave-night-out
GroupKFold: environmental GBT **CV R² = +0.62** (dimensionless) vs a negative
S/N-only baseline. `z_gradient` (M1M3 vertical thermal gradient) dominates
permutation importance; r rises as z_gradient → 0.

### 4.4 The retrieval-bias algebra, and what it predicts

Derived in `MIW_COADD_EQUATIONS.md` §3–§4. Two results matter:

**(a) Unbiased case — the u-modes cancel.** Differencing two MIW fixed points built
from the same batoid starting intrinsic gives
`I_b − I_B ≈ R[(1−P)(<W>_b − <W>_B)]`: the residual is driven by the **discarded**
(1−P) part of the state difference, so the stored u-mode amplitudes a — which span
exactly the removed subspace — cancel at first order.

**(b) Biased case — they do not.** With a bias `B_+ = 1 + B` acting on the pupil
index j (same at every focal order k), the build removes an *effective* state
`B_+W` which is **not reachable**, and
`I_b − I_B ⊃ R[L·Δa]` with
`L = (1−P) B_+ S_hat (U_effᵀ B_+ S_hat)⁺`. **L ≡ 0 identically when B = 0** (the
proof collapses to `V_35:50ᵀ V_34 = 0`), so L is a direct estimator of the retrieval
bias with a null hypothesis of exactly zero.

### 4.5 Measurements of L

**Block level** (`code/analyze_miw_bias_regression.py`): regressing the per-block
residual maps on Δa, ridge, leave-night-out, 1568 map cells, ~180 non-build blocks —
Δa-only **CV R² = +0.322**, permutation null mean −0.144 → p = 0.000. **But** thermal
alone gives +0.344 and thermal+Δa +0.371, so Δa adds only **+0.028** over thermal:
largely redundant, and this test alone cannot separate bias from thermal proxying.

**Per-visit level** (`code/analyze_umode_null_coupling.py`) — the stronger test.
Ridge (lambda/n ≈ 0.11) of n_null [µm] on the 34 **signed** a_m [µm],
leave-night-out CV over 19 nights, n = 1126 visits:

- observed **CV R² = +0.408** (dimensionless)
- simulated noise null (keep each visit's real reachable part `U_all U_allᵀw`, redraw
  the remainder from Sigma_emp): mean +0.014, 95th percentile +0.021, **0 of 100
  simulations exceeded observed → p < 0.01**

The null is essential: a and n_null are orthogonal projections of the same w, so they
would be independent only under isotropic noise, and Sigma_emp is strongly
non-isotropic. Per-mode drop-one CV ΔR² (dimensionless): u13 +0.040, u31 +0.025,
u2 +0.020, u21 +0.014, u20 +0.012.

Supporting amplitudes [µm]: mean-of-per-visit-norms ‖a‖ 0.734 (true RMS 0.801),
‖d‖ 0.082, ‖n_null‖ 0.084. n_null decomposes into a **static** mean vector of
0.0395 µm (20.5% of its power) and a **varying** part of 0.0777 µm; Sigma_emp
predicts noise-only 0.0425 µm, so ~70% of the varying part is real structure, and
R² = +0.408 refers to the varying part only.

**Mode 1 (focus) carries 41.4% of a's total variance** (a_1 std = 0.491 µm) from
open-loop focus drift — so never correlate against total ‖a‖; use per-mode signed a.

### 4.6 The error model — this was the crux

`fits.parquet`'s `_err` columns are Huber RLM formal errors, sigma ≈
scale/sqrt(N_donuts), which treat ~2800 donuts as independent. Turbulence breaks
that. Using them gives chi²_50/dof ≈ 500, which is **an artifact**.

Fix (`code/analyze_dz_goodness_of_fit.py`): **Sigma_emp**, the empirical 126×126
coefficient covariance [µm²] from **within-block visit-to-visit scatter**. Inside a
stable block the optical state is essentially fixed, so that scatter measures the
coefficient error with turbulence, dome seeing, retrieval noise and their spatial
correlations all included, with no modelling. 833 residual dof from 82 blocks.

- **Restrict to stable programs.** `T614_triplets` are stable.
  `T377/T378/T379/T380/T381` are S-matrix tests that deliberately **step the optical
  state within a block**, so their scatter is real DOF motion. (They are not in this
  build's visit set, but the guard matters generally.)
- `sigma_emp/sigma_RLM` = **8.5** (dimensionless; empirical over formal coefficient
  error) → **~39 of 2838 effective independent donuts**.
- Cross-check with a consecutive-visit-difference covariance: ratio 7.9, and
  sigma_adjacent/sigma_block-mean = 0.93 (dimensionless) — so the within-block
  scatter really is stationary noise and the block-mean estimator is sound.
- **Calibrated chi²_50/dof = 4.2** (dof = 76, vs any reachable 50-DOF state);
  chi²_34/dof = 5.2 (dof = 92). A real inconsistency, ~100× smaller than the artifact.

**Shrinkage matters:** Ledoit–Wolf shrinks toward a *scaled identity*, which is wrong
here — the 126 coefficients span ~30× in sigma (0.00038 to 0.186 µm), so it leaks
Z5's variance into Z26's. Use `_shrink_cov`: keep the empirical variances exactly,
shrink only the **correlation** matrix (lambda ≈ 0.16).

### 4.7 The correction truncation (a separate, fixable defect)

`ts_ofc`'s `SensitivityMatrix.evaluate()` sums **all** field orders, so a hardware
correction removes a DOF change's entire field dependence. The MIW build does not:
it reconstructs only the fitted k≤6 coefficients, so each mode's k>6 response is
never subtracted. Fraction of a mode's total wavefront power left behind: ~0.000 for
modes 1–7, 0.0007–0.001 for 8–27, **0.086–0.179 for modes 28–34**.

k≤6 **is** sufficient to determine the state (S_hat(k≤6) is full rank 50,
condition number 1.12e4 vs 0.998e4 for k≤30; top-50 v-mode subspace identical to
0.00 deg principal angle), so this is fixable: recover the DOF and subtract
`S_full·d` over all orders. Note `S_{k≤6}·d == P·w` **exactly**, so the fix is purely
additive. Magnitude for the build's own mean state: 0.022 µm total; per Zernike
11.5% / 10.1% of MIW Z5/Z6 amplitude and 1.5% / 2.9% of Z7/Z8. The leakage's field
shape matches MIW **coma** (r = +0.80 Z7, +0.64 Z8) but not astigmatism
(+0.19 Z5, −0.02 Z6), because coma's k>6 power peaks at k=7,8 (just above the cut)
while astigmatism's sits at k=15,22,23,25.

---

## 5. Retracted claims — do not repeat these

1. **"A δS sensitivity-matrix error causes a gain error in the removal."** Wrong for
   the coadd−MIW residual: the removal is pure software, subtracting exactly the
   `R[P·w]` it just fitted, so there is no gain to be in error, and the §4.4(a)
   result holds for whatever P is. **However** δS *is* a valid mechanism for §4.5's
   per-visit test, because there the null space is defined by the *tabulated* S_hat —
   if the real response leaves col(S_hat), genuine states leak in proportional to a.
   Keep these two cases distinct.
2. **"chi²_50/dof ≈ 500, a huge inconsistency."** Artifact of the RLM
   independent-donut errors. The calibrated value is 4.2.
3. **"The 3 iterations make the MIW independent of the starting intrinsic."** It
   lands on a fixed-point *manifold* in one step and retains
   `G = P·F[I_true − I_batoid]`; iteration 2 ≈ 3 shows it landed, not *which* point.
   Moot for differences (both builds share I_batoid), and the MIW's headline
   high-field-order excess is outside the k≤6 fit basis so it cannot be this gauge.
4. **"Filter/L3 footprints don't overlap across field points."** They do (~88%/81%
   adjacent). The limitation is geometric — near focus, a surface figure's effect is
   mostly per-field tilt/defocus, the gauge the stitch removes.
5. Quoting the k>6 leakage as the ratio |k>6|/|k≤6| (up to 0.47) rather than the
   fraction of total power (up to 0.18) — overstated it.
6. Calling a mean-of-per-visit-norms an "RMS", and quoting "29× the null" (a bare
   ratio of two R² values) as though it were a statistic.

---

## 6. Code inventory

Local (laptop, no LSST stack unless noted):

| file | purpose |
|---|---|
| `code/analyze_miw_dz_full_k.py` | MIW fitted to focal Noll k=1..45; the §4.1 result. Needs ts_ofc? No — galsim only |
| `code/analyze_dz_goodness_of_fit.py` | per-visit chi² vs the sensitivity matrix; Sigma_emp; `--stable-programs`; `--fits-biw` |
| `code/analyze_umode_null_coupling.py` | the §4.5 per-visit L test with the simulated noise null |
| `code/analyze_miw_bias_regression.py` | block-level L from the residual maps; auto-detects the new npz keys |
| `code/recompute_coadd_metrics.py` | **the single** coadd time-series renderer; `--rebin 1 3` (one page per factor), u-mode pages, ML |
| `code/check_k_truncation.py` | §4.7 checks (a) and (b) |
| `code/analyze_miw_field_order.py` | is the MIW the un-subtracted k>6 term? (§4.7 shape test) |
| `code/analyze_sparse_observability.py` | sparse-mode DOF observability, both samplings |
| `code/analyze_sensitivity_sparse.py` | which DOF drive secondary/tertiary aberrations |

RSP-only: `code/run_coadd_blocks_miw.py` (+ `run_coadd_blocks_miw.sbatch` — roma,
128 GB, 12 h; it OOMs a notebook pod). `--param-set` is **required**.

Calibration: `calibration/miw/umode_k_leakage_50_34.npy` (per-mode k>6 leakage
ratios, dimensionless).

---

## 7. Pending work

1. **RSP rerun of `run_coadd_blocks_miw.py`** (submitted 2026-09-03, status unknown).
   It now additionally saves, in `block_grids.npz`: `raw_dz` and `resid_dz`
   (block-median w and (1−P)w, 126-vectors, µm), plus `U_eff`/`U_all`/`U_disc`,
   `kj_grid`, `iZs`, `sigma`, `V`, `norm_weights` — so the subspace split is
   derivable off-RSP. And a sidecar `block_grids_ext.npz` with coadd+MIW maps for the
   secondary/tertiary pupil Zernikes (`--ext-terms secondary` = 12,13,16,17,18,19,
   22,23,24,25,26), float32, row-aligned by `block`. Legacy keys unchanged.
   `analyze_miw_bias_regression.py` lights up its discriminator section automatically
   once these exist.
2. **Decide the n_null definition** given §4.1. Strong recommendation: rebuild on a
   **k=1..30 refit** (630 coefficients, 580-dim null) before drawing further
   conclusions from the k≤6 version. Requires a build-side refit with a wider focal
   basis — cheap per visit.
3. **The k-independence test** (Aaron asked for this; not yet built). Discriminates
   retrieval bias B from sensitivity error δS: a pupil-index bias is **k-independent
   by construction**, so the fitted L must factor as a 21×21 pupil mixing replicated
   across the focal orders. δS carries no such constraint. Best done on the extended
   k basis from item 2.
4. **BIW-vs-MIW comparison.** `fits.parquet` is the MIW-referenced fit. To get the
   batoid-intrinsic version, rerun the build with `--n-iter 1` (iteration 1 fits
   against the batoid tabulated intrinsic by construction) and pass it as
   `--fits-biw`. Caveat: the MIW case is partly self-fulfilling for build visits, so
   compare on non-build visits.
5. **Fix the correction truncation** (§4.7): subtract `S_full·d` instead of
   re-expanding only the fitted k≤6 coefficients.
6. **Sparse-fit study, Pieces 2–3.** Piece 1 done: nearly every DOF drives primary
   *and* secondary of a family with field correlation ≈ ±1, so the orders are
   near-degenerate per DOF. Observability done: the operational 22/12 scheme keeps
   all 12 controlled v-modes under primary-only measurement (median 1.00, min 0.99,
   both 4-corner-WFS and field-complete sampling); only the extended 50/34 scheme
   loses ground, in the high-order M2 bending modes (0.37–0.63). Piece 2 = the
   leakage matrix from injected-secondary donut sims; Piece 3 = a sparse MIW.

---

## 8. Build selection facts worth not re-deriving

The 5rot MIW build selection (from `mi_config.yaml`, verified against the build's
`fits.parquet`): i-band, program `BLOCK-T614_triplets`, elevation **65–75 deg**, the
5 in-family rotator windows, `day_obs <= 20260513`. On the coadd that is **16 blocks
over 4 nights** (2026-03-15 .. 03-24) at elevation ~70 deg. `program` is not in
`block_grids.npz` — read it from the sibling `blocks_summary.parquet` (BLOCK- prefix
stripped there).

Build knobs: `k_min=1, k_max=6, n_iter=3, n_bins=73, fp_radius_basis=1.8,
fp_radius_grid=1.8`.
