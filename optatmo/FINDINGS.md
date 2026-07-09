# Accuracy of momfit analytic moments vs HSM weighted moments

**Question.** Can the analytic phase-gradient moment method of Zanmar Sanchez et
al. (SPIE 2026) — or a Gaussian-*weighted* variant of it — replace per-star
Galsim/Fraunhofer + HSM moment evaluation inside an Optics+Atmosphere PSF fit
for Rubin, including the 3rd/4th-order moments needed to break degeneracies?

**Setup.** Aberration states (Z4–Z22, per-term rms scaled to the MIW field
variation, × amplitude tiers 0.5–3.0) × seeing FWHM ∈ {0.6,0.8,1.0,1.2}″. Truth
= HSM adaptive-weighted moments (faithful PIFF `calculate_moments` port) of
`OpticalPSF(aberrations) ⊗ VonKarman(L0=25)`, λ=750 nm, 0.2″/pix. Compared to:
- **unweighted analytic** — momfit's exact geometric phase-gradient moments;
- **weighted analytic** — the diffraction-free geometric spot ⊗ atmosphere
  *rendered* and measured with the same HSM estimator (`render_geometric.py`).

Images are noiseless, so scatter = pure model discrepancy; per-star HSM error
(S/N≈300) is a reference scale only.

## Two corrections / clarifications found along the way
1. **Parity.** The image-plane ray displacement is −f·∇W, so the transverse-
   aberration coordinate carries a sign relative to ∇W. This is invisible to the
   even (2nd, 4th) moments but sets the sign of all odd (3rd) moments; without it
   the analytic 3rd moments are anti-correlated with HSM. Fixed in
   `moments_analytic.py`.
2. **The higher-order wall is diffraction, not weighting.** Testing the
   *weighted* geometric variant separated the two error sources cleanly.

## Result 1 — weighting fixes 2nd order; diffraction does not touch it
On the diffraction-free weighted render, 2nd-order moments match the Fraunhofer
truth well (e.g. pure-mode e1: render −1.0e-4 vs Fraunhofer −1.1e-4), while the
*unweighted* analytic value is ~2–3× too large. So the ~0.5 analytic→HSM slope
seen for e1/e2 is the HSM weighting suppression, and it is largely calibratable.

## Result 2 — 3rd/4th-order moments are intrinsically diffractive (optics-only, exact)
| single mode (0.2 µm) | dominant 3rd moment: geometric | Fraunhofer HSM |
|---|---|---|
| coma Z7 | M12 = +5.4e-4 | −1.5e-4  (sign-flip + 0.27×, and diffraction adds an M03 term absent geometrically) |
| trefoil Z9 | M03 = **1e-21 (zero)** | −1.2e-4  (**100 % diffractive**) |
| 2nd coma Z16/17 | huge | ~0  (geometric over-predicts ~20–35×) |

No geometric model — weighted or not — can reproduce these: the trefoil skew
exists *only* because of diffraction, and coma's is reduced 4× and mixed by it.

## Result 3 — full ensemble (Z4–Z22, amplitudes 0.5–3×), analytic → HSM
| Moment | slope | R² | resid/sig | resid/err | slopeVar(seeing) | slopeVar(amp) | Verdict |
|--------|------:|----:|----------:|----------:|-----:|-----:|---------|
| e0 (size, +atm) | 0.52 | 0.93 | 0.26 | 13.8 | 0.08 | 0.11 | ✅ good, mild amplitude curvature |
| e1 | 0.37 | 0.67 | 0.58 | 3.0 | 0.10 | 0.08 | ⚠️ degrades with amplitude |
| e2 | 0.32 | 0.76 | 0.49 | 2.9 | 0.13 | 0.15 | ⚠️ |
| M21,M12 (3rd) | 0.04 | 0.64–0.70 | 0.55–0.60 | 1.2–2.0 | 0.19–0.26 | 0.22–0.36 | ❌ tiny/unstable slope |
| M30,M03 (3rd) | 0.07 | 0.65–0.70 | 0.54–0.60 | 1.2–1.6 | 0.14–0.19 | 0.12–0.22 | ❌ |
| M22 (4th radial) | 0.21 | 0.05 | 0.97 | 44 | 0.47 | 0.70 | ❌ atmosphere-dominated |
| M31,M13 (4th) | 0.09–0.14 | 0.10–0.42 | 0.76–0.95 | 5 | 0.47–0.58 | 0.63–0.86 | ❌ |
| M40,M04 (4th) | 0.02–0.04 | 0.27–0.29 | 0.84–0.85 | 1.4–1.8 | 0.21–0.37 | 0.16–0.42 | ❌ |

Compared to the earlier small-amplitude run, **2nd-order accuracy degrades with
aberration amplitude** (e1 R² 0.88→0.67; model error grows from 0.3× to ~3× the
per-star measurement error) — the unweighted geometric moment is a linearized
approximation that frays as the PSF departs from Gaussian.

## Conclusions
- **Weighting** (the "weighted analytic variant") is the right fix for 2nd order
  and confirms diffraction is negligible there — but it does **not** rescue the
  higher orders.
- **The 3rd/4th-order moments are dominated by diffraction** (trefoil: entirely
  so). Since these are exactly the moments wanted to break optical degeneracies,
  no geometric/analytic engine (weighted or not) is adequate for a higher-order
  fit, at any calibration, over a realistic amplitude range.
- The analytic method remains useful only as a fast **2nd-order initializer**
  (size + ellipticity, with a fitted, amplitude-aware calibration).

**Recommendation (unchanged, now firmly established):** build the standalone
**JAX `Fraunhofer × atmospheric-kernel × HSM-weighted-moment`** forward model —
it reproduces the truth estimator by construction (diffraction included) and
gives autodiff gradients of the moment-comparison cost for the DZ + atmosphere
fit. The analytic moments seed it; the weighted render (`render_geometric.py`)
is a useful diffraction-free cross-check.

## Files
- `moments_analytic.py` — momfit unweighted phase-gradient moments (parity-corrected, + optional atm).
- `moments_hsm.py` — standalone PIFF `calculate_moments` port (truth estimator).
- `psf_model.py` — Galsim `OpticalPSF ⊗ VonKarman` forward model.
- `render_geometric.py` — diffraction-free geometric ⊗ atmosphere render ("weighted analytic").
- `decompose_modes.py` — per-Zernike geometric vs diffractive moment content → `output/optatmo_geom_vs_diffraction.png`.
- `evaluate.py` — ensemble driver → `output/optatmo_moment_accuracy.{png,npz}`.
