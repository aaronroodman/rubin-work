# optatmo_moments

Standalone Optics+Atmosphere PSF moment tools for a Rubin wavefront/PSF fit —
a modern, differentiable rebuild of the ideas in the old PIFF `optatmo3` branch.
Fits a Double-Zernike optical wavefront (on top of the measured intrinsic
wavefront, MIW) plus a constant atmospheric kernel to observed stellar HSM
moments (2nd/3rd/4th order).

## Status

**Phase 1 — accuracy study of the momfit analytic method** (SPIE 2026,
Zanmar Sanchez et al.): *complete*. Verdict: the analytic (geometric,
phase-gradient) moments are adequate only for 2nd-order size/ellipticity (with
a calibration slope); 3rd/4th-order moments are dominated by diffraction
(trefoil skew is 100% diffractive) and cannot be reproduced by any geometric
model. See [FINDINGS.md](FINDINGS.md).

**Phase 2 — differentiable JAX forward model**: *complete & validated*. A
standalone `Fraunhofer × atmosphere × HSM-adaptive-moment` model in JAX with
exact autodiff, reproducing galsim/PIFF `calculate_moments` to <1% (well below
per-star noise) across seeing and aberration amplitude.
See `output/jax_validation.png`.

**Phase 3 — the fit** (DZ + MIW + atmosphere + jitter to stellar moments):
*built & validated on simulation*. VonKarman kernel, MIW OCS/CCS+rotator
loader, Double-Zernike field model, constant moment-offset (jitter/wind-shake)
term, YAML config, and an SVD-degeneracy-controlled fit driver. On simulated
stars it recovers injected coma/trefoil/field-variation and atmospheric size to
noise-limited precision; the SVD exposes the expected degeneracies (defocus↔
size, astigmatism-at-focus, shear↔moment-offset) and fits only the identifiable
subspace. Next: run on real DM star catalogs.

## Modules

Forward model (JAX, the production path):
- `jax_optatmo.py` — `JaxOptAtmoPSF`: pupil→Fraunhofer PSF from Zernike coeffs,
  atmospheric MTF (Kolmogorov + VonKarman, galsim-calibrated table),
  `.moments_adaptive()` (HSM-matching) and `.moments()` (fixed weight). Autodiff.
- `jax_hsm.py` — pure-JAX adaptive (HSM) moments via the Bernstein-Jarvis
  `M ← 2⟨xxᵀ⟩` fixed point; matches `FindAdaptiveMom` to 5 digits.
- `validate_jax.py` — validates JAX vs galsim over an ensemble + autodiff check.

Fit system:
- `dz.py` — Double-Zernike field model (pupil × focal Noll terms → per-star
  wavefront design tensor; linear in the free params).
- `miw.py` — MIW loader: OCS/CCS + rotator → Noll Zernike vector (microns),
  matching the ts_intrinsic_wavefront reconstruction convention.
- `config.py` + `config.yaml` — YAML config and flat-parameter layout
  (DZ + free atmosphere params + moment offsets), bounds, initial values.
- `model.py` — batched differentiable forward model + moment χ² cost.
- `fit.py` — fit driver (jax value-and-grad + scipy L-BFGS-B), with optional
  SVD degeneracy control (`svd_analysis`, `run_fit(use_svd=True)`).
- `sim_fit_test.py` — parameter-recovery validation on simulated stars.
- `sim_svd_test.py` — SVD degeneracy demonstration on the full model.

## The jitter / wind-shake term
Modelled as configurable additive **constant offsets in moment space**
(`moment_offsets.moments` in the config) — the generalisation of the old
constant-ellipticity term to any of the 12 moments. These are degenerate with
the atmospheric shear (for e1/e2) and size (for e0); use the SVD control or
constrain them with priors / a subset of moments.

Reference / study (numpy + galsim):
- `moments_hsm.py` — faithful port of PIFF `util.calculate_moments` (truth).
- `psf_model.py` — galsim `OpticalPSF ⊗ VonKarman` truth PSF.
- `moments_analytic.py` — momfit unweighted phase-gradient moments (parity-corrected).
- `moments_fixedweight.py` — numpy twin of the fixed-weight JAX estimator.
- `render_geometric.py` — diffraction-free geometric⊗atmosphere render.
- `evaluate.py`, `decompose_modes.py` — Phase-1 accuracy studies.

## Conventions
Rubin: D=8.36 m, obscuration 0.61, λ=750 nm, 0.2″/pix. Zernikes Noll-indexed in
microns. Moments named as in PIFF: `e0=M11, e1=M20, e2=M02`, then 3rd (M21,M12,
M30,M03), 4th (M22,M31,M13,M40,M04). Circular Noll Zernikes (matching
`galsim.OpticalPSF`); switch to annular for MIW ingestion.

## Layout & how to run
```
rubin-work/optatmo/
  code/        all .py modules (a flat package; siblings import by bare name)
  config.yaml  fit configuration
  data/        parquets + npz (gitignored)          — inputs/products
  output/      rendered plots (gitignored)
```
Run scripts **from the `optatmo/` directory** as `python code/<script>.py` — Python
puts the script's dir (`code/`) on `sys.path` so the flat imports resolve, and
`config.yaml`, `data/`, `output/`, `../aos/...` stay relative to `optatmo/`. e.g.:
```bash
cd rubin-work/optatmo
python code/build_svd_local.py                 # build SVDs from data/ofc_raw.npz
python code/run_vmode_fit.py 1 data/ofc_svd_50_34_k6.npz
python code/plot_data_model.py data/ofc_svd_50_34_k6.npz
```
For interactive use, add `code/` to `sys.path` first.

## Quick start (library)
```python
import sys; sys.path.insert(0, 'code')
from jax_optatmo import JaxOptAtmoPSF
import jax, jax.numpy as jnp
m = JaxOptAtmoPSF(jmax=22, stamp=32, oversample=16, annular=True)
z = jnp.zeros(23).at[4].set(0.2)          # 0.2 µm defocus (Noll, µm)
atm = jnp.array([0.8, 0.0, 0.0])          # [fwhm_arcsec, g1, g2]
mom = m.moments_adaptive(z, atm)          # 12 HSM moments (arcsec^n)
J = jax.jacobian(lambda z: m.moments_adaptive(z, atm))(z)   # d(moments)/d(zernike)
```
