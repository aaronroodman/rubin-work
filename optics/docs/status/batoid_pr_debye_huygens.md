# Draft batoid PR: NUFFT evaluation and Debye weights for huygensPSF

> **Status:** current · **Last updated:** 2026-10-04 · **Kind:** status (draft PR text, not yet opened)

Local branch `debye-huygens` in `~/Software/batoid`, one commit on top of `main`
(a923c3e, which is also `releases/0.9`). Target: `jmeyers314/batoid:main`. The text
below the line is the PR body.

---

## Summary

Adds three options to `batoid.analysis.huygensPSF`. The defaults reproduce the current
behavior exactly.

- `method='nufft'` evaluates the Huygens sum at every output point at once with a
  type-1 non-uniform FFT (NUFFT), using the optional `finufft` package. It agrees with
  the existing per-point `sumAmplitude` loop to about 1e-11 of the peak, and is
  1.5e3 (`nx=256`) to 1.7e4 (`nx=1024`) times faster on the default lattice.
- `weights='debye'` weights each ray by sqrt(dΩ/d²u), the energy-conserving
  Debye-Wolf amplitude for rays that sample the entrance pupil uniformly.
- `irradiance='flux'` returns the energy flux through the detector plane,
  Re(U* V), instead of |U|².

The motivation is defocused (donut) images for wavefront sensing. A Rubin donut at
±1.5 mm needs a 2560² ray grid and about 10⁷ output points, which the per-point loop
cannot do in practical time.

## What huygensPSF and fftPSF compute

They are different algorithms for the same physics. Both evaluate a sum of plane waves
over a grid of rays filling the entrance pupil,

    U(x) = Σ_j a_j exp(i φ_j) exp(i k_j·x),

which is a discretization of the Debye integral. They differ in how they get each
ray's direction k_j, its phase φ_j, and the output positions x.

| | `huygensPSF` | `fftPSF` |
|---|---|---|
| Ray direction k_j | each ray's own traced direction (non-uniform in k) | a linear map k = (dk/du)·u, one 2×2 Jacobian fitted over the pupil (`dkdu`) |
| Ray phase φ_j | optical path at the focal-plane intersection. The plane-wave phase k₀t_j − k_j·r_j is constant along the ray, so this is exact. | wavefront on a reference sphere of radius `sphereRadius`, centered on the reference point |
| Evaluation | per point (`method='direct'`), or a type-1 NUFFT (`method='nufft'`) | uniform FFT of the padded pupil array |
| Output grid | any lattice | fixed by the pupil grid and `pad_factor` |

The traced wavefront includes the full defocus phase, so `fftPSF` is not a far-field
(Fraunhofer) approximation in the usual sense. It is the same Debye sum with two
approximations, and in the limit of an exact linear map and infinite `sphereRadius` the
two methods coincide. The approximations matter in three ways:

1. **Finite reference sphere.** `fftPSF` assigns each ray's phase to the ray's own
   direction. A Debye treatment would assign it to the direction of the sphere point
   seen from the sphere center. For a donut with about 0.66 mm of transverse ray
   aberration, the phase error is about k(0.66 mm)²/(2R): about 0.44 rad at R = 5 m and
   0.11 rad at R = 20 m, falling as 1/R.
2. **Linear pupil-to-direction map.** This is exact only for an optic obeying the Abbe
   sine condition (sin θ = h/f). Rubin obeys it to 8.6e-6 (dimensionless: standard
   deviation of sin θ/h over the pupil, divided by its mean). An f/1.23 annular
   paraboloid violates it at 7.2e-3, and `fftPSF` then underestimates the defocused
   98%-flux radius by 8.7%.
3. **Fixed lattice.** Rebinning `fftPSF`'s lattice (0.766 µm for a Rubin donut) into
   10 µm pixels beats at about 1.5% of the peak pixel. `huygensPSF` can sample directly
   on a pixel-commensurate lattice.

## Implementation

- On an output lattice x = x₀ + m a₁ + n a₂, the sum is
  Σ_j c_j exp(i[m (k_j·a₁) + n (k_j·a₂)]). That is a type-1 NUFFT with frequencies
  k_j·a₁ and k_j·a₂.
  - Because m and n are integers, the frequencies can be wrapped into [−π, π) exactly.
    So `method='nufft'` works for every lattice `huygensPSF` accepts, including
    coarse and non-orthogonal ones.
  - The phase convention and time reference (`rays.t[0]`) are those of
    `RayVector.sumAmplitude`.
- The Debye weight is computed by finite differences of the normalized direction
  cosines over the `nx`×`nx` pupil grid, using dΩ = dα dβ/γ (γ is the direction cosine
  along the detector normal). It is normalized to unit mean over unvignetted rays.
  - The weight multiplies `RayVector.flux`, so it works with both methods.
  - With `irradiance='flux'`, the second sum V carries an extra factor γ.
- For an optic obeying the sine condition, dα dβ/d²u is constant. The Debye weight then
  cancels against the γ in the flux projection, and unit weights with |U|² are correct.
  This is why the current default is accurate for Rubin.
- `finufft` is imported only when `method='nufft'` is used, with a clear ImportError if
  it is missing. It is added to `test_requirements.txt`.

## Validation

New tests in `tests/test_analysis.py`:

- `test_huygensPSF_nufft` compares `nufft` against `direct` with `LSST_r.yaml` at 0.1°.
  It covers the default lattice, a 10 µm square lattice (|k·dx| ≫ π, so the wrapping is
  exercised) and a non-orthogonal lattice, each with all weight and irradiance options.
  The maximum difference is 2e-12 to 5e-11 of the peak. The test also checks that
  invalid option values raise ValueError.
- `test_huygens_debye_paraboloid` uses an f/1.2 annular paraboloid (obscuration 0.5)
  defocused by 50 µm at 500 nm, which violates the sine condition. It compares the image
  with the semi-analytic Debye integral, a 1D Bessel integral over pupil radius:
  - Debye weights with flux irradiance give an RMS residual of 1.8e-3 of the peak at
    `nx=512`, and 4.9e-3 at `nx=256`, so the error is limited by pupil sampling;
  - unit weights with |U|² stay at about 9.4e-3 of the peak at both grid sizes, a
    systematic error.

All other tests pass locally (199 passed, 12 skipped). The one exception is the
`Analytic despace aberration` notebook test, whose kernel dies in my local source
build both with and without this change.

Separate studies with the 2026-09 Rubin design (r band, 622 nm, on axis, camera
pistoned ±1.5 mm, 2560² rays):

| Comparison | Result (fraction of total flux, Σ\|Δ\|, unless stated) |
|---|---|
| Debye-weighted vs unit-weight Huygens | < 1e-4 of the peak (sine condition holds) |
| Huygens, 2560² vs 3584² ray grid | 1.3e-4 RMS of the peak |
| `fftPSF` vs Huygens on the same lattice, `sphereRadius` 5 m (yaml default) | 0.7–0.8% |
| Same, `sphereRadius` 20 m | 0.10–0.18%; no further gain at 100–1000 m |
| Huygens vs geometric ray trace, 10 µm pixels | 13% |

For defocused images, `sphereRadius` ≳ 20 m is worth recommending for `fftPSF`.

## Timing

On an Apple M3 Max, with this branch built from source (`huygensPSF` on `LSST_r.yaml`
in focus; times include the ray trace):

| Pupil grid | Default lattice | `direct` | `nufft` | Ratio, direct/nufft (dimensionless) |
|---|---|---|---|---|
| 256² | 512² | 131 s (0.50 ms per point) | 0.09 s | 1.5e3 |
| 1024² | 2048² | 22 400 s, extrapolated (5.35 ms per point) | 1.3 s | 1.7e4 |

With batoid 0.8.1 PyPI wheels and the same rays and lattice, `fftPSF` and the
NUFFT-evaluated, Debye-weighted sum take about the same time:

| Case | Pupil grid | `fftPSF` | Debye-Huygens |
|---|---|---|---|
| In focus | 512² | 0.33 s | 0.46 s |
| 1.5 mm donut | 2560² | 7.1 s | 7.5 s |

The ray trace is 70–95% of both.

## Points to discuss

- **OpenMP on macOS.** The PyPI wheels of batoid and finufft each bundle their own
  `libomp`. Loading both aborts unless `KMP_DUPLICATE_LIB_OK=TRUE` is set. Separately,
  finufft must be imported before batoid has run a threaded trace, or the threaded
  finufft calls segfault.
  - I have not tried to handle this inside batoid, since a library should not set that
    environment variable for its users.
  - A source build of batoid with Apple clang has no OpenMP and no clash, but its trace
    is single-threaded.
  - This may be worth fixing in how the wheels are built.
- **Naming.** The new arguments are `method`, `weights`, `irradiance` and `eps`. Happy to
  change them to fit batoid's conventions.
- **Defaults.** I left the defaults as before (`direct`, `unit`, `intensity`). Making
  `nufft` the default when finufft is installed would be an easy follow-up.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
