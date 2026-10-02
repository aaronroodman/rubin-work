# Diffraction models of defocused Rubin donuts

> **Status:** current · **Last updated:** 2026-09-30 · **Kind:** reference (investigation writeup)

Comparison of geometric ray tracing, batoid's Fourier-transform point-spread function
(`fftPSF`), and Huygens/Debye diffraction for defocused star images ("donuts") in the
Rubin telescope. Two questions drive it: whether batoid's Huygens code
(`batoid.analysis.huygensPSF`) gives reliable diffraction images, and how large
diffraction effects are at the 10 µm pixel scale that donut wavefront fitting sees.
All images are monochromatic (622 nm, r band).

## Method

Every model starts from the same batoid ray trace:

- **Huygens/Debye sum.** `huygensPSF` evaluates
  U(x) = Σ_j f_j exp{i[k_j·(x − r_j) + k₀ t_j]} over the traced rays, one output pixel at
  a time in a Python loop. Here f_j is `RayVector.flux`, which is 1 by default. On a
  uniform output lattice this sum is exactly a type-1 non-uniform FFT (NUFFT). The
  module `code/fresnel/fresnel_donut.py` evaluates it with FINUFFT, which agrees with
  batoid's own `sumAmplitude` to 2.6×10⁻¹² in relative amplitude (dimensionless) and is
  ~10⁶ times faster (dimensionless ratio of wall-clock times).
- **Debye-Wolf weights.** For rays sampled uniformly over the entrance pupil, the
  energy-conserving amplitude weight is sqrt(dΩ/d²u), where Ω is the ray-direction
  solid angle and u the entrance-pupil position. It is computed from finite
  differences of the direction cosines over the ray grid. Irradiance is the scalar
  energy flux through the detector plane, Re(U* ∂U/∂z)/k.
- **References.** The first reference is an on-axis annular paraboloid with Rubin's
  focal ratio (f/1.234), diameter (8.36 m) and obscuration (0.612, inner/outer diameter
  ratio). It is stigmatic, so the scalar Debye integral reduces to one dimension and can
  be computed to 10⁻¹³ precision. The paraxial Fresnel (Lommel) integral is
  cross-checked independently with HCIPy.

## Results

| Test | Paraboloid (1.5 mm) | Rubin on axis (camera piston ±1.5 mm) |
|---|---|---|
| Huygens with Debye weights vs reference: RMS residual as a fraction of the peak irradiance | 1.0×10⁻³ vs the semi-analytic Debye integral | converged in grid size to 1.3×10⁻⁴ |
| Sine condition: std(sin θ/h)/mean (dimensionless) | 7.2×10⁻³ | 8.6×10⁻⁶ |
| Unit-weight Huygens (batoid default) vs Debye-weighted: Σ\|Δ\| at 10 µm pixels as a fraction of total flux | 2.5×10⁻² | ~10⁻⁴ (max sample 1.7×10⁻⁴ of peak) |
| `fftPSF` vs reference: 98%-flux radius | 576.9 µm vs 632.1 µm | agrees to 0.2 µm |
| `fftPSF` vs Huygens on the same lattice: Σ\|Δ\| as a fraction of total flux, default `sphereRadius` 5 m → 20 m | — | 7.4–8.2×10⁻³ → 1.0–1.8×10⁻³ (no further gain at 100–1000 m) |
| Paraxial Fresnel (HCIPy, Lommel): 98%-flux radius | 606.4 µm (4.1% small) | — |
| Geometric vs diffraction: Σ\|Δ\| at 10 µm pixels as a fraction of total flux | 0.14 | 0.13 (shot-noise floor 0.035) |

Conclusions:

- **batoid's Huygens algorithm is correct.** Its only real defect is speed, and the
  NUFFT evaluation removes that. Unit ray weights are exact for a system obeying the
  Abbe sine condition, which Rubin does to 4×10⁻⁶. For other optics, set
  `RayVector.flux` to sqrt(dα dβ/d²u) before `sumAmplitude`.
- **batoid `fftPSF` is accurate for Rubin donuts only with a large reference sphere.**
  The residual falls roughly as 1/`sphereRadius` down to a ~0.1%-of-flux floor reached
  by ~20 m; use ≳ 20 m for defocused images, not the 5 m in the yaml files. Its linear pupil-to-direction map is exact for Rubin but
  fails badly for optics that break the sine condition, such as the paraboloid.
  Re-binning its 0.766 µm lattice into 10 µm pixels adds a ~1.5%-of-peak beat
  artefact.
- **Paraxial Fresnel codes** (HCIPy, POPPY, PROPER, prysm, GalSim with a defocus
  Zernike) are not accurate at f/1.23: the donut edge is 4% too small.
- **Diffraction redistributes ~13% of the donut flux** (Σ|Δ| over 10 µm pixels,
  fraction of total flux) relative to geometric optics. The difference is concentrated
  in ~3-pixel rings at the inner and outer edges (√(λΔz) = 30.5 µm), plus interior
  fringes, in these monochromatic images.
- **Intra- and extra-focal donuts differ for the design optic.** With the camera
  pistoned ±1.5 mm, the geometric inner radius is 389.5 µm intra versus 378.3 µm
  extra, and the traced Z11 is −0.163 µm versus +0.267 µm of wavefront
  (−0.125 / +0.229 µm for a detector-only shift). The difference is defocus-induced
  spherical aberration from the fast beam.

## Seeing

`optics_fresnel_rubin_seeing_v1.ipynb` adds long-exposure von Kármán seeing (0.7″
FWHM at 622 nm, outer scale 20 m) to the on-axis donuts. The main geometry is the
camera piston (full array mode, FAM); the detector-only shift (corner wavefront
sensors, CWFS) is also included in the sensitivity tests.

- **Separable convolution is accurate.** For a long exposure, the exact wave-optics
  ray kick is an average of the optics-only image over field angle, weighted by the
  atmospheric PSF, for turbulence at any altitude. It differs from convolution only
  through anisoplanatism: ≤ 3.4×10⁻⁴ of total flux (Σ|Δ| over 10 µm pixels) for
  3–4.2″ offsets. Each defocused plane needs its own plate scale (10.3144 m/rad
  intra-focal, 10.3054 m/rad extra-focal).
- **Danish reference TA.** `batoid.analysis.zernikeTA` of each defocused optic with
  jmax = 66 reproduces the traced ray positions to 0.004 µm RMS. With jmax = 28 the
  residual is 0.21 µm RMS, which produces a spurious ~0.005″ intra/extra FWHM
  difference. Camera piston and detector shift need different TAs.
- **Huygens versus geometric, both with seeing:** Σ|Δ| of 3.2% of total flux per side,
  in rings at the edges.
- **No intra/extra seeing asymmetry is reproduced.** The fitted Kolmogorov FWHM
  differs between intra and extra by ≤ 0.001″ with the correct TA (Huygens + seeing:
  0.8239″ / 0.8234″), and by ≤ 0.002″ for every reference variant tested. The data
  show intra ≈ 0.9″ against extra ≈ 0.8″. Diffraction inflates the fitted FWHM by
  0.07″ equally on both sides.
- **Apparent Z11** (TA reference → apparent, µm of wavefront): −0.1631 → −0.1282
  intra-focal, and +0.2669 → +0.2308 extra-focal. That is fitted offsets of
  +34.9 / −36.1 nm, reducing the antisymmetric part from 0.215 to 0.180 µm; the
  intra/extra mean is unchanged. Z4 offsets are +8.0 / −4.3 nm; Z22 changes by ≤ 1 nm.

## Silicon sensor (i band)

`optics_fresnel_sensor_iband_v1.ipynb` traces i-band photons (676–833 nm) through
`LSST_i.yaml` with the camera pistoned ±1.5 mm and per-ray 0.7″ seeing kicks, and
accumulates them with GalSim's `SiliconSensor` (ITL and e2v). Each photon converts at
a wavelength-dependent depth, travels laterally along its incidence direction, and
diffuses; brighter-fatter is off. GalSim does not refract at the silicon surface, so
the directions are Snell-refracted first: the 14–24° air incidence angles become
3.7–6.4° in silicon, and the mean lateral travel is 0.78 µm (median absorption length
8.1 µm).

- **No intra/extra blur difference.** Intra − extra fitted FWHM is +0.0003 to
  +0.0004″ with refracted angles. That has the observed sign but is ~250 times
  smaller than the observed 0.1″.
- **Diffusion** adds 0.023″ to the fitted FWHM, equally on both sides.
- **The sensor's signature is an apparent focus offset:** +34 nm of Z4 wavefront on
  both sides (+31.6 intra, +36.4 extra). It is not modelled by a batoid detector plane
  at the silicon surface, and it should grow steeply toward z and y.
- **Passing batoid's air directions unrefracted** would give a +127 nm Z4 offset and a
  spurious +0.0017″ intra − extra difference.

## Outstanding work

- Off-axis field positions, especially the corner wavefront sensors (CWFS), where lens
  vignetting creates edges not conjugate to the pupil. There the Debye model itself
  can fail, and a multi-plane Fresnel code (POPPY/PROPER) or Gaussian beamlet
  decomposition (poke) is needed as the check.
- The origin of the observed intra/extra difference in fitted donut blur, which is
  seen consistently in both CWFS and FAM data and is not reproduced here by scalar
  diffraction or by errors in the reference TA.
- Broadband (r-band) images, to see which fringes survive bandwidth averaging.
- The bias that a geometric forward model (as in Danish or ts_wep) produces in fitted
  Zernikes when applied to diffraction-limited donuts.

## Files

- `notebooks/optics_fresnel_analytic_benchmark_v1.ipynb`: annular paraboloid versus the
  semi-analytic Debye and Lommel integrals and HCIPy.
- `notebooks/optics_fresnel_rubin_onaxis_v1.ipynb`: Rubin on axis, camera pistoned ±1.5 mm.
- `notebooks/optics_fresnel_rubin_seeing_v1.ipynb`: the same donuts with von Kármán seeing,
  the separability tests, and Danish fits of apparent Zernikes and FWHM.
- `code/fresnel/donut_seeing.py`: von Kármán kernel, plate scale, per-ray kicks,
  Danish fit wrapper, TA ray-residual check, edge widths.
- `notebooks/optics_fresnel_sensor_iband_v1.ipynb`: i-band silicon-sensor effects on intra- and
  extra-focal donuts with GalSim `SiliconSensor`.
- `code/fresnel/donut_sensor.py`: bandpass photon tracing, Snell refraction into
  silicon, sensor accumulation.
- `code/fresnel/fresnel_donut.py`: NUFFT Huygens sum, Debye weights, `fftPSF` and
  geometric wrappers, semi-analytic references.
- Figures: `output/fresnel/optics_fresnel_*_<date>.png`.

Dependencies beyond batoid: `finufft` and `hcipy` (installed with
`/opt/local/bin/pip3 install --user finufft hcipy`), and `galsim` and `danish` for the
seeing notebook. On macOS the PyPI wheels of batoid and finufft each vendor their own
`libomp`. The module therefore sets `KMP_DUPLICATE_LIB_OK` and imports finufft
before batoid; in the other order, multi-threaded finufft calls segfault.
