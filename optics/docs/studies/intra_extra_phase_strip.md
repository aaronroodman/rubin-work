# Static phase strips and the intra/extra difference in donut blur

> **Status:** current · **Last updated:** 2026-10-04 · **Kind:** reference (investigation writeup; hypothesis, not yet simulated or tested on data)

The Danish donut model fits a smearing kernel (the Danish `fwhm`) to each defocused star
image (donut). In Rubin data the fitted kernel is consistently larger for intra-focal
donuts than for extra-focal ones: about 0.9″ against 0.8″ full width at half maximum
(FWHM). This writeup derives a symmetry constraint on the possible causes, and proposes
that the cause is a static phase strip along the spider vanes and the pupil edges.

## Observation

- The difference appears in the corner wavefront sensors (CWFS) and in full array mode
  (FAM). It appears in all bands, and is probably not field dependent. The same kind of
  difference was seen with DECam at the Blanco telescope.
- CWFS intra- and extra-focal images are exposed simultaneously, so the difference is not
  a change in seeing between exposures.
- The spider-vane shadows are visibly higher in contrast in extra-focal donuts, for
  example in exposure 2026071300702. Danish does not model the spiders.
- **Size.** In quadrature the intra side has √(0.9² − 0.8²) ≈ 0.41″ of extra blur. At the
  50.0 µm/″ plate scale that is about 21 µm FWHM on the detector, or about 2 pixels of
  10 µm.

The diffraction study ([fresnel_donuts.md](fresnel_donuts.md)) does not reproduce the
difference. Scalar diffraction, errors in the reference Zernikes, and i-band silicon
effects give at most 0.002″ between intra and extra.

## Symmetry constraint

For an exit pupil with real amplitude P(**u**) (obscurations and vignetting are real),
taking the complex conjugate of the Debye integral gives

  I(**x**, −Δz)[Φ] = I(−**x**, +Δz)[−Φ],

where Φ is the pupil phase and Δz the detector defocus. The relation holds without the
paraxial approximation. **An intra-focal donut is the reflected extra-focal donut of the
optic with the opposite pupil phase.** An intra/extra difference therefore requires a
phase that is odd under Φ → −Φ and static, a departure from the single-pupil (Debye)
model, or an asymmetry in the detector or the analysis.

Consequences:

| Effect | Why it cannot produce the difference |
|---|---|
| Atmospheric turbulence at any altitude, including scintillation | A long-exposure image depends on the turbulence only through the mutual coherence Γ(**u**₁ − **u**₂). For Kolmogorov turbulence Γ is real and even, and it does not change with propagation. Intra and extra images are therefore identical. |
| Seeing changes between exposures, wind shake, tracking | CWFS images are simultaneous. |
| Scattering or turbulence inside the camera | The lever arm from the scatterer to the detector is d ± Δz, which blurs the extra side more. That is the wrong sign. In FAM the lenses move with the detector, so the lever arm does not change. |
| Silicon conversion depth, diffusion, brighter-fatter | Symmetric. The i-band study gives ≤ 0.0004″. |
| Smooth static aberrations, CCD height | Fitted by the Zernike terms, or cancel between the two sides. |
| Fresnel diffraction by the spider vanes | The vanes are 8.6–9.8 m above M1. The pupil-space propagation distance for ±1.5 mm defocus is f²/Δz = 70.9 km, so the asymmetry is of order 9 m / 70.9 km ≈ 1e-4 (dimensionless). |
| Chromatic effects | The radial spread from longitudinal color is the same on both sides to first order. |

The constraint excludes the *fluctuating* part of the atmosphere, but not a static,
nonzero-mean refractive structure. Such a structure enters as a fixed Φ, and the intra
side sees −Φ.

## Geometry of an edge-localized slope

A ray from entrance-pupil point **u** lands at **x** = R·**u** − f∇W(**u**). Here R is the
defocus magnification: about +1.5e-4 for intra-focal and −1.5e-4 for extra-focal
(detector distance per unit pupil distance, dimensionless). The image is upright before
focus and inverted after it. W is the wavefront and f = 10.31 m.

- The displacement δ = −f∇W depends only on the pupil point, so it is the same vector on
  both sides.
- A displacement of 20 µm corresponds to a ray deflection of β = δ/f ≈ 2 µrad. In the
  focal region the displaced ray is therefore a parallel translation of the nominal ray,
  at the true ray angle: 14.4° at the inner pupil edge and 23.9° at the outer edge
  (sin θ = h/f).
- The intra and extra planes are at f ∓ 1.5 mm from the pupil. Since Δz/f ≈ 1.5e-4, the
  two planes see the same δ to that precision.
- The outer pupil rim maps to the outer donut rim on both sides (radius 665 µm; inner
  radius about 385 µm; annulus width about 280 µm). A slope at the outer edge zone moves
  light **just outside** the outer edge on one side, where a geometric model reads it as
  blur. On the other side it moves light **just inside** the same edge, where it forms a
  bright rim and the edge stays sharp.
- The displacement is a few percent of the annulus width: 20 µm is about 7% of 280 µm.
  Light never moves across the annulus to the inner rim. A 5 cm edge zone on M1 maps to
  about 7.6 µm on the detector.

Both the displacement and the width of the edge zone are smaller than the Fresnel width
on the detector, √(λΔz) = 30.5 µm at 622 nm. The real intensity pattern is therefore
softened by diffraction, and the size of the effect has to come from the Huygens/Debye
calculation, not from this geometric picture.

## Phase-strip hypothesis

**Vanes.** The spider vanes in the batoid model (`LSST/ComCamSpiders_*.yaml`) are 0.05 m
wide, in pairs offset by ±0.566 m, between 8.62 m and 9.82 m above M1. The DECam vanes
are 0.019 m wide at 9.75 m. Both are much narrower than the pupil-space Fresnel zone,
√(λf²/Δz) = 0.21 m. The Fresnel number of a Rubin vane is w²/(λZ) = 0.057 (dimensionless),
with w the vane width and Z = f²/Δz. The geometric shadow is only 7.6 µm on the detector,
so the visible dark line is a Fresnel diffraction pattern.

To linear order, a vane surrounded by a phase strip φ(**u**) (radians) acts in the pupil
as P ≈ 1 − V(**u**) + iφ(**u**). V is 1 inside the vane and 0 outside. After Fresnel
propagation, the depth of the shadow at its centre is

  ΔI ∝ [ w_V cos(π/4) ∓ (k·OPD)·w_φ sin(π/4) ] / √(λZ),

where w_V and w_φ are the widths of the vane and the strip (m), OPD is the optical path
difference of the strip (m), and k = 2π/λ. The vane term is the same on both sides. The
phase term changes sign, so the strip deepens the shadow on one side and fills it on the
other.

**Magnitude.**

- An extra/intra contrast ratio of 1.5 requires k·OPD·w_φ ≈ 0.2·w_V. That is
  OPD·w_φ ≈ 1e-9 m², for example **about 10 nm of OPD over a 10 cm wide strip**.
- In air at Cerro Pachón, dn/dT ≈ −7e-7 /K. Across the 1.2 m depth of the vane that gives
  about 0.84 µm of OPD per kelvin. 10 nm therefore corresponds to a temperature
  difference between vane and air of **about 0.01 K**.
- Vanes cool radiatively to the night sky, and cold, dense air flowing down them is
  expected.

**Sign.** By the geometric argument, extra-focal vanes being sharper requires a
converging strip: higher refractive index at the vane, so colder air. This sign is still
to be confirmed by simulation.

**Edges.** The Danish kernel is set mainly by the donut edges, which are the outer and
inner edges of M1 (r = 4.18 m and 2.558 m). The same mechanism at these edges would
change the fitted kernel. It could be a thermal sheath at the mirror cell or baffles, or
a static turned-down edge on the mirror figure; again about 10 nm of OPD over about
10 cm is enough. Unmodelled vane shadows may also bias the Danish kernel directly.

## Predictions

1. **Bright flanks.** On the sharper (extra) side, light deflected away from the vane
   piles up just outside the shadow, brighter than the surrounding annulus. A blur
   kernel cannot produce this.
2. **Equivalent width.** The flux deficit integrated across a vane is the same on both
   sides.
3. **Wavelength dependence.** The ratio of the phase term to the vane term scales as 1/λ,
   so the contrast asymmetry is stronger in u and g than in z and y.
4. **Dependence on conditions.** The asymmetry varies with the temperature difference
   between the spider or top end and the air, and with wind, which flushes the sheath.
   It does not scale with seeing.
5. **Field and mode.** CWFS and FAM agree, and the effect is roughly independent of
   field position until vignetting changes the pupil edges.

## Other candidates

The DECam observation disfavours causes specific to Rubin or to Danish, but does not
exclude them:

- an error in the pupil model, such as a misplaced vignetting edge on one side only
  (linked to the pupil-model dependence of the intra/extra Z11 split);
- production reference Zernikes that differ from those of the correctly defocused optic
  away from the field centre;
- asymmetric processing of the intra and extra stamps in the pipeline;
- diffraction from vignetting edges that are not at a pupil conjugate, which is small
  (of order Δz/d, where d is the distance from the clipping surface to focus).

## Outstanding work

- **Simulation.** Add vane obscurations and a per-ray phase e^{iφ(**u**)} to the NUFFT
  Huygens sum in `code/fresnel/fresnel_donut.py`. Sweep OPD over ±30 nm and the strip
  width over 2–20 cm, at ±1.5 mm defocus, with and without 0.7″ seeing. Measure the
  vane contrast and profiles and the Danish kernel on each side. Repeat with phase strips
  at the M1 edges.
- **Data.** Stack the profiles across each vane in the CWFS stamps, intra against extra,
  and look for bright flanks, equal equivalent widths, and the band dependence.
- **Telemetry.** Correlate the vane-contrast ratio and the difference in Danish kernel
  with the top-end temperature minus the ESS (environmental sensor system) air
  temperature, and with wind.
