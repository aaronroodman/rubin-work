# Item 7 — giant donuts, pupil models and the intra/extra Z11 split

> **Status:** Q1 answered, steps 4-6 run; notebook complete at `binning = 1`; no blocker ·
> **Last updated:** 2026-10-06 · **Kind:** handoff

Specification is item 7 of [`notes/todos/todo-ideas.md`](../todos/todo-ideas.md). Study
lives in `wfs/code/giant_donuts/` (Aaron's choice, 2026-10-05).

## Done and committed

- `850c87d` — saved the uncommitted `wfs_giant_donut_fit.ipynb` work before archiving it.
- File moves (Q4, authorized 2026-10-05), staged not yet committed:
  `wfs/docs/danish_pupil_mask_findings.md` → `wfs/docs/giant_donuts_pupil_findings.md`;
  `wfs_giant_donut_fit.ipynb`, `wfs_batoid_pupil_compare.ipynb`, `wfs_diffraction.ipynb`
  → `wfs/notebooks/archive/`. Nothing deleted.

## Settled facts

**Q7 — the 8 mm apportionment. The headline result.** Against the bracketing in-focus
exposure (seq 336, M2 dz Trim −1288.6 µm, camera dz Trim −1168.9 µm), the giant exposures
sit at **±4000 µm on the M2 hexapod and ±4000 µm on the camera hexapod**, an 8000 µm throw
on each. So it is a symmetric **4 + 4 mm split, not camera-only 8 mm** — the case `wfs/`
predicted gives a different pupil (donut span 6.7 mm against 7.0 mm, M2 being powered).
The pupil model must carry both hexapod offsets.

**Q2 — the night, settled label-independently.** `day_obs` **20251023**, BLOCK-T626, `r`
band, 60 s. Clean giant exposures: **extra 337, 338; intra 340, 341**. Aaron confirms Bryce
used 337 and 340, which are in this set.

| candidate | verdict | numbers it was judged on |
| --- | --- | --- |
| 20251023 T626 | **chosen** | 4 clean exposures, 2 per side; only dof0/dof5 move across seq 330–354; rotator angle 0.012 to 0.173 deg; bracketing in-focus `psf_sigma_median` 2.22 and 2.42 pixels |
| 20250520 | rejected, no giants | camera dz Trim spans 12000 µm with 15 DOF moving including bending modes 30–34 — an active-optics night. No ±4000 µm plateau |
| T389 | not yet checked | deferred; 20251023 already satisfies the scope |

The item's Q2 answer says "Bryce used 20250520". That was a misremembered night —
Aaron corrected it to 20251023 on 2026-10-05. The item has been updated.

**Exposures rejected, and why it matters that selection is Trim-first:**

- **seq 339** sits at the giant intra Trim state (−4000/−4000) but is 30 s with
  `science_program` "unknown" — not part of the T626 giant sequence. Trim alone would have
  wrongly admitted it.
- **seq 351, 352**, labelled `intra_8mm_m1m3_b4`, show **no bending-mode motion in the
  Trim at all**. Across seq 330–354 only dof0 and dof5 ever move. Either the b4 mode was
  commanded to M1M3 as forces outside the OFC aggregated DOF, or it was never applied. The
  aggregated Trim cannot distinguish these, so they are excluded pending a check of the
  M1M3 applied forces. **This check is still open.**
- seq 334/335 and 346/347 are ordinary FAM at **camera-only ±1500 µm** (dof5 only, dof0
  flat). Same night, so it carries both camera-only FAM and the 4+4 split giant donuts —
  a built-in contrast worth using.

**Q1 — does blitz handle 8 mm donuts? Mostly yes, and better than the item assumed.**

- The real tags are **`blitz-prototype-v1` and `blitz-prototype-v2`**. There is **no v3**
  (`git fetch --tags` on 2026-10-05) and no tag named `donut_blitz_v2` — the item's name
  for it does not exist. v2 (2026-09-11) is the newest.
- Detection is **size-agnostic**: `blindDetectTask` builds an annular template from
  `donutRadius` and scales `min_distance` / `exclude_border` by it, with an override
  argument. An 8 mm donut is a parameter change, not a code change.
- **`cameraOffset` and `m2Offset` are both config fields**, in meters, signed per exposure.
  So blitz v2 *can* model the measured 4+4 mm split directly. This contradicts the item's
  note that ts_wep could not apportion the offset — that capability arrived in v2.
- **`modelSpiderShadows` is already a config field** (default False). It gates `rtp`, which
  gates danish's `spider_angle`. Step 6 is a config toggle, as the item hoped.
- **Rotator angle is ≈ 0 deg** for all four exposures (337: +0.012, 340: +0.149 deg). The
  shipped pupil YAMLs were built at `rtpp0`, i.e. rotator 0, so the spider geometry matches
  the data almost exactly. No large rotator extrapolation.

**The blur bound (Q5).** `wavefrontFittingTask` hardcodes `fwhm=[0.1, 5.0]` arcsec with the
start value at 1.0, **not** the 0.5–1.5 arcsec of the production config the item cites. The
5.0 arcsec ceiling is exactly the runaway regime `wfs/` warned about. It is a literal in
the source, not a config field, so bounding it means patching the task — the "overriding a
pipeline config" case the item anticipated. `binning` defaults to 2, which also keeps the
galsim FFT out of trouble — but **the notebook now runs at `fit_binning = 1`** so the model
images share the data stamp's 10 µm pixel; see the Tried-and-rejected entry on binning.

**Thermal telemetry for the later pass**, recorded now as the item asks: camera average
temperature ≈ 9.01 to 9.06 °C across the four exposures; M1M3 radial gradient ≈ −0.0017 to
−0.0020 °C/m; sonic temperature ≈ 11.02 to 11.08 °C; altitude 65.4 to 67.4 deg; azimuth
267.6 to 268.5 deg.

## The finding that changes the plan

**ts_wep's blitz does not use danish's pupil YAMLs at all.** The factory is built with
`mask_params=_INSTRUMENT.maskParams`, where `_INSTRUMENT` is a module-level singleton from
`policy/instruments/LsstCam.yaml`. That file carries `diameter: 8.36 m` and M1 inner
`2.558 m` — i.e. **v3.14 numbers**; v1000 is `pupilSize 8.33 m`, M1 inner `2.5833 m`. There
is no v1000 variant of ts_wep's `maskParams`.

So step 4's v3.14-against-v1000 comparison is **not a file swap**. It needs v1000
`maskParams` polynomials generated from the v1000 batoid model (cubic-in-θ centre and
radius per optical-element edge). That is real work the item did not anticipate, and it is
the one place new code is genuinely required. The archived
`wfs_batoid_pupil_compare.ipynb` has the batoid-boundary tooling to build them from.

This also bears on **item 5**, which tests the same v3.14/v1000 question through the MIW:
if item 5 goes through ts_wep rather than danish directly, it faces the same missing
`maskParams`. Worth noting there. (Another session has `aos/code/miw/compare_pupil_models.py`
in progress, untouched here.)

## The physics result so far

**Josh's ring is real, it is extra-focal, and it looks like an OPD term rather than a
pupil-model error.** Measured on seq 337 (extra) and 340 (intra), R22_S10, field radius
0.235 deg:

| zone (normalised radius) | extra | intra | extra − intra |
| --- | --- | --- | --- |
| inner ring, 0.62–0.70 | 1.0073 | 0.8532 | **+0.1541** |
| outer, 0.94–1.00 | 0.8803 | 0.9760 | **−0.0957** |

All dimensionless normalised flux, relative to the mean over the illuminated annulus,
at the converged 1.5 pixel per bin. The intra-minus-extra difference peaks at **−0.3330
at normalised radius 0.648**, just outside the inner pupil edge (0.612 v3.14, 0.620
v1000) and **not** at the rim.

`ring_excess` also discriminates the two inner radii from the images alone: referenced to
v1000's 0.6204 edge the extra-focal ring reads **+0.0132**, referenced to v3.14's 0.6120 it
reads **−0.0626** because the band lands partly in the central hole. Independent support
for v1000's inner radius, separate from the fits.

Flux is conserved and redistributed radially in **opposite directions on the two sides of
focus**. That is the OPD signature from the item's step 7: a mask boundary error would move
an edge *position* the same way on both sides rather than swap the flux balance between
zones. The location at the inner edge is consistent with a turned-down edge inside M1 —
which is also where v1000 moves the inner radius, outward by 25.3 mm.

**The donut sizes do NOT confirm the 4+4 mm split — an earlier claim here was wrong.**
Measured diameters are **6.95 mm extra and 6.85 mm intra** (bounding box), with the
profile's fitted outer edge at 341.5 and 335.7 pixel, i.e. 6.83 and 6.71 mm. `wfs/`
predicts 6.7 mm for a 4+4 mm split and 7.0 mm for camera-only 8 mm, so the measurements sit
between the two and nearer the camera-only value. The images do not discriminate on size
alone; **the Trim is the evidence for 4+4 mm.** A previous version of this handoff quoted
6.85/6.72 mm and called it independent confirmation — those numbers were not reproducible
and the conclusion did not follow.

**Caveats.** One donut per side, one sensor, one night — the item asks for a distribution.
The 5.8 pixel (0.06 mm) outer-edge difference between the two donuts is removed by the
normalisation and has not been separated from the effect.

## Binning: both projections are resolution-limited, not noise-limited

Chosen by convergence scan (`rp.bin_convergence`), not by rule of thumb. Per-pixel noise
is 47.7 electrons against an annulus signal of 1485.6 electrons per pixel, so even a
1 pixel radial bin reaches a signal-to-noise ratio of about 1268 (dimensionless) per bin.
Shrinking bins therefore costs nothing but resolution gains are real.

- **Radial: 1.5 pixel per bin** (287 bins over the 430 pixel stamp half-size). The
  ring-zone mean converges there and holds at 1.0 and 0.5 pixel. Coarser bins are
  actively wrong: 2.15 pixel per bin reads the ring excess about 12 per cent high, and
  3.6 pixel per bin collapses it to +0.054 (dimensionless) and mislocates the peak to the
  outer edge. The 90–10 per cent outer-edge roll-off width, 12.9 pixel, also stops
  shrinking here.
- **Azimuthal: 0.25 deg per bin** (1440 bins). A 0.05 m spider vane at 0.8 of the pupil
  radius subtends only 0.86 deg, so 1.0 deg resolves a vane with a *single* bin;
  0.25 deg puts about 3.4 bins across it and so resolves its profile. Peak-to-peak has
  converged — 1.0150 at 0.25 deg against 1.0182 at 0.20 deg (dimensionless, extra-focal)
  — and at 122 pixels per bin peak-to-peak over median-error is still 82 (dimensionless).

The **lag-1 autocorrelation is diagnostic azimuthally only.** Radially it reads 0.98–1.00
at every bin width, because a smooth monotonic curve always correlates between neighbours;
an earlier version of this scan reported it and it was useless. The radial discriminator is
the edge roll-off width, validated on a synthetic donut with a known 13.0 pixel edge
(reads 16.1 pixel at 5.4 pixel per bin, converges to 12.9 pixel by 1.5 pixel per bin).

## Azimuthal profiles: the extra-focal donut is far more structured

At 0.25 deg per bin, peak-to-peak normalised flux is **1.0150 extra-focal against 0.5711
intra-focal** (dimensionless) — against 0.4392 and 0.2632 at the 5 deg bins this study
started with, so coarse bins were hiding most of the structure. The intra-minus-extra
difference peaks at **+0.5920 (dimensionless) at 140.4 deg** (counterclockwise from the
+x pixel axis) against a robust scatter of 0.0543. Localising that to a sector of M1 needs
more than one donut pair.

## The 2-D residual images: the sign flip, measured directly

Per pupil zone, as a fraction of the model's annulus level (dimensionless), ranges over
the four fit configurations:

| zone (normalised radius) | extra, signed mean | intra, signed mean |
| --- | --- | --- |
| inner, 0.62–0.70 | +0.033 to +0.047 | −0.030 to −0.035 |
| flat, 0.70–0.94 | −0.032 to −0.035 | −0.003 to −0.001 |
| outer, 0.94–1.00 | −0.006 to +0.009 | −0.002 to +0.009 |

The inner zone is positive extra-focally and negative intra-focally in **every** case,
with the flat zone taking the compensating deficit on the extra side. Neither the pupil
model nor the spiders shifts it — which is what makes it a residual OPD term rather than
mask geometry, and it answers next-action 1 below for the radial direction. The nMAD is
largest at **both** pupil boundaries (about 0.19 inner, 0.17 outer) and smallest in the
flat middle (about 0.12), so the forward model fails hardest at the edges.

## The fit result: v1000 halves the Z11 split

Committed in `449d928`. `wfs/code/giant_donuts/fit_giant_donut.py` drives blitz's own
factory and danish model directly on stamps cut from the raws, rather than running the
butler pipeline — this engineering night has no reference catalog or matched calibs, and
the question is about the forward model. The blitz checkout is
`/sdf/home/r/roodman/u/LSST/packages/ts_wep_blitz` at `blitz-prototype-v2`, branch
`giant-donuts-study`.

**Q1 is answered: yes.** blitz detects, cuts and fits the 8 mm donuts end to end with the
4+4 mm split configured (`cameraOffset` and `m2Offset` both 4.0e-3 m, signed per side).
Both sides report `fit_success`.

R22_S10, seq 337/340, **binning 1**, unpaired, field angle (−0.263, −0.062) deg:

Free blur, bound 0.1 to 3.0 arcsec:

| pupil model | spiders | chi2/dof extra | chi2/dof intra | Z11 split [µm wf] |
| --- | --- | --- | --- | --- |
| v3.14 | off | 8.850 | 4.418 | **+0.8145** |
| v3.14 | on | 7.117 | 3.775 | +0.7972 |
| v1000 | off | 8.797 | 4.423 | **+0.3725** |
| v1000 | on | 7.018 | 3.764 | +0.3593 |

Blur fixed at the measured seeing, 0.7 arcsec:

| pupil model | spiders | chi2/dof extra | chi2/dof intra | Z11 split [µm wf] |
| --- | --- | --- | --- | --- |
| v3.14 | off | 8.937 | 4.817 | +0.8144 |
| v3.14 | on | 7.294 | 4.687 | +0.7953 |
| v1000 | off | 8.890 | 4.853 | +0.3690 |
| v1000 | on | 7.195 | 4.700 | **+0.3522** |

dof is 741295 for every row. Z11 split is intra minus extra, in µm of wavefront.
**The Z11 split is robust to the blur treatment** — it moves 0.0001 to 0.007 µm wf
between the two, against the 0.25 µm wf that remains unexplained.

These replace the earlier binning-2 table, where `chi2/dof` ran 12–15 extra and 6–8 intra
on dof 184015. Going to `binning = 1` cut `chi2/dof` by about 40 per cent at four times
the pixels and moved the Z11 splits by 0.01 µm wf at most, so **the physics conclusions
are unchanged by the binning**; only the pixel-frame spider numbers were.

**The two effects are orthogonal, which is the useful part.** The pupil model moves the
Z11 split (0.81 to 0.37 µm, a 54.3 per cent reduction) and barely moves `chi2/dof`.
Modelling the spiders moves `chi2/dof` (about −20 per cent extra, −15 per cent intra) and
barely moves Z11. So v1000 is the better pupil *geometry* for the Z11 diagnostic, and
spiders are a genuine image-fidelity improvement that does not confound it.

**Step 6 is answered:** spiders can be included with good fidelity. `modelSpiderShadows`
is a plain config toggle and switching it on improves the fit at both sides of focus.

**This does not close item 7's physics question.** v1000's 0.3593 µm still exceeds the
0.10 µm diffraction term by 0.26 µm, and the observed split is about 0.30 µm. So a
residual OPD term remains after the best pupil model and the spiders, which is what the
radial profiles independently point at.

**Z22 splits far less than Z11, and in the opposite sign.** Secondary spherical splits
−0.0744 µm wf under v3.14 and −0.0147 µm wf under v1000 (spiders off, free blur), against
Z11's +0.8145 and +0.3725 µm wf. So the Z11 split is not a general spherical-family mismatch;
the v1000 pupil improves both, but Z11 is where the residual lives.

**Z11 is robust; Z4 was corrupted by a basis bug, now fixed.** Z11 moves less than
0.005 µm across loosened blur bound, tightened tolerance (1e-3 to 1e-8), the
radius-consistency choice below, and the ε fix. See the ε-basis bug section below for Z4.

## The ε-basis bug: Z4 was not a degeneracy, it was wrong

**A previous version of this handoff reported a v1000 Z4 split of +4.1147 µm wf and
explained it as a pupil-scale/defocus degeneracy. That explanation was wrong; it was a
bug.** `fit_stamp` called `batoid.zernikeTA` with `eps=telescope.pupilObscuration`, and
**both** YAML models carry that as the nominal 0.612 — v1000 keeps 0.612 even though its
traced inner edge is at 0.6204. danish meanwhile got `R_inner/R_outer = 0.6204` from the
traced mask. So under v1000 the reference wavefront and the fitted deviation sat on
**different annular Zernike bases**, and Z4, the term most sensitive to annulus width,
absorbed the entire mismatch. Under v3.14 the two bases agree exactly, which is precisely
why only v1000 looked broken and the artifact was mistaken for physics.

Fix: take ε from `Instrument.obscuration`, the same number danish uses. The v1000 Z4 split
collapses from **+4.1147 to +0.0205 µm wf**. Z11 moves 0.005 µm wf (+0.3669 to
+0.3618), so no Z11 conclusion changes.

**Lesson for item 5 and anything else swapping pupil models:** swapping a pupil model in
blitz is *three* things, not two — the analytic mask, the batoid telescope, **and** the ε
of the annular Zernike basis. A YAML's `pupilObscuration` is a nominal label, not the
traced boundary, and the two disagree by 8.4e-3 (dimensionless) for v1000.

Z4 is still not a measurement of the telescope's defocus: these fits are unpaired and
single-sided, so Z4 trades against flux, centroid and blur. Report it as a consistency
check on the pupil basis.

## The blur: relaxed to 3.0 arcsec, then fixed at the DIMM seeing

**Relaxing the ceiling from 1.5 to 3.0 arcsec did not resolve the intra/extra asymmetry —
the pinning was a symptom, not the cause.** Off the bound the intra fits sit at 1.512 to
1.691 arcsec against 0.992 to 1.020 arcsec extra-focally, in every pupil configuration.
The intra donut genuinely prefers about 60 per cent more blur.

That is roughly twice the measured seeing: the DIMM read 0.65 to 0.80 arcsec on these
visits and nearby in-focus acquisition images had about 0.9 arcsec. So a second full set
of fits holds the blur **fixed at 0.7 arcsec** (`fit_stamp(fwhm_fixed=...)`, imposed as a
degenerate bound so danish's parameter packing is untouched).

**Fixing the blur costs the intra side 25 per cent in `chi2/dof` and the extra side 3 per
cent** (v1000 spiders on: 3.764 → 4.700 intra, 7.018 → 7.195 extra). The excess blur the
intra fit wants is therefore real unmodelled structure on that side, and it is not
representable by the fitted Zernikes through Z26.

## Spiders: the azimuthal projection was the wrong cut

**Superseded result.** An earlier pass measured vane depth as the deepest azimuthal dip
below the profile median and concluded the modelled vanes were about half as deep as the
data's. That conclusion was an artifact of the projection, not a statement about the
optical model.

The vanes run from M2's support ring outward, so each shadow is a **chord offset from the
pupil centre**, not a radial spoke, and its azimuth drifts with radius. Measured drift
between normalised radius 0.68 and 0.92 is **1.62–2.16 deg extra-focally and 1.21–2.28 deg
intra-focally**, against the **0.86 deg** a vane subtends at 0.8 of the pupil radius. The
drift is 2–3 times the vane width, so a fixed-annulus azimuthal projection averages each
vane across its own drift and cannot resolve it at *any* azimuthal bin size. The data's
fitted impact parameters run **19.2–33.2 pixel**, confirming offset chords rather than
spokes, with the two vane families at line directions near 44.9/135.0 deg extra-focally.

`wfs/code/giant_donuts/spider_trajectory.py` replaces the projection: trace each shadow's
centre at seven narrow radii, fit a total-least-squares line, then profile perpendicular to
that line. Validated by `test_spider_trajectory.py` on a synthetic pupil with known vane
geometry — depth recovered as 1.0000 (dimensionless) exactly, impact parameter to 0.9 pixel
of truth with 1.7 pixel spread among the eight shadows, direction with a **common**
+0.32 deg bias that cancels in model-minus-data differences.

### Depth and width along the shadow

Fractional flux deficit referenced to the unobscured shoulder 10–20 pixel either side
(dimensionless), mean over eight shadows:

Models and data are both on the native 10 µm detector pixel (`fit_binning = 1`), so the
pixel numbers compare directly:

| | extra-focal | intra-focal |
| --- | --- | --- |
| data | **0.8603 ± 0.0114** | **0.5264 ± 0.0223** |
| spiders on, free blur | 0.5878–0.5891 | 0.3919–0.3943 |
| spiders on, blur fixed 0.7 arcsec | 0.7427–0.7458 | 0.7422–0.7452 |

Spiders-off fits are not traced at all — with no vanes in the pupil there is nothing to
follow, and the dip finder latches onto diffraction rings and Zernike modulation instead.

**The discrepancy is width, not depth.** Data shadow FWHM is **11.35 ± 0.43 pixel extra
and 10.11 ± 0.33 pixel intra**; the fixed-blur models give **5.10 and 5.13 pixel**, a
factor of 2.0–2.2 too narrow. A wide, nearly opaque shadow is what a vane plus its
diffraction skirt looks like, and the geometric shadow model has no diffraction in it. With
blur fixed at 0.7 arcsec the depth comes out roughly right (0.743–0.746 extra,
0.742–0.745 intra) — 13 per cent shallow against the data extra-focally but
**overshooting intra-focally by 41 per cent** — so the free-blur fits were using inflated
blur to pad a too-sharp vane out toward the right depth.

### The spiders sit 4–6 pixel too far out in the model

In every spiders-on configuration the model's shadows sit **further** from the donut centre
than the data's:

| | d angle [deg] | d \|impact\| [pixel] | d azimuth [deg] |
| --- | --- | --- | --- |
| extra, spiders on, four configs | +0.315 to +0.380 | +4.25 to +5.45 | −0.012 to +0.001 |
| intra, spiders on, four configs | −0.266 to −0.395 | +5.19 to +6.40 | −0.042 to +0.009 |

Model minus data, median over eight shadows. v1000 is about 1 pixel closer to the data than
v3.14 in every pairing, consistent with its wider inner obscuration.

**Not a rotation and not an azimuth error — a radial offset.** Direction agrees to
0.27–0.40 deg and azimuth to 0.042 deg, while the impact parameter is off by 4–6 pixel the
same sign on both sides of focus and under both pupil models. At the 10 µm detector pixel
5 pixel is 0.05 mm, or about **0.06 m at the pupil** (5 pixel / 341.5 pixel outer edge ×
4.18 m pupil radius) — about three times the method's own 1.7 pixel spread.

**This supersedes an earlier "11 pixel inward" reading, which was a binning artifact.**
The fits ran at blitz's default `binning = 2`, so the model images were on a 20 µm pixel
while the data stamp stayed on 10 µm; the comparison differed by a factor of two for that
reason alone. `fit_binning = 1` puts every image on one grid. The sign also flipped —
inward became outward — which the binning factor alone would not do, so the earlier
reading was wrong about direction as well as magnitude. At three times the method's spread
the surviving offset is weaker evidence than 11 pixel appeared to be, and wants a second
sensor before it is used to change the vane geometry.

## v1000 maskParams: generated

`wfs/output/giant_donuts/maskParams_v1000.yaml` (not committed, `*/output/*` is ignored).
Validated by regenerating v3.14 and recovering the shipped coefficients: **M1 outer to
0.42 mm on 4.18 m, M1 inner to 0.02 mm on 2.558 m**. Generated v1000 values against the
batoid model: M1 inner 2.58397 against 2.5833 m (+0.7 mm), M1Baffle1 4.16479 and M1Baffle2
4.16460 against 4.165 m (−0.2 and −0.4 mm).

Two things to know about it:

- **Only M1 is refitted.** M3's aperture also differs between the models (outer 2.508 to
  2.48511 m, inner 0.55 to 0.52735 m) but it sits inside M1's inner shadow and **never sets
  the boundary** — traced on-axis through the full system, the surviving annulus is
  2.5580–4.1796 m in v3.14 and 2.5840–4.1650 m in v1000, with the inner edge tracking M1's
  annulus to within the ray grid's 0.4 mm. M2, L1 and the filter are byte-identical between
  the models and are copied through.
- **M1's outer coefficient is redundant in v1000.** The generated 4.17247 m is 7.5 mm off
  M1's 4.18 m rim because the new baffles at 4.165 m now clip inside it, so the measured
  outer edge is the baffle. Physically right, but it means the v1000 outer edge is the
  baffle entry, not M1's.

## In progress

**Uncommitted, the `binning = 1` pass.** `fit_binning` is now a notebook parameter set to
1, so the fitted stamps and model images share the data stamp's native 10 µm detector
pixel. The spider section skips `spiders = False` fits. Every fit number in this handoff
and in the notebook's interpretation has been regenerated; the figures and PDFs are rerun.
Modified: `wfs/notebooks/giant_donuts/giant_donut_radial_profiles.ipynb`,
`wfs/code/giant_donuts/fit_giant_donut.py` (docstring only), this file.

Nothing uncommitted in this repo. Commits in order: `fe9fa1a` (selection module, file
moves, todo edits), `1f37ac1` (maskParams generator, radial profiles), `8f79f50` (the
profile measurement), `ee07cc6` (profile result in this handoff), `449d928` (fit driver
and image helpers), `6369dc9` (the four fits and azimuthal profiles in the notebook),
`05ac889` (bin sizes by convergence, table of contents, residual images), `d235b9a` (the
ε-basis fix and the fixed-blur fits). In the blitz checkout, `ab85fc3d` on branch
`giant-donuts-study`.

`wfs/notebooks/giant_donuts/giant_donut_radial_profiles.ipynb` is complete and fully
executed: 0 unexecuted code cells, 0 error outputs, 13 embedded figures, a 13-entry table
of contents with every anchor resolving. Twelve PDFs in `wfs/output/giant_donuts/`
(gitignored): `radial_profiles_`, `azimuthal_profiles_`, the blur-suffixed
`{radial,azimuthal}_fits_{free,fixed}blur_` and `residual_images_{free,fixed}blur_`, and
the spider set `spider_trajectories_data_`, `spider_placement_`,
`spider_cross_profiles_{extra,intra}_`, all suffixed `R22_S10`.

The study's code is `wfs/code/giant_donuts/`: `donut_images.py` (ISR, donut finding,
stamps), `radial_profile.py` (radial and azimuthal projections, bin convergence),
`fit_giant_donut.py` (the blitz/danish driver and pupil override), `spider_trajectory.py`
(shadow tracing and cross-vane profiles) with `test_spider_trajectory.py`, and
`gen_mask_params.py`.

Run the notebook with `--ExecutePreprocessor.kernel_name=python3`: its stored kernelspec
is `lsst`, which is not registered on s3df.

## Next concrete action

1. **Why is `chi2/dof` twice as large extra-focally as intra-focally?** Consistent across
   all four configurations (about 7-9 against 3.8-4.9 at `binning = 1`). The residual
   images now localise the
   *radial* part to both pupil boundaries, with the inner zone flipping sign between
   sides, but they do not explain the extra/intra magnitude asymmetry itself. Note the
   extra-focal donut is also the azimuthally more structured one (peak-to-peak 1.0150
   against 0.5711, dimensionless) — the two asymmetries may be the same thing.
2. **Convert the radial-profile zone statistics into an implied wavefront amplitude in
   µm**, so the +0.1541 (dimensionless) normalised-flux ring can be set against the
   0.26 µm of Z11 split that v1000-with-spiders leaves unexplained (0.3593 µm measured
   less about 0.10 µm from diffraction). These are currently two pieces of evidence for
   the same thing in different units, and this is the step that joins them.
3. Extend to seq 338 and 341 and to more sensors for a distribution rather than one pair.
   `fit_giant_donut.py --detector` already takes any sensor.
4. Test the figure roll-off hypothesis directly (item 7 step 7) by perturbing M1's inner
   edge in the batoid model and refitting, to see whether it absorbs the residual Z11.
   This is the main remaining physics step.
5. Explain the intra/extra **blur** asymmetry. Now better characterised and still
   unexplained: 1.51 to 1.69 arcsec intra against 0.99 to 1.02 arcsec extra with a
   3.0 arcsec ceiling, i.e. not a bound artifact, and forcing both to the measured
   0.7 arcsec costs intra 25 per cent in `chi2/dof` against extra 3 per cent.
6. **Get a second sensor before acting on the 4-6 pixel outward spider offset.** The
   model's vanes sit about 0.06 m too far out at the pupil in every configuration, with
   direction and azimuth correct, but that is only about three times the method's own
   1.7 pixel spread — much weaker evidence than the 11 pixel binning artifact it
   replaced. If a second sensor confirms it, the places to look are the vane mount radius
   on M2's support ring and the spider's axial position in `LsstCam.yaml` / the Rubin
   YAMLs, which set how the vane projects into the pupil.
7. **The model's spider shadow is about 2.0-2.2 times too narrow** (5.10 pixel FWHM against
   the data's 11.35 pixel extra-focally) while the fixed-blur depth is roughly right. Candidates
   are the 0.05 m vane width in `LsstCam.yaml` and, more likely, diffraction at the vane
   edges, which the geometric shadow model omits. This is probably the same deficiency that
   forces the fitted blur up, so it couples to action 5.
8. Check the M1M3 applied forces for seq 351/352 to settle whether the b4 mode was applied.

## Decisions needed from Aaron

**None.** All three earlier decisions are made and done: the second ts_wep checkout
(2026-10-05, re-cloned with `GIT_LFS_SKIP_SMUDGE=1`, confirmed on `blitz-prototype-v2`
at `9651cd23`), the v1000 `maskParams`, and the file moves.

The blur-bound patch is committed **in the blitz checkout, not this repo**:
`ab85fc3d` on branch `giant-donuts-study`, adding `fwhmMin`/`fwhmMax` config fields in
place of the hardcoded `fwhm=[0.1, 5.0]`. Worth upstreaming if the blitz authors want it.

The shared `~/u/LSST/packages/ts_wep` on `develop` has **not** been touched.

## Tried and rejected, and why

- **Fitting at blitz's default `binning = 2` while measuring the data stamp unbinned —
  rejected, it produced a wrong result that survived a full write-up.** `fit_stamp` returns
  `img` and `model_img` on the binned grid, so the model images were on a 20 µm pixel while
  the stamp the spider tracing measured stayed on 10 µm. Everything expressed in normalised
  radius was fine; everything in pixels was not. It reported the model's vanes as 11 pixel
  **inward** of the data's and 3.7× too narrow; at `binning = 1` they are 4–6 pixel
  **outward** and 2.0–2.2× too narrow. The sign flip is the warning: a factor of two alone
  would not reverse a direction, so the comparison was not merely mis-scaled. Aaron spotted
  it from the placement plot. `binning = 1` costs four times the pixels per fit (about
  25 min for all sixteen fits) and moves the Z11 splits by at most 0.01 µm wf, so there is
  no reason to go back. Anything quoted in pixels must come from one grid.
- **Measuring spider depth from a fixed-annulus azimuthal projection — rejected.** The
  vanes are offset chords, so a shadow's azimuth drifts 1.2–2.3 deg across the annulus
  against the 0.86 deg it subtends. No azimuthal bin size fixes this: the smearing comes
  from averaging over radius, not from the binning. It made the data's vanes look 0.76 deep
  when they are 0.86, and made the models look uniformly too shallow when in fact their
  depth at fixed blur is roughly right and their *width* is the problem. Profile along the
  fitted shadow trajectory instead (`spider_trajectory.py`).
- **Widening the trace radii to 0.64–0.96 to beat down the impact-parameter error —
  rejected, it made things worse.** The outermost radius lands in the edge roll-off, which
  biases the dip centroid; the common direction bias grew from +0.32 to +0.39 deg. The
  bias is common to both vane families so it cancels in model-minus-data differences, which
  is all the placement numbers are used for. Trace radii stay inside 0.68–0.92.
- **Tracing spiders in the spiders-off models at all — rejected, and the gating that
  propped it up with it.** The first attempt gated on the trace-time dip depth, which does
  not work: a vane-free model still has diffraction rings and Zernike modulation for the dip
  finder to latch onto, so it reports several per cent of depth. A second gate on the depth
  measured *along the fitted trajectory* did work (0.011–0.015, dimensionless, with no
  vane), but the whole exercise is pointless — a pupil with no vanes has no vane placement
  to compare. The spider section now skips `spiders = False` fits outright.
- **Taking the `8mm` labels at face value — rejected.** `observation_reason` is free text
  and wrong in both directions here: seq 339 has the giant Trim state without being a giant
  exposure, and 351/352 claim a bending mode the Trim does not show. Select on Trim first,
  then use metadata only to reject.
- **Using seq-330 as the Trim baseline — rejected** in favour of the per-DOF night median.
  Both give the same ±4000 µm throw, but a single reference exposure is fragile if that one
  exposure is itself offset. The median is unaffected by the symmetric defocal excursions.
  Reported numbers above use the bracketing in-focus exposure for interpretability.
- **Assuming the item's `donut_blitz_v2` tag name — rejected, it does not exist.** Always
  `git fetch --tags` and read the actual tag list; Aaron also asked about a v3, which does
  not exist either.
- **Assuming spiders needed new pupil code — rejected.** The item was right that it is a
  toggle. An early worry that `spider_angle=group.rtp` is passed unconditionally (so
  spiders might always be on) was **unfounded**: the gating is one level up, where
  `rtp_deg` is set to `None` unless `modelSpiderShadows` is true.
- **Expecting to swap danish pupil YAMLs — rejected, see above.** blitz never reads them.
  Checking which pupil model a fit actually used means reading `maskParams`, not the YAML
  filename. The item's note that danish 1.3's default *is* v1000 remains true but is
  irrelevant inside blitz.
- **20250520 — rejected as a giant-donut night**, on the Trim, not the label.
- **A smoothed-peak donut finder — tried and rejected.** `uniform_filter` at a 200 pixel
  scale landed about 80 pixels off the true centre, which truncated the stamp and put the
  fitted outer edge at 157 pixels instead of the true 343, making the profile meaningless.
  Replaced by connected-region labelling above sky, which finds the giant donut directly
  and returns its diameter as a by-product. **`stamp_half_pix` must exceed about 347
  pixels**; the first attempt used 320 and silently cut the donut off.
- **Refitting M3's `maskParams` — attempted, then dropped as unnecessary.** Neither the
  surface nor the pupil frame reproduced the shipped M3 coefficients (best residual 1.04 m
  on the outer edge), because the shipped non-M1 radii are back-projected along the chief
  ray with per-element magnifications (measured about 2.63 for M3, 2.21 for L1, 20.2 for the
  filter). Rather than reproduce that convention, note that M3 never sets the pupil
  boundary, so it does not need refitting. **Do not spend time on the general frame
  convention** unless an element other than M1 starts to matter.
- **`ring_excess` referenced across the inner edge — rejected, it was measuring the wrong
  thing.** Differencing the band outside the edge against the band inside put the reference
  in the central hole, so the statistic read +0.83 on a ring-free synthetic donut. Now
  referenced to a local baseline further out in the annulus: reads +0.0001 with no ring and
  responds linearly to injected amplitude.
- **`common/camera_utils.py` does not exist.** The item cites it for `pixel_to_focal`; that
  helper was defined inline in the archived notebook. Use the camera geometry API
  (`getTransform(PIXELS, FIELD_ANGLE)`) directly.
- **Swapping only `maskParams` to change the pupil model — rejected, it is half the
  swap.** blitz carries the pupil **twice**: the analytic mask danish uses
  (`Instrument.radius`, `Instrument.obscuration`, `Instrument.maskParams`) *and* a real
  batoid telescope, `LSST_{band}.yaml`, hardcoded as an f-string in `donutBlitzFamTask`
  and `donutBlitzMonolithTask` with no config field. The telescope supplies the reference
  wavefront via `batoid.zernikeTA` and is z-shifted by `withGloballyShiftedOptic` to apply
  the defocus. Changing only the mask leaves the reference wavefront on the old model.
  `pupil_override` changes both together and clears the `telescope_by_offsets` memo, which
  would otherwise hand back shifted telescopes built from the previous model.
- **`LSST_r.yaml` is byte-equivalent to `Rubin_v3.14_r.yaml` in pupil geometry**
  (`pupilSize` 8.36 m, `pupilObscuration` 0.612). So blitz today is self-consistently
  v3.14 across both representations — worth knowing before assuming the two disagree.
- **Trusting v1000's `pupilObscuration` — rejected.** v1000 keeps the nominal 0.612 even
  though its traced inner edge moves outward by 26.0 mm, giving a true obscuration of
  0.6204. Use the traced boundary, which is what `gen_mask_params` fits.
- **Leaving `Instrument.radius`/`obscuration` at the v3.14 values while injecting v1000
  `maskParams` — rejected as an inconsistency, but note it barely matters for Z11.** It
  hands danish `R_outer`/`R_inner` from one model and mask circles from another; Z4
  absorbed the mismatch (extra-focal +1.44 against +0.64 µm). Setting both from the traced
  M1 edges fixes the inconsistency, and Z11 moves only 0.002 µm either way. Z4 gets
  *worse* (±4.11 µm), which is the degeneracy above, not a regression.
- **Blaming the loose solver tolerance for the poor `chi2/dof` — rejected.** blitz's
  `lstsqKwargs` default of `xtol=ftol=gtol=1e-3` stops a giant-donut fit after about 5
  function evaluations, which looked like the cause. Tightening to 1e-8 raised `nfev` to
  13-20 and moved Z11 by 0.001 µm. `chi2/dof` stayed at 14.99 and 7.13, so the fit is at a
  genuine local minimum and the high `chi2/dof` is **model mismatch, not non-convergence**
  — consistent with a residual OPD term. `--tol` is exposed to re-check this cheaply.
- **A late `sys.path.insert` to pick up the blitz checkout — rejected, it silently fails.**
  `lsst` and `lsst.ts` are namespace packages: the first import of `lsst.ts` freezes its
  `__path__` from whatever `sys.path` held then, and the shared EUPS `ts_wep` (on
  `develop`, no `blitz` subpackage) is already on `PYTHONPATH`. A later insert is ignored
  and `lsst.ts.wep.blitz` raises `ModuleNotFoundError`. **Deleting the `lsst.ts.wep`
  entries from `sys.modules` does not help** — the finder consults the frozen parent
  `__path__`. The insert must happen before *any* `lsst` import, so it sits at module
  scope. `_import_blitz` then asserts the imported file really is under the checkout,
  comparing **resolved** paths because `~/u` is a symlink to `/sdf/data/rubin/user/roodman`
  and a literal substring check fails.
- **Running the butler blitz pipeline for Q1 — deliberately not done.** It needs a
  reference catalog and matched calibrations this engineering night does not have, and the
  question is about the forward model. Driving blitz's own factory and danish model gives
  the same answer with no pipeline infrastructure. A pipeline run is still the right test
  for the detection and pairing stages if those ever matter here.
- **Reporting numbers from a throwaway shell as if the notebook had produced them —
  a real mistake, corrected 2026-10-05.** The radial-profile results were first obtained by
  running the profiling code in a scratch process and were written into the notebook's
  Interpretation cell and this handoff, but **the notebook itself was committed unexecuted**
  (every `execution_count` null, no outputs, no PDF). On finally executing it, most zone
  statistics shifted in the third decimal and the donut diameters were wrong by 0.10 and
  0.13 mm, which killed the "images independently confirm the 4+4 mm split" claim. The
  physics conclusion survived; one supporting claim did not. **Execute a notebook before
  quoting its numbers, and check `execution_count` is non-null before committing.**
- **The notebook was not runnable as first committed**, two separate bugs. It called
  `repo_root()` imported *from* `common.utils` before the repo root was on `sys.path` —
  chicken-and-egg, `ModuleNotFoundError`. The repo's established notebook idiom (see
  `aos/notebooks/fam_processing/*.ipynb`) walks up to the topic dir and inserts paths before
  any repo import; use that, not `repo_root()`. It also called `fig.tight_layout()` after
  creating a colorbar, which `setup_plotting()`'s constrained layout engine rejects with
  `RuntimeError`.
- **The template's `lsst` kernel does not exist in the USDF terminal**, only `python3`
  (same interpreter). Execute with `--ExecutePreprocessor.kernel_name=python3` and leave the
  committed metadata alone, so the notebook still opens correctly on the RSP.
- **Re-deriving the `wfs/` geometry results — deliberately not done**, per the item. The
  99.50% matched-config agreement, 95.97% giant-intra, +261 mm filter-dominated intra outer
  edge, the circle-refit recovery to 99.5%, and the ~100x-too-small chromaticity and static
  mismatch terms are taken as inputs.
- **Coarse bins were quietly corrupting the results, in both projections.** Numbers quoted
  from 2.15 pixel radial bins read the ring excess about 12 per cent high, and the 5 deg
  azimuthal bins the study started with reported peak-to-peak 0.4392 where the resolved
  binning gives 1.0150 (dimensionless, extra-focal) — more than half the structure was
  being averaged away. Set bin width from a convergence scan on the sharpest feature
  present, not from a default. Here that is the 12.9 pixel edge roll-off radially and the
  0.86 deg spider vane azimuthally, and the vane needs several bins across it, not one.
- **The lag-1 autocorrelation is not a resolution diagnostic for a monotonic profile.**
  It reads 0.98–1.00 at every radial bin width, because neighbouring bins on a smooth
  curve always correlate. It *is* diagnostic azimuthally, where the structure is not
  monotonic. The radial substitute is the 90–10 per cent edge roll-off width.
- **Z4 was nearly reported as a result.** Its split swings from −0.2151 to +4.1147 µm wf
  between pupil configurations, purely from the pupil-scale/defocus degeneracy. Keep it in
  the output table as a diagnostic of that degeneracy, labelled as such, and never as a
  measurement of the telescope's defocus.
- **A nominal `pupilObscuration` in a batoid YAML is not the traced pupil boundary, and
  trusting it produced a fake physics result.** v1000 carries 0.612 while its traced inner
  edge is 0.6204; using the nominal for the annular-Zernike ε while danish used the traced
  value put reference and deviation on different bases and inflated the Z4 split to
  +4.1147 µm wf. It was explained away as a pupil-scale/defocus degeneracy for a full
  session. When one coefficient moves by 20x and its neighbours do not, suspect the basis
  before the physics.
- **Relaxing the blur bound from 1.5 to 3.0 arcsec did not fix the intra/extra blur
  asymmetry.** The hypothesis that the bound was biasing the fits is **rejected**: the
  intra blur simply moves to 1.51-1.66 arcsec. Do not revisit this by loosening the bound
  further.
