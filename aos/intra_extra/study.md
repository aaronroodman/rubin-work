---
study: intra_extra
topic: aos
status: active
question: "Why do intra- and extra-focal donuts differ, in Danish blur (intra larger by about 0.34 arcsec), in Z11 and Z14, and in the spider and pupil-edge shadows, when a static pupil would make them mirror images?"
answer: "Not a static pupil or model mismatch: the blur difference (median +0.3440 arcsec, n 873 FAM triplets) tracks camera-body minus M1M3 air temperature (Spearman rho -0.7828), v1000 cuts the giant-donut Z11 split from 0.81 to 0.37 um of wavefront, and diffraction gives about one third of it. A phase that flips sign with focus (a thermal strip or a turned-down M1 edge) is the open candidate."
uses:  # the blur notebook reads value_added; its FAM blur comes from blitz_reader on danish_1_3_test raws, not a product build
  value_added: v1
started: 2026-06-12
---

# `intra_extra`: intra- against extra-focal donuts: blur, Z11 and the pupil edges

> **Code:** the library `rubinwork.giant_donuts` (stamps, danish fits through blitz,
> masks, radial profiles, spider tracing) and `rubinwork.fresnel` (Huygens donuts) ·
> **Notebooks:** `notebooks/` (giant-donut profiles, blur offset, the two 4 mm Huygens
> notebooks), `notebooks/archive/` (the June pupil-mask and diffraction notebooks) ·
> **Write-ups:** [`intra_extra_phase_strip.md`](intra_extra_phase_strip.md) (symmetry
> constraint and phase-strip hypothesis), [`giant_donuts_pupil_findings.md`](giant_donuts_pupil_findings.md)
> (danish against Batoid pupil masks) · **Handoffs, now Log:**
> [`item7_giant_donuts_handoff.md`](item7_giant_donuts_handoff.md),
> [`blur_intra_extra_handoff.md`](blur_intra_extra_handoff.md)

## Question

Why do intra- and extra-focal donuts differ, in Danish blur (intra larger by about 0.34
arcsec), in Z11 and Z14, and in the spider and pupil-edge shadows, when a static pupil
would make them mirror images?

An intra-focal donut is the reflected extra-focal donut of the optic with the opposite
pupil phase (`intra_extra_phase_strip.md`, "Symmetry constraint"). So a static amplitude
error, a static aberration or a detector height cannot make the two sides differ; it takes
a phase that changes sign with focus, a departure from the single-pupil model, or an
asymmetry in the analysis.

## Current answer

- **The blur difference is real, not constant, and thermal.** Over 873 Danish 1.3 FAM
  triplets (15 nights, 20260315–20260619) the median intra − extra blur is +0.3440 arcsec,
  intra larger in 0.9954 of them (dimensionless). Camera body minus M1M3 air is the
  strongest single driver (Spearman rho −0.7828 full sample, −0.2934 within night).
- **Z11 split on giant donuts:** fitting the 20251023 pair (seq 337 extra, 340 intra)
  through blitz, v1000's pupil cuts the intra − extra Z11 split from +0.8145 to +0.3725 µm
  of wavefront against v3.14. Diffraction gives about one third of the split (+0.10 µm in
  unpaired fits of simulated donuts, `notebooks/archive/wfs_diffraction.ipynb`).
- **The extra-focal ring is an OPD signature:** flux is moved radially in opposite
  directions on the two sides, peaking at normalized radius 0.648, just outside the inner
  pupil edge (difference −0.3330, dimensionless normalized flux).
- **A turned-down M1 inner edge gives the right sign:** 50 nm over 5 cm gives +0.82 /
  −0.34 arcsec of extra blur intra / extra in quadrature (Huygens simulation, 4 + 4 mm
  defocus), stronger than the 0.41 arcsec observed.
- **Open:** whether a thermal strip along the vanes and edges, or a static edge, makes the
  phase. The pass 6 vane-shadow result in [`../giant_donut_pupil`](../giant_donut_pupil/summary.md)
  (a +0.10 to +0.15 µm optical-path strip along the vanes matches the shadow widths) points
  the same way.

The unpaired FAM Zernike version of this question is the separate study
[`../unpaired_intra_vs_extra`](../unpaired_intra_vs_extra/study.md). The Z11/Z14 split in the
corner sensors is recorded in [`../cwfs/cwfs.md`](../cwfs/cwfs.md).

## Log

### 2026-10-23 — backfilled from docs and history

Built in the phase 5 conversion (decision 6) from four lines of work:

- **Giant donuts, item 7** (2026-10-04 to 10-06): `item7_giant_donuts_handoff.md` and
  `notebooks/giant_donut_radial_profiles.ipynb`, formerly `wfs/`. The code is the library
  `rubinwork.giant_donuts` (commit `c7f1325`), which `aos/giant_donut_pupil` also imports.
  Settled: the 4 + 4 mm split (Trim ±4000 µm on both hexapods against seq 336), the Z11
  split numbers above, the ring.
- **Pupil masks and diffraction** (2026-06-12 to 06-17): `giant_donuts_pupil_findings.md`
  and the three notebooks in `notebooks/archive/`. danish's circle mask matches Batoid to
  99.5 % (dimensionless area fraction) when model and defocus match.
- **Phase strip** (2026-10-09 to 10-18): `intra_extra_phase_strip.md` and the two
  `optics_fresnel_hexapods_4mm` notebooks, formerly `optics/`. They import
  `rubinwork.fresnel` (commit `08f7479`).
- **Blur offset** (2026-10-03 to 10-06): `blur_intra_extra_handoff.md` and
  `notebooks/blur_intra_extra_offset.ipynb`, formerly aos `fam_processing`.

Outputs copied into `output/`: `blur_intra_extra/` (from
`aos/output/fam_processing/blur_intra_extra/`), `giant_donuts/` and the
`wfs_giant_donut_*.pdf` (from `wfs/output/`). The notebooks now write there.
`aos/unpaired_intra_vs_extra` still reads the blur cache from the old aos path.

Next, from the handoffs: the M1 inner-edge perturbation refit (item 7 step 7); the
thermal vane study; the Z11/Z14 corner split against the giant-donut split.
