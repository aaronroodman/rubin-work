# Giant-donut pupil: summary

> **Status:** parked 2026-10-23, pending the CAD review (week of 2026-10-26) and a Batoid
> model rebuild from CAD and Zemax · **Kind:** summary

Figure references are `figure_index.csv` ids (`P6-pupil_pass6-39`: pass 6, `pupil_pass6.pdf`,
page 39) or files in `figures/`. The row ids (P5, V3, ...) are in `findings.md`; the methods
are in `method.md`.

## Question

Does the v4.02 CAD-matched pupil fit giant donuts better than v1000, across the full focal
plane including the corner wavefront sensors (WFS)? What wavefront, coma (Z7, Z8) and
astigmatism (Z5, Z6), do the donuts give, and where does the v4.02 pupil disagree with the
data?

## Data

- **Night:** 20251023, BLOCK-T626, r band, 60 s exposures.
- **Visits:** 2025102300337 (extra-focal) and 2025102300340 (intra-focal), the camera and
  M2 hexapods both at ±4000 µm from the in-focus seq 336, about 7.6 mm effective defocus.
- **Sample:** stars of the_monster refcat, 7–14.5 AB mag, selected by an automatic rule
  scored against Aaron's hand labels (science purity 0.976, completeness 0.965,
  dimensionless). Pass 6 uses 188 donut sides in two bands, 0–0.8 deg and 1.4–1.75 deg.
- **Scale:** 1 donut pixel (10 µm) is about 12.3 mm in the stop plane.

## Method

- **Fit:** danish with the v4.02 obscuration mask: v4.02g (masks fitted at the giant
  configuration) as the baseline, v4.02s (camera silhouette refitted above 1.4 deg) where
  noted. Fit a is Z4 only; fit c frees Z4–Z11, Z14, Z15, Z20–Z22. Blur fixed at 1.0 arcsec
  FWHM.
- **Conventions in force:** reference wavefront intrinsicTA (ts_wep), Zernikes as
  deviations in µm of wavefront; rotator angle = ConsDB `physical_rotator_angle`; ISR with
  flat and CTI correction; relative azimuth ψ (0 points away from the field center) to
  stack the outer band. Full list: `method.md` "Conventions in force".
- **Edges:** robust harmonic fits of the inner and outer donut edges about the fit-a
  center, spider and vignetted bins dropped.
- **Spider tracer:** the model shadow guides the data search; a per-side Huber fit of
  rotation plus translation to the shadow azimuth residual at the half radius.
- **Simulations:** Batoid design images of 26 sides through the same pipeline (nominal, a
  0.20 deg rotation, a 10 mm TEA shift), and Batoid Huygens images with and without phase
  strips along the vanes.

## What is settled

- **v4.02 fits better than v1000** (P1, P2): lower residual RMS on 16 of 16 clean sides,
  median ratio v4.02 / v1000 0.94 (dimensionless); 35 of 38 sides with Z4 only. Most of the
  gain is the spider: v1000's vanes sit 50 mm off the CAD shadow.
  `resid_rms_v1000_v402.png`, `spider_overlay.png`.
- **No field trend inside 1.725 deg** (P3): fitted Z5–Z8 within ±3 µm of wavefront; fits
  break down beyond 1.9 deg.
- **The large outer-band Z5–Z7 is pupil bias, not wavefront** (P5): the central-band
  medians (Z5 −0.22, Z6 −0.13, Z7 +0.22, Z9 −0.30 µm) cost only 6 % in chi2/dof (dof about
  7.4e5) against fit c's 1–3.9 µm. `P4-pupil_pass4-10`, `P6-pupil_pass6-12`.
- **The edge offset looks like coma but is not** (P7, P13): the +3.3 pixel inner-minus-outer
  shift reads as Z7 +0.7–0.8 µm, but whole-donut fits find +0.02 to +0.24 µm. Coma moves
  the inner edge and the spiders the same way; the data move them opposite.
  `pass2_coma.png`, `P3-pupil_pass3-106`.
- **The inner shadow is larger than the model** (P11): r_inner / r_outer is 0.0104 above it
  (median, dimensionless), flat in field. Freeing Z11 removes it, but Z11 flips sign with
  focus (−0.15 / +0.15 µm), so it is the pupil. `pass3_ratio_vs_field.png`.
- **The central edge mismatch is in the image frame** (P14): extra and intra agree in OCS
  image azimuth (Pearson r +0.76) and anti-correlate in the pupil frame (−0.95).
  `pass4_central_image.png`, `pass4_central_pupil.png`.
- **The camera silhouette sets the vignetted inner edge at 1.45–1.75 deg** (P16, P17, M1):
  v4.02g's circle sat 3.3–3.9 pixels inside Batoid; v4.02s matches Batoid to 0.0 pixels
  median. v1000's single CameraBody disk is within 0.2–1.1 pixels of v4.02s.
  `P6-pupil_pass6-39`, `P6-pupil_pass6-40`, `pass5_vignetting_intra.png`.
- **The spider offset is a translation, not a rotation** (V3, V6): dy −1.65 / −2.07 pixels
  OCS image (extra / intra) removes the m = 1 term (0.35 / 0.42 deg to 0.003 deg). Sims show
  no danish bias (dy −0.01 / −0.05 pixels). `pass5_spider_m1.png`, `P6-pupil_pass6-7`.
- **Not the TEA, the detector, ISR or the rotator** (V8, V9, P20, V10): a TEA shift flips
  sign with focus, which puts the TEA within about 1.5–2 mm of center; the translation is
  the same in both amplifier halves and vendors; full ISR moves nothing; rotator motion in
  the exposure is 0.1 pixels. `P6-pupil_pass6-10`, `P6-pupil_pass6-44`.
- **The vane shadows are not opaque strips** (V11, V12): a free vane width fits 102.8 mm
  extra and 61.3 mm intra against CAD 50 mm (sims return 49.7 / 49.9 mm); blur does not
  explain it. `P6-pupil_pass6-42`, `pass4_spider_profiles_G0005.png`.
- **Diffraction gives about half the edge miss** (V13): +0.31 / +0.26 pixels inner,
  −0.37 / −0.28 outer, ratio excess +0.002 / +0.001. `P6-pupil_pass6-44`.

## What is open

- **The common image-frame translation** (V14): spider dy −1.65 / −2.07 pixels and
  inner-edge dy +2.1 / +1.8 pixels, same sign on both sides of focus. Anything fixed in the
  pupil flips with focus, so the cause must act in the image frame.
- **The intra-focal silhouette miss** (P18): +1.93 ± 0.39 pixels with v4.02s; it wants a
  silhouette circle 14.3 ± 4.4 mm larger, center −10.7 ± 4.6 mm (stop plane).
- **Missing baffles in the mask** (P19): chi2/dof 2.1–2.8 remains at 1.69–1.70 deg on rays
  first clipped by the TMASection and M1Baffle1, which the danish mask lacks.
- **The vane shadow difference** (V15): a phase strip of +0.10 to +0.15 µm of optical path
  along the vanes matches the data (equivalent width 8.8–9.6 / 3.6–4.5 against 10.0 / 5.7
  pixels). The physical cause, vane or air temperature, is not measured. `pass6_huygens.png`.
- **The residual rotation** (V5): −0.026 / +0.087 deg after the sim recovery; the 0.11 deg
  intra − extra difference is unexplained.
- **Night 2** (N1, N2): the review found no other night with a usable intra-focal set.
  Extra-focal 20251027 i band repeats night 1's spider dy (−1.69 to −1.80 pixels) and
  inner-edge dy (+1.81 to +2.05 pixels), with a narrower vane width (76–92 mm). 20260127 is
  still being fitted. `PN2-night2_20251027-2`.

## What would move it forward

- **The CAD rebuild:** a Batoid model rebuilt from CAD and Zemax, with the camera envelope
  at z 4.0–4.5 m, the true vane cross-section, the TMA, mid-level and top-end baffles, the
  r-band filter frame and the M1 clear aperture. Questions in `cad_review_summary.md`.
- **Night 2 on the rebuilt model:** 20260127 T680 r band (16 exposures at rotator −48 to
  +59 deg, plus three at 8 mm) tests whether the translation turns with the rotator.
- **A thermal vane study:** vane and air temperatures against the shadow width.
- **A giant pair at rtp ±90 deg:** an intra and extra pair at a quarter turn separates
  an image-frame effect from one fixed to the telescope.
- **To do: the M1 inner-edge roll-off in the model** (Aaron, 2026-11-18).
  - *Done elsewhere:* `rubin-optical-model`, `studies/mirror_figure_errors/SUMMARY.md`
    (commits `a479d15`, `c855075`). The mirror-lab M1 map has the inner edge rolled down at
    every azimuth: h(r) = c − A exp(−(r − 2.558 m)/w), A = 76.4 ± 8.9 nm, w = 27.1 ± 2.8 mm,
    30 nm deep at the v4.x inner clear aperture (2.5833 m); A and w each vary by a factor
    of 3.0 (dimensionless, largest over smallest of twelve 30° sectors). `LSST_r_v4.03.yaml`
    is v4.02 plus this roll-off on M1 (a `Bicubic` in a `Sum`); vignetting is identical to
    v4.02, ΔW = −59 nm at 2.5833 m, and its annular Zernike terms are under 2 nm, so the
    fitted Z ≤ 22 cannot absorb it. Predicted, if positive map values mean high glass: a hot
    rim on the extra-focal inner edge and a softened inner edge with a skirt into the hole
    on the intra-focal side; focal-plane throw about 23 µm (2.3 pixels) at 2.5833 m.
  - *Seen here* (`20261118_edge_review`, star S4848323907389759872, 20261107): extra inner
    rim up to 1.3 of the flat top; intra inner edge a soft toe rising from about 10 pixels
    below the edge. That is the predicted sign, on one star. SUMMARY.md expects the rim to
    fade above about 6 mm of detector defocus; a giant donut (r_out 340 pixels) is about
    8.4 mm, yet the rim is strong, so its contrast is a quantitative test of A and w.
  - *Not explained by it:* the outer-edge rims (intra outer up to 1.6 of the flat top with a
    dip inside). Candidates in SUMMARY.md: M3 outer edge turned up about +80 nm over its last
    50 mm; M1 outer edge −40 to +90 nm by sector.
  - *Next, Batoid:* trace giant donuts (±4 mm M2 and camera z, i band, the night 3 fields and
    rotator angles) through v4.02 and v4.03 (or v4.02 with the runtime
    `withPerturbedSurface('M1', Bicubic)` of SUMMARY.md section 4); confirm the rim side and
    contrast; measure the edge shift v4.03 − v4.02 under each edge rule.
  - *Next, data:* the rim should appear only where M1 inner sets the inner edge (psi 60–300
    deg, `psi_labels`), not where M2 inner or the silhouette does. Its strength against M1
    azimuth tests the per-sector A and w, once the map azimuth is tied to the batoid, CCS and
    OCS frames (open there). Fitting A and w to the giant-donut edge profiles is item 2 of
    SUMMARY.md section 6.
  - *Next, danish:* danish 1.4 maps pupil to focal by x = −f ∇W (`_pupil_to_focal`), with
    Zernike-only `x_offset`/`y_offset` hooks. Its forward mesh path,
    `DonutFactory._image_from_mesh(verts_px=...)`, takes pre-projected vertices, so the
    roll-off deflection −f ∇ΔW(ρ), ΔW = 2 × 0.982 × dz(r) with r the M1 radius, can be added
    to the Zernike projection there, with A and w fixed from v4.03 or fitted (degenerate
    with the inner clear aperture radius). Two danish changes: a mesh that resolves the
    27 mm zone at the inner edge, and keeping folded triangles: the rim is a fold, and the
    C++ kernel skips triangles with projected area ≤ 0.

## Ruled out

Coma as the edge and spider shift (P8); the 1–3 µm outer-band Z5–Z7 as wavefront (P4);
seq 324 as the reference (P9); the telescope state (P10); a rotator offset (V2); the
visitInfo rotator angle (V4); a TEA decenter (V7).

## See also

- `cad_review_summary.md`: where v4.02 disagrees with the data, and the questions for CAD.
- `camera_body_review.md`: the camera body in v1000, v3.14 and v4.02.
- rubin-optical-model `studies/tea_decenter`: the TEA decenter and tilt study.
- `figures/pupil_mask_viewer.html` and `figures/rubin_model_geometry.html`: 3D viewers of
  the masks and model geometries.
