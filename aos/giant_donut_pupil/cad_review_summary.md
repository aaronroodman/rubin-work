# Giant donuts against LSST_r_v4.02: what the CAD review should check

> **Status:** summary for the CAD review with Josh and Hannah, week of 2026-10-26 ·
> **Written:** 2026-10-23 · **Kind:** note

**Data:** giant donuts from 20251023, BLOCK-T626, r band, 60 s, with the camera and M2
hexapods both at +4000 µm (extra-focal, visit 2025102300337) or −4000 µm (intra-focal,
2025102300340). There are 188 donut sides, from the field center out to 1.75 deg.

**Fit:** danish with the v4.02 obscuration mask (v4.02g, or v4.02s where noted), the blur
fixed at 1.0 arcsec, and a sparse set of Zernikes. ISR includes flat and CTI correction,
and the rotator angle is ConsDB `physical_rotator_angle`.

**Units:** 1 donut pixel (10 µm) is about 12.3 mm in the stop plane. "Extra / intra" gives
the two sides of focus.

**Figure references:** IDs like `P6-pupil_pass6-39` are rows of `figure_index.csv`, giving
the pass, the PDF and the page. The `.png` files are in `figures/`. The two 3D viewers,
`figures/pupil_mask_viewer.html` and `figures/rubin_model_geometry.html`, show the masks and
both model geometries.

## Where v4.02 disagrees with the data

### 1. Camera silhouette

At 1.45–1.75 deg, the inner edge on the vignetted side is set by the camera body. In
v4.02 that's the 165 `CameraSilhouette` slices, z 3.481–5.121 m, radii 796–825 mm.

- **Mask against Batoid:** the danish v4.02g circle sat 1.8–5.3 pixels inside Batoid's
  edge. v4.02s refits it above 1.4 deg and matches Batoid to 0.0 pixels (median), worst
  −0.5 pixels.
- **Data against model (v4.02s):**
  - extra-focal, ψ 160–200 deg: −0.28 ± 0.57 pixels, consistent with zero;
  - intra-focal, ψ 330–30 deg: **+1.93 ± 0.39 pixels**.
- **What the remaining intra-focal miss wants:** a silhouette circle 14.3 ± 4.4 mm larger in
  radius, with its center shifted −10.7 ± 4.6 mm in the stop plane. The extra side gives
  +8.7 ± 7.4 mm and +14.5 ± 7.8 mm, consistent with zero.
- **v1000's single `CameraBody` disk** (radius 0.80469 m at z 3.500 m), through danish, is
  within 0.2–1.1 pixels of v4.02s at 1.4–1.75 deg. The 165 slices add little over one disk
  for this edge.
- **Which slices block:** in a ray fan at 1.7 deg, the rays stopped by the silhouette hit
  slices at z 4.05 and 4.46 m.
- **Figures:** `P6-pupil_pass6-39`, `P6-pupil_pass6-40`, `P5-pupil_pass5-3`,
  `P5-pupil_pass5-4`; `pass5_vignetting_intra.png`; `camera_body_review.md`.

### 2. Spider vanes: width and depth of the shadows

- **Fitted width:** one vane width fitted per side, on the central-band science sides, gives
  **102.8 mm extra-focal** (nMAD 3.6 mm, 26 sides) and **61.3 mm intra-focal** (nMAD 2.0 mm,
  27 sides). The CAD says 50 mm. Batoid simulations through the same fit return
  49.7 / 49.9 mm, so the method is not biased.
- **Shadow profiles, as depth × FWHM ("equivalent width"):** the data give 10.0 pixels extra
  and 5.4 pixels intra; the model gives 4.1 pixels on both sides. An opaque vane gives the
  same equivalent width on both sides of focus, so the vanes are not acting as opaque
  strips alone.
- **What the Huygens simulations say:**
  - diffraction widens both sides equally, to 6.8 / 6.0 pixels;
  - a phase strip of **+0.10 to +0.15 µm of extra optical path along the vanes** (colder or
    denser air) gives 8.8–9.6 / 3.6–4.5 pixels, close to the data.

  This may be thermal, not geometric. It's a separate study.
- **Placement is right in v4.02:**
  - v1000's vanes sit one vane width (50 mm) off the CAD shadow, and v4.02's match it.
  - The TEA is centered on the optical axis to about 1.5–2 mm (rubin-optical-model
    `studies/tea_decenter`).
  - With `physical_rotator_angle`, the remaining spider rotation is −0.026 deg (extra) and
    +0.087 deg (intra).
- **Figures:** `P6-pupil_pass6-42`, `P6-pupil_pass6-43`, `P6-pupil_pass6-44`,
  `P6-pupil_pass6-45`, `P4-pupil_pass4_spiders-37`; `pass6_huygens.png`,
  `pass4_spider_profiles_G0005.png`.

### 3. TMA structure baffle

- **Not in the danish mask:** the `TMASection` planes have no circle in it. In pass 5, a
  TMASection sets the outer edge over the central 6 deg of ψ on both sides, around ψ 0 deg
  (intra) and 180 deg (extra), at 1.7 deg field.
- **The sims show the gap:** danish fitted to Batoid's own simulated donuts leaves residual
  at 1.69–1.70 deg (P1693, CR00SW0_0), on rays first clipped by TMASection, and by
  M1Baffle1 the other way.
- The **mid-level baffle** and **top-end ring** that the CAD review covers sit in the same
  class: in the beam only at the field edge.
- **Figures:** `P5-pupil_pass5-1`, `P6-pupil_pass6-7`.

### 4. Filter frame

- **Too tight in v4.02 at the field edge:** at 1.73–1.86 deg, Batoid v4.02 clips at the
  filter entrance 7–21 pixels inside where the data edge is. This is on the corner and
  edge-of-field sides (G0663, G0664, G0678, G0681, G0684 intra).
- **Likely cause:** v4.02's in-camera openings, including the filter frame, come from the
  **z-band** camera CAD, and these visits are r band.
- **Figures:** passes 1 and 2 (unmodelled-surface table, run `20261018_pupil`).

### 5. Inner/outer radius ratio and the edge offsets

- **Away from the silhouette:** with v4.02s, data − model is +0.75 / +0.42 pixels on the inner
  edge and −0.51 / −0.46 pixels on the outer edge. That is flat in field radius.
- **The ratio:** r_inner / r_outer in the data is +0.004 / +0.002 above the model
  (dimensionless).
- **Diffraction explains about half:** +0.31 / +0.26 pixels inner, −0.37 / −0.28 pixels
  outer, and a ratio excess of +0.002 / +0.001.
- **The rest** may be the M1 inner and outer clear radii, or a slope effect at the mirror
  edges (the M1 inner-edge roll-off is in v4.03).
- **Figures:** `P6-pupil_pass6-44`; `pass3_ratio_vs_field.png`.

### 6. The unexplained translation

- **What it is:** the spider shadows are shifted by **−1.65 / −2.07 pixels** in OCS image y,
  and the inner edge by +2.1 / +1.8 pixels, both relative to the donut center. Both have
  the same image sign on both sides of focus.
- **Ruled out:**
  - danish modeling: Batoid simulations give −0.01 / −0.05 pixels;
  - ISR, flat and CTI;
  - CCD readout half or sensor vendor;
  - a TEA decenter or tilt, which would flip sign between the sides of focus;
  - coma, which moves the inner edge and the spiders the same way.
- **Open:** anything fixed in the pupil flips with focus, so this needs an effect in the image
  frame.
- **Figures:** `P5-pupil_pass5-54`, `P5-pupil_pass5-56`, `P6-pupil_pass6-7`,
  `P6-pupil_pass6-9`; `pass5_spider_m1.png`, `pass4_central_image.png`.

## Questions for the CAD review

1. **Camera body:** what sets the outer envelope at z about 4.0–4.5 m, where the slices block
   the M2 → M3 beam? Are there parts outside the 796–825 mm outline, such as hoses, cable
   wraps or the utility trunk, that would add the intra-side 14 mm?
2. **Spider vanes:**
   - the true cross-section and anything mounted along them (cable trays, piping) that
     widens the shadow;
   - the exact angle and extent of each vane, which v4.02 does not capture exactly.
3. **Top-end ring, mid-level baffle and TMA structure baffle:** their radii and heights where
   they reach the beam at 1.65–2.0 deg.
4. **Filter frame:** the r-band filter frame and holder geometry, against the z-band frame
   used in v4.02.
5. **M1 clear aperture:** the inner and outer edges as built (the baffle radius, and the
   inner-edge roll-off).
