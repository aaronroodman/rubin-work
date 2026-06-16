# WFS — Wavefront Sensors

Studies of the Rubin LSST Camera corner wavefront sensors (CWFS): sky-foreground
shape, ISR-processed image inspection, and related AOS diagnostics.

## Notebooks

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `wfs_corner_postisr_visualize.ipynb` | Locate and visualize ISR-processed (`post_isr_image`) frames for the eight corner-WFS half-chips. Lists available exposures/nights in the AOS `cwfs` collection, then plots all eight half-chips for a chosen exposure with a sky-foreground-tuned stretch (sigma-clipped / ZScale / percentile + asinh) and writes a multi-page PDF (one page per visit). Start of a sky-foreground-shape study. | 2026-06-12 | 2026-06-12 |
| `wfs_sky_foreground_radius.ipynb` | Sky-foreground level vs. focal-plane radius on the eight corner-WFS CCDs. Selects high-galactic-latitude visits (\|b\| > `b_min`, from butler `tracking_ra`/`tracking_dec` → galactic), masks pixels above the N-sigma clipped sky level (`clip_sigma` parameter), and computes the sigma-clipped mean sky flux in focal-plane radius bins. Writes 4 PDF pages per visit: sky-stretched images, masked-pixel diagnostic, the 8 clipped-mean-vs-radius curves, and a normalized fine-binned (0.25 mm) profile out to the 1.75° donut radius (310.5 mm). | 2026-06-12 | 2026-06-12 |
| `wfs_donut_selection_stages.ipynb` | Corner-WFS images with **open circles** overlaid on the donuts, coloured by ts_wep stage (selected → fit → used) for each intra/extra half-chip; circle radius a bit larger than a donut (~150 px across). Builds an extended `aggregateDonutTable` with `fit`/`used` flags. Also an **interactive pair inspector** (`%matplotlib widget`): both halves of a corner raft side by side — click a donut in either panel to highlight its paired donut in the other, re-run the **joint Danish fit** of the pair (`WfEstimator`, `jointFitPair=True`), and show **data / model / residual** stamps for both, plus a RubinTV-style Zernike bar chart. Stages joined on `donut_id`: `donutTable` → `aggregateAOSVisitTableRaw` membership (fit) → `used==True`. | 2026-06-13 | 2026-06-14 |
| `wfs_giant_donut_fit.ipynb` | Pupil-comparison tool for the large (8 mm) FAM defocus images, which have no donut tables/fits. Runs **minimal ISR** (overscan + nominal gains, no calibs) on the raw, displays a CCD, and lets you **click a giant donut**; cuts a ~1000 px stamp and runs a **single-sided, Z4-only, pure-defocus** Danish fit (`startWithIntrinsic=False`) at the labelled defocus; the **blur (fwhm) is fixed at 1″ by default** (fix-fwhm toggle; letting it float runs to ~3″, over-blurring to hide pupil mismatch). Shows **data / model / data−model** (large), a **Zernike bar chart**, the fitted **fwhm**, a **row-slice grid** (25 px horizontal bands), and a **pie-slice grid** (24 × 15° wedges: data vs model as a function of radius from the donut centre). Also computes the **sky foreground vs focal-plane radius** (3σ-clipped mean, via `pixel_to_focal`) for the sensor and can **subtract** that radial sky model before fitting. Live `ipywidgets` controls for max-Zernike term, binning, fix-fwhm, and subtract-sky, plus a **Save PDF** button that writes all panels to `output/` with an auto name (`wfs_giant_donut_{dayobs}_seq{seq}_{sensor}_x{cx}_y{cy}.pdf`). `PRESETS` select good intra (seq340) / extra (seq349) giant donuts on R30_S21 (same star). The residual reveals where the modelled optical pupil disagrees with the donut. Defocus from the `observation_reason` `intra_8mm`/`extra_8mm` label (tunable). | 2026-06-15 | 2026-06-15 |
| `wfs_batoid_pupil_compare.ipynb` | **Batoid** optical-model study: is the giant-donut pupil different when an 8 mm defocus is made by moving the **Camera hexapod 8 mm** vs the **Camera + M2 hexapods 4 mm each** (M2 is powered)? Traces a dense pupil ray grid (a spot diagram with many spots) from a point source at a chosen field angle through the defocused system; the 2-D histogram of ray positions is the geometric pupil/donut image. Hexapod pistons are rigid z-shifts of the `LSSTCamera` group and `M2` (`withGloballyShiftedOptic`, no `batoid_rubin` data download). Compares donut images, difference, and azimuthal radial profiles, for **both extra- and intra-focal** 8 mm, and a **FAM-mode** intra-vs-extra comparison (camera hexapod ±1.5 mm). Field angle is taken from the real R30_S21 donut **pixel (1167, 2915)** (≈ 1.76°). Also **extracts the pupil danish uses in its fit** (DonutFactory + SingleDonutModel built from the ts_wep `Instrument`, off-axis `z_ref`, `z_fit=()`, tiny fwhm) and overlays it on the batoid donut (giant camera-only 8 mm + FAM 1.5 mm) with a difference panel and radial profiles. Model options: design `LSST_r.yaml` or as-built `Rubin_v3.12_r`/`Rubin_v3.14_r`. Also **localizes the donut-fit rim mismatch**: a **rim-radius-vs-field-angle** sweep shows danish and batoid agree on-axis but diverge off-axis (inner −36 µm / outer +37 µm at the 1.76° field edge for the giant donut — danish's annulus stays wider), and a **per-surface vignetting attribution** (batoid `traceFull`) names the cause — at 1.76°, 25% of the rays an ideal annulus would pass are vignetted, dominated by the M1/M2/M3 mirror edges (15/7/2.5%) plus the filter (1.3%), field-dependent vignetting that danish's parametric pupil mask only approximates. Finally **compares danish's `maskParams` mask directly to the batoid vignetting boundary** in ts_wep's own pupil frame (rays on `optic.stopSurface`, danish circles evaluated on the same points): ~98.6% agreement at R30, with the **inner edge exact** and the **outer edge non-circular** — danish models each element edge as a single shifted circle, so its outer edge over-extends (mean +16 mm, up to +90 mm on the field-ward side, dominated by the filter aperture then M1), the pupil-plane origin of the donut-fit rim residual; a uniform radius tweak can't close a non-circular shape mismatch. | 2026-06-15 | 2026-06-15 |

## Data dependencies

- **wfs_corner_postisr_visualize**: Run on the USDF RSP. Reads `post_isr_image`
  from collection
  `LSSTCam/runs/aos/cwfs/danish_1_0/wep_17_3_0/dv_4_2_0/bin_x2` in repo `/repo/main`.
  Corner-WFS detectors: 191/192 (R00), 195/196 (R04), 199/200 (R40), 203/204 (R44).
- **wfs_sky_foreground_radius**: Same collection/detectors. Also uses
  `common/camera_utils.py::pixel_to_focal` for the per-pixel focal-plane radius and
  the butler exposure record (`tracking_ra`/`tracking_dec`) for the galactic-latitude
  cut. The corner CCDs span focal-plane radii ~273–333 mm.
- **wfs_donut_selection_stages**: Same collection. Reads `post_isr_image` (keyed by
  `exposure`), the ts_wep products `donutTable` (per detector, round-1 detection),
  `donutStampsExtra`/`donutStampsIntra` (stamps, on the SW0 detector),
  `aggregateDonutTable` (per visit), and `aggregateAOSVisitTableRaw` (per visit, one
  row per fit donut pair with the `used` flag). CWFS pairing: SW0=extra, SW1=intra;
  paired products are stored on the SW0/extra detector. The interactive pair inspector
  needs `ipympl` (`%matplotlib widget`) and `lsst.ts.wep` (`WfEstimator`,
  `getTaskInstrument`) + `danish` for the on-the-fly joint refit that produces the
  model/residual stamps (the pipeline does not persist the forward-model image).
  Currently uses danish 1.0.0 (the version in the `lsst-scipipe-13.0.0` / `d_latest`
  stack env; danish is part of rubin-env, not an eups product).
- **wfs_giant_donut_fit**: Reads `raw` from `LSSTCam/raw/all` and runs ISR in-notebook
  (`lsst.ip.isr.IsrTask`, overscan + nominal gains; no calibs). Single-sided fit via
  `lsst.ts.wep` `WfEstimator`/`getTaskInstrument` (FAM instrument, `defocalOffset`
  overridden to the labelled defocus) + `danish`; field angle from `lsst.afw.cameraGeom`
  `PIXELS→FIELD_ANGLE`. Needs `ipywidgets` + `ipympl` for the live controls. Default
  exposure `2025102300340` (`intra_8mm`); neighbours `337/338` are `extra_8mm`. The
  donut's effective defocus is ~7.6 mm (vs the 8 mm label); set `defocus_mm` to drive
  the fitted Z4 toward zero. The fit **always passes `bounds`** via `lstsqKwargs`
  (flux ≥ 0; fwhm in 0.1–5″ when floating, or pinned when fix-fwhm is on) — ts_wep's
  single-sided path otherwise leaves fwhm unbounded, and a runaway fwhm blows up the
  galsim FFT (so binning=1 fails on giant corner donuts unless fwhm is bounded). The
  sky-foreground section uses `common/camera_utils.py::pixel_to_focal` for the per-pixel
  focal-plane radius; subtraction matters most on corner sensors (large radial gradient).

- **wfs_batoid_pupil_compare**: Pure `batoid` for the trace; imports `lsst.obs.lsst` +
  `lsst.afw.cameraGeom` only to get the field angle of the R30_S21 donut pixel
  (PIXELS→FIELD_ANGLE; correct radius, orientation approximate). Uses the LSST model
  yaml (`LSST_r.yaml` design, or `Rubin_v3.12_r`/`v3.14_r` as-built) shipped with batoid;
  hexapod pistons via `withGloballyShiftedOptic` on `LSSTCamera` and `M2` (no
  `batoid_rubin` download). Finding at the R30_S21 field (1.76°): camera-only vs
  camera+M2 give different donut spans (8 mm: 7.04 vs 6.73 mm extra, 7.06 vs 6.75 mm
  intra) — the pupil differs; FAM ±1.5 mm donut span ~1.3 mm. danish's own fit pupil
  (extracted via DonutFactory/SingleDonutModel with z_fit=0) matches batoid closely
  (outer radius 3.42 vs 3.38 mm giant, 0.65 vs 0.62 mm FAM) and registers without a flip
  (shared field-angle convention); danish assumes camera-only defocus, so it's compared
  to batoid camera-only. §8 sweeps rim radius vs field angle (sub-bin half-max edges);
  §9 attributes off-axis vignetting per surface via `batoid.Optic.traceFull`; §10 compares
  danish's `maskParams` mask to the batoid vignetting boundary in ts_wep's pupil frame
  (rays on `optic.stopSurface` via `asPolar`; danish circles per `ts_wep maskUtils._fitEdges`
  — polyval in θ_deg, centre `c·thx/thr`). Needs `danish` + `lsst.ts.wep`
  (`getTaskInstrument`, `DefocalType`).

## Reference

- [`ts_wep_cwfs_dataflow.md`](ts_wep_cwfs_dataflow.md) — how the corner-WFS ts_wep /
  donut_viz dataset types are produced and flow together (ISR → detection → cutout →
  pairing → Zernike fit → per-visit aggregation), with the selected/fit/used stage
  definitions.

## Notes

- **No `preliminary_visit_image` (PVI) here.** PVI is a full-focal-plane / FAM data
  product and is not produced by the corner-WFS-only `cwfs` pipeline. The available
  ISR output for the corner sensors is `post_isr_image`, which is the correct product
  for studying sky-background shape (ISR-corrected, sky background retained).
- Exposures in this collection are tagged `infocus_aos_stability_test` (in-focus
  FAM-triplet data). The corner sensors are physically offset in focus, so stars
  still appear as donuts even on in-focus exposures.
