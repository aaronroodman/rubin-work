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
| `wfs_batoid_pupil_compare.ipynb` | **Batoid ↔ danish pupil/mask comparison** for the LSST Camera. Traces a dense pupil ray grid (spot diagram) through the defocused system; the 2-D histogram of un-vignetted ray positions is the geometric pupil/donut. Defaults to the **design model `LSST_r` (~v3.3)** — the model ts_wep/danish actually use (`inst.batoidModelName == "LSST_{band}"`); as-built `Rubin_v3.12_r`/`Rubin_v3.14_r` selectable. Hexapod/CCD pistons via `withLocallyShiftedOptic` (matching ts_wep). Field angle from the R30_S21 donut **pixel (1167, 2915)** (≈ 1.76°). Sections: 8 mm camera-only vs camera+M2 spot diagrams (§5); FAM ±1.5 mm intra/extra (§6); **danish's fit pupil** extracted (DonutFactory + SingleDonutModel from the ts_wep `Instrument`, `z_fit=()`) and overlaid on the batoid donut (§7); **rim radius vs field angle** (§8); **per-surface vignetting attribution** via `traceFull` (§9); and **§10 the three-case danish-mask-vs-batoid-boundary comparison** in ts_wep's stop-surface pupil frame, offsetting the optic ts_wep offsets per case — **WFS** (Detector ±1.5 mm), **FAM** (Camera ±1.5 mm), **giant** (Camera ±8 mm + the 4+4 mm camera+M2 split), intra & extra. Key results: danish's circle mask reproduces the batoid boundary to **99.5%** when the model and defocal config match (inner edge exact; ~3 mm M1 outer circle-shape residual); the **WFS case is exact** (CCD piston, camera fixed); the mask mismatch is purely a **camera-hexapod** effect (filter/L1/L2 ride with the camera → intra/extra asymmetry: ~+57 mm at FAM intra, ~+261 mm at giant intra); and against the **as-built** model the filter moves the edge ~+16 mm (matters for data). **§11** tests a per-element **ellipse** edge model (generalizing ts_wep's circle fit) and finds it does *not* help — M1/M2 are already circular, and a closed ellipse over-clips the far camera-borne elements; the real fix is making the mask **defocal-aware** (a refit *circle* at the camera position recovers giant-intra 96→99.5%). **§12** visualizes **non-telecentricity** (chief-ray AOI ~6.6° at 1.72° → intra/extra donut centroids shift ±~140 µm oppositely). **§13** is reusable **synthetic-donut + ts_wep paired-danish-fit tooling** (batoid spot diagram → seeing → pixelate → Poisson → `WfEstimator` jointFitPair); used for the **M1M3 aperture-update bias** study (draw donuts with the measured aperture 2.5833/4.165 m, fit with the default 2.558/4.18) → small bias, ~3 nm RMS, ~9 nm spherical (Noll 11) at FAM 1.72°. **§14/§15** are **Danish/Batoid/Difference pupil grids** vs field radius 1.60–1.725° (0.025° steps), for the **WFS** (R04, Detector-offset) and **FAM** (R03 azimuth, Camera-offset) cases, intra & extra — showing directly how the donut flux pattern matches on both sides of focus (corr ~0.96–0.98). **§16** documents the **intra-focal filter-vignetting onset** (~1.73°, just outside the 1.725° cut). **§17** captures the **pupil-mismatch Zernike-bias simulation** (filter bias ~0 at the cut, ~tens of nm coma just beyond). Note: the donut off-axis ellipticity (~7% at 1.72°) is dominated by the field-dependent pupil mask, not the cos(AOI) foreshortening (~0.3%). See `danish_pupil_mask_findings.md` for the mask writeup. | 2026-06-12 | 2026-06-16 |

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
  yaml — defaults to `LSST_r.yaml` (design ~v3.3, = `inst.getBatoidModel()`, the model
  ts_wep/danish actually use), with `Rubin_v3.12_r`/`v3.14_r` as-built selectable.
  Pistons via `withLocallyShiftedOptic` (matching ts_wep). §5 camera-only vs camera+M2
  spot diagrams; §6 FAM ±1.5 mm; §7 danish fit pupil (DonutFactory/SingleDonutModel,
  `z_fit=0`) overlaid on the batoid donut. §8 rim radius vs field angle (sub-bin half-max
  edges); §9 per-surface vignetting via `batoid.Optic.traceFull`; **§10 the three-case
  danish-mask-vs-batoid-boundary comparison** in ts_wep's stop-surface pupil frame
  (`asPolar` → `optic.stopSurface`; danish circles per `ts_wep maskUtils._fitEdges`,
  polyval in θ_deg, centre `c·thx/thr`), offsetting the optic ts_wep offsets per case
  (`inst.batoidOffsetOptic`: **Detector** for the corner WFS, **LSSTCamera** for FAM/giant).
  Result: 99.5% agreement when model+config match (WFS exact; mismatch is a camera-hexapod
  filter/L1/L2 effect, intra/extra-asymmetric). §11 (per-element ellipse fit) additionally
  needs `scipy` (`optimize.least_squares`, `interpolate.griddata`) and
  `sklearn.neighbors.NearestNeighbors`. §12/§13 (non-telecentricity, synthetic-donut fit)
  need `scipy.ndimage.gaussian_filter` and `lsst.ts.wep` `Image` + `WfEstimator`
  (paired danish fit). Needs `danish` + `lsst.ts.wep`
  (`getTaskInstrument`, `DefocalType`, `BandLabel`).

## Reference

- [`danish_pupil_mask_findings.md`](danish_pupil_mask_findings.md) — writeup of the
  batoid↔danish pupil-mask comparison for Josh Meyers / danish: method (stop-surface
  frame), the three-case findings (WFS exact; camera-hexapod filter/L1/L2 mismatch;
  design-vs-as-built model; cam+M2 split), and the ranked proposed improvements, with a
  self-contained reproduction recipe.
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
