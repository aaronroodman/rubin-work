# WFS — Wavefront Sensors

Studies of the Rubin LSST Camera corner wavefront sensors (CWFS): sky-foreground
shape, ISR-processed image inspection, and related AOS diagnostics.

## Notebooks

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `wfs_corner_postisr_visualize_v1.ipynb` | Locate and visualize ISR-processed (`post_isr_image`) frames for the eight corner-WFS half-chips. Lists available exposures/nights in the AOS `cwfs` collection, then plots all eight half-chips for a chosen exposure with a sky-foreground-tuned stretch (sigma-clipped / ZScale / percentile + asinh) and writes a multi-page PDF (one page per visit). Start of a sky-foreground-shape study. | 2026-06-12 | 2026-06-12 |
| `wfs_sky_foreground_radius_v1.ipynb` | Sky-foreground level vs. focal-plane radius on the eight corner-WFS CCDs. Selects high-galactic-latitude visits (\|b\| > `b_min`, from butler `tracking_ra`/`tracking_dec` → galactic), masks pixels above the N-sigma clipped sky level (`clip_sigma` parameter), and computes the sigma-clipped mean sky flux in focal-plane radius bins. Writes 3 PDF pages per visit: sky-stretched images, masked-pixel diagnostic, and the 8 clipped-mean-vs-radius curves. | 2026-06-12 | 2026-06-12 |

## Data dependencies

- **wfs_corner_postisr_visualize_v1**: Run on the USDF RSP. Reads `post_isr_image`
  from collection
  `LSSTCam/runs/aos/cwfs/danish_1_0/wep_17_3_0/dv_4_2_0/bin_x2` in repo `/repo/main`.
  Corner-WFS detectors: 191/192 (R00), 195/196 (R04), 199/200 (R40), 203/204 (R44).
- **wfs_sky_foreground_radius_v1**: Same collection/detectors. Also uses
  `common/camera_utils.py::pixel_to_focal` for the per-pixel focal-plane radius and
  the butler exposure record (`tracking_ra`/`tracking_dec`) for the galactic-latitude
  cut. The corner CCDs span focal-plane radii ~273–333 mm.

## Notes

- **No `preliminary_visit_image` (PVI) here.** PVI is a full-focal-plane / FAM data
  product and is not produced by the corner-WFS-only `cwfs` pipeline. The available
  ISR output for the corner sensors is `post_isr_image`, which is the correct product
  for studying sky-background shape (ISR-corrected, sky background retained).
- Exposures in this collection are tagged `infocus_aos_stability_test` (in-focus
  FAM-triplet data). The corner sensors are physically offset in focus, so stars
  still appear as donuts even on in-focus exposures.
