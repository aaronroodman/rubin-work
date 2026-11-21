# Giant-donut pupil: pairs review plan

> **Status:** approved and built 2026-11-20 (run `20261120_pairs_review`); reviewed and adopted 2026-11-20 · **Kind:** plan

The final pass of this study on the v4.02 masks (v4.02g, v4.02s). The work then moves to
Josh's CAD-based Batoid model and danish mask. Every output goes into one study run, so it
can be reviewed in one place. Only existing fits are used, so there are no new fits.

## Pairs

Each pair is an intra-focal and an extra-focal exposure at ±4 mm in camera z and M2 z. The
absolute Trim differs within a pair only in dof0 and dof5 (−8000 µm intra − extra; no other
DOF over 0.5 µm; value_added `visit_telemetry`, checked 2026-11-20).

| pair | day_obs | intra seq | extra seq | band, exposure | camera dx, dy Trim (µm) |
|---|---|---|---|---|---|
| 0 | 20251023 | 340 | 337 | r, 60 s | 0, 0 |
| 1 | 20261107 | 366 | 388 | i, 30 s | +1006, −2056 |
| 2 | 20261108 | 14 | 36 | i, 30 s | +108, −2231 |
| 3 | 20261108 | 152 | 147 | i, 30 s | −537, −2817 |
| 4 | 20261108 | 168 | 190 | i, 30 s | −537, −2817 |

Rotator angle (`physical_rotator_angle`, intra / extra, deg): pair 1 0.77 / 5.02, pair 2
2.74 / 5.66, pair 3 38.58 / 36.05, pair 4 46.28 / 55.46; pair 0 about 0.2–0.3.

Inputs: pair 0 from pass 6 (`20261023_pupil6`, `isr_phys` = v4.02g, `isr_phys_v402s`);
pairs 1–4 from `20261118_night2_<day_obs>` (`v402g`, `v402s`).

## Field bands and masks

| band (deg) | sides | masks |
|---|---|---|
| 0.25–0.5, science | 5–14 per visit | v4.02g (v4.02s is the same mask below 1.4 deg) |
| 1.6–1.725, science | 16–19 per visit | v4.02g and v4.02s |
| 1.4–1.75, corner (SW0, SW1) | 1–7 per visit | v4.02g and v4.02s, as in the existing psi pages |

## Layout

```
output/<date>_pairs_review/
  README.md
  conclusions.pdf
  summary.pdf
  pair0_20251023_340i_337e/
    focal_plane.pdf
    fit_gallery.pdf
    edges.pdf
  pair1_20261107_366i_388e/   (same three files)
  pair2_20261108_014i_036e/
  pair3_20261108_152i_147e/
  pair4_20261108_168i_190e/
```

### focal_plane.pdf (per pair; intra visit, then extra)

The candidates PDF layout, in DVCS (the DM view).

- Page 1: full focal-plane mosaic of the 4×4-binned ISR, with CCD outlines. Used donuts are
  circled and coloured by band; other selected donuts are thin grey. Field-radius circles
  are dashed at 0.25, 0.5, 1.6 and 1.725 deg.
- Page 2: donuts used, per band and sensor class.
- Pages 3–27: one page per raft, with the same markers and each used donut labelled with
  star, field radius and magnitude.

### fit_gallery.pdf (per pair)

One row per side, as in `psi_gallery`: data, model, and (data − model) / noise, with model
edges coloured by element, data edges as dots, and an arrow at psi 0.

- 1.6–1.725 deg, science and corner: columns for v4.02g and v4.02s side by side (data,
  model g, residual g, model s, residual s).
- 0.25–0.5 deg: v4.02g only.
- Intra visit, then extra; rows ordered by OCS field azimuth; 4 rows per page.

### edges.pdf (per pair)

The existing `psi_edges_report` pages, unchanged. The 5 panels are inner edge, outer edge,
r_inner / r_outer, median data − model and margin, plus the element strips and the element
table page. One group per pair, mask, band and side, in this order:

1. 1.6–1.725 science: v4.02g intra, v4.02g extra, v4.02s intra, v4.02s extra.
2. Corner groups, v4.02g and v4.02s.
3. 0.25–0.5 science: v4.02g intra, v4.02g extra.

### summary.pdf (all pairs)

- Page 1: the pairs table above, with reference PSF FWHM and sides per band.
- Page 2: median data − model per element against rotator angle, intra and extra, both masks,
  1.6–1.725 deg. Page 3: the same at 0.25–0.5 deg, v4.02g.
- Page 4 (pending Aaron): spider translation and rotation against rotator angle.

### conclusions.pdf

One section per conclusion of the study summary (settled and open). Each section opens with
a statement page giving the claim and its numbers, followed by the supporting pages. Those
pages are copied from the run PDFs that `findings.md` cites, or are new pages from the pairs
where the pairs add evidence (outer edges across nights, the M1 and M3 rims).

## Steps

1. psi labels for 0.25–0.5 deg, v4.02g, all five pairs (about 110 sides; batch, minutes on
   32 CPUs), and for pair 0 at 1.6–1.725 deg if the pass 6 label runs lack a group.
2. Gallery renders, about 560 at about 40 s each (batch).
3. Focal-plane, edges, summary and conclusions PDFs (local).
