---
study: spot_through_focus
topic: optics
status: done
question: "What does the Rubin design spot diagram look like through focus, across the field, in r band?"
answer: "A batoid through-focus spot diagram in the Zemax layout: 4 field points (0 to 1.75 deg) by 5 detector shifts (-14 to +14 µm), at 552, 620 and 691 nm, in a 40 µm box."
uses: {}  # no rubin-work products; batoid ray trace
started: 2026-09-30
---

# `spot_through_focus`: design spot diagram through focus

> **Code:** `code/lsst_spot_through_focus.py` (writes `lsst_spot_through_focus_r.png` to the current directory) · **Output:** `output/20260930_r_band/`

## Question

What does the Rubin design spot diagram look like through focus, across the field, in r band?

## Current answer

A figure, not a measurement: batoid `LSST_r.yaml`, rows are field points (0, 0), (0, 1.225),
(0, 1.75) and (1.121, 1.231) deg; columns are detector z shifts of -14, -7, 0, +7 and
+14 µm. Three wavelengths (552, 620, 691 nm: r-band blue edge, center, red edge), spots
centered on the per-field centroid in a 40 µm box, made to match a Zemax "Spot Diagram:
Cosine Space" figure.

## Log

### 2026-11-14 — filed from the top-level working area

Made in an ad-hoc session on 2026-09-30 ("use Batoid to make a spot diagram through focus
like the attached figure") and left in `~/Astrophysics/Claude/`. Moved here as a study;
the script was not changed and does not use `study_run`.
