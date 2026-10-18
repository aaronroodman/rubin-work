---
study: telecentricity
topic: optics
status: done
question: "What is the chief-ray angle of incidence on the focal plane against position, for the design and the as-built v3.14 prescription?"
answer: "Rubin is non-telecentric: the chief-ray angle grows linearly at 76.45 arcsec/mm of focal-plane radius, to 6.71 deg at r = 315 mm. As-built v3.14 equals the design to < 4.5 arcsec once the telecentricity null is moved 0.66 mm off centre."
uses: {}  # no rubin-work products; batoid ray trace
started: 2026-09-04
---

# `telecentricity`: chief-ray angle across the focal plane

> **Code:** `code/telecentricity.py` (library), `code/run_telecentricity.py` (driver; writes `output/`) · **Write-up:** [`../README.md`](../README.md), Telecentricity

## Question

What is the chief-ray angle of incidence on the focal plane against position, for the design and the as-built v3.14 prescription?

## Current answer

- The chief-ray angle grows linearly with focal-plane radius, from 0 on axis to **6.71 deg at r = 315 mm** (the field corner), at **76.45 arcsec/mm** (design; 76.46 arcsec/mm as-built). The tilt is inward; the exit pupil sits 2.69 m behind the focal plane.
- The flux-weighted mean direction of the cone runs about 0.6 to 0.8 deg steeper than the chief ray over most of the field.
- **As-built against design** is a dipole of 105 arcsec peak to peak (36 arcsec RMS) at fixed focal-plane position, which is a shift of the telecentricity null to (+0.589, +0.298) mm. Removing it leaves **< 4.5 arcsec** anywhere. Sensor modelling can use the design radial law re-centred on that null.

r band, 622 nm, out to 1.75 deg field radius. Method and figures in [`../README.md`](../README.md).

## Log

### 2026-09-04 — backfilled from docs and history

Backfilled 2026-10-18 from the Telecentricity section of `optics/README.md` and commit `d467c87`. Done in the phase 5 triage. The library and driver moved here from `optics/code/`; the driver's default output is now this study's `output/`.
