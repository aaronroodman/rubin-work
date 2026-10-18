---
study: interference_filter_design
topic: filters
status: answered
question: "How narrow can a multilayer dielectric interference filter be in the Rubin f/1.23 annular beam while keeping throughput and OD4 out-of-band blocking?"
answer: "The beam-averaged FWHM floors at about 9.4 nm (cone smear over 14 to 23 deg incidence); the practical narrow-band limit is about 25 to 30 nm FWHM. A 25 nm band at 410 nm keeps about 92% in-band throughput with OD4 blocking except within 15 to 25 nm of the band."
uses: {}  # no rubin-work products; thin-film simulation
started: 2026-09-04
---

# `interference_filter_design`: multilayer interference filters in the f/1.23 beam

> **Code:** `code/` (run `python code/<script>.py`) · **Write-up:** [`../README.md`](../README.md), sections 0 to 5 · **Output:** `output/`

## Question

How narrow can a multilayer dielectric interference filter be in the Rubin f/1.23 annular beam, which illuminates the filter over 14 to 23 deg of incidence, while keeping in-band throughput and optical density 4 (OD4, transmission below 1e-4) blocking across the silicon range, 320 to 1080 nm? Designs are optimized with `tmm_fast` (a PyTorch transfer-matrix method, autograd on the layer thicknesses).

## Current answer

- **The beam-averaged FWHM floors at about 9.4 nm** for an Nb2O5/SiO2 cavity: the cone sweeps the passband by about 11 nm and blue-shifts it by about 7 nm. A different high-index material lowers the floor only to about 8.3 nm.
- **The practical narrow-band limit is about 25 to 30 nm FWHM.** At 10 nm the cone shift equals the bandwidth, so the band mis-centres and varies over the field.
- A 25 nm band at 410 nm, with the blocker optimized against the beam, keeps about 92% in-band throughput (fraction of transmitted power) and OD4 everywhere except within about 15 to 25 nm of the passband; that guard band needs an angle-independent absorptive blocker.
- Normal-incidence OD4 blocking falls to about OD2 across the cone; beam-averaged optimization restores it.

Full results and tables are in [`../README.md`](../README.md).

## Log

### 2026-09-04 — backfilled from docs and history

Backfilled 2026-10-18 from `filters/README.md` (the write-up) and the commits `d467c87` to `f9f9ff1`. The whole topic is this one study; its code moved from `filters/code/` into `code/` here, and each driver still writes to `../output`, now the study's `output@`. `tmm_fast` was checked against Byrnes' `tmm` to better than 1e-8 (dimensionless, transmission) on a 15-pair quarter-wave reflector.

Next, if resumed:

- an absorptive near-band blocker in series, for the 388 to 395 nm and 425 to 435 nm shoulders;
- a degradation map over centre wavelength and bandwidth: centre shift, peak loss, FWHM broadening and achievable guard band;
- redder centre wavelengths, where the fractional cone shift is smaller.
