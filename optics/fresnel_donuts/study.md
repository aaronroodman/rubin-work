---
study: fresnel_donuts
topic: optics
status: done
question: "Is batoid's Huygens diffraction correct for defocused Rubin donuts, and how large are diffraction, seeing and sensor effects at the 10 µm pixel scale?"
answer: "batoid huygensPSF is correct, and a NUFFT evaluation makes it about 1e6 times faster (dimensionless, wall-clock ratio). Diffraction moves about 13% of donut flux (fraction of total flux) against geometric optics, but gives at most 0.002 arcsec intra/extra difference in fitted blur."
uses: {}  # no rubin-work products; batoid simulation
started: 2026-09-30
---

# `fresnel_donuts`: diffraction models of defocused Rubin donuts

> **Write-up:** [`fresnel_donuts.md`](fresnel_donuts.md) · **Notebooks:** `notebooks/optics_fresnel_*_v1.ipynb` · **Code:** `rubinwork.fresnel` (library, shared with the intra/extra notebooks) · **batoid PR text:** [`batoid_pr_debye_huygens.md`](batoid_pr_debye_huygens.md)

## Question

Is batoid's Huygens diffraction (`batoid.analysis.huygensPSF`) correct for defocused Rubin star images (donuts), and how large are diffraction, seeing and silicon-sensor effects at the 10 µm pixel scale that donut wavefront fitting sees? Monochromatic, 622 nm, on axis, camera pistoned ±1.5 mm.

## Current answer

- **batoid's Huygens algorithm is correct**; its only defect is speed. A type-1 non-uniform FFT (NUFFT, FINUFFT) evaluation agrees with batoid's `sumAmplitude` to 2.6e-12 (relative amplitude) and is about 1e6 times faster (dimensionless, wall-clock ratio). Unit ray weights are exact because Rubin obeys the Abbe sine condition to 4e-6 (dimensionless).
- `fftPSF` needs `sphereRadius` of at least 20 m for donuts (residual 1.0 to 1.8e-3 of total flux, against 7.4 to 8.2e-3 at the default 5 m).
- Paraxial Fresnel codes make the donut edge 4% too small (fraction of radius) at f/1.23.
- **Diffraction moves about 13% of the donut flux** (Σ|Δ| over 10 µm pixels, fraction of total flux) against geometric optics, in rings at the edges.
- **No intra/extra blur asymmetry is reproduced** by diffraction (≤ 0.002 arcsec fitted FWHM difference) or by an i-band silicon sensor (+0.0003 to +0.0004 arcsec, about 250 times smaller than the observed 0.1 arcsec). The data show intra about 0.9 arcsec against extra about 0.8 arcsec; that question continues in the intra/extra study.

Details and tables in [`fresnel_donuts.md`](fresnel_donuts.md).

## Log

### 2026-10-04 — backfilled from docs and history

Backfilled 2026-10-18 from `fresnel_donuts.md` (formerly `optics/docs/studies/`) and the commits `52a21b2` (notebooks) and `da1a127` (Huygens-minus-geometric Danish offsets). Done in the phase 5 triage. The batoid pull request is open as jmeyers314/batoid#127.

The four notebooks moved here; they now write to this study's `output/`. They import the library `rubinwork.fresnel`, which the intra/extra notebooks also use.

Outstanding, if resumed: off-axis positions, especially the corner wavefront sensors, where vignetting edges are not conjugate to the pupil; broadband r-band images.
