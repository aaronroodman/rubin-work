---
study: v4speed_danish_closure
topic: aos
status: active
question: When the donuts, the reference wavefront and the obscuration file all come from the v4_speed model, how well do Danish fits match the donuts, and how far are the fitted Zernikes from the Batoid wavefront, by defocus configuration, side and field position?
answer: ''
uses:
  batoid_intrinsic: v4_speed@20261121b
  sim_donut_fits: v4_speed_dense@20261121, v4_speed_sparse@20261121, v4_speed_sparse_bin1@20261121, v4_speed_sparse_prod@20261121
  sim_donuts: v4_speed_mono@20261120, v4_speed_mono_edge@20261121
started: 2026-11-20
---

# `v4speed_danish_closure`: v4_speed: Danish fits to Batoid-traced donuts

## Question

When the donuts, the reference wavefront and the obscuration file all come from the v4_speed model, how well do Danish fits match the donuts, and how far are the fitted Zernikes from the Batoid wavefront, by defocus configuration, side and field position?

## Current answer

None yet.

## Log

### 2026-11-22 — sim_donut_fits built; examples with OPD against intrinsic

`sim_donut_fits` built (`6d381db`, cache fix `c70245d`): four variants, build `20261121`,
2342 stamps each (both `sim_donuts` variants), r band, noise-free stamps at 1e6 electrons,
fit variance sky 1000 electrons per 10 µm pixel plus the model. Medians within 1.725 deg;
deviation = |OPD - intrinsic|, quadrature sum over the fitted terms (µm of wavefront);
l1 = fraction of the flux misplaced (dimensionless); true FWHM 1.0 arcsec:

| variant | config | l1 | fitted FWHM (arcsec) | deviation median / max (µm) |
|---|---|---|---|---|
| sparse (bin 2) | fam | 0.0037 | 0.962 | 0.019 / 0.035 |
| sparse (bin 2) | cwfs | 0.0038 | 0.961 | 0.026 / 0.032 |
| sparse (bin 2) | giant | 0.0021 | 0.968 | 0.187 / 0.274 |
| sparse_bin1 | fam | 0.0016 | 0.988 | 0.009 / 0.020 |
| sparse_bin1 | cwfs | 0.0017 | 0.988 | 0.011 / 0.017 |
| sparse_bin1 | giant | 0.0021 | 0.992 | 0.186 / 0.271 |
| dense | fam | 0.0035 | 0.962 | 0.067 / 0.156 |
| dense | giant | 0.0019 | 0.968 | 0.133 / 0.248 |
| sparse_prod (no spider) | fam | 0.0185 | 0.972 | 0.071 / 0.091 |
| sparse_prod (no spider) | giant | 0.0254 | 1.001 | 0.274 / 0.335 |

Binning x2 doubles the FAM and CWFS deviation and lowers the fitted FWHM by 0.03 arcsec;
the giant deviation is the same binned or not (pupil edges). The dense set roughly triples
the FAM deviation; leaving the spider out, as production does, raises l1 fivefold.

Examples PDF (`fit_examples.py`, `93ec8da`, run `20261122_fit_examples`): data, model and
data - model for four field points per configuration and variant, plus the OPD per fitted
term against the batoid intrinsic with photon-noise 1-sigma errors from the fit Jacobian.
FAM 1-sigma at 1e6 electrons is about 0.023 µm (z5, z6 about 0.06 µm), so the binned
sparse bias is about 1 sigma. FAM residuals sit on one diagonal pair of spider vanes,
from 0.86 deg out: the vane shadow is misplaced in Danish.

Next:
- maps of deviation over field per Noll term; decide whether the product needs stored errors;
- the spider-vane mismatch on one diagonal;
- misaligned sims (Aaron's follow-up).

### 2026-11-20 — started

Study created.
