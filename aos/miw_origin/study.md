---
study: miw_origin
topic: aos
status: parked
question: "What produces the high-field-order Z5-Z8 (OCS, about 0.1 um RMS of wavefront) in the measured intrinsic wavefront that the batoid design does not predict: a static optic, a thermal state, or a bias in the wavefront retrieval?"
answer: "Retrieval bias leads: the per-visit L test gives leave-night-out CV R2 = +0.408 (dimensionless) against a null of +0.014, n = 1126 visits. No physically plausible static figure explains more than about 0.11 of the MIW (dimensionless, explained variance), and 83 % of MIW power sits at focal orders above the k <= 6 the fit uses."
uses: {}  # the coadds and miw products through the old aos/output tree; see coadd.md
started: 2026-08-18
---

# `miw_origin`: where the MIW's high-field-order astigmatism and coma come from

> **Code:** `code/` (the coadd-against-MIW scripts, the static-optics back-projections and
> camera-gravity model, and the smatrix field-order and thermal tests) · **Write-ups:**
> [`miw_investigation_handoff.md`](miw_investigation_handoff.md) (the portable state, with
> **retracted claims**), [`coadd.md`](coadd.md), [`static_optics.md`](static_optics.md),
> [`camera_gravity.md`](camera_gravity.md),
> [`miw_astig_coma_investigation.md`](miw_astig_coma_investigation.md) · **Equations:**
> [`../docs/miw_coadd_equations.md`](../docs/miw_coadd_equations.md)

## Question

What produces the high-field-order Z5–Z8 (OCS, about 0.1 µm RMS of wavefront) in the
measured intrinsic wavefront (MIW) that the batoid design does not predict: a static
optic, a thermal state, or a bias in the wavefront retrieval?

## Current answer

Parked since 2026-10-11 with its next steps listed (handoff §7).

- **The excess:** in the `danish_1_2_A_50_34_i_5rot` MIW, Z5–Z8 carry about 0.10–0.13 µm
  RMS of wavefront more than the design, at field radial order n = 3–5, telescope-fixed
  (OCS). The CCS part is significant only for Z4.
- **Focal-order truncation:** 0.169 of MIW power (dimensionless, power-weighted over 21
  pupil Zernikes) is at the k ≤ 6 focal orders the per-visit fit uses; 0.675 is at
  k = 7–30. Any subspace analysis inside k ≤ 6 sees at most 17 % of the MIW.
- **No static optic:** single-surface back-projection explains at most 0.11 of the MIW
  (dimensionless, explained variance); a joint four-surface fit reaches 0.43 only with
  20–24 µm RMS figures. A mirror would need about 1.7 µm of mid-spatial figure. Camera
  gravity and thermal sensitivities give low field orders, not n = 3–5.
- **Retrieval bias leads:** the per-visit L test gives leave-night-out CV R² = +0.408
  (dimensionless) against a null of +0.014, n = 1126 visits, 19 nights.
- **Coadd against MIW is thermal-associated:** environmental CV R² = +0.62
  (dimensionless), dominated by the M1M3 vertical gradient.

## Log

### 2026-10-11 — backfilled from docs and history

Built 2026-10-23 in the phase 5 conversion (decision 4) from four sources, dated by the
last real work (the handoff, 2026-10-11):

- aos `coadd` (8 scripts, `coadd.md`), formerly `aos/code/coadd/`. The builder
  `run_coadd_blocks_miw` stays in the `coadds` product, with its shim at the old path;
- aos `static_optics` (8 scripts, `static_optics.md`, `camera_gravity.md`), formerly
  `aos/code/static_optics/` and `aos/docs/`;
- the MIW handoff, formerly `aos/docs/status/miw_investigation_handoff.md`;
- smatrix `miw_astig_coma` (`miw_astig_coma_investigation.md`, 2026-08-18) with
  `pupil_zernike_study.py`, `thermal_study.py`, `thermal_sensitivity.py`,
  `demo_field_order.py` and `miw_ocs_analysis.py`, formerly `smatrix/code/`.

Moved scripts import `rubinwork.smatrix` and `rubinwork.products.miw` instead of the
shims. The camera-gravity and smatrix scripts now write to this study's `output/`; their
earlier outputs (`camera_gravity/`, `demo_field_order.png`) were copied there.
`pupil_zernike_study.py` still reads the full-mode sensitivity from `smatrix/output/`
(smatrix-internal data, triage decision 3). The coadd scripts keep reading and writing the
old `aos/output/coadd/` tree and the `coadds` product, as `coadd.md` describes. `rule
fam_coadd_miw` in `aos/Snakefile` now runs `miw_origin/code/run_fam_coadd_miw.py`.

Next (handoff §7): the k = 1–30 refit and the n_null definition; the k-independence test;
the batoid-intrinsic (BIW) against MIW comparison; the correction-truncation fix.
