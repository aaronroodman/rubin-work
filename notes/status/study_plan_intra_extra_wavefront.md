# Plan: the `unpaired_intra_vs_extra` study (Danish 1.3 blitz FAM)

> **Status:** proposal, awaiting Aaron · **Last updated:** 2026-10-15 · **Kind:** status (plan)

Aaron, 2026-10-15: extend the earlier per-visit blur study to the Zernikes. For each FAM
intra/extra pair, compare the wavefront measured from the intra-focal donut against the
extra-focal donut, and carry the blur study's thermal argument over to the Zernikes —
Z11 expected to be the interesting one.

**Scope: the one prior study incorporated here is aos `blur_intra_extra`**
(`aos/output/fam_processing/blur_intra_extra/`). `wfs/giant_donuts`,
`optics/intra_extra_phase_strip` and aos `cwfs` stay exactly as they are and none of
their work is included (Aaron, 2026-10-15). Triage question 6 is **not** answered here.

Nothing below has been created. Read-only checks already run are marked **checked**.

---

## 1. What the earlier blur study established

`aos/intra_extra/notebooks/blur_intra_extra_offset.ipynb`, output in
`aos/output/fam_processing/blur_intra_extra/`, handoff
`aos/intra_extra/blur_intra_extra_handoff.md`. Its argument, which this study repeats on
the Zernikes:

1. Median donut blur per visit per side of focus; difference intra − extra per FAM
   triplet. Median +0.3440 arcsec over 873 quality triplets, intra blurrier in 0.9954 of
   them (dimensionless).
2. **The difference is not constant.** A Huber fit of (intra − extra) blur on extra blur
   gives slope −0.5764 ± 0.0344 (dimensionless), 17σ from zero — the quadrature
   signature, q = +0.5614 arcsec².
3. Correlated against thermal telemetry: camera body minus M1M3 air is the strongest
   single quantity, Spearman rho = −0.7828 full sample, −0.2934 within night.
4. A linear Huber model on the thermal variables, subtracted, **narrowed the spread and
   removed the residual correlation against the extra-focal blur** — the Huber slope
   flattened from −0.5788 ± 0.0344 to −0.0868 ± 0.0171 arcsec per arcsec.
5. Conclusion: **thermal effects drive the intra − extra difference, not the overall
   seeing.**

Two corrections to carry forward, both checked:

- **That notebook ran on `danish_1_3_test`**, not v1000 — `PARAM_SET = 'danish_1_3_test'`
  at its top. Its +0.3440 arcsec reproduces on `test`; v1000 gives +0.3445 arcsec.
- Its thermal-variable selection added variables in order of their own correlation with
  the target, which double-counts because the thermal variables are themselves highly
  correlated. Item 4 below replaces that algorithm.

---

## 2. Placement

**Study: `aos/unpaired_intra_vs_extra/`**, per section 7 of
`notes/status/organization_plan.md`. First study created in the new layout.

```yaml
---
study: unpaired_intra_vs_extra
topic: aos
status: active
question: Does the wavefront measured from the intra-focal donut differ from the
  extra-focal donut of the same FAM pair, and is that difference thermally driven as the
  blur difference is?
answer: (open)
uses:
  fam_tables: danish_1_3_v1000@20261013, danish_1_3_test@20261013
started: 2026-10-15
---
```

Directory: `study.md`, `notebooks/`, `figures/`, and
`output@ -> /sdf/group/rubin/u/roodman/LSST/rubin-work/studies/aos/unpaired_intra_vs_extra`
(gitignored symlink). Per item 6 below: **one notebook, one PDF, one `study.md`** — no
`code/` unless something outgrows the notebook.

The earlier blur study's handoff becomes the **first log entry** of `study.md`, per
section 7 ("Handoffs become log entries"). Its thermal derivation moves beside `study.md`
as `blur_thermal.md`. The old `aos/intra_extra/blur_intra_extra_handoff.md` and the old
notebook stay on disk until Aaron says otherwise — no deletions.

---

## 3. Data

**Primary: `danish_1_3_v1000@20261013`.** The pupil model Aaron wants going forward
(as-built + M1M3 + M1 baffles) and the sample he expects to update as Danish and the
pupil model advance. `danish_1_3_test@20261013` is kept as a second column, for the
+0.3440 arcsec reproduction check and to see which Zernike differences the pupil model
moves.

Both builds already exist — no reprocessing needed. **Checked:**

| | `danish_1_3_v1000@20261013` | `danish_1_3_test@20261013` |
|---|---|---|
| Butler collection | `u/jmeyers3/t614_fam_unpaired_v1000` | `u/jmeyers3/t614_fam_unpaired` |
| pupil model | v1000: as-built + M1M3 + M1 baffles | blitz default |
| `donuts.parquet` rows | 3,155,836 | 3,155,734 |
| columns / row groups | 66 / 966 | 66 / 966 |
| size | 13.296 GB | 13.296 GB |
| visits, nights | 966, 15 | 966, 15 |
| `visit_quality_pass` | 873 visits, 15 nights | 873 visits, 15 nights |
| `day_obs` range | 20260315–20260619 | 20260315–20260619 |
| median(intra − extra) visit blur | +0.3445 arcsec | +0.3440 arcsec |

**v1000 full-scan completeness, checked:** all 966 row groups read, 3,155,836 rows, 966
distinct `(day_obs, seq_num)`, **zero** rows with a non-finite per-side blur or per-side
Zernike. 21 Noll indices, [4..19, 22..26], both builds. All 18 `SIDE_COLUMNS` present.

Selection: `visit_quality_pass` (873 of 966 visits, as the blur study); paired rows
(`matched_intra_extra`); field radius ≤ 1.725 deg in OCS as a guard (checked: max on the
sampled visit is 1.7240 deg, so it removes nothing).

**The per-side Zernikes are independent single-donut Danish fits** — the blitz product is
unpaired, one row per donut per exposure, and `pair_sides`
(`rubinwork/products/fam_tables/builders/blitz_reader.py:317-408`) joins the two sides on
`(det_name, donut_id)` *after* each is fitted. So intra − extra is a difference of two
independent measurements of the same star.

**The intrinsic cancels in the difference, so this is a measurement and not a model
artifact.** Each side's intrinsic is evaluated at its own field angle (median 4.36 arcsec
apart between sides). Checked on one visit, median(intra − extra): intrinsic Z11
−0.00008 µm of wavefront against a total of +0.2691 µm of wavefront. Total and deviation
therefore give the same difference to 4 decimals, and **the primary quantity is the
total, `zk_OCS_intra − zk_OCS_extra`, in µm of wavefront** — no dependence on which MIW
build supplied the intrinsic. Also checked: `zk_OCS` = 0.5·(intra + extra) to
1.8e-15 µm of wavefront, so the paired wavefront is exactly the two-side mean.

Per-donut Danish fit covariance is **not stored** (only `lstsq_cost_{intra,extra}` and
`lstsq_optimality_{intra,extra}`), so no fit-covariance diagnostic is available from the
tables.

---

## 4. The plots, in Aaron's order

Everything is **per FAM visit**: the median over that visit's donuts, per side, as the
blur study did. `ΔZ_k ≡ median(Z_k,intra) − median(Z_k,extra)` in µm of wavefront;
`mean_k ≡ (median(Z_k,intra) + median(Z_k,extra))/2`; `Δblur` in arcsec. n = 873 visits.

### Plot 1 — per-side medians and the difference, for all 21 Zernikes

Two panels per Noll index, matching the blur study's layout:

- **(a)** median intra and median extra against **visit ordinal**, two markers/colours,
  night boundaries marked so within-night drift is separable by eye.
- **(b)** median(intra − extra) per FAM visit against visit ordinal, same night
  boundaries.

21 Noll indices × 2 panels. Laid out as a grid over several PDF pages (one page per 3–4
Noll indices keeps the ordinal axis readable), all in the single study PDF.

### Plot 2 — which Zernikes have a substantial difference

Ranking figure plus a table: for each Noll index, `median(ΔZ_k)` in µm of wavefront
against `median(|mean_k|)` in µm of wavefront, and the **ratio
`median(ΔZ_k) / median(|mean_k|)` (dimensionless, difference over two-side mean)** — the
"substantial compared to the size of the mean" test. The nMAD of `ΔZ_k` over visits is
reported with each, so a large ratio that is pure scatter is distinguishable from a real
offset.

Expectation is Z11 only. Early evidence says more may qualify: on one visit of `test`,
median(intra − extra) over 3,381 donuts in µm of wavefront was **Z11 +0.2691 (nMAD 0.0167
over donuts)**, **Z14 −0.3877 (0.0759)**, Z6 −0.2095 (0.2171), Z7 +0.1186 (0.1274),
Z9 −0.1083 (0.0980), Z15 −0.1000 (0.0743), Z26 +0.0975 (0.0165), Z4 −0.0303 (0.1564).
Z11's nMAD is 1/16 of its median. Whether these survive at visit level over 873 visits,
and how they compare against their own means, is what the plot decides — **I am not
assuming Z11 is the only one.**

### Plot 3 — Δblur against ΔZ11

Scatter of `Δblur` in arcsec against `ΔZ11` in µm of wavefront, one point per FAM visit,
n = 873. With Spearman rho and Pearson r (both, with n) and a Huber RLM slope ± its
standard error in µm of wavefront per arcsec. Points coloured by night, since the blur
study found roughly half the full-sample signal is night-level covariance — so the same
numbers are also reported **within-night de-meaned**. Repeated for any other Noll index
plot 2 flags.

### Plot 4 — the thermal selection, generalised

A general routine, developed in this notebook and later moved into `rubinwork`:

```
select_thermal_predictors(target, predictors, n_max, ...) -> ordered list, signs, fit
```

Given a target time series (per visit) and a set of telemetry time series on the matching
index, pick predictors by **greedy forward selection on the incremental gain**, which is
the fix to the earlier ordering:

- **a.** Rank all candidates by their own correlation with the target. Record each one's
  **sign** from this first round, so the signs are fixed before any orthogonalisation.
- **b.** Take the leading variable. Then **re-rank every remaining candidate by how much
  it adds given variable 1** — not by its own correlation with the target — and take the
  biggest addition. "Addition" measured as the reduction in residual nMAD of the Huber
  fit, reported in the target's units, with partial correlation alongside.
- **c.** Continue to **`n` predictors, `n` settable**.

Reported per step: the variable, its sign from step (a), the incremental reduction in
residual nMAD (target units), the partial correlation, and a collinearity guard (VIF,
dimensionless) so a step that is nearly a repeat of an earlier variable is visible rather
than silent. Fit is Huber RLM throughout (`statsmodels RLM(y, X, M=HuberT())`), per the
repo default.

Run on **`Δblur` in arcsec first**, to reproduce the earlier study's conclusion under the
better algorithm, then on **`ΔZ11` in µm of wavefront**, and on any other Noll index plot
2 flags. The thermal variables and the per-exposure join already exist, cached at
`aos/output/fam_processing/blur_intra_extra/thermal_join_per_side_cam.parquet` (966 rows);
the earlier notebook ranked 22 thermal quantities, which is the candidate set.

### Plot 5 — after the correction

For each target (`Δblur`, `ΔZ11`, plus any others): the Huber-corrected residual
`Δ − model(thermal)` against **the two-side mean** and against **the extra-focal value**,
before and after, with the Huber slope ± error on each. This is the step that carried the
blur study's conclusion — the earlier flattening was −0.5788 ± 0.0344 to
−0.0868 ± 0.0171 arcsec per arcsec. If `ΔZ11`'s residual correlation flattens the same
way, thermal drive extends to the wavefront; if it does not, the blur and the Z11
asymmetries are different effects, which is equally worth knowing.

---

## 5. Output layout (item 6)

**One notebook, one PDF, one Markdown file.** Proposed as the standing rule for studies:

- `aos/unpaired_intra_vs_extra/notebooks/unpaired_intra_vs_extra.ipynb` — the analysis.
- `output@/<YYYYMMDD>_<label>/unpaired_intra_vs_extra.pdf` — **one** multi-page PDF
  carrying every figure above, via `matplotlib.backends.backend_pdf.PdfPages`.
- `aos/unpaired_intra_vs_extra/study.md` — front matter, Question, Current answer, Log.
- `figures/` holds at most two PNGs pulled from the PDF for the log entry.

This is a change to the conventions, so it needs writing into the rules rather than just
followed here: section 7 of `notes/status/organization_plan.md` and the `/start-study`
skill that replaces `rubin-new-study`. Both are **phase 5 work owned by another session**,
so this plan records the rule and does not edit those files. Suggested wording:

> A study's default output is one notebook, one multi-page PDF and one `study.md`. Reach
> for a second notebook or a `code/` directory only when something genuinely outgrows the
> first, and say why in the log.

---

## 6. Checks to reproduce

| check | target | status |
|---|---|---|
| quality sample | 873 visits, 15 nights, 20260315–20260619 | **reproduced** |
| median(intra − extra) visit blur, `test` | +0.3440 arcsec | **reproduced** |
| paired wavefront = two-side mean | to 1.8e-15 µm of wavefront | **reproduced** |
| per-side intrinsic cancels in the difference | ≤ 0.0001 µm of wavefront in the median | **reproduced** |
| fraction with intra > extra | 0.9954 (dimensionless, of 873) | to do |
| quadrature parameter | q = +0.5614 arcsec² | to do |
| Huber slope, Δblur on extra blur | −0.5764 ± 0.0344 (dimensionless) | to do |
| thermal correction flattens the blur slope | −0.5788 ± 0.0344 → −0.0868 ± 0.0171 arcsec/arcsec | to do, under the new selection algorithm |

The last one will not reproduce exactly — the selection algorithm is deliberately
different — so the check is that the *conclusion* survives, not the coefficients.

---

## 7. Cost

Interactive on an s3df node. No batch job. **Calibrated, checked:** a full pass over all
966 row groups of `donuts.parquet` with 6 columns projected took about 3 minutes, so the
~25-column passes below are 10–15 minutes, not the 25 estimated earlier.

| step | passes over the 13 GB table | wall clock | memory |
|---|---|---|---|
| per-visit per-side medians, 21 Noll + blur, v1000 | 1 | 10–15 min | < 8 GB, one row group at a time |
| same for `test` | 1 | 10–15 min | < 8 GB |
| plots 1, 2, 3 | none — read the cached per-visit table | 5 min | < 1 GB |
| plot 4, thermal selection | none — the 966-row thermal cache exists | 5 min per target | < 1 GB |
| plot 5 | none | 2 min | < 1 GB |

The reduced per-visit table is 873 visits × 21 Noll × 2 sides plus blur — a few hundred
kB — cached once under the run directory, so only the two initial passes touch the big
file. **Total about 45 min**, which fits "results tonight".

---

## 8. Questions for Aaron

1. **Visit-level median only, or also pair level?** The plots above are all per-visit
   medians (n = 873), matching the blur study. The per-donut pair level (n ≈ 2.8 M) would
   give the within-visit slope of ΔZ11 on Δblur — a useful bound on fit degeneracy, since
   within one visit the wavefront is common and only the blur varies donut to donut. Worth
   adding as one extra panel, or keep this study purely visit-level?
2. **"Substantial" in plot 2** — is `median(ΔZ_k) / median(|mean_k|)` (dimensionless) the
   right test, and is there a threshold you want drawn on the figure, or just the ranking?
3. **`n` for the thermal selection** — default to 5, as the earlier study's 5-channel
   thermal_focus model? And keep the VIF < 10 collinearity guard as a reported diagnostic
   rather than a hard gate?
4. **Does the de-meaning stay within-night?** The blur study de-meaned by night and found
   half the full-sample signal was night-level. Same for the Zernikes, or do you want the
   full-sample numbers primary with within-night alongside?
5. **v1000 primary with `test` as a second column**, as proposed — or v1000 only, with
   `test` used solely for the +0.3440 arcsec reproduction check?
