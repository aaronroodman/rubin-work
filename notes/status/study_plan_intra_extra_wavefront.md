# Plan: the `unpaired_intra_vs_extra` study (Danish 1.3 blitz FAM)

> **Status:** proposal, awaiting Aaron · **Last updated:** 2026-10-15 · **Kind:** status (plan)

Aaron, 2026-10-15: compare the wavefront measured from the intra-focal and the
extra-focal donut of each FAM intra/extra pair, then correlate the intra − extra Zernike
differences with the intra − extra blur differences. Extends the earlier blur study
(`aos/docs/status/blur_intra_extra_handoff.md`).

Nothing below has been created. The only read-only checks already run are reported inline,
each marked **checked**.

---

## 1. Placement

**Proposed study: `aos/unpaired_intra_vs_extra/`**, per section 7 of
`notes/status/organization_plan.md`. This is the first study created in the new layout.

Front matter for `aos/unpaired_intra_vs_extra/study.md`:

```yaml
---
study: unpaired_intra_vs_extra
topic: aos
status: active
question: Does the wavefront measured from the intra-focal donut differ from the
  extra-focal donut of the same FAM pair, and is the difference the same thing as the
  intra/extra blur difference?
answer: (open; the first entry is the blur result)
uses:
  fam_tables: danish_1_3_test@20261013, danish_1_3_v1000@20261013
started: 2026-10-15
---
```

Directory, per section 3: `study.md`, `code/`, `notebooks/`, `figures/`, and
`output@ -> /sdf/group/rubin/u/roodman/LSST/rubin-work/studies/aos/unpaired_intra_vs_extra`
(gitignored symlink; run directories `<YYYYMMDD>_<label>/` under it).

### The four prior studies triage question 6 names

Question 6 of `notes/status/phase5_study_triage.md:210-212` names these four. Inventory
checked on disk, 2026-10-15:

| # | study | docs | code | notebooks | output on disk | state |
|---|---|---|---|---|---|---|
| 1 | aos `blur_intra_extra` | `aos/docs/status/blur_intra_extra_handoff.md` (455 lines); the notebook row and per-side paragraph in `aos/docs/studies/fam_processing.md` | none of its own — reads `fam_tables` | `aos/notebooks/fam_processing/blur_intra_extra_offset.ipynb` | `aos/output/fam_processing/blur_intra_extra/` → `/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/fam_processing/blur_intra_extra/`: 10 PDFs + 3 parquet caches, **3 PDFs and 1 parquet orphaned** by the 2026-10-04/06 rewrites | active |
| 2 | wfs `giant_donuts` (todo item 7) | `wfs/docs/giant_donuts_pupil_findings.md` (401 lines); `notes/status/item7_giant_donuts_handoff.md` | `wfs/code/giant_donuts/` — 7 modules + 1 test (`fit_giant_donut.py`, `radial_profile.py`, `spider_trajectory.py`, `donut_images.py`, `gen_mask_params.py`, `select_exposures.py`) | `wfs/notebooks/giant_donuts/giant_donut_radial_profiles.ipynb` (**modified, uncommitted**) | `wfs/output/giant_donuts/` — 10 PDFs, R22_S10, *in the repo tree*, not the data root; plus 3 loose `wfs_giant_donut_*.pdf` in `wfs/output/` | active, own open next steps |
| 3 | optics `intra_extra_phase_strip` | `optics/docs/studies/intra_extra_phase_strip.md` only | **none** | **none** | **none** — `optics/output/` holds only `.gitkeep` | writeup only: a hypothesis plus a symmetry argument. Its status line says "not yet simulated"; the triage says it was, on 2026-10-09, but no code or output for that simulation is in `optics/` |
| 4 | the Z11/Z14 part of aos `cwfs` | the "State and open questions" section of `aos/docs/studies/cwfs.md`; `aos/docs/ts_wep_zernike_intrinsics.md` for the CCD-height argument | `aos/code/cwfs/` — but shared with the optical-state comparison, which is a *different* question | `aos/notebooks/cwfs/` (4, shared the same way) | `aos/output/wfs_corner_compare/danish_1_2/`, `aos/output/danish_tarts_compare_20260713/` (6 PDFs) — again shared with the other track | parked; only the Z11/Z14 *finding* is in scope, not the study |

What each contributes to the one question:

- **1** is the only one with ensemble statistics: 873 quality FAM triplets, 15 nights. It
  measures the blur asymmetry (+0.3440 arcsec) and says nothing yet about the Zernikes.
- **2** is the one deep dive: 6 giant donuts at 8 mm hexapod defocus, 3 pupil models, and
  the Z11 split 0.81 → 0.37 µm of wavefront under `v1000`. One detector, 2 exposures.
- **3** is the only candidate *mechanism* with the right sign: a 50 nm, 5 cm turned-down
  M1 inner edge giving +0.82 / −0.34 arcsec intra / extra in quadrature.
- **4** is the independent instrument check: the same Z11/Z14 split appears in the corner
  sensors, so it is not special to FAM.

The new study adds the fifth leg, and the one none of them has: the **per-side Zernikes
over the whole FAM ensemble**.

**Recommendation on triage question 6: merge, and put it in `aos`.** One question — why
the two sides of focus disagree — asked of four data sets. Making it one study is what
lets the giant-donut Z11 split, the M1-edge-strip mechanism, the CWFS Z11/Z14 offset and
the FAM blur asymmetry be checked against each other instead of each being re-explained.
`aos` because the FAM ensemble (873 visits, 3.16 M donut rows) is the only sample with
statistical power, and it lives in `aos`.

How each one moves, concretely:

- **1 `blur_intra_extra` → merged in**, as the first log entry (below). It has no code of
  its own, so only the notebook moves to `aos/unpaired_intra_vs_extra/notebooks/`.
- **3 `intra_extra_phase_strip` → merged in** as a derivation beside `study.md`
  (`m1_edge_strip.md`). It is a writeup with no code or output, so this is a file move and
  a status-line fix. Lowest-risk of the four.
- **2 `giant_donuts` → Aaron's call.** It is *active*, has 7 code modules and its own open
  next steps. Merging it means moving `wfs/code/giant_donuts/` into the new study and
  carrying the item-7 next steps into the log verbatim. Keeping it separate and
  cross-linking costs nothing now. **My recommendation: keep `wfs/giant_donuts` as its own
  study and cross-link it**, changing what I proposed earlier in this file — its code and
  its 8 mm single-detector question are distinct enough from the 1.5 mm FAM ensemble that
  merging buys a shared `study.md` and little else, and it is the one merge that could
  lose live work.
- **4 `cwfs` → cross-link only.** The Z11/Z14 finding is cited in the new study's log; the
  study itself stays, since its code, notebooks and output are shared with the
  optical-state comparison and cannot be split cleanly.

So: merge 1 and 3 into `aos/unpaired_intra_vs_extra`, cross-link 2 and 4.

**The earlier blur study's handoff becomes the first log entry** of
`aos/unpaired_intra_vs_extra/study.md`, dated 2026-10-06, per section 7 ("Handoffs become log
entries"). It is 455 lines and much of it is thermal-regression detail, so: the result
tables, the per-side negative test and the whole "Tried and rejected" section go into the
log entry; the thermal model derivation moves beside `study.md` as
`aos/unpaired_intra_vs_extra/blur_thermal.md`. Nothing is deleted without Aaron's say-so — the old
`aos/docs/status/blur_intra_extra_handoff.md` stays until he confirms.

---

## 2. What the per-side Zernikes are — **checked, and this is the key result of the session**

**Each side is an independent single-donut Danish fit.** The blitz product
(`donutBlitzFamResults`) is *unpaired*: one row per donut per exposure, each fitted on its
own. There is no joint pair fit to take parts of.
`rubinwork/products/fam_tables/builders/blitz_reader.py:5-10` (module docstring) and
`pair_sides` at `:317-408`, which joins the two sides on `(det_name, donut_id)` *after*
each has been fitted. The paired row's wavefront is then the arithmetic mean of the two
(`:24-36`, verified below).

So `intra − extra` is a difference of two independent measurements of the same star, not a
decomposition of one fit.

**Where each side's intrinsic comes from.** The blitz product supplies
`zk_deviation_*` and `zk_intrinsic_*` per donut and no total; the total is reconstructed as
`zk = zk_deviation + zk_intrinsic` (`blitz_reader.py:56-59`). The intrinsic is evaluated at
**each side's own field angle**, which differ because the two exposures point slightly
differently.

Checked on `danish_1_3_test@20261013`, row group 0 (`day_obs` 20260315 `seq_num` 73,
3,381 paired donuts):

| quantity | value |
|---|---|
| per-side field-angle offset, OCS | median 4.36 arcsec, max 7.74 arcsec |
| intrinsic Z4, median(intra − extra) | −0.00007 µm of wavefront |
| intrinsic, max abs(intra − extra) over 21 Noll terms, all donuts | 0.0256 µm of wavefront |
| `zk_OCS` − 0.5·(`zk_OCS_intra` + `zk_OCS_extra`), max abs | 1.8e-15 µm of wavefront |

**The side-dependent intrinsic cancels.** The intrinsic difference is 4 orders of
magnitude below the signal, so total and deviation give the *same* intra − extra to 4
decimal places:

| Noll | median(intra − extra), total [µm wf] | same, deviation [µm wf] | same, intrinsic [µm wf] |
|---|---|---|---|
| Z4 | −0.0303 | −0.0303 | −0.00007 |
| Z11 | +0.2691 | +0.2692 | −0.00008 |
| Z14 | −0.3877 | −0.3877 | −0.00000 |
| Z22 | −0.0318 | −0.0319 | +0.00001 |

**So intra − extra is a measurement, not a model artifact.** This is the question that
decided whether the study is worth running, and the answer is yes.

**The CCD-height term does not enter the difference.** `aos/docs/ts_wep_zernike_intrinsics.md`
records `zkStart[Z4] += Z4_height` with sign "extra +, intra −"
(`HEIGHT_TO_Z4_UM_PER_MM = 15`), which *would* put 2·15.8 nm of Z4 per µm of height into
the difference — but the same document's batoid check (2026-10-11) finds a height h adds
+15.8·h nm of Z4 on **both** sides, so it cancels, leaving only the height difference
between the two donut positions, below 0.25 nm of Z4 for the same star. In FAM both sides
of a pair are the *same* sensor and the same star, so the corner-pair case (about −72 nm)
does not apply. **The two conventions are not reconciled in that document and this study
should settle it**: if ts_wep's "extra +, intra −" were what the data saw, Z4_intra −
Z4_extra would carry −2·15.8 nm/µm × the per-detector height, giving a per-detector
pattern in ΔZ4 of RMS about 0.10 µm of wavefront (from the 3.1 µm RMS per-CCD piston). The
measured median ΔZ4 is −0.0303 µm of wavefront with a per-detector map available — so
analysis (b) below tests this directly, and that is the open check closed.

**Primary quantity: the total wavefront, `zk_OCS_intra − zk_OCS_extra`, in µm of
wavefront.** Reason: for the difference it is numerically identical to the deviation
(table above), and the total needs no statement about which MIW build the intrinsic came
from. The deviation is reported alongside for Z4 only, where the intrinsic is largest, as
a consistency line.

---

## 3. Data and selection

Both builds exist and carry every per-side column — **checked**:

| | `danish_1_3_test@20261013` | `danish_1_3_v1000@20261013` |
|---|---|---|
| `donuts.parquet` rows | 3,155,734 | 3,155,836 |
| columns / row groups | 66 / 966 | 66 / 966 |
| size | 13.296 GB | 13.296 GB |
| `visits.parquet` rows / cols | 966 / 24 | 966 / 24 |
| `fits.parquet` rows / cols | 873 / 449 | 873 / 449 |
| visits, nights | 966, 15 | 966, 15 |
| `visit_quality_pass` visits, nights | 873, 15 | 873, 15 |
| `day_obs` range | 20260315–20260619 | 20260315–20260619 |
| median(intra − extra) visit blur [arcsec] | +0.3440 | +0.3445 |

All 18 `SIDE_COLUMNS` present in both. Noll indices in `visits.nollIndices`: **[4..19,
22..26], 21 terms** in both. The variant contrast is the **pupil model** (`v3_14`-era
default against `v1000`); the blur reproduces the handoff's +0.3440 arcsec exactly on
`test`, so the two builds differ in the fit, not the sample.

Selection:
- `visit_quality_pass` on `visits.parquet`, as the blur study (873 of 966 visits). Per
  memory, `median_blur_arcsec <= 1.2 arcsec` is usually the cut that bites.
- field radius `<= 1.725 deg` in OCS, computed from `thx_OCS`/`thy_OCS` (radians in the
  parquet). **Checked:** max field radius on the sample row group is 1.7240 deg, so this
  limit removes nothing there and is a guard, not a cut.
- paired rows only (`matched_intra_extra`), and finite per-side Zernikes on both sides.

Pair counts per variant are reported in the first run, not guessed here; the paired donut
count over the 873 quality visits is roughly 2.85 M rows if the sample row group
(3,381 pairs per visit) is typical.

Column projection only — the donut table is 13.3 GB and has one row group per visit, so
every pass reads the ~25 needed columns visit by visit.

---

## 4. Analyses

Pair level = one paired donut. Visit level = median over the donuts of one visit.
`ΔZ_k ≡ Z_k,intra − Z_k,extra` in µm of wavefront; `Δblur ≡ blur_intra − blur_extra` in
arcsec.

**(a) Per Noll index, the difference.** For each of the 21 Noll indices: median and nMAD of
`ΔZ_k` over pairs, per visit (873 values) and pooled, in µm of wavefront. Plus the
wavefront RMS of the difference vector, `sqrt(sum_k ΔZ_k²)` in µm of wavefront (amplitude,
not power), per pair and its median over the sample.

Already visible on the one sampled visit (20260315 seq 73, 3,381 pairs, `test`), median and
nMAD over pairs in µm of wavefront: **Z11 +0.2691 (nMAD 0.0167)**, **Z14 −0.3877
(nMAD 0.0759)**, Z6 −0.2095 (0.2171), Z7 +0.1186 (0.1274), Z9 −0.1083 (0.0980),
Z15 −0.1000 (0.0743), Z26 +0.0975 (0.0165), Z4 −0.0303 (0.1564). Z11's nMAD is 1/16 of its
median — this is a precise, systematic offset, not noise.

**(b) Field dependence.** Focal-plane maps of median `ΔZ_k` per detector (189 science
CCDs) for the terms that survive (a), in **OCS and in CCS**, pooled over the sample. A
pattern fixed in OCS is telescope-fixed (optics, M1 edge); fixed in CCS is camera-fixed
(detector heights, pupil model, camera optics). The rotator spans enough angle over 15
nights to separate them; `rotator_angle` (= `rotTelPos`, phase 3b) and
`rotator_angle_source` are in `visits.parquet`. Also the ΔZ4-against-per-detector-height
regression named in section 2, in nm of wavefront per µm of detector height, against the
predicted 0 nm/µm (batoid: the term cancels) and −31.6 nm/µm (= −2 × 15.8, my arithmetic
from the ts_wep `zkStart` "extra +, intra −" convention, not a number either document
states).

**(c) ΔZ_k against Δblur.** At pair level (`blur_intra`/`blur_extra` per donut) and visit
level (`median_blur_intra_arcsec`/`median_blur_extra_arcsec`), for each Noll index:
Spearman rho with n, Pearson r with n, and a **Huber RLM slope with its standard error**
in µm of wavefront per arcsec (`statsmodels RLM(y, X, M=HuberT())`, as `dz_fitting.py`).
Each reported twice: full sample and **within-night de-meaned** (subtract the per-night
median of both variables) — the blur study found roughly half the full-sample signal is
night-level covariance with the seeing, so the full-sample number alone overstates it.

**(d) `test` against `v1000`.** The same table of per-Noll median `ΔZ_k` for both builds,
and the difference between them in µm of wavefront, over the identical 873 visits. Z11 is
expected to move. **Already checked on the one sampled visit**, median(intra − extra) in µm
of wavefront: Z11 **+0.2691 → +0.1916** (−29%), Z14 −0.3877 → −0.3855 (unchanged),
Z4 −0.0303 → −0.0279, Z22 −0.0318 → −0.0238. So the pupil model moves Z11 and leaves Z14
alone, which is the giant-donut result appearing in the FAM ensemble.

**(e) The fit-degeneracy null test.** Danish fits blur jointly with the Zernikes, so the
+0.344 arcsec blur asymmetry could leak into Z4, Z11 or Z22 through fit covariance rather
than being a wavefront difference.

**No stored covariance exists** — **checked**: `donuts.parquet` carries only
`lstsq_cost_{intra,extra}` and `lstsq_optimality_{intra,extra}` per side, no per-donut
parameter covariance or Jacobian; `fits.parquet`'s 449 columns are DZ-fit coefficients and
their errors, a later stage. So the test must be a forward simulation:

1. Take a handful of real donuts spanning field radius. Simulate with `danish` at known
   Zernikes (the measured paired mean) and a known blur, once per side with the matching
   defocal sign.
2. Refit with the same configuration and free blur, and record the recovered Zernikes —
   the zero-asymmetry baseline.
3. Re-simulate with a blur difference imposed between the sides matching the data
   (`b_intra² = b_extra² + 0.5614 arcsec²`, the blur study's quadrature model), refit, and
   measure how much ΔZ4, ΔZ11 and ΔZ22 the fit invents from blur alone, in µm of
   wavefront.
4. Compare against the measured ΔZ_k. The item-7 handoff already has a strong prior that
   this leak is small: the giant-donut Z11 split moves by 0.0001 to 0.007 µm of wavefront
   between free blur and blur fixed at the measured 0.7 arcsec seeing, against a 0.25 µm
   of wavefront unexplained residual. The simulation turns that into a number for the FAM
   geometry.

A cheap data-side proxy first, because it costs one pass: the Huber slope of `ΔZ11` on
`Δblur` *within* a visit (where the wavefront is common and only the blur varies donut to
donut) bounds the leak without any simulation.

---

## 5. Existing results to reproduce as checks

| check | target | source |
|---|---|---|
| quality sample | 873 triplets, 15 nights, 20260315–20260619 | blur handoff — **already reproduced** |
| median(intra − extra) visit blur | +0.3440 arcsec | blur handoff — **already reproduced**, 0.3440 on `test` |
| fraction with intra > extra | 0.9954 (dimensionless, of 873) | blur handoff |
| quadrature parameter | q = +0.5614 arcsec² | blur handoff |
| Huber slope, (intra − extra) blur on extra blur | −0.5764 ± 0.0344 (dimensionless) | blur handoff |
| paired wavefront = mean of sides | max abs deviation 1.8e-15 µm of wavefront | **already reproduced** |
| Z11 split, pupil model | 0.81 → 0.37 µm of wavefront (giant donuts, 8 mm, R22_S10) | item-7 handoff |
| Z11/Z14 sign | Z11 positive intra, negative extra; Z14 splits the same way | `aos/docs/studies/cwfs.md` |
| Z22 splits opposite in sign to Z11 and far less | −0.074 (v3.14), −0.015 (v1000) µm of wavefront | item-7 handoff |

The giant-donut numbers are 8 mm hexapod defocus on one detector and will not reproduce
numerically against 1.5 mm FAM over 189 detectors — they are **sign and ordering** checks
(Z11 positive intra, v1000 roughly halving it, Z22 opposite and small). Reproducing the
0.80/0.37 µm of wavefront magnitude is not expected and its absence is not a failure.

---

## 6. Cost

All interactive on an s3df node. No batch job. One pass over `donuts.parquet` with ~25 of
66 columns projected, 966 row groups, is the unit of cost.

| analysis | passes | wall clock | memory |
|---|---|---|---|
| (a) per-Noll difference | 1 per variant | 15–25 min each | < 8 GB (one row group at a time) |
| (b) field maps, OCS and CCS | reuses (a)'s per-detector accumulation | + 5 min | < 8 GB |
| (c) ΔZ vs Δblur, pair and visit | reuses (a) if the pair-level arrays are cached | + 10 min | < 16 GB for the cached pair table |
| (d) test vs v1000 | the two (a) passes | — | — |
| (e) null test, data proxy | within (a) | + 5 min | < 8 GB |
| (e) null test, danish simulation | none over the table | 1–2 h, a few donuts × 3 configs × 2 sides | < 4 GB |

Total about 1.5 h of table passes plus 1–2 h of simulation. The per-pair reduced table
(873 visits × ~3,400 pairs × 21 Noll × 2 sides) is cached once under the study's run
directory as parquet (roughly 1–2 GB) so (c) and (e) need no second pass.

---

## 7. Figures and the first answer

Planned, to `figures/` (a few committed PNGs) with the PDFs under `output@`:

1. `delta_zk_per_noll.png` — median ± nMAD of `ΔZ_k` over 21 Noll indices, µm of
   wavefront, both variants on one panel.
2. `delta_zk_field_maps.png` — focal-plane maps of `ΔZ4`, `ΔZ11`, `ΔZ14` in OCS and CCS,
   µm of wavefront.
3. `delta_zk_vs_delta_blur.png` — `ΔZ11` and `ΔZ4` against `Δblur`, visit level, full
   sample and within-night de-meaned, with the Huber lines.
4. `delta_z4_vs_detector_height.png` — `ΔZ4` in nm of wavefront against per-detector
   height in µm, with the batoid (0 nm/µm) and ts_wep-convention (−31.6 nm/µm, derived) predictions drawn.
5. `blur_quadrature.png` — the blur result, carried from the earlier study as the first
   log entry's figure.

`study.md`'s first **Current answer**, as it stands today from the checks above:

> The intra-focal and extra-focal donuts of the same FAM pair give different wavefronts,
> and the difference is a measurement rather than a model artifact: the side-dependent
> intrinsic cancels to below 0.0001 µm of wavefront in the median, so the total and the
> deviation give the same difference. On one visit of `danish_1_3_test@20261013`
> (`day_obs` 20260315 `seq_num` 73, 3,381 paired donuts) the median intra − extra is
> +0.2691 µm of wavefront in Z11 (nMAD 0.0167 over pairs) and −0.3877 µm of wavefront in
> Z14 (nMAD 0.0759). The `v1000` pupil model cuts the Z11 difference to +0.1916 µm of
> wavefront, 29 per cent, and leaves Z14 unchanged — the same pupil-model dependence the
> 8 mm giant donuts show. The intra side is also blurrier by a median +0.3440 arcsec over
> 873 quality visits, a quadrature term q = +0.5614 arcsec²; whether the two asymmetries
> are the same effect is the open question.

---

## 8. Questions for Aaron

1. **Triage question 6**, now that the four are inventoried in section 1 — merge
   `blur_intra_extra` (1) and `intra_extra_phase_strip` (3) into
   `aos/unpaired_intra_vs_extra` and cross-link `giant_donuts` (2) and `cwfs` (4), as
   recommended? Or merge all four, which means moving `wfs/code/giant_donuts/`'s 7 modules
   and folding an active study with its own open next steps?
2. **Robust method** — Huber RLM for every ΔZ-against-Δblur slope, per the default? Or a
   different robust method for the pair-level fits, where n is ~3 M and the leverage of a
   few bad donuts is different from the visit-level n = 873?
3. **Primary quantity** — total wavefront for the difference, as recommended (identical to
   the deviation for ΔZ, and no MIW-build dependence)? Or report both throughout?
4. **The null test** — is the `danish` forward simulation in scope now, or do the cheap
   within-visit proxy and the item-7 blur-robustness result settle it well enough to defer?
5. **The handoff** — does `aos/docs/status/blur_intra_extra_handoff.md` get superseded in
   place (left on disk, marked superseded, pointing at the new log entry), and the thermal
   derivation moved to `aos/unpaired_intra_vs_extra/blur_thermal.md`?
