# Study: `miw` — the Measured Intrinsic Wavefront

> **Status:** current · **Last updated:** 2026-10-05 · **Kind:** reference (study)

> **Code:** `code/miw/` · **Notebooks:** `notebooks/miw/`
> **Output:** `output/miw/<P>_<M>/intrinsic_split_{maps,decomp,rms}.parquet`, `output/miw/<P>_<M>/intrinsic_split.pdf`, `output/miw/<P>_<M>/study_radialbins.pdf`, `output/miw/<P>_<M>/fits.parquet`

Construction and validation of the Measured Intrinsic Wavefront (MIW) — the static
wavefront of the telescope and camera, measured on sky.

The MIW is the empirical, on-sky replacement for the batoid design intrinsic: the static
wavefront that remains after the reachable (OFC-controllable) part of the optical state
has been removed. It splits into a telescope-fixed component **O** (OCS frame) and a
camera-fixed component **C** (CCS, rotating with the rotator).

This study covers the pipeline's own products and the QA around them. The *library* that
builds the MIW is **not here** — it is in the external `ts_intrinsic_wavefront` package
(`lsst.ts.intrinsic.wavefront`). `aos/code/` holds only the analysis and QA scripts.

## Code

| file | role |
|---|---|
| `run_study_radialbins.py` | pipeline `study_radialbins` rule — OCS measured intrinsic in four WFS radial shells, overlaid by rotator bin |
| `compare_miw_versions.py` | two MIW builds term by term as a PDF — one page per Noll term holding both field maps on a shared colour scale and their difference on its own |
| `check_dof_ranges.py` | the build's per-visit recovered degrees of freedom (DOF) against the allowed range `r_j`, per DOF and as the wavefront the over-range amplitudes carry |
| `compare_pupil_models.py` | two builds as inferred full width at half maximum (FWHM) in arcsec, by field annulus, and by pupil Zernike term — the image-quality and localization half of a pupil-model comparison |
| `compare_build_dof.py` | two builds' recovered DOF and v-modes, differenced per visit on the visits common to both |
| `compare_rbr_arms.py` | an unconstrained build against its Range-Bounded Recovery arm — recovered DOF against `r_j`, achieved residual, the MIW as inferred FWHM in arcsec, and the change per DOF and per v-mode |
| `test_rbr_against_prototype.py` | cross-checks the `ts_ofc` Range-Bounded Recovery against the `smatrix` prototype, and the two independent routes to the allowed range `r_j` |

The build itself is in the external `ts_intrinsic_wavefront` package
(`measured_intrinsic.build_measured_intrinsic_uconstrained`, driven by the
`build_intrinsic` and `intrinsic_split` rules), not here.

Validation of the per-visit DZ fit that feeds the build, and quality checks on the donut
data, are the [`dzfit`](dzfit.md) study.

## Inputs and outputs

Reads the combined `output/fam_processing/<P>/{donuts,fits,visits}.parquet`; the per-rotator-bin
grids come from the package's `build_intrinsic`. Writes `study_radialbins.pdf` and the
`intrinsic_split_{maps,decomp,rms}.parquet` products that the other studies consume.

The **canonical MIW product** for downstream use is the `_5rot` `intrinsic_split_maps`
(OCS columns) — see `../../../notes/claude-memory/miw-products-and-m3-backprojection.md`.

## Which builds exist

Each MIW build is one `(param_set, mi_name)` pair from `mi_config.yaml`, written to the
joined directory `output/miw/<P>_<M>/`. Two wavefront versions are configured, with
identical knobs so that they differ only in the wavefronts they were built from:

| param_set | wavefront version | entries |
|---|---|---|
| `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` | Danish 1.2.0_alpha0, paired | `A_50_34_i`, `A_50_34_i_5rot` |
| `danish_1_3_test` | Danish 1.3 "blitz", unpaired, legacy pupil model | `A_50_34_i`, `A_50_34_i_5rot` |
| `danish_1_3_v1000` | Danish 1.3 "blitz", unpaired, **v1000 pupil model** | `A_50_34_i`, `A_50_34_i_5rot` |

In each pair the `_5rot` entry carries `build_from`, reusing the parent's nine
per-rotator-bin grids and re-running only the OCS/CCS split over the five in-family
rotator bins — so it has no `build/` directory of its own and the parent is a required
input.

The Danish 1.3 pair is requested **by explicit target path, not through `rule all`**: its
`visits.parquet` carries only 19 columns and none of the engineering telemetry, so the
`correlations` and `bounce` targets that `rule all` expands over every pair have no
thermal or Trim columns to read. The MIW chain itself reads eleven visits columns, all
present. The comparison of the two builds rests on 182 `(day_obs, seq_num)` visits common
to both processings inside the five rotator bins. Detail, including the per-bin visit
counts and the column audit, is in
[`../status/miw_danish_1_3_proposal.md`](../status/miw_danish_1_3_proposal.md).

### Comparing two builds

`compare_miw_versions.py` writes one page per pupil Zernike Noll term with three field
maps: build A, build B on the **same** colour scale, and B minus A on its own scale set
from the 2nd to 98th percentile of the difference. Both builds must sit on the same field
grid, which two builds sharing a `rotator_select` do; the script checks that row-for-row
rather than interpolating.

```bash
cd ~/notebooks/rubin-work/aos
python code/miw/compare_miw_versions.py \
  --miw-a output/miw/danish_1_2_A_50_34_i_5rot/intrinsic_split_maps.parquet \
  --miw-b output/miw/danish_1_3_test_A_50_34_i_5rot/intrinsic_split_maps.parquet \
  --label-a "Danish 1.2 (paired)" \
  --label-b "Danish 1.3 blitz (unpaired)" \
  --out-dir output/miw/danish_1_2_vs_1_3 \
  --out-name miw_danish_1_2_vs_1_3_OCS
```

Colour scales are computed inside a field radius of 1.70 deg, set by
`--scale-r-max-deg`. The outermost ring carries the convex-hull edge defect — across the
1.70 to 1.75 deg step the Z5 OCS difference root-mean-square rises from 0.0499 to 0.1109
µm of wavefront and reaches 0.6441 µm — and would otherwise set the range and flatten the
real structure. Those 240 of 3985 field points are still plotted, and saturate.

Alongside the PDF the script writes a `_summary.parquet` carrying, per Noll term, the
root-mean-square of each build and of the difference in µm of wavefront, the difference
normalized median absolute deviation, and both colour limits.

### The pupil model: legacy against v1000

The `danish_1_3_v1000` build is the same Danish 1.3 blitz unpaired retrieval on the
**v1000 pupil model** — the Batoid as-built model plus the updated M1M3 measurements and
the M1 outer and inner baffles, whose `M1Baffle1`/`M1Baffle2` `ClearCircle` surfaces at
radius 4.165 m sit 15 mm inside M1's 4.18 m rim. Paired against `danish_1_3_test` it
measures what modelling that baffle does to the measured intrinsic. The scheme is held
fixed at 50/34; only the pupil model varies.

In Josh Meyers' T614 collections **the suffix is the pupil model**, and the unsuffixed
`u/jmeyers3/t614_fam_unpaired` that `danish_1_3_test` reads is the *legacy* model. That is
measured, not assumed: the unsuffixed and `_legacy` chains share 12 of their 13 flattened
RUN children, each having one output RUN of its own. Note this does **not** generalize —
danish 1.3.0's own default `RubinObsc.yaml` is byte-identical to the v1000 file, so
danish's default *is* v1000.

The pair is not a pure pupil-model difference, and the size of the confound is known.
Reading the task configuration out of both output RUNs:

| | `t614_fam_unpaired` (baseline) | `t614_fam_unpaired_v1000` |
|---|---|---|
| pupil mask | no `maskModel` field — predates it | `RubinObsc_v1000_r_rtpp0_azp45_pp0d0.yaml` |
| `danish` | `5037d9f3` | `ca41ae8c` |
| `ts_wep` | `9651cd23` | `639a89d9` |
| task label | `donutBlitzFamTask` | `donutBlitzFam` |
| visits | 966 over 15 nights | 966 over the same 15 nights |

So the baseline is an older code version as well as an older pupil model. Carried
deliberately: the code changes between those commits are small, so the pupil model is the
leading term. `u/jmeyers3/t614_fam_unpaired_legacy` is the airtight baseline if one is
ever wanted — it differs from the v1000 run in **exactly one config line**, `maskModel`,
on the same `danish` commit — at the cost of a third build.

The v1000 tables are built by hand, not by `rule all`, because this param_set declares no
chunks in `snake_config.yaml`: the blitz recast writes
`output/fam_processing/<P>/{donuts,visits,fits}.parquet` directly and Snakemake then
treats them as terminal inputs.

```bash
cd ~/notebooks/rubin-work/aos
python code/fam_processing/run_blitz_mktable.py \
  --param-set danish_1_3_v1000 \
  --fit \
  --overwrite
```

Two scripts compare the builds beyond `compare_miw_versions.py`'s term-by-term maps.
`compare_pupil_models.py` converts each MIW to an inferred FWHM in arcsec with ts_wep
`convertZernikesToPsfWidth`, splits the difference by field annulus, and gives each pupil
Zernike term's share of the difference power — the spherical terms Noll 11 and 22 being
the pupil-rim diagnostic a field-map product can offer. `compare_build_dof.py` differences
the two builds' recovered DOF and v-modes **per visit on the visits common to both**,
since a pupil change that moves the MIW but leaves the subtracted optical state alone is
acting on the static wavefront, while one that moves both is partly being absorbed by the
fit.

```bash
cd ~/notebooks/rubin-work/aos
python code/miw/compare_pupil_models.py \
  --miw-a output/miw/danish_1_3_test_A_50_34_i_5rot/intrinsic_split_maps.parquet \
  --miw-b output/miw/danish_1_3_v1000_A_50_34_i_5rot/intrinsic_split_maps.parquet \
  --label-a "legacy pupil" \
  --label-b "v1000 pupil" \
  --out-dir output/miw/danish_1_3_legacy_vs_v1000

python code/miw/compare_build_dof.py \
  --build-a output/miw/danish_1_3_test_A_50_34_i/build \
  --build-b output/miw/danish_1_3_v1000_A_50_34_i/build \
  --label-a "legacy pupil" \
  --label-b "v1000 pupil" \
  --out-dir output/miw/danish_1_3_legacy_vs_v1000
```

#### What the comparison found

Both builds select the **same 205 visits** in the five in-family rotator bins — identical
`(day_obs, seq_num)`, on a field grid identical to 1e-12 deg — so the comparison runs on the
full common set with no sample caveat. Products are in
`output/miw/danish_1_3_legacy_vs_v1000/`; the term-by-term pages are
`miw_legacy_vs_v1000_OCS.pdf`. The CCS pages are near-empty by construction: this split
forces the camera-fixed component to zero on every term except Z4.

The pupil model moves the MIW, by a small amount:

| quantity | legacy | v1000 | difference |
|---|---|---|---|
| wavefront RMS inside the hull, µm of wavefront | 0.0518 | 0.0528 | 0.0053 |
| inferred FWHM, arcsec | 0.1645 | 0.1678 | +0.0033 |

The difference RMS over build-A's RMS is 0.1027 (dimensionless, both amplitudes) — about a
third the size of the Danish 1.2-to-1.3 paired-to-unpaired retrieval change, which moved the
same quantity by 0.0156 µm of wavefront and −0.0146 arcsec.

**The signature is not the one a baffle predicts, and that is the result.** An axisymmetric
change to the pupil radius should load the spherical terms. It does not: Noll 11 and 22
together carry **0.0143 of the difference power** (dimensionless), *less* than the 0.0609 the
Danish 1.2-vs-1.3 retrieval change carried — the deliberate baseline for this test. The
difference is instead 0.70 of its power in astigmatism, Noll 6 at 0.4032 and Noll 5 at
0.2969, with coma Noll 7 and 8 a further 0.105.

Nor is it cleanly at the field edge. The difference rises outward from 0.0020 µm of wavefront
inside 0.60 deg to 0.0077 µm at 1.55–1.70 deg, a factor of 3.9, but non-monotonically — it
dips to 0.0040 µm at 1.35–1.55 deg. The 0.0204 µm in the 1.70–1.75 deg ring is the convex-hull
edge defect present in both builds, reported as its own annulus and excluded from every number
above.

The subtracted optical state moves too, so part of the pupil change is being absorbed by the
fit rather than appearing in the MIW. Per visit on the 205 common visits, median difference
v1000 minus legacy: M2_dy −139.1 µm (nMAD 60.88 µm) and M2_dx +53.48 µm, both mechanically
trivial against their ±6700 µm range, 0.0208 and 0.0080 of it. The bending modes are the ones
that matter relative to range — B1_16 moves 1.1276 of its allowed `r_j`, B1_12 0.4968 — and
they are the same modes already known to sit far outside range in both builds. In the v-modes,
29 and 31 move by −0.0742 and +0.0574 (dimensionless), roughly twice their own nMAD scatter.

Two readings fit this, and these products cannot separate them: either the baffle's effect is
absorbed into the fitted optical state rather than the MIW, or the astigmatism shift is the
confounded **code-version** change (`danish` `5037d9f3` to `ca41ae8c`) rather than the pupil
model. The second is live, since astigmatism is also where the paired-to-unpaired change was
largest. Treat the astigmatism-led difference as unexplained rather than attributed to the
pupil. `u/jmeyers3/t614_fam_unpaired_legacy` is the clean discriminator — one config line
apart from v1000 — at the cost of a third build, which has not been done.

Consequence for Range-Bounded Recovery (RBR): because the two MIW agree to 0.1027 of
amplitude, RBR carries forward on **`danish_1_3_v1000` only** (Aaron, 2026-10-05); two arms
would buy little.

### Whether the subtracted optical state is physically reachable

The build recovers a 50-DOF optical state per visit with a truncated SVD keeping 34
v-modes, and truncation is that recovery's only regularizer — nothing holds a recovered
amplitude inside the stroke the mirror or hexapod can reach. `check_dof_ranges.py`
compares those states against the allowed range `r_j` that the
[`regularized_inversion`](../../../smatrix/docs/studies/regularized_inversion.md) study
defines, reading the build's own per-visit DOF from `build/rot_*/dz_fits.parquet` and
back-deriving `r_j` from the same SVD normalization weights the recovery already uses, so
no new input enters.

```bash
cd ~/notebooks/rubin-work/aos
python code/miw/check_dof_ranges.py \
  --build-dir output/miw/danish_1_3_test_A_50_34_i/build \
  --label "Danish 1.3 blitz (unpaired)" \
  --build-dir output/miw/danish_1_2_A_50_34_i/build \
  --label "Danish 1.2 (paired)" \
  --out-dir output/miw/dof_ranges
```

It is not reachable, and the two wavefront versions agree closely on that. Over the five
in-family rotator bins — 205 visits for Danish 1.3, 193 for Danish 1.2 — **33 of 50 DOF
have at least 5 % of visits outside ±`r_j`** in both builds, and the same 33: every one is
a bending mode, 17 of 20 on M1M3 and 16 of 20 on M2. All ten rigid-body DOF are inside
range on every visit, by a wide margin — the hexapod ranges are thousands of µm against
recovered amplitudes of hundreds. The worst modes are outside on essentially every visit:
on Danish 1.3, B1_20 (`r_j` = 0.002206 µm) has a median `|d_j| / r_j` of 41.55
(dimensionless) reaching 61.14, and B1_12 (`r_j` = 0.009088 µm) a median of 24.43 reaching
54.31.

Clipping each amplitude into ±`r_j` and forward-propagating both states through the same
rank-limited sensitivity matrix puts a wavefront scale on it: the over-range part of the
subtracted state carries 0.0315 µm of wavefront against the full state's 0.0587 µm (median
over visits of the RMS over the DZ `(k, j)` grid, Danish 1.3), a ratio of 0.4753
(dimensionless, both amplitudes); Danish 1.2 gives 0.0269 against 0.0567 µm and 0.4378.
That clipped state is an accounting of the wavefront at stake, not a proposed MIW —
clipping is not what Range-Bounded Recovery does.

Absolute states violate the range harder than the paired differences the `bounce` study
inverts, which reach `|d_j| / r_j` of 8.59 to 11.27 over all 50 DOF: a difference between
two visits cancels the common part of the state, and these do not. The two measurements
are consistent, not in conflict.

Whether and how to add a penalty term to the MIW optical state fitting is open.

### Range-Bounded Recovery: where the code lives

Range-Bounded Recovery (RBR) is the penalty that answers the reachability problem
above. It is implemented in **`ts_ofc`**, not in this repository:
`lsst.ts.ofc.range_bounded_recovery`, on branch `tickets/RSO-1007`. That is
deliberate — RBR would ultimately be used by MTAOS, and `ts_ofc` already holds the
quadratic `motion_penalty`, so a second home would drift. The prototype at
`smatrix/code/regularized_inversion.py` remains the derivation and the cross-check
reference, not the implementation.

The penalty, over normalized DOF `x` with physical `d = w * x`:

```
min_d  || dW - S x ||^2  +  sum_j ( |d_j| / (kappa * r_j) ) ^ (2 * power)
```

with `kappa` = 4.0 and `power` = 3, so the penalty equals unity at
`|d_j| = 4 r_j` and grows as the sixth power of the amplitude. It is a **barrier,
not a shrinkage term**: a DOF comfortably inside its range is left essentially
untouched, unlike a quadratic penalty which pulls on every DOF everywhere. Solved
by iteratively reweighted least squares in the retained mode coefficients, with a
backtracking line search — a fixed step size enters a limit cycle at `power` 3.

`r_j` now comes from **`OFCData.dof_ranges()`**, added on the same branch, which
builds it from the configured hexapod strokes and mirror force ranges. Two
independent routes to `r_j` agree to 1.5e-15 relative: that one, and the
prototype's back-derivation `r_j = w_j^2 f_j` from the shipped normalization
weights. The agreement holds only under `w_j = r_j^0.5 f_j^-0.5` — note the
shipped weights file `range0.5_fwhm-0.15.yaml` is **misnamed**, the exponent being
-0.5 rather than -0.15.

On a representative DZ wavefront, RBR takes the worst DOF from 59.30 to 1.28 of
its allowed range (dimensionless, `|d_j| / r_j`) for a 5.5e-02 fractional increase
in the achieved residual.

The MIW build reaches this through `ts_intrinsic_wavefront`, whose `build_ofc_svd`
now delegates its decomposition to `lsst.ts.ofc.DoubleZernikeStateEstimator`
(branch `tickets/RSO-809`), so **`ts_ofc` holds the only sensitivity-matrix SVD in
the code base**. That delegation is verified bit for bit, including an end-to-end
rebuild of a `danish_1_3_v1000` rotator-bin grid.

A caution on relating the two estimators: `DoubleZernikeStateEstimator` and
`StateEstimator` decompose the same matrix — the DZ slab omits 1.0e-05 of its
power — but their **individual v-modes past about mode 9 are not comparable**,
because the singular values cluster and the truncation at 34 modes cuts through a
cluster. The leading-34 subspaces overlap at 0.9156 (dimensionless) while
individual vectors can agree at `|dot|` of 2e-04. Compare subspaces, not vectors.

### What RBR does to the MIW

Measured on the `danish_1_3_v1000` `rot_-3_3` bin, 49 visits, `kappa` 4.0 and
`power` 3, against the unconstrained arm on the same visits
(`compare_rbr_arms.py`). **This is one rotator bin of nine, and the RBR arm is not
converged — read the caveat below before using these numbers.**

RBR does what it was built to do. The recovered state comes inside range:

| quantity (dimensionless, `|d_j| / r_j`) | unconstrained | RBR |
|---|---|---|
| worst over all DOF and visits | 60.25 | 1.713 |
| median for B1_20 | 47.28 | 0.881 |
| median for B1_12 | 32.69 | 1.038 |
| fraction of (visit, DOF) pairs outside range | 0.4645 | 0.0939 |

The cost is modest in the fit and large in the MIW. The achieved residual, each
arm against its own fitted wavefront, rises from 0.1131 to 0.1744 as a fraction
of that wavefront (dimensionless, amplitude). But the MIW itself moves by 0.2764
µm of wavefront RMS over the field, and the inferred FWHM goes from 0.1774 to
0.3147 arcsec.

That is a factor of 50 larger than the 0.0053 µm the pupil model moved the MIW,
so **the recovery constraint matters far more to the MIW than the pupil model
does** — which also means step A's attribution question is not the limiting
uncertainty here.

The mechanism is visible in the v-modes. RBR moves the *smallest*-singular-value
retained modes: v33 (`sigma` 0.0231) by a median 3.85 in its coefficient, v21
(0.0965) by 1.855, v34 (0.0228) by 1.632. Those are the poorly-conditioned
directions where a small wavefront signal implies enormous mirror motion, which
is exactly where the unconstrained recovery was placing unreachable amplitudes.
RBR subtracts 0.8784 of the wavefront amplitude the unconstrained arm removed,
and the remaining 0.1216 stays in the MIW.

So the MIW grows because RBR **declines to attribute wavefront to motion the
mirrors cannot make**. Whether that larger MIW is the better estimate of the
telescope's intrinsic wavefront is the open question: the unconstrained MIW is
smaller because it absorbed real aberration into a physically impossible state,
but RBR's residual now contains whatever that aberration actually was.

**Convergence caveat.** The build iterates, feeding the MIW grid back onto the
donuts, and RBR converges markedly more slowly. At the configured `n_iter` 3 the
unconstrained arm had settled to 8.92e-04 µm of wavefront between iterations
(inside the 1.0e-03 µm tolerance) while the RBR arm was still moving by 5.42e-03
µm. Raising `n_iter` to 8 left it at 1.37e-03 µm, decaying roughly as
1/iteration, and moved the MIW a further 0.0483 µm of wavefront — to an inferred
FWHM of 0.3550 arcsec. The direction and the scale of the effect are therefore
robust (0.3077 µm between arms against 0.0483 µm of convergence drift), but the
RBR arm's exact numbers are an iterate, not a settled value, and the full
nine-bin arm will need a larger `n_iter`.

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh -n                                    # what is stale
./run_snake.sh --until plots
python code/miw/check_chunk.py --help                    # pre-flight, before mktable
```

Phase 2 needs `lsst.ts.ofc`/`lsst.ts.wep`, `$TS_CONFIG_MTTCS_DIR`, and batoid height
maps — RSP only. `mktable` is the expensive Butler step and is deliberately **not**
re-triggered by code edits.

## State and open questions

- **83 % of MIW power sits above the `k<=6` focal orders the build actually fits.** This
  reframes every DZ-subspace analysis: the fit constrains 34 v-modes from k≤6, then
  subtracts only the k≤6 part of that state's wavefront. Quantified in
  `../../code/coadd/analyze_miw_dz_full_k.py` and `analyze_miw_field_order.py` (`coadd` study).
- The MIW's high-field-order astigmatism/coma excess (Z5–Z8, OCS, ~0.1 µm) is **not**
  predicted by the batoid design model. Whether static optics can explain it is the
  [`static_optics`](static_optics.md) study; the write-up is
  [`../../../smatrix/docs/miw_astig_coma_investigation.md`](../../../smatrix/docs/miw_astig_coma_investigation.md).

## Notebooks

`notebooks/miw/aos_miw_ocs_ccs_maps.ipynb` reads the MIW OCS/CCS split maps — a viewer for
`<mi>/intrinsic_split_maps.parquet`, no repo-module imports.

## See also

- [`../miw_pipeline.md`](../miw_pipeline.md) — every rule, config file, output path
- [`../miw_coadd_equations.md`](../miw_coadd_equations.md) — the notation and derivations
- [`../status/miw_investigation_handoff.md`](../status/miw_investigation_handoff.md) —
  portable state, **with an explicit list of retracted claims**
- [`../double_zernike_convention_validation.md`](../double_zernike_convention_validation.md)
