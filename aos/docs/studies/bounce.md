# Study: `bounce` — elevation and rotator bounce tests

> **Status:** current · **Last updated:** 2026-10-01 · **Kind:** reference (study)

> **Code:** `code/bounce/` · **Notebooks:** `notebooks/bounce/`
> **Output:** `output/bounce/<P>_<M>/bounce_*.pdf`, `output/bounce/<P>_<M>/bounce_kj_stats.parquet`, `output/bounce/<P>_<M>/bounce_dof_stats.parquet`, `output/bounce/<P>_<M>/bounce_fwhm_metric.parquet`, `output/bounce/bending_mode_test_meta.parquet`

Analysis of elevation and rotator bounce test data, for Look-Up-Table (LUT)
development. A bounce test moves the telescope to a position and back; a repeatable
difference between the two visits measures hysteresis or gravity-driven flexure rather
than noise.

Two BLOCKs supply the data: **BLOCK-T720** (elevation) and **BLOCK-T724** (rotator
0↔60 deg). The analysis is a *paired* Δ — time-ordered comp−ref pairs **within a night**,
which cancels slowly-varying terms.

The Δ wavefront at each bounce point is inverted onto physical degrees of freedom (DOF) **four
ways** — the default truncated 50 DOF / 34 v-mode recovery, that same recovery with a per-DOF
range penalty, a reduced 22 DOF / 12 v-mode scheme, and the quadratic motion penalty `ts_ofc`
already ships — with a fifth camera-hexapod-only 5 DOF / 5 v-mode recovery on the rotator bounce.
Each is reported as the DOF value per point against its allowed range, and as the image quality
the correction delivers.

## The elevation bounce is a sweep, not a single throw

BLOCK-T720 carries **one reference leg at elevation 70 deg and five comparison legs**, each a
±3 deg window about a measured elevation. Different nights throw to different elevations, so a
night contributes only to the leg(s) it actually exercises:

| comparison leg | throw from 70 deg (deg) | nights with pairs | n pairs |
|---|---|---|---|
| elev 40 deg | −30 | 20260418, 20260419, 20260513 | 22 |
| elev 60 deg | −10 | 20260709 | 4 |
| elev 50 deg | −20 | 20260711 | 6 |
| elev 30 deg | −40 | 20260713 | 5 |
| elev 75 deg | +5 (upward) | 20260713 | 6 |

Only the 40 deg leg has more than one night, so it is the only leg carrying a night-to-night
repeatability measurement; the night-pair products are drawn only where 2 or more nights
qualify.

The 75 deg leg uses `alt_range: [73.0, 78.0]` deg so it does not overlap the 67–73 deg
reference window; being a small *upward* throw it serves as a near-null control rather than a
flexure measurement. BLOCK-T724 has **no July nights**, so the rotator bounce carries its
single `Rot=60` leg unchanged, with `camera_hexapod_only: true`, over nights 20260420 and
20260513 with 31 pairs.

**The per-visit quality cut has two active parts**, and the blur one is what removes visits
here. `quality_visit_mask` applies a donut-count floor (`--min-detectors 160`) *and*
`median_blur_arcsec <= 1.2` arcsec, the latter at its library default
`DEFAULT_MAX_MEDIAN_BLUR_ARCSEC`. Every BLOCK-T720 visit on all six nights clears the CCD
floor at 176–180 CCDs, so relaxing `--min-detectors` changes nothing; three visits fail the
blur cut, both of 20260711's elevation 30 deg visits at 1.525 and 1.934 arcsec and one of
20260713's at 1.799 arcsec. That is why the 30 deg leg is a single-night 5-pair leg rather
than a two-night 8-pair one, and it is a seeing limitation, not a processing one.

Because a multi-leg bounce need not exercise every leg on every night, `bounce_lib.bounce_nights`
qualifies a night on the legs it actually has — at least one populated leg, and every populated
leg clearing `night_min_visits`. Requiring *all* configured legs would leave the per-night
breakdown empty for any multi-leg bounce.

The legs live in `analysis_config.yaml` under `overrides:
fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x: bounce:`, at the **param_set** level rather than
under one `mi_name`, so the measured-intrinsic refit table and the Phase-1 batoid-intrinsic
table resolve identical bounce definitions and their results are directly comparable. That
block also sets `pupil_j_range` explicitly, because the Phase-1 table carries no
`nollIndices` column for the script to infer the grid from.

Results, including the throw-vs-FWHM trend and the measured bound on the intrinsic choice, are
written up in [`../../../notes/aos-bounce-test-summary/note.md`](../../../notes/aos-bounce-test-summary/note.md).

## Code

| file | role |
|---|---|
| `bounce_lib.py` | library: pairing, per-(k,j) statistics, significance, leg-night coverage, plotting, and the solver-agnostic per-pair reduction `solver_dof_per_pair` / `solver_deltas` with the RBR wrappers `rbr_dof_per_pair` / `rbr_deltas` bound on top |
| `run_bounce.py` | pipeline `bounce` rule — paired Δ for DZ coefficients, OFC v-modes, and physical DOF under every recovery scheme, with vs-ordinal pages, night cross-scatter, the per-DOF-vs-B-set panels and the achieved-FWHM-vs-B-set comparison; `--rbr-kappa`, `--rbr-power` and `--no-rbr` control the RBR arm |
| `../../../smatrix/code/regularized_inversion.py` | the solvers themselves — truncated, damped, range-penalty and OIC — shared code in `smatrix`, imported from the study that validated them rather than copied |

## Notebooks

| file | role |
|---|---|
| `notebooks/bounce/bending_mode_test_lut_trim.ipynb` | commanded hexapod LUT (10 axes) and Trim (all 50 DOF) against `seq_num` for the CWFS exposures of the bending-mode test BLOCKs T377–T380, read from the value-added telemetry DuckDB |

### Bending-mode test BLOCKs

BLOCK-T378 (M1M3 bending modes) and BLOCK-T379 (M2 bending modes) command an individual
mirror bending mode and take Corner Wavefront Sensor (CWFS) pairs, with the mode named in
the `science_program` suffix (`BLOCK-T378_M1M3B12`, `BLOCK-T378_M1M3B20`,
`BLOCK-T379_M2B18`). The notebook shows what the hexapod LUT had loaded and what the Trim
had accumulated while each mode was exercised.

The LUT and the Trim are **different index spaces** that agree only over their first ten
entries. Both order the hexapods M2 first (indices 0–4, `M2_dz/dx/dy/rx/ry`) then camera
(5–9, `Cam_dz/dx/dy/rx/ry`); the Trim continues with M1M3 bending (10–29) and M2 bending
(30–49), which the hexapod LUT has no counterpart for. Labels, units and index groups come
from `lsst.ts.intrinsic.wavefront.ofc_svd` (`LABELS_50DOF`, `DOF_UNITS_50`, `DOF_GROUPS`),
and the LUT axis order is documented at `common/dof_telemetry.py:fetch_hexapod_lut_for_visits`.

One unit trap: the LUT angular axes are **deg**, as the hexapod reports them, while the
Trim rotations are **arcsec**, the OFC convention. Translations are µm in both.

## The vs-ordinal marker scheme

The three vs-ordinal PDFs encode three visit properties in one marker, defined in
`lsst.ts.intrinsic.wavefront.intrinsics_lib` (the external package, not `aos/code/`) so that
every study drawing these pages shares one scheme:

| property | encoding |
|---|---|
| elevation | marker **colour**, one per grid centre at 30, 40, 50, 60, 70, 75 deg |
| camera rotator angle | marker **shape** — an arrow whose direction gives the rotator angle |
| filter band | a small **dot overlaid at the centre of the arrow shaft**, in the band's colour |

A visit is assigned to the nearest grid centre within a half-width, `ELEV_HALFWIDTH_DEG = 2.0`
deg in elevation, tightened from 5 deg so that the 70 and 75 deg legs separate. The half-width
is a parameter (`elev_halfwidth_deg` / `ab_elev_halfwidth_deg`, and the rotator equivalent)
rather than a constant.

The band is a dot rather than the marker edge because the edge colour competes with the
elevation fill: at plotted marker sizes an edge-encoded band makes the elevation unreadable.
Band colours come live from `lsst.utils.plotting.get_band_dicts()['colors']`, the canonical
Rubin palette (DM-51122, revised DM-51690), with the current hex values kept as a fallback for
environments where `lsst.utils.plotting` is unavailable.

## Why the O/C split matters here

Any intrinsic that is **fixed in the fitting frame cancels in a Δ**. So the
telescope-fixed **O** term drops out, and it is the **rotating camera term C** that
changes a rotator-bounce result. This is precisely why the MIW's O + C decomposition
matters — a bounce analysis run against a frame-fixed intrinsic would show an artifact.

That cancellation is now measured on **every leg**, not just argued. The MIW-referenced fit was
run over the July nights as well (`output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet`),
which needed no MIW rebuild — the build stays frozen at `day_obs` 20260513 and the fit merely
references it. Comparing its Δ against the batoid design intrinsic
(`output/fam_processing/danish_1_2/fits.parquet`) per (k, j), with n = 126 DZ coefficients on
each leg, the `nmad` of the difference over the leg's own RMS(Δ) is 0.9% to 1.6% on the four
single-night elevation legs (Pearson r ≥ 0.999), 3.1% on the 40 deg leg — where the two tables
select different visit sets, 22 pairs against 10, so that row mixes intrinsic choice with a
genuine sample change — and 5.2% on the BLOCK-T724 rotator bounce, where C *does* rotate
within a pair and the difference is expected to be largest. So at present precision the
intrinsic choice is not a limiting systematic for either bounce. The full table is in the
results note.

## Outputs

Four parquet tables and nine PDFs per run.

| product | content |
|---|---|
| `bounce_kj_stats.parquet` | per-(k, j) Δ in µm of wavefront, its error and significance, per night and pooled |
| `bounce_dof_stats.parquet` | per-DOF and per-v-mode Δ, per night and pooled, one row per (bounce, leg, night, quantity) with a `kind` naming the recovery scheme — `dof` / `vmode` for 50/34, `dof22` / `vmode22` for 22/12, `dof_oic` for the OIC solution, `dof5` / `vmode5` for the camera-hexapod-only 5/5 — and a `unit` column: µm for translations and bending-mode amplitudes, arcsec for hexapod rotations, dimensionless for v-modes. The `dof`, `dof22`, `dof_oic` and `dof5` rows are keyed by **global** DOF index, so a reduced scheme's rows sit at the indices it actually solves for and are absent elsewhere. Carries the *measured* `elevation_deg` and `rot_angle_deg` of the comparison leg and `ref_elevation_deg` / `ref_rot_angle_deg` of the reference, plus `day_obs`, `block`, `n_visits` and `n_pairs`, so it reproduces `bounce_dof_night_values.pdf` and can be shared standalone. Every DOF row carries `dof_range` (the allowed range `r_j`, same unit as the row) and the dimensionless `ratio_to_range`; `kind = dof` rows additionally carry the Range-Bounded Recovery (RBR) result as columns — `delta_rbr` and `delta_rbr_err` in the row's own unit and `ratio_to_range_rbr` dimensionless — since RBR solves the same 50 DOF the `dof` rows do |
| `bounce_fwhm_metric.parquet` | `fwhm_before` and one achieved-residual column per recovery scheme — `fwhm_after_50_34`, `fwhm_after_rbr`, `fwhm_after_22_12`, `fwhm_after_oic`, `fwhm_after_5_5` — all arcsec FWHM, per leg, pooled over nights. `fwhm_after_5_5` is populated only on a `camera_hexapod_only` bounce |
| `bounce_fwhm_vs_bvalue.parquet` | one row per (bounce, leg, night), with `b_value` (elevation or camera rotator angle in deg), `n_pairs`, and the achieved-residual FWHM in arcsec for every scheme: `fwhm_before`, `fwhm_after_default`, `fwhm_after_rbr`, `fwhm_after_22_12`, `fwhm_after_oic` and `fwhm_after_5_5` |

`bounce_summary.pdf` opens with a leg-night coverage table — which nights back each leg and how
many night-pair scatter pages follow — then the night-vs-night Δ cross-scatter, wide and
zoomed, for every leg with 2 or more qualifying nights. `bounce_dof_night_scatter.pdf` carries
the same coverage table and the per-DOF night-pair scatter. `bounce_dof_night_values.pdf` shows
the paired Δ DOF as per-DOF panels against the B-set position — elevation in deg for BLOCK-T720,
camera rotator angle in deg for BLOCK-T724 — with one point per (night, leg), laid out **2
columns × 5 rows per page and paginated** across pages so the overlaid schemes stay readable
(`dof_ncols` and `dof_rows_per_page` in `analysis_config.yaml`). The 50 DOF therefore take five
pages per bounce. **Colour and marker both encode the recovery scheme**, which is what has to be
separable at a glance: the 50/34 recovery is a filled blue circle, and each other scheme an open
marker of its own colour — red square for RBR, green triangle for 22/12, purple diamond for OIC,
orange inverted triangle for 5/5, matching the colours used for the same schemes in
`bounce_fwhm_vs_bvalue.pdf` so a scheme looks the same in every product. The night is **not** on
colour; nights sharing a B set are fanned out in x instead and named by the per-point annotation
on the base series (`day_obs` and B value in deg), with the full night list in the figure title.
Each panel also carries the allowed range ±`r_j` as a shaded band; where the band is wider than
the data it is annotated as a value in the corner instead of being allowed to set the y scale. A
scheme is simply absent from the panels of DOF it does not solve for: 22/12 from the 28 DOF
outside its index set, 5/5 from the 45 DOF outside the camera hexapod. The scheme legend sits
below the panels rather than in a corner, so it cannot overlap the multi-line title. `bounce_fwhm_vs_bvalue.pdf` is one page per bounce of the
achieved-residual FWHM series against the B-set position, one point per (night, B set), with the
nights at a shared B set fanned out in x for legibility and the trend line drawn through the
per-B-set median over nights. `bounce_dz_vs_ordinal.pdf` opens with the marker legend and an A/B
bounce-position table per bounce.

With `add_dof_trim` enabled the run additionally queries the EFD live for the MTAOS Trim
overlay, so that mode needs RSP/EFD access.

Two live output directories, so neither shadows the other:
`output/bounce/danish_1_2_A_50_34_i_5rot_july/` from the MIW-referenced fit over all six nights
(the lead result, every number in the note), and `output/bounce/danish_1_2_batoid/` from the
Phase-1 batoid-intrinsic fit (the intrinsic-choice comparison). Both are hand-run with
`--out-dir`, `--fits` and `--min-detectors 160`, matching the `bounce` rule.

The same MIW build over April/May nights only, and the earlier non-rotated `pathA_50_34_i`
build, are superseded and parked under `output/archive/bounce/` — see that tree's `README.md`.
Note `output/miw/danish_1_2_A_50_34_i_5rot/` is a different path and is current: it is the MIW
fit table this study reads.

These PDFs currently share `<mi>/plots/` with three other studies' output; splitting
them per study is outstanding work.

## Recovered bending-mode amplitudes exceed their allowed range

The S-matrix SVD normalization weight is `w_j = r_j^0.5 * f_j^-0.5`, with `r_j` the allowed
range of degree of freedom `j` — µm for translations and bending-mode amplitudes, arcsec for
hexapod rotations — and `f_j` the FWHM response in arcsec per DOF unit. The shipped
`range0.5_fwhm-0.15.yaml` stores only the product, so `output/bounce/dof_normalization_split.parquet`
records the split: `range`, `fwhm_per_unit_arcsec` back-derived as `r_j / w_j^2`, and `weight`.
The hexapod ranges are the `rb_stroke` literals (M2 5900/6700/6700 µm and 0.12 arcsec, camera
8700/7600/7600 µm and 0.24 arcsec); the 40 bending-mode ranges are
`(force_range / 20) / max|force per µm|` from the 134 N M1M3 and 45 N M2 force ranges over a
20-mode budget. Reconstruction satisfies `w_j = sqrt(r_j / f_j)` to machine precision.

Compared against those ranges, **the recovered high-order bending-mode amplitudes are
unphysically large**, and increasingly so with throw. Counting only DOF at over 3σ:

| leg | n DOF over 3σ | n exceeding full range | largest ratio |
|---|---|---|---|
| elev 30 deg | 27 | 10 | 11.7 (B1_20) |
| elev 40 deg | 29 | 10 | 7.3 (B1_20) |
| elev 50 deg | 10 | 2 | 2.8 (B2_12) |
| elev 60 deg | 7 | 3 | 3.1 (B2_17) |
| elev 75 deg | 11 | 1 | 1.4 (B1_20) |
| rotator 60 deg | 31 | 8 | 3.1 (B1_11) |

The ratio is `abs(delta)/r_j`, dimensionless, the recovered amplitude over the allowed range. B2_12 reaches
−0.0612 ± 0.0032 µm at elevation 30 deg against a range of 0.01447 µm, a ratio of 4.2, at
significance 19.2; B1_20 reaches −0.0258 ± 0.0032 µm of mode amplitude against a range of
0.00221 µm, a ratio of 11.7, at significance 8.0. Since a mirror physically cannot exceed its actuator-force-limited range, these
amplitudes are not real mirror figure changes. The monotonic growth with throw — ratios
dropping to about 1 on the near-null upward 75 deg leg — points to the unconstrained recovery
absorbing something that scales with the bounce signal into the weakly-constrained high-order
modes, rather than the modes themselves being excited. The rigid-body terms, which dominate the
Δ in FWHM terms, stay far inside their ranges. The next section measures that interpretation
rather than leaving it as a reading of the pattern.

This is a property of the open-loop recovery, not of the bounce measurement: the DZ Δ itself
(`bounce_kj_stats.parquet`) and the correctable-FWHM metric are unaffected, since the FWHM
metric projects onto the correctable subspace rather than reading individual amplitudes.

**Which count.** The table above pools each leg over its nights and counts only DOF significant
at over 3σ; the next section counts all 50 DOF, per (leg, night). Both are correct under their
own definition — the second is the larger set, so its counts and largest ratios are higher, and
neither is a correction of the other.

## Four recovery schemes at every bounce point

The same Δ wavefront is inverted onto DOF four ways at every bounce point — five on the rotator
bounce. All four share the (k, j) grid, so one Δ wavefront feeds every scheme with no
re-projection, and all four are scored by the same achieved-residual FWHM so the comparison is
like for like.

| scheme | DOF | v-modes | regularizer | why it is here |
|---|---|---|---|---|
| **50/34 truncated** | 50 | 34 | truncation only | the default recovery, what the OFC does today |
| **50/34 RBR** | 50 | 34 | superlinear per-DOF range penalty | bounds the amplitudes to what the telescope can apply |
| **22/12 reduced** | 22 | 12 | truncation only | the reduced set the AOS is expected to operate in, so this is the operationally relevant answer |
| **50/34 OIC** | 50 | 34 | quadratic motion penalty | the knob `ts_ofc`'s Optimal Integral Controller already ships, so this says what the existing code buys |
| **5/5 camera hexapod** | 5 | 5 | truncation only | only the camera hexapod moves in a rotator bounce, so this is the scheme the BLOCK-T724 result is used in |

The 22 DOF are an **index set, not the first 22** — the 10 rigid-body DOF, M1M3 bending modes
1–7, and M2 bending modes 1–5, fixed in `run_bounce.py` as `DOF22`. The singular values show no
gap at the 12-mode cut (`Sigma[11] = 0.16723`, `Sigma[12] = 0.14383`, both dimensionless), so 12
is an operational choice exactly as 34 is. BLOCK-T724 carries all five schemes in every product:
although only 5/5 will be used operationally there, it is worth seeing what is lost by not using
all DOF.

The OIC penalty weight is `oic_rho = 1e-3`, dimensionless, read from `analysis_config.yaml`.
`ts_ofc` ships `motion_penalty = 0.0`, so the penalty is inactive as delivered and a value has to
be chosen; this study **consumes** the value and does not derive it. The scan over rho that
produced it — feasibility, image quality *and* amplitude retention — is in the
[`regularized_inversion`](../../../smatrix/docs/studies/regularized_inversion.md) study, and the
criterion was to match RBR's feasibility while accepting a few legs slightly over range.

**The OIC arm's rigid-body amplitudes are suppressed and are not measurements.** That scan shows
no rho reaches feasibility without crushing them. At `rho = 1e-3` the rigid-body amplitude
retention — the dimensionless regression slope of the penalized amplitudes on the truncated ones
— is 0.045 to 0.073 across the six legs, a suppression by a factor of 14 to 22. Retention only
recovers to 0.98 at `rho = 1e-5`, where feasibility is no better than unregularized
(`max_j |d_j|/r_j` of 2.8 to 11.0 dimensionless), and the two transitions sit within a factor of
ten of each other because the penalty on camera `dx` alone equals the entire wavefront misfit at
`rho ≈ 1.03e-3`. The suppression is visible directly in `bounce_dof_night_values.pdf`: the OIC
series sits near zero on the rigid-body panels where the other three schemes show a substantial
trend with elevation — camera `dx` runs −50.3 to −684.2 µm of hexapod translation across the
elevation legs under the truncated recovery but only −3.2 to −38.9 µm under the OIC. That
near-zero trend is the penalty, not the telescope. The arm is kept at this rho because the
comparison being made is between penalties at *matched compliance*, and the OIC's behaviour at
matched compliance is the finding.

### What RBR does

The default recovery inverts the measured wavefront onto DOF with a truncated SVD, keeping
34 of 50 singular modes. Truncation is its only regularizer, and it is a blunt one — a mode
is either fully trusted or fully discarded. The normalization weight `w_j = sqrt(r_j / f_j)`
puts the allowed range into the *metric* of the fit, but nothing puts it into the *feasible
set*, which is why the amplitudes above are free to run past what the mirror can reach.

RBR adds a penalty on each DOF's physical amplitude that is negligible while the amplitude
stays well inside its range and climbs steeply as it approaches and passes it. Writing `d` for
the physical DOF, `x = d / w` for the normalized DOF the SVD is taken in, `S` for the rank-34
forward operator in µm of wavefront per unit normalized DOF, and `dW` for the measured DZ
wavefront in µm of wavefront, RBR minimizes

```
||dW - S x||^2  +  sum_j ( |d_j| / (kappa * r_j) ) ^ (2 p)
```

Two knobs, both dimensionless. `kappa` is the amplitude-over-range ratio at which the penalty
reaches unit weight, and `p` sets how fast it climbs. The bounce runs use `kappa = 4`,
`p = 3` — the setting the [`regularized_inversion`](../../../smatrix/docs/studies/regularized_inversion.md)
study found best or near-best on five of the six legs. `kappa` is deliberately well above 1:
the penalty only has to bound the largest amplitude, and putting the knee at the range itself
taxes the other 49 DOF for no gain.

The penalty is smooth, not a hard bound, so a recovered amplitude can still finish slightly
outside its range — it does, by at most a factor of 1.157 on these legs. Preferring a hard
constraint (`scipy.optimize.lsq_linear` with `bounds=(-r, r)`) is a reasonable alternative
that has not been run.

### How it is applied here

Two points that matter for reading the numbers:

- **Per pair, not once on the median.** RBR and the OIC solve are nonlinear or
  amplitude-coupled, so they do not commute with the median over pairs, and inverting a single
  median wavefront would give no error bar. Each (reference, comparison) pair's Δ wavefront is
  inverted on its own, for every scheme, and the same median / median-SEM-of-a-median reduction
  used everywhere else in this study is applied to the resulting DOF. Every scheme's error
  therefore means the same thing. `bounce_lib.solver_deltas` is solver-agnostic and takes the
  scheme as a callable, so the four arms differ only in which solver is passed.
- **Every FWHM series is the achieved residual, not the subspace projection.** The subspace
  projection `(I - U U^T) dW` is independent of the recovered amplitudes, and so structurally
  unable to see a regularizer trade wavefront for amplitude; a scheme that gives up amplitude
  for reachability would score identically to one that does not. Every series instead scores
  `dW - S (d / w)` in its own scheme's SVD. The two metrics agree exactly for an unregularized
  truncated solution (measured: 1.7e-16 arcsec FWHM on `fwhm_after_50_34`, 4.2e-17 arcsec FWHM
  on `fwhm_after_5_5`), so switching every series to the achieved residual changed no existing
  number while making all five comparable.

### The code

Every solver is **shared code in `smatrix`** at `smatrix/code/regularized_inversion.py`, imported
rather than copied so it cannot drift from the study that validated it. `bounce_lib.py` supplies
the bounce-side reduction:

```python
import bounce_lib as bl

ri = bl.rbr_module()                       # the shared smatrix solver module
r_j = ri.dof_range_vector(svd)             # allowed range per DOF, µm or arcsec
a_j = ri.oic_authority()[0]                # OIC per-DOF authority, inverse DOF units

# One DOF vector from one Δ wavefront (µm of wavefront -> µm / arcsec of DOF).
# Each takes the SVD of its own scheme, so `svd22` gives 22 entries and `svd5` five:
d_default = ri.invert_truncated(dW, svd)
d_rbr = ri.invert_range_penalty(dW, svd, r_j, kappa=4.0, power=3)
d_oic = ri.invert_oic(dW, svd, a_j, rho=1.0e-3)

# Or, over a bounce leg's pairs, with the median / median-SEM reduction.  The
# solver is a callable, so one function serves every scheme; `keys` maps the
# solver's own DOF rows onto global DOF indices for a reduced scheme:
oic = bl.solver_deltas(W_all, pairs, svd,
                       lambda dW, s: ri.invert_oic(dW, s, a_j, 1.0e-3))
d22 = bl.solver_deltas(W_all, pairs, svd22,
                       lambda dW, s: ri.invert_truncated(dW, s),
                       keys=list(svd22.dof_idx))
#   -> {dof_index: {'delta', 'err', 'sig', 'n'}}, same form as paired_deltas_matrix
```

`invert_oic`'s `authority` must be subset to `svd.dof_idx` by the caller when the SVD carries
fewer than 50 DOF; `run_bounce.py` does that in its `_solve_oic` closure. `dof_range_vector`
needs no such care — it already indexes `f_full[svd.dof_idx]`, so it returns the right rows for a
reduced SVD on its own, and the run asserts that every scheme's own `r_j` matches the
corresponding entries of the global 50-DOF vector before using it.

`run_bounce.py` runs all of this for every leg and night; `--rbr-kappa`, `--rbr-power` and
`--no-rbr` override the config. The IRLS solve needs a backtracking line search to converge
at `p >= 3` and must be restricted to the retained mode coefficients; both are handled inside
the solver and are documented in the `regularized_inversion` study, which is where to look
before changing them. The OIC solve uses the same retained-mode subspace, so across the three
regularizers only the penalty differs.

### Result: RBR bounds the amplitudes at a small cost in FWHM

Per (leg, night), from `bounce_dof_stats.parquet` and `bounce_fwhm_vs_bvalue.parquet`. FWHM
values are the achieved correctable FWHM in arcsec, median over the focal plane; ratios are
dimensionless, recovered amplitude over allowed range, maximized over the 50 DOF.

| leg | night | n pairs | max ratio default | max ratio RBR | n over range default | n over range RBR | FWHM default | FWHM RBR | FWHM cost |
|---|---|---|---|---|---|---|---|---|---|
| elev 30 deg | 20260713 | 5 | 11.70 | 1.16 | 15 | 2 | 0.0598 | 0.0756 | +0.0158 |
| elev 40 deg | 20260418 | 6 | 11.93 | 1.04 | 16 | 1 | 0.0463 | 0.0489 | +0.0027 |
| elev 40 deg | 20260419 | 8 | 10.63 | 1.04 | 14 | 1 | 0.0487 | 0.0492 | +0.0005 |
| elev 40 deg | 20260513 | 8 | 5.95 | 1.03 | 14 | 1 | 0.0354 | 0.0459 | +0.0105 |
| elev 50 deg | 20260711 | 6 | 5.46 | 0.98 | 13 | 0 | 0.0664 | 0.0715 | +0.0052 |
| elev 60 deg | 20260709 | 4 | 3.15 | 0.76 | 8 | 0 | 0.0326 | 0.0357 | +0.0032 |
| elev 75 deg | 20260713 | 6 | 1.40 | 0.56 | 3 | 0 | 0.0153 | 0.0151 | −0.0002 |
| rotator 60 deg | 20260420 | 12 | 3.77 | 1.11 | 12 | 2 | 0.0247 | 0.0273 | +0.0026 |
| rotator 60 deg | 20260513 | 19 | 3.15 | 0.88 | 11 | 0 | 0.0185 | 0.0199 | +0.0014 |

Across the nine (leg, night) points the FWHM cost has a median of +0.0027 arcsec and a range
of −0.0002 to +0.0158 arcsec. Counting `kind = dof` rows, 106 of the 450 per-(leg, night) rows
exceed their range under the default recovery against 7 under RBR; on the leg-pooled rows it is
61 of 300 against 2. The run prints those two counts separately, since a single total would
double-count the same physics.

### Result: what each of the four schemes costs

Per leg, pooled over nights, from `bounce_dof_stats.parquet` and `bounce_fwhm_metric.parquet`.
The ratio is `max_j |Δ_j| / r_j` over the scheme's own DOF — dimensionless, recovered amplitude
over allowed range — taken on the leg-pooled median Δ. FWHM values are the achieved correctable
FWHM in arcsec, median over the focal plane, of the median Δ wavefront; `before` is the
uncorrected Δ.

| leg | n pairs | max ratio 50/34 | RBR | 22/12 | OIC | 5/5 |
|---|---|---|---|---|---|---|
| elev 40 deg | 22 | 7.253 | 0.980 | 0.367 | 1.006 | — |
| elev 60 deg | 4 | 3.147 | 0.760 | 0.182 | 0.461 | — |
| elev 50 deg | 6 | 5.456 | 0.976 | 0.367 | 0.431 | — |
| elev 30 deg | 5 | 11.705 | 1.157 | 0.320 | 1.590 | — |
| elev 75 deg | 6 | 1.403 | 0.563 | 0.053 | 0.316 | — |
| rotator 60 deg | 31 | 3.147 | 0.920 | 0.138 | 1.336 | 0.131 |

| leg | FWHM before | 50/34 | RBR | 22/12 | OIC | 5/5 |
|---|---|---|---|---|---|---|
| elev 40 deg | 0.2986 | 0.0381 | 0.0414 | 0.0933 | 0.1519 | — |
| elev 60 deg | 0.1409 | 0.0326 | 0.0357 | 0.0563 | 0.0666 | — |
| elev 50 deg | 0.2516 | 0.0664 | 0.0715 | 0.1193 | 0.1347 | — |
| elev 30 deg | 0.3987 | 0.0598 | 0.0756 | 0.1344 | 0.2169 | — |
| elev 75 deg | 0.0908 | 0.0153 | 0.0151 | 0.0302 | 0.0457 | — |
| rotator 60 deg | 0.2082 | 0.0191 | 0.0203 | 0.0446 | 0.1417 | 0.0482 |

FWHM cost over the 50/34 truncated recovery, in arcsec, over the nine (leg, night) points — two
of them for 5/5, which only BLOCK-T724 populates:

| scheme | median cost | min | max |
|---|---|---|---|
| RBR 50/34 (kappa = 4, power = 3, both dimensionless) | +0.00266 | −0.00022 | +0.01576 |
| 22/12 reduced | +0.04200 | +0.01482 | +0.07463 |
| OIC 50/34 (rho = 1e-3 dimensionless) | +0.09517 | +0.03037 | +0.15713 |
| 5/5 camera hexapod | +0.03293 | +0.03023 | +0.03564 |

DOF rows over the allowed range (`|Δ|/r_j > 1`, dimensionless), split per-(leg, night) and
leg-pooled because a single total would double-count the same physics. The denominators differ
because each scheme contributes only the DOF it solves for:

| scheme | per (leg, night) | leg-pooled |
|---|---|---|
| 50/34 truncated | 106 of 450 | 61 of 300 |
| RBR 50/34 | 7 of 450 | 2 of 300 |
| 22/12 reduced | **0 of 198** | **0 of 132** |
| OIC 50/34 (rho = 1e-3) | 5 of 450 | 3 of 300 |
| 5/5 camera hexapod | 0 of 10 | 0 of 5 |

**Three results follow, and they order the schemes differently than the regularizer question
alone would.**

*The reduced set is feasible with no penalty at all.* 22/12 never exceeds a DOF's range on any
leg — 0 of 198 rows, worst case `max |Δ_j|/r_j = 0.367` dimensionless — because the 28 DOF it
drops are exactly the weakly-constrained high-order bending modes the unconstrained 50-DOF
inversion was pushing past their range. It does not need a regularizer; restricting the DOF set
*is* the regularizer. The price is +0.042 arcsec FWHM median, about sixteen times RBR's, so
restriction is a far blunter instrument than the range penalty: both deliver a reachable
solution, and RBR delivers it for a sixth of the wavefront.

*The OIC penalty is the most expensive way to reach feasibility.* At the rho that matches RBR's
feasibility, the OIC costs +0.095 arcsec FWHM median — 36 times RBR — and still leaves 5 rows
over range against RBR's 7 at a fifth the cost, with the worst leg at `max |Δ_j|/r_j = 1.590`
against RBR's 1.157. A quadratic penalty cannot distinguish "comfortably inside the range" from
"nowhere near it", so it taxes all 50 DOF to bound the few that need bounding; RBR's superlinear
penalty is near-zero until a DOF approaches its own range. The FWHM understates the damage: the
same indiscriminate tax drives rigid-body amplitude retention to 0.045 to 0.073 dimensionless
against RBR's 0.73 to 0.99, so the OIC reaches feasibility largely by *not recovering* the
hexapod motion rather than by bounding it. This is a measured statement about the knob `ts_ofc`
already ships, and it is the argument for the `range_authority` proposal written up in the
`regularized_inversion` study.

*The camera-hexapod-only recovery loses little on the rotator bounce.* On BLOCK-T724, 5/5 costs
+0.033 arcsec FWHM over the full 50/34 recovery (0.0482 against 0.0191 arcsec on the pooled leg)
while staying far inside range at `max |Δ_j|/r_j = 0.131`. That it costs anything at all is worth
noting — a pure camera-hexapod motion would cost nothing — so the rotator Δ does carry wavefront
outside the camera-hexapod subspace. But 5/5 is cheaper than 22/12 here, so on this bounce the
right five DOF beat a larger set that is not aligned with the motion.

**This measures what the section above could only infer.** The over-range amplitudes carry
almost no wavefront: removing them entirely costs a median 0.0027 arcsec FWHM, a few percent of
the residual and well under 2% of the uncorrected FWHM the bounce produces. They are an
ill-conditioning artifact of the unconstrained inversion, not a real high-order mirror figure
change. The elevation 30 deg leg — largest throw, most extreme excursion — is the worst case at
+0.0158 arcsec, and the near-null upward 75 deg leg is very slightly *better* under RBR, which
is what one expects when the discarded amplitude was noise.

**Caveat on individual rigid-body DOF.** Both penalized schemes bias toward zero by construction,
and the reshuffling is not confined to the bending modes: the `regularized_inversion` study
measures the M2-versus-camera hexapod split moving (M2_dz can flip sign) while the rigid-body
*wavefront* is preserved to within a few percent. A penalized rigid-body amplitude is a
constrained estimate and should not be quoted as a measurement of hexapod motion; for the LUT fit,
which wants the amplitudes themselves, the default recovery remains the estimator. RBR and the OIC
answer whether a physically reachable correction exists and what image quality it delivers.

**The two penalties are not equally biased, and the difference is large.** RBR's superlinear
penalty is near-zero until a DOF approaches its own range, so it leaves the rigid body almost
intact: measured amplitude retention 0.73 to 0.99 dimensionless across the six legs. The OIC's
fixed quadratic curvature taxes every DOF regardless of how far inside its range it sits, so at
the adopted `rho = 1e-3` it returns rigid-body retention of 0.045 to 0.073 dimensionless — a
suppression by 14 to 22×. **An OIC rigid-body amplitude is therefore not usable even as a
constrained estimate**; read the OIC arm for its feasibility and FWHM only. This is quantified in
the `regularized_inversion` study's rho scan.

The 22/12 and 5/5 schemes are unpenalized truncated inversions, so their amplitudes are ordinary
least-squares estimates over a restricted DOF set and carry no such bias — but they are estimates
of a *different* quantity, the best fit available within that set.

## Statistics note — SEM of a median

A paired Δ reports `median(diffs)`, whose standard error is `1.2533 * sigma_mad /
sqrt(n)`, **not** `sigma_mad / sqrt(n)`. The review backlog flagged
`bounce_lib.py:652` for using the mean form; checked 2026-09-05 and **it now has the
1.2533 factor**, consistent with `stats_per_kj` at line 136. That item is fixed —
do not "re-fix" it.

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh --until bounce
```

Knobs in `analysis_config.yaml` under `bounce`: `rbr_enable`, `rbr_kappa` and `rbr_power` for the
range penalty, `reduced_enable` and `n_keep_reduced` for the 22/12 arm (the 22 DOF themselves are
an index set in code, not a count that could be read from config), `oic_enable` and `oic_rho` for
the OIC arm, and `dof_ncols` / `dof_rows_per_page` for the panel pagination.

The lead result here is hand-run against the July fit table, which the Snakemake `bounce` rule
does not target:

```bash
cd ~/notebooks/rubin-work/aos
python code/bounce/run_bounce.py \
  --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
  --mi-name pathA_50_34_i_5rot \
  --fits output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet \
  --out-dir output/bounce/danish_1_2_A_50_34_i_5rot_july \
  --min-detectors 160
```

Add `--no-rbr` to skip the range-bounded arm, or `--rbr-kappa 5 --rbr-power 2` to change the
penalty shape. Every penalized scheme needs `smatrix/code/regularized_inversion.py` on disk; the
run prints `(RBR unavailable [...])` and drops those arms rather than failing if the import does
not resolve.

## See also

- [`../miw_pipeline.md`](../miw_pipeline.md) — the `bounce` rule in context
- [`../../../smatrix/docs/studies/vmode.md`](../../../smatrix/docs/studies/vmode.md) — where the v-modes and DOF come from
- [`../../../smatrix/docs/studies/regularized_inversion.md`](../../../smatrix/docs/studies/regularized_inversion.md) — the shared solver module, the `(p, kappa)` sweep behind the RBR setting used here, the damped-SVD alternative it was chosen over, and the scan over the OIC `rho` that fixed the `oic_rho = 1e-3` dimensionless value this study consumes
- [`../../../smatrix/docs/vmode_normalization.md`](../../../smatrix/docs/vmode_normalization.md) — where the allowed range `r_j` and the FWHM response `f_j` come from
- [`../../../notes/aos-bounce-test-summary/note.md`](../../../notes/aos-bounce-test-summary/note.md) — results summary: elevation sweep 30–75 deg and rotator 0→60 deg
- `../../../notes/claude-memory/apr-2026-50dof-lut.md` — the fixed 50-DOF LUT on sky Apr 24–28 2026, which explains the 20260424/28 anomaly
