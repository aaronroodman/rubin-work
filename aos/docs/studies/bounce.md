# Study: `bounce` — elevation and rotator bounce tests

> **Status:** current · **Last updated:** 2026-09-23 · **Kind:** reference (study)

> **Code:** `code/bounce/` · **Notebooks:** `notebooks/bounce/`
> **Output:** `output/bounce/<P>_<M>/bounce_*.pdf`, `output/bounce/<P>_<M>/bounce_kj_stats.parquet`, `output/bounce/<P>_<M>/bounce_dof_stats.parquet`, `output/bounce/<P>_<M>/bounce_fwhm_metric.parquet`, `output/bounce/bending_mode_test_meta.parquet`

Analysis of elevation and rotator bounce test data, for Look-Up-Table (LUT)
development. A bounce test moves the telescope to a position and back; a repeatable
difference between the two visits measures hysteresis or gravity-driven flexure rather
than noise.

Two BLOCKs supply the data: **BLOCK-T720** (elevation) and **BLOCK-T724** (rotator
0↔60 deg). The analysis is a *paired* Δ — time-ordered comp−ref pairs **within a night**,
which cancels slowly-varying terms.

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
| `bounce_lib.py` | library: pairing, per-(k,j) statistics, significance, leg-night coverage, plotting, and the RBR wrappers `rbr_dof_per_pair` / `rbr_deltas` |
| `run_bounce.py` | pipeline `bounce` rule — paired Δ for DZ coefficients, OFC v-modes, and physical DOF, with vs-ordinal pages, night cross-scatter, the per-DOF-vs-B-set panels and the achieved-FWHM-vs-B-set comparison; `--rbr-kappa`, `--rbr-power` and `--no-rbr` control the RBR overlay |
| `../../../smatrix/code/regularized_inversion/regularized_inversion.py` | the RBR solver itself, imported from the study that validated it rather than copied |

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
| `bounce_dof_stats.parquet` | per-DOF and per-v-mode Δ, per night and pooled, one row per (bounce, leg, night, quantity) with a `kind` of `dof`, `vmode`, `dof5` or `vmode5` and a `unit` column — µm for translations and bending-mode amplitudes, arcsec for hexapod rotations, dimensionless for v-modes. Carries the *measured* `elevation_deg` and `rot_angle_deg` of the comparison leg and `ref_elevation_deg` / `ref_rot_angle_deg` of the reference, plus `day_obs`, `block`, `n_visits` and `n_pairs`, so it reproduces `bounce_dof_night_values.pdf` and can be shared standalone. For `kind = dof` rows it also carries the Range-Bounded Recovery (RBR) result — `delta_rbr` and `delta_rbr_err` in the row's own unit — alongside `dof_range` (the allowed range `r_j`, same unit) and the two dimensionless ratios `ratio_to_range` and `ratio_to_range_rbr` |
| `bounce_fwhm_metric.parquet` | `fwhm_before`, `fwhm_after_50_34`, `fwhm_after_rbr`, `fwhm_after_5_5`, all arcsec FWHM, per leg, pooled over nights |
| `bounce_fwhm_vs_bvalue.parquet` | one row per (bounce, leg, night), with `b_value` (elevation or camera rotator angle in deg), `n_pairs`, and the three achieved-residual FWHM values `fwhm_before`, `fwhm_after_default` and `fwhm_after_rbr` in arcsec |

`bounce_summary.pdf` opens with a leg-night coverage table — which nights back each leg and how
many night-pair scatter pages follow — then the night-vs-night Δ cross-scatter, wide and
zoomed, for every leg with 2 or more qualifying nights. `bounce_dof_night_scatter.pdf` carries
the same coverage table and the per-DOF night-pair scatter. `bounce_dof_night_values.pdf` shows
the paired Δ DOF as small per-DOF panels against the B-set position — elevation in deg for
BLOCK-T720, camera rotator angle in deg for BLOCK-T724 — with one point per (night, leg); for a
`camera_hexapod_only` bounce only the 5-DOF / 5-v-mode camera-hexapod scheme is drawn, since
that is the scheme the result is used in. Each panel carries the default recovery as a filled
circle, the RBR recovery as an open square in the same night colour, and the allowed range ±`r_j`
as a shaded band; where the band is wider than the data it is annotated as a value in the corner
instead of being allowed to set the y scale. `bounce_fwhm_vs_bvalue.pdf` is one page per bounce
of the three achieved-residual FWHM series against the B-set position, one point per (night, B
set), with the nights at a shared B set fanned out in x for legibility and the trend line drawn
through the per-B-set median over nights. `bounce_dz_vs_ordinal.pdf` opens with the marker
legend and an A/B bounce-position table per bounce.

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

## Range-Bounded Recovery (RBR) — the same Δ, bounded to what the telescope can apply

### What the method does

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

- **Per pair, not once on the median.** RBR is nonlinear, so it does not commute with the
  median over pairs, and inverting a single median wavefront would give no error bar. Each
  (reference, comparison) pair's Δ wavefront is inverted on its own, and the same
  median / median-SEM-of-a-median reduction used everywhere else in this study is applied to
  the resulting DOF. The RBR error therefore means the same thing as the default recovery's.
- **The FWHM comparison uses the achieved residual, not the subspace projection.**
  `fwhm_after_50_34` comes from `aos_fwhm.residual_dW`, which is the projection
  `(I - U U^T) dW` — independent of the recovered amplitudes, and so structurally unable to
  see a regularizer trade wavefront for amplitude. The RBR comparison scores
  `dW - S (d / w)` for both recoveries instead. The two agree exactly for the truncated
  solution (measured: 1.7e-16 arcsec FWHM), so the default series still reproduces
  `fwhm_after_50_34`.

### The code

The solver lives in the `regularized_inversion` study
(`smatrix/code/regularized_inversion/regularized_inversion.py`) and is imported from there
rather than copied, so it cannot drift from the study that validated it. `bounce_lib.py`
supplies the bounce-side wrapper:

```python
import bounce_lib as bl

ri = bl.rbr_module()                       # the smatrix solver
r_j = ri.dof_range_vector(svd)             # allowed range per DOF, µm or arcsec

# One DOF vector from one Δ wavefront (µm of wavefront -> µm / arcsec of DOF):
d_default = ri.invert_truncated(dW, svd)
d_rbr = ri.invert_range_penalty(dW, svd, r_j, kappa=4.0, power=3)

# Or, over a bounce leg's pairs, with the median / median-SEM reduction:
rbr = bl.rbr_deltas(W_all, pairs, svd, r_j, kappa=4.0, power=3)
#   -> {dof_index: {'delta', 'err', 'sig', 'n'}}, same form as paired_deltas_matrix
```

`run_bounce.py` does this for every leg and night; `--rbr-kappa`, `--rbr-power` and
`--no-rbr` override the config. The IRLS solve needs a backtracking line search to converge
at `p >= 3` and must be restricted to the retained mode coefficients; both are handled inside
the solver and are documented in the `regularized_inversion` study, which is where to look
before changing them.

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

**This measures what the section above could only infer.** The over-range amplitudes carry
almost no wavefront: removing them entirely costs a median 0.0027 arcsec FWHM, a few percent of
the residual and well under 2% of the uncorrected FWHM the bounce produces. They are an
ill-conditioning artifact of the unconstrained inversion, not a real high-order mirror figure
change. The elevation 30 deg leg — largest throw, most extreme excursion — is the worst case at
+0.0158 arcsec, and the near-null upward 75 deg leg is very slightly *better* under RBR, which
is what one expects when the discarded amplitude was noise.

**Caveat on individual rigid-body DOF.** RBR biases toward zero by construction, and the
reshuffling is not confined to the bending modes: the `regularized_inversion` study measures
the M2-versus-camera hexapod split moving (M2_dz can flip sign) while the rigid-body *wavefront*
is preserved to within a few percent. An RBR rigid-body amplitude is a constrained estimate and
should not be quoted as a measurement of hexapod motion; for the LUT fit, which wants the
amplitudes themselves, the default recovery remains the estimator. RBR answers whether a
physically reachable correction exists and what image quality it delivers.

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

Knobs in `analysis_config.yaml` under `bounce`, including `rbr_enable`, `rbr_kappa` and
`rbr_power`.

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

Add `--no-rbr` to skip the range-bounded overlay, or `--rbr-kappa 5 --rbr-power 2` to change the
penalty shape. RBR needs `smatrix/code/regularized_inversion/` on disk; the run prints
`(RBR unavailable [...])` and drops the overlay rather than failing if the import does not
resolve.

## See also

- [`../miw_pipeline.md`](../miw_pipeline.md) — the `bounce` rule in context
- [`../../../smatrix/docs/studies/vmode.md`](../../../smatrix/docs/studies/vmode.md) — where the v-modes and DOF come from
- [`../../../smatrix/docs/studies/regularized_inversion.md`](../../../smatrix/docs/studies/regularized_inversion.md) — the RBR solver, the `(p, kappa)` sweep behind the setting used here, and the damped-SVD alternative it was chosen over
- [`../../../smatrix/docs/vmode_normalization.md`](../../../smatrix/docs/vmode_normalization.md) — where the allowed range `r_j` and the FWHM response `f_j` come from
- [`../../../notes/aos-bounce-test-summary/note.md`](../../../notes/aos-bounce-test-summary/note.md) — results summary: elevation sweep 30–75 deg and rotator 0→60 deg
- `../../../notes/claude-memory/apr-2026-50dof-lut.md` — the fixed 50-DOF LUT on sky Apr 24–28 2026, which explains the 20260424/28 anomaly
