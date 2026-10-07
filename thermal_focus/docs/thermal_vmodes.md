# Thermal response of all 34 v-modes

> **Status:** current · **Last updated:** 2026-10-06 · **Kind:** reference (study)

> **Code:** `code/thermal_vmodes.py`, `code/run_thermal_vmodes.py`,
> `code/thermal_vmodes_figures.py`, `code/thermal_vmodes_channels.py`,
> `code/thermal_vmodes_channel_figures.py`, `code/test_thermal_vmodes.py`,
> `code/test_thermal_vmodes_channels.py`
> **Output:** `output/thermal_vmodes/`
> **Notebooks:** `notebooks/`

Extends the [thermal-focus](thermal_focus.md) deliverable from v-mode 1 to every v-mode the
50-degree-of-freedom (DOF) recovery retains, asking **which other v-modes carry a component
predictable from telescope thermal telemetry**.

The thermal-focus model predicts one quantity: v-mode 1, which is almost pure uniform
defocus, from five thermal channels. Nothing in that work says whether astigmatism, coma or
any higher mode behaves the same way. This study runs the same fit on all 34 and reports
per-mode skill against a null.

## The response

For mode `k` the response is the **optical state**, `Trim − Deviation`, which is the negative
of the stored open-loop column:

```
y_k [dimensionless v-mode amplitude] = −v{k}_olr = v{k}_trim − v{k}
```

Verified against `thermal_focus_lib.MEASURED_SIGN` to 6.7e-16 (dimensionless v-mode
amplitude) over modes 1 through 34, so this is the v-mode-1 convention generalized rather
than a new one.

**The response is left dimensionless**, not divided into µm of equivalent hexapod dz as the
v-mode-1 deliverable is. That conversion (`DZ_UM_PER_UM_WF`, −63.9902 µm of equivalent
hexapod dz per µm of wavefront) is specific to defocus; no single physical axis stands in for
the higher modes, so a per-mode physical conversion would be invented rather than derived.

## Why this is not just the existing fit run 34 times

**The null matters more than the fit.** Most modes are expected to carry no thermal signal,
so the question is per-mode skill against an intercept-only model, not goodness of fit. Skill
is the fractional reduction in out-of-fold residual normalized median absolute deviation
(nMAD), both terms night-grouped. The null's intercept is the **median** of the training
folds, not the mean — the robust counterpart of the Huber fit it is compared against. A
mean-based null would be pulled by the same one-sided tail the Huber fit is chosen to resist,
flattering the fit.

**34 simultaneous tests need a correction.** `mode_table` applies a Benjamini-Hochberg
false-discovery-rate cut at `q = 0.05` (dimensionless), so a mode is called thermal only if it
survives. Skill has no analytic null distribution here — the response is spatially correlated
between modes and the folds are not independent — so the empirical null is taken from the 34
modes themselves. **This is a screening rule for which modes deserve a closer look, not a
calibrated significance claim**; a mode near the threshold should be confirmed with
`thermal_focus_fit.nested_comparison` on that mode alone.

**The high modes are measurement noise.** Four corner wavefront sensors (CWFS) constrain 84
Zernike values and the recovery's scatter grows with mode index. `noise_floor_table` reports
per-mode within-night against between-night scatter, so a reader does not take a high-mode
slope at face value. Within-night scatter is estimated from successive-visit differences
divided by sqrt(2), which does not assume the state is constant across a night: a slow
thermal drift contributes to the between-night term instead. A mode whose between/within
ratio is near 1 (dimensionless) carries no more night-to-night structure than its own
measurement noise and is not interpretable however it scores.

## Nights are held out whole

Only 2.7% of the Telescope Mount Assembly truss temperature's variance is within-night
(dimensionless, within-night over total), so consecutive visits are near-duplicates in
feature space and a visit-level split leaks. Every fit here uses
`thermal_focus_fit.evaluate`, which is `GroupKFold` on `day_obs`. The measured optimism of a
visit-level split is a factor of 3.1 (dimensionless) for boosted trees and about 1.04 for the
Huber linear fit; see [`thermal_focus.md`](thermal_focus.md).

## Variants

| role | variant | why |
|---|---|---|
| primary | `v50_34_rbr__batoid__consdb_v1` | range-bounded recovery, physically realizable on 99.5% of visits |
| intrinsic check | `v50_34__batoid__consdb_v1` and `v50_34__miw__consdb_v1` | the two intrinsic routes at fixed solver |

The primary result uses the range-bounded recovery because the unconstrained 50/34 solution
asks a median 33x the available actuator stroke on every visit, so its high-mode amplitudes
are partly fitting an unreachable state (`olr/docs/scheme_comparison.md`).

The intrinsic check holds the solver fixed and varies only the intrinsic — there is no
range-bounded arm on the measured intrinsic wavefront (MIW) route. **The expectation is that
the two routes agree on which modes are thermal**: the MIW differs from the batoid prediction
by a static offset per rotator angle, and a static offset moves a fitted intercept, not a
thermal slope. A mode where they disagree is either dominated by the rotator-angle-dependent
part of the MIW−batoid difference — which correlates with elevation through the observing
pattern, and so with temperature — or is marginal on both routes.

## Features and selection

The five thermal channels of the thermal-focus deliverable, `DELIVERABLE_GROUPS`: the TMA
truss temperature and the four M1M3 bulk thermal gradients. The whole selection funnel is
reused from `run_thermal_focus.load_science`, including the LUT-epoch night exclusion and the
20 °C truss-temperature cut, so the sample is the one the published v-mode-1 result is fitted
on. `load_science` gained a `keep_extra` argument to carry the 34 `v*_olr` columns through;
its default behaviour is unchanged.

## Code

| function | what it does |
|---|---|
| `run_thermal_vmodes.py` | the run: the per-mode table, the noise floor, both intrinsic routes, the per-channel screen, the figures |
| `thermal_vmodes_figures.py` | the five figure pages, one function each |
| `thermal_vmodes_channels.py` | the per-channel screen: one channel at a time, the duplicate guard, the combined fits, the v-mode Zernike content |
| `thermal_vmodes_channel_figures.py` | the screen's figures: both heatmaps, the prediction test, one page per followed-up mode |
| `attach_mode_response` | sets `y` to one mode's optical state from the stored open-loop column |
| `null_nmad` | out-of-fold residual nMAD of a median-intercept null, nights held out |
| `fit_mode` | night-grouped Huber fit and null for one mode |
| `bh_threshold` | Benjamini-Hochberg cut over the 34 per-mode statistics |
| `mode_table` | the per-mode table, sorted by skill, with the cut and the noise flag applied |
| `noise_floor_table` | per-mode within-night against between-night scatter |
| `intrinsic_comparison` | per-mode skill on the two intrinsic routes, side by side |

`code/test_thermal_vmodes.py` has 8 tests. Two matter: the response sign is pinned against
`thermal_focus_lib.MEASURED_SIGN`, and the screening rule is checked on a synthetic frame
where one planted thermal mode must be the only one called — and on a pure-noise frame where
nothing must be.

## The per-channel screen

The first pass fitted five channels jointly per mode and reported one skill number, which hid
two things. **Skill carries no sign**, so v-modes 3 and 18 scored positively while
*anti*-correlating with truss temperature; and a joint fit cannot say *which* channel a mode
responds to, which is what a physical explanation needs.

So `thermal_vmodes_channels` fits **one channel at a time** — a Huber line per (v-mode, channel)
pair with Spearman rho alongside — over a much wider telemetry set than the deliverable's five:

| family | channels |
|---|---|
| structure | TMA truss mean temperature |
| M1M3 shape | the four thermal gradients (z, radial, x, y) and the three quadratic radial terms (M1M3, M1, M3) |
| camera | average and ambient air, three body, four housing, two L1 and three L2 lens temperatures |
| differences | camera − ambient, truss − ambient, camera − truss, L1 − L2 at X−, and the housing X and Y asymmetries |

A difference is carried as its own channel because an absolute temperature and an excess over
ambient are different physical drivers: a uniform warming of camera and air together changes
neither spacing nor figure, while an excess does.

The four ESS sonic temperatures are **excluded**. They cover 90,814 of 213,704 rows, so requiring
them would cut the sample by more than half — the same reason `run_thermal_focus.TELEMETRY_COLS`
leaves the turbulence group out. The camera air-handling channels (shroud ring, plenum, shutter,
charger return air) are also left out: they track `cam_AverageTemp` and carry no separate optical
structure.

### Near-duplicate channels are skipped, not just ranked

Taking a mode's top four channels by |rho| gives four copies of one signal. For v-mode 1 those
were camera body Y−, Y+, X+ and L2 Y+, mutually correlated at rho 0.92 to 0.95 — four
thermometers on the same structure. Fitting them together **lost** skill, +0.740 on the single
best channel against +0.694 on four, because the Huber fit splits one real coefficient across
four collinear columns and pays variance for it.

`select_lead` therefore walks the ranking greedily and admits a channel only if it correlates
below `CHANNEL_DUP_RHO = 0.9` with every channel already kept. With that, v-mode 1's leading set
becomes four distinct drivers and the combination *gains*: +0.740 → +0.784. The count of skipped
duplicates is reported per mode.

### Which v-mode is which aberration cannot be guessed

The natural reading of mode index is wrong, so `vmode_zernike_content` computes each v-mode's
field-averaged Zernike content from `aos_state.corner_recovery_basis` — whose `U` is (84, 50),
four corners by 21 Noll terms, corner-major (verified: v-mode 1 carries 0.500 of Z4 at all four
corners, which is uniform defocus, and norm 1.0 overall).

| Noll term | strongest v-modes (field-averaged amplitude, dimensionless) |
|---|---|
| Z4 defocus | v1 0.500, v4 0.437, v5 0.437 |
| Z11 spherical | v21 0.338, v16 0.250, v18 0.113 |
| Z22 second spherical | v14 0.104, v31 0.096, v28 0.072 |

**V-mode 12 is not spherical** — it is dominated by Z15 (0.470), a trefoil-family term. Spherical
Z11 lives mainly in v21 and v16, and Z22 in v14 and v31. `prediction_table` therefore tests the
S-matrix expectation from `smatrix/docs/plots.md` — z-gradient drives Z4 + Z11, radial gradient
drives Z4 + Z11 + Z22 — against the modes that actually carry those terms rather than against
mode index.

## Results

**V-mode 1 is the only thermal mode.** Over 72,835 science visits across 175 nights on
`v50_34_rbr__batoid__consdb_v1`, one mode of 34 survives the false-discovery-rate cut at
`q = 0.05` (dimensionless), and it is the defocus mode the thermal-focus deliverable already
models. The gap below it is wide, not marginal: v1 scores a skill of +0.808 and the next mode,
v10, scores +0.243 (both dimensionless, fractional reduction in out-of-fold residual nMAD).

The per-channel screen below widens this picture without changing the conclusion. **Seven modes
correlate with at least one thermal channel at |Spearman rho| ≥ 0.4**, and v-mode 18's response
to the M1M3 z-gradient (rho −0.511) is a real relation that the five-channel joint fit missed.
But only v-mode 1 behaves like a mode with a thermal *origin*: for the rest a thermal model
cannot reproduce the mode's range, and the four-channel combination buys as little as 1% over a
single channel. The per-channel section is the stronger statement of the result and supersedes
the sign argument below.

| mode | skill | nMAD null | nMAD fit | between/within | well constrained | thermal |
|---|---|---|---|---|---|---|
| v1 | +0.808 | 0.3092 | 0.0593 | 37.07 | yes | **yes** |
| v10 | +0.243 | 0.6818 | 0.5162 | 4.01 | yes | no |
| v18 | +0.222 | 0.1870 | 0.1455 | 1.78 | no | no |
| v15 | +0.219 | 0.6605 | 0.5161 | 9.72 | no | no |
| v19 | +0.208 | 0.6168 | 0.4888 | 3.75 | no | no |
| v13 | +0.169 | 0.5766 | 0.4794 | 8.68 | no | no |
| v22 | +0.160 | 0.1648 | 0.1384 | 2.70 | no | no |
| v3 | +0.152 | 0.0741 | 0.0628 | 5.74 | yes | no |

Skill and the between/within ratio are dimensionless; the nMAD columns are dimensionless v-mode
amplitude. The v1 truss coefficient is +0.1132 (dimensionless v-mode-1 amplitude per °C of mean
TMA truss temperature), the same channel and sign the deliverable reports.

The modes scoring between +0.15 and +0.25 are **not** a weak thermal signal to be chased. Four
things place them: none survives the multiple-comparison cut; the empirical null's own scale is
set by that cluster, so they define the noise rather than stand out from it; most of them leave
a residual an order of magnitude larger than v1's in absolute terms (v10 at 0.516, v15 at
0.516, v13 at 0.479 against v1's 0.059, dimensionless v-mode amplitude), so taken at face value
they would predict little of what is there; and their skill does not come from a temperature
relation at all.

The absolute-residual argument does not cover the whole cluster, and saying it did would be
wrong: across the eight modes scoring +0.15 to +0.25 the fitted residual spans 0.063 to 0.516
(dimensionless v-mode amplitude), so v3 at 0.063 and v18 at 0.146 are not large-residual modes.
For those two the other three arguments carry the case — in particular the next one.

That last point is visible on the figures and is the clearest of the four. Per-visit
correlations against mean TMA truss temperature, over the same 68,690 visits (all dimensionless):

| mode | skill | Pearson r | Spearman rho |
|---|---|---|---|
| v1 | +0.808 | **+0.529** | **+0.789** |
| v10 | +0.243 | −0.057 | −0.090 |
| v18 | +0.222 | −0.178 | −0.170 |
| v3 | +0.152 | −0.171 | −0.223 |

V-mode 10 has no relation to truss temperature at all. V-modes 3 and 18 — the two cluster modes
whose absolute residuals are small, so the argument above does not reach them — correlate
*negatively* with truss temperature, the opposite sign to v-mode 1.

**The per-channel screen qualifies this for v-mode 18.** Its negative truss correlation is not
the whole story: its actual strongest channel is the M1M3 z-gradient at rho −0.511, a stronger
relation than it has with truss temperature, and one the five-channel joint fit could not
expose. So "anti-correlates with truss temperature" is the right reading of this table but the
wrong reason to dismiss the mode; the reason it is still not a thermal mode is that no
combination of channels reproduces its range (see the per-channel section). V-mode 3 stays
dismissed: its strongest channel anywhere in the 28 is a camera body temperature at rho −0.260,
well below the follow-up threshold.

So a mode can post a moderate grouped skill with no thermal relation, because holding out whole
nights lets a night-level offset be partly predicted by whatever the features happen to do that
night. Skill against a median-intercept null is the right statistic for *ranking* modes, but it
is not by itself evidence of a thermal relation. The sign and the scatter are what settle it,
which is why the figures carry both.

**The noise floor is not monotonic in mode index**, which is why the ratio is measured rather
than assumed. 19 of 34 modes carry between-night structure at more than twice their own
within-night scatter; the 15 that do not are v6, v11, v12, v17, v18 and v23 through v32. So v34
(ratio 5.15) is better determined night to night than v11 (ratio 0.96) or v12 (ratio 1.02).
`WELL_CONSTRAINED_MAX = 12` remains a useful flag for the recovery's conditioning, but it is not
the same statement as this ratio and the table reports both.

**The two intrinsic routes agree, as predicted.** On the unconstrained 50/34 pair, 32 of 34
modes get the same thermal flag and the median absolute skill difference is 0.0016
(dimensionless). Both disagreements are threshold artefacts rather than physics: v3 scores
+0.172 on both routes to three decimals and falls on opposite sides of the cut, and v18 differs
by 0.030 (dimensionless) with neither route calling it thermal in the primary result. Neither
route finds a thermal mode the other misses.

The practical consequence: the published v-mode-1 correction is the whole thermal feed-forward
available from these five channels. There is no second mode to add to it.

### Per-channel screen: seven modes correlate, none has a thermal origin but v-mode 1

Over 952 single-channel fits (34 modes × 28 channels), **45 pairs reach |Spearman rho| ≥ 0.4**
and **seven modes** have at least one such channel — against the one mode the joint five-channel
screen found. The seven, each with its strongest channel:

| mode | strongest channel | Spearman rho | nMAD null → single → combined | skill single → combined | gain |
|---|---|---|---|---|---|
| v1 | TMA truss mean | **+0.789** | 0.302 → 0.138 → 0.104 | +0.553 → **+0.656** | +24.9% |
| v18 | M1M3 z gradient | **−0.511** | 0.188 → 0.149 → 0.146 | +0.204 → +0.225 | +1.9% |
| v15 | truss − ambient air | +0.456 | 0.679 → 0.557 → 0.492 | +0.172 → +0.275 | +11.6% |
| v10 | M1M3 quadratic radial | +0.449 | 0.679 → 0.558 → 0.552 | +0.182 → +0.187 | +1.1% |
| v19 | M1M3 z gradient | +0.421 | 0.612 → 0.532 → 0.500 | +0.131 → +0.183 | +5.9% |
| v13 | truss − ambient air | +0.405 | 0.584 → 0.513 → 0.487 | +0.121 → +0.166 | +5.1% |
| v21 | L2 lens X+ | −0.405 | 0.188 → 0.168 → 0.154 | +0.151 → +0.180 | +8.1% |

nMAD columns are dimensionless v-mode amplitude; rho, skill and gain are dimensionless. The
`gain` column is what the four combined channels buy over the single best one. Sample sizes
differ by channel coverage: 72,835 visits where only the gradients are needed, 67,430 to 67,519
where `cam_AmbAirtemp` enters.

**V-mode 1's skill reads +0.656 here against +0.808 in the primary table.** Not a
contradiction — the combined fit leads with `truss_minus_ambient_c`, which requires the ambient
air channel and drops the sample to 67,519 visits. The primary five-channel fit runs on 68,690.

**A correlation is not a thermal origin, and the mode pages are what separate them.** For
v-mode 18 the single-channel relation against the M1M3 z-gradient is real — rho −0.511, slope
−0.5317 ± 0.0033 (dimensionless v-mode amplitude per °C/m). But the out-of-fold combined
prediction spans roughly 0.0 to +0.6 while the response spans −0.5 to +1.0, so it cannot track
the mode: the residual nMAD falls only 0.188 → 0.146 and the four-channel combination buys
+1.9% over one channel. The same holds for v10 (+1.1%). Those are correlations with a thermal
channel, not modes a thermal model would predict.

Only v-mode 1 shows the signature of a thermal origin: the prediction follows the response along
the 1:1 line across its whole range, the residual collapses by a factor near three, and the
combination still adds a quarter on top of the best single channel.

### The M1M3 prediction holds for one pair out of fifteen

Tested on the modes that actually carry each Zernike, **1 of 15 predicted pairs reaches
|rho| ≥ 0.4**:

| prediction | measured, on the three modes carrying that Zernike |
|---|---|
| Z4 defocus from the z-gradient | v1 −0.153, v4 −0.107, v5 +0.205 |
| **Z11 spherical from the z-gradient** | v21 +0.066, v16 −0.179, **v18 −0.511** |
| Z4 defocus from the radial gradient | v1 −0.002, v4 −0.014, v5 +0.170 |
| Z11 spherical from the radial gradient | v21 −0.004, v16 −0.229, v18 −0.181 |
| Z22 second spherical from the radial gradient | v14 −0.069, v31 +0.168, v28 +0.008 |

All dimensionless. The one hit is the spherical-from-z-gradient pair on v-mode 18 — which is
also the strongest non-v1 correlation anywhere in the grid, and v18 is the third-largest carrier
of Z11 (amplitude 0.113). So the predicted *channel* for spherical is the one that shows up, but
it shows up on the third carrier rather than the first two: v21 and v16, which hold most of the
Z11, give +0.066 and −0.179.

The Z22 prediction fails outright on all three carriers. And defocus Z4 shows no z-gradient
response on the modes that carry it, including v-mode 1 at −0.153 — v-mode 1's thermal signal is
in the truss bulk temperature (rho +0.789), not in a gradient.

Worth noting what this does **not** test: the prediction is about the mirror's figure response,
while the measured quantity is the open-loop state after the AOS has been correcting. A gradient
term the loop removes well would be absent here whatever the mirror does.

Reproduce with:

```bash
python code/run_thermal_vmodes.py
python code/run_thermal_vmodes.py --no-channels      # the first-pass products alone
```

Products in `output/thermal_vmodes/`: `mode_table_<variant>.parquet`, `noise_floor.parquet`,
`intrinsic_comparison.parquet`, `thermal_vmodes.pdf`, and from the per-channel screen
`channel_grid_<variant>.parquet`, `channel_combined.parquet`, `channel_nmad_summary.parquet`,
`channel_prediction.parquet`, `channel_expectation.parquet`, `vmode_zernike_content.parquet`
and `thermal_vmodes_channels.pdf`.

## Figures

`thermal_vmodes.pdf`, five pages, written by the same run. `--no-figures` writes the tables
alone.

| page | what it shows |
|---|---|
| skill per mode | every mode's skill with the false-discovery-rate cut drawn on it, and the same values as a rank plot beside it — the cut comes from the median and nMAD of those very points, so the distribution it was taken from is on the page |
| absolute residual | residual nMAD per mode, null against fit, log scale. Skill is a *fraction*, so this is the page that shows the +0.15 to +0.25 cluster leaving an order of magnitude more residual than v-mode 1 |
| noise floor | within-night and between-night scatter per mode, and their ratio against the ratio-2 line. The ratio is not monotonic in mode index, which is why it is measured; `WELL_CONSTRAINED_MAX` is drawn for contrast and is deliberately a different line |
| the relation | v-mode-1 optical state against mean truss temperature as a per-visit hexbin with night medians over it, beside the highest-skill mode the cut rejects. Signal against best non-signal at matched scale |
| intrinsic routes | per-mode skill on one route against the other with the 1:1 line, flag disagreements circled and labelled; a threshold artefact sits *on* the line, a real disagreement does not |

The Huber line on the fourth page is fitted on the per-visit points, which is **not** the
night-grouped out-of-fold number the skill column reports. Each panel title carries the grouped
skill alongside it so the two are not read as the same quantity.

`thermal_vmodes_channels.pdf`, 11 pages, from the per-channel screen:

| page | what it shows |
|---|---|
| rank-correlation heatmap | the whole screen: 34 modes by 28 channels, Spearman rho on a diverging scale centred at zero so the **sign** reads, cells past \|rho\| 0.4 labelled, a rule separating base channels from derived differences. The camera block is visibly one signal; v18's isolated response on the M1M3 shape columns alone is visibly not |
| skill heatmap | the same grid as night-grouped out-of-fold skill, blank where a channel does not beat the null. Shown beside the first because a per-visit correlation from a few nights buys nothing out of fold, and only a cell strong in both deserves a physical story |
| prediction test | where Z4, Z11 and Z22 actually live across the v-modes, beside the measured response of those modes to the channel the S-matrix work names |
| M1M3 shape channels | per-channel rho against mode index for the four gradients and three quadratic radial terms, thresholds drawn |
| one page per qualifying mode (7) | the response against its strongest channel; the out-of-fold combined prediction against the response with the 1:1 line; and the residual histogram before and after, with all three nMAD values |

**The middle panel of each mode page is the discriminator.** A mode with a real thermal
dependence gives a prediction that tracks the response across its whole range; a mode whose
correlation is a night-level artefact gives a prediction clustered in a narrow band whatever the
response does. V-mode 1 shows the first, v-modes 10 and 18 the second.

## Reference

- [`thermal_focus.md`](thermal_focus.md) — the v-mode-1 deliverable: the five features, the
  Huber pipeline, the night-grouped evaluation and the physical conversion.
- `olr/docs/scheme_comparison.md` — the three solvers on 96,278 paired visits.
- `value_added/docs/schema.md` — `optical_state` columns and the two sign conventions.
- `aos/docs/studies/cwfs_lut.md` — the pointing-dependence study specified alongside this one.
- `notes/status/vmode_thermal_and_lut_handoff.md` — working state for both studies.
