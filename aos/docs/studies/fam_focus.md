# Study: `fam_focus` — focus drift within a FAM block

> **Status:** stale — verify before acting · **Last updated:** 2026-09-18 · **Kind:** reference (study)

> **Two v-mode sign errors were found on 2026-09-18 and are not yet fixed.**
> `v1_per_um_dz` discards the sign of a negative sensitivity, and `fam_dz.v_modes` is built by
> a different engine than `optical_state.v_modes`, with the opposite sign for v-mode 1. Every
> conclusion below that involves the **thermal prediction** — in particular "the `science_lut`
> thermal correction makes within-block scatter worse" — must be re-derived after the fix.
> The Double Zernike (DZ) coefficients, the selection, the triplet structure and the commanded-focus
> validation are unaffected. See
> [`../status/rerun_needed.md`](../status/rerun_needed.md).

Change in focus — v-mode 1 of the Active Optics System (AOS) sensitivity matrix, essentially
uniform defocus — against exposure sequence number within a single Full Array Mode (FAM) block. A
FAM block is a run of triplets taken at one fixed pointing over tens of minutes; the in-focus
`acq` visit of each triplet carries a Corner Wavefront Sensor (CWFS) optical state in the
Consolidated Database (ConsDB), so v-mode 1 can be read per triplet and followed across the
block.

The study also compares that in-focus v-mode 1 against the Double Zernike (DZ) fit of the
triplet's own defocused FAM pair, term DZ(k=1, j=4) — the field-constant component of pupil
Noll index 4, defocus — read from the `fam_dz` table of the value-added database. Both are put
into **µm of equivalent hexapod dz**, so the two sensors' focus estimates and the `science_lut`
thermal prediction share one y-axis.

The measurement matters twice over. A FAM coadd averages the wavefront over a whole block, so any
within-block focus drift enters the coadd as a systematic. And it is a timescale the
[`science_lut`](science_lut.md) thermal model was never fitted on: that model is fitted and scored
between whole nights, because within a night the thermal telemetry barely moves.

**Result.** The within-block drift is real — median within-set peak-to-peak **36.2 µm of
equivalent hexapod dz** over 82 sets, against the `science_lut` out-of-fold residual normalized
median absolute deviation (nMAD) of 61.3 µm on the same quantity between nights. The drift is not
monotonic: last minus first is median **+4.0 µm** with nMAD **24.6 µm**, scattering both ways.
**The `science_lut` thermal correction makes within-block scatter worse**, raising the median
peak-to-peak to 47.6 µm — a ratio of **1.31 (dimensionless, corrected over uncorrected)** — and
reducing the within-set standard deviation in only **8 of 82 sets**.

## The response

Identical to `science_lut`, so the two studies cannot drift apart:

```
response [um of equivalent hexapod dz] = (v1_trim + MEASURED_SIGN * v1) / v1_per_um_dz
```

with `MEASURED_SIGN = -1`, so the response is `Trim − measured` — the focus error the closed loop
had accumulated but not yet corrected. It is expressed as **equivalent hexapod dz [µm]**: total
defocus travel, 0.5 µm on the camera hexapod and 0.5 µm on the M2 hexapod, through
`v1_per_um_dz = 9.00851e-04` dimensionless v-mode-1 amplitude per µm of total dz travel.

The elevation- and temperature-dependent hexapod look-up-table (LUT) baseline is **deliberately
excluded**, as in `science_lut`, so the known commanded elevation dependence does not enter a
measurement about drift at fixed pointing.

## The triplet

Each triplet of a `BLOCK-T614` block is three consecutive visits, in ascending `seq_num`:

| member | `seq_num` | commanded Trim camera-hexapod dz, offset from the `acq` value [µm] |
|---|---|---|
| intra-focal cwfs | *n* | **−1500.0** |
| extra-focal cwfs | *n* + 1 | **+1500.0** |
| in-focus acq | *n* + 2 | 0 |

The **`acq` is the last** member, not the first. The two defocused members are the FAM donut pair
the DZ fit is made from, and `fits.parquet` is keyed on the **extra-focal** one, so the in-focus
visit of the same triplet is `seq_num + 1` from a FAM fit row and `seq_num − 2` from the intra-focal
one.

Measured over the 919 selected triplets that have complete telemetry (of 984 selected): the intra-
and extra-focal offsets match −1500.0 and +1500.0 µm with a maximum residual of **0.0000 µm**, and
the M2 hexapod dz holds fixed across the triplet to a maximum |offset| of **0.0000 µm**. **The
±1500 µm defocus is applied on the camera hexapod alone**; it is not split between the two
hexapods. The 65 triplets without a full set of three rows in `visit_telemetry` are excluded from
the count rather than treated as failures.

One consequence for the response's units: `v1_per_um_dz` is stated per µm of *total* hexapod dz
travel, 0.5 µm on each hexapod, but the two dz axes carry the same sign and their v-mode-1
sensitivities differ by only **2.1% (dimensionless, camera over M2)** — v-mode 1 per µm is
−8.9144254e-04 for the camera hexapod (degree of freedom 5) and −9.1026032e-04 for M2 (degree of
freedom 0), against the mean magnitude 9.00851e-04 the constant uses. So the same constant converts
a camera-only motion, **1.1% low** — ±17 µm on a 1500 µm offset, far below any scatter here. One
constant, not two.

## The commanded focus is constant within a set

Within-set spread of the three terms of the response, over the 82 sets:

| quantity | median | p90 | max | unit |
|---|---|---|---|---|
| commanded `v1_trim`, standard deviation | 0.00 | 0.00 | 2.43 | µm equiv hexapod dz |
| commanded `v1_trim`, peak-to-peak | 0.00 | 0.00 | 8.43 | µm equiv hexapod dz |
| measured `v1`, standard deviation | 11.23 | 19.43 | 32.33 | µm equiv hexapod dz |
| response, standard deviation | 11.23 | 19.43 | 32.33 | µm equiv hexapod dz |

`v1_trim` is **exactly constant in 81 of 82 sets** — the one exception is set 65 on `day_obs`
20260404, where it moves 8.43 µm peak-to-peak. So the within-set variation of the response is
entirely the measured optical state, not the command, which is what makes the drift a measurement
of the telescope rather than of the loop's own motion.

## Sample

82 sets of 12 triplets — 984 `acq` visits over 24 nights, `day_obs` 20251104 to 20260619, at
elevations 29.5 to 80.2 deg. The band mix is dominated by i: i 67 sets, r 9, u 3, g 2, z 1.

Selection, in order:

| step | count |
|---|---|
| `img_type = 'acq'` in `BLOCK-T614_triplets` or `BLOCK-T614`, `day_obs >= 20251101` | 1,261 visits, 30 nights |
| with a valid `v50_34__batoid__consdb_v1` optical state | 1,238 visits, 29 nights |
| assigned to a contiguous fixed-pointing block | 121 blocks |
| blocks holding exactly 12 visits | 84 |
| with a constant `seq_num` step of 3 | 84 (0 rejected) |
| excluding `LUT_EPOCH_OFFSET_NIGHTS` | **82 sets, 984 visits, 24 nights** |

Blocks come from the [`coadd`](coadd.md) study's greedy pointing-set walk: within one
`(science_program, day_obs)`, a new block starts when `seq_num` reaches 36 past the block's first
visit, or when altitude, azimuth or camera rotator angle drifts beyond **2.0 deg**. Azimuth is
compared circularly. The tolerance is not load-bearing — the `coadd` default of 5.0 deg gives 85
sized blocks against 84, because within a selected set the pointing holds to about 0.01 deg.

The `seq_num` step of 3 validates the triplet structure; a 12-triplet set spans
exactly 33. It currently rejects nothing, because the one block of consecutive `acq` visits
(step 1, 18 visits, 20251219) is already excluded by the size cut. It is kept as a guard.

Two sets fall on `LUT_EPOCH_OFFSET_NIGHTS` (20251211 and 20251219), the nights running a different
hexapod LUT configuration. Their responses sit thousands of µm from the rest because the commanded
baseline itself changed, so they are dropped by default
(`--keep-lut-epoch-offset-nights` restores them).

## The thermal correction is applied, not refitted

The correction is the `science_lut` model — one band-independent Huber robust linear fit on the
Telescope Mount Assembly (TMA) truss temperature and the four M1M3 thermal gradients — fitted on
the 68,317 science visits and applied **unchanged** to the `acq` sample through the fitted pipeline
returned by `run_science_lut_analysis.fit_full`. Nothing is fitted to FAM data. Its truss
coefficient is **+124.64 µm of equivalent hexapod dz per °C**.

## Why the correction does not help inside a block

Median within-set peak-to-peak of the response and of every model input:

| quantity | median within-set peak-to-peak | unit |
|---|---|---|
| response (`Trim − measured`) | 36.2 | µm equiv hexapod dz |
| model prediction | 19.4 | µm equiv hexapod dz |
| truss temperature | 0.073 | °C |
| M1M3 z gradient | 0.027 | °C per m |

The prediction swings 54% as much as the response within a set (dimensionless, median prediction
peak-to-peak over median response peak-to-peak) while the truss temperature moves only a few hundredths
of a °C. The model's coefficients are large because they were fitted between nights, where the
truss temperature moves degrees: +124.64 µm per °C on the truss and −3334.94 µm per (°C per m) on
the x gradient. Inside one block the inputs barely move, so those coefficients turn minutes-
timescale telemetry noise into a prediction swing of the same order as the drift being measured,
uncorrelated with it — the within-set prediction and response peak-to-peak correlate at only
Pearson r +0.551, Spearman rho +0.437 (dimensionless, n = 82).

The model describes night-to-night thermal drift, which is what it was built for. It does not
describe what happens inside one block, and applying it there is not a correction but an addition
of noise.

## The FAM pair's own defocus, DZ(k=1, j=4)

The defocused pair of each triplet carries its own measurement of the same physical quantity: the
DZ fit's term at focal (field) Zernike order k=1 and pupil Noll index j=4, in µm of wavefront. The
study plots that against the in-focus `acq` v-mode 1 response and against the `science_lut` thermal
prediction, one panel per set, `seq_num` on x.

All three series are drawn **in µm of equivalent hexapod dz on one shared y-axis**, so vertical
distance means the same thing everywhere on a panel. DZ(k=1, j=4) is converted through

```
DZ_UM_PER_UM_WF = -63.9902 um of equivalent hexapod dz per um of wavefront
```

derived in [`../../notebooks/smatrix_vmode/ofc_conversion_constants.ipynb`](../../notebooks/smatrix_vmode/ofc_conversion_constants.ipynb)
from the OFC sensitivity matrix. It is the **0.5 µm on each hexapod** inverse, the same convention
`v1_per_um_dz` uses for the response, so both series share one definition of "equivalent hexapod
dz". The camera-only inverse is −62.8389 and the singular-value-decomposition minimum-norm total is
−63.2195 µm of dz per µm of wavefront; all three agree to 1.2% (dimensionless, spread over the
mean), because the camera and M2 dz axes are near-degenerate — their forward sensitivities differ
by only 3.7%. The choice of convention therefore does not affect any conclusion here.

Each series has its own within-set median removed, so the panels show change rather than offset.

Within-set spread over the 62 sets with a DZ fit on every triplet:

| quantity | median within-set peak-to-peak | max | within-set standard deviation, median | unit |
|---|---|---|---|---|
| DZ(k=1, j=4) | 0.3350 | 0.8458 | 0.1044 | µm of wavefront |
| DZ(k=1, j=4), converted | **21.4** | 54.1 | 6.7 | µm equiv hexapod dz |
| `acq` v-mode 1 response | **36.2** | 94.7 | 11.8 | µm equiv hexapod dz |
| `science_lut` thermal prediction | 19.9 | 61.4 | 6.5 | µm equiv hexapod dz |

**The in-focus corner-sensor state swings about 1.7 times as much as the FAM pair's own defocus
over the same set** — the ratio of medians is 1.69 on peak-to-peak and 1.77 on the standard
deviation (both dimensionless, `acq` over FAM). With both axes now in one unit the robust fit is
dimensionless too: a Huber fit of the `acq` response spread on the FAM spread gives slope
**0.690** on peak-to-peak and **0.613** on the standard deviation (dimensionless, `acq` over FAM),
and every panel of the summary scatter sits above the equality line.

The two move together, but only moderately: across those 62 sets the within-set peak-to-peak of
DZ(k=1, j=4) against that of the response gives **Pearson r +0.529, Spearman rho +0.518
(dimensionless, n = 62)**, and the within-set standard deviations **Pearson r +0.520, Spearman rho
+0.538 (dimensionless, n = 62)**. So neither series is a restatement of the other: the two sensors
see a common focus motion plus substantial independent scatter.

Two readings of the amplitude difference are open, and this measurement does not separate them. The
FAM pair is a **defocused** exposure pair whose fit spans the whole focal plane, while the `acq`
v-mode 1 comes from the four corner sensors at best focus, so the two differ in both what they
average over and how they are retrieved; and DZ(k=1, j=4) is one term of the FAM wavefront rather
than the whole of it. The direct route is available — `fam_dz` stores the FAM pair's own v-mode 1,
recovered from the whole wavefront in the same basis as the `acq` optical state — and would
distinguish "this DZ term carries only part of the focus motion" from "the FAM retrieval sees less
motion than the corner sensors do".

Coverage: **870 of 984** selected `acq` visits have a FAM DZ fit, touching **78 of 82** sets, of
which **62** are complete at 12 of 12 triplets. The shortfall is FAM processing coverage — the
`param_set` was built over a narrower date range than the `acq` selection spans — not a quality cut:
no matched row is flagged `bad_fit`.

### Where the DZ coefficients live

The DZ coefficients are in `output/<param_set>/fits.parquet`, one row per FAM extra/intra-focal
pair, with columns named `<prefix>_z<j>_c<k>` — so DZ(k=1, j=4) at `prefix = z1toz6` is
`z1toz6_z4_c1` [µm of wavefront], with its formal error in `z1toz6_z4_c1_err`.
`output/<param_set>/dz_fit_check.parquet` holds **residual diagnostics only** (`resid_nmad_um`,
`resid_rms_um`, `resid_median_um`, `dev_nmad_um` per `pupil_j`) and carries no DZ coefficient.

The two prefixes, `z1toz3` (k=1..3) and `z1toz6` (k=1..6), agree here to 0.0003 µm of wavefront on
the median within-set peak-to-peak, so the focal-order truncation does not drive the result.

## Code

| file | role |
|---|---|
| `code/fam_focus/run_fam_focus.py` | the whole study: selection, the applied correction, the DZ comparison, and the document |
| `../../common/scripts/build_fam_dz.py` | writes the `fam_dz` table this study reads |

**This script needs ConsDB, so it runs on the Rubin Science Platform (RSP) or USDF only** — unlike
`code/science_lut/run_science_lut_analysis.py`, which reads parquet alone. The reason is
`truss_temp_mean_c`: it is derived inside `common/efd_db.py` (`join_consdb`) from two TMA truss
resistance thermometers and then interpolated within the night, rather than stored, so there is no
offline route to it. `--cache <path>` writes the assembled `acq` table and reuses it, making the
network cost one-time.

The script **imports from `code/science_lut/`** — `load_target`, `fit_full`, `page_text`,
`v1_per_um_dz_value` and the response constants — a declared one-directional dependence, so that
both studies share one definition of the response and one fitted model. It also reads
`output/science_lut/science_lut.parquet`, which `code/science_lut/run_science_lut.py` writes.

Options: `--variant`, `--day-obs-min`, `--programs`, `--set-size`, `--seq-step`, `--pointing-tol`,
`--max-seq-span`, `--keep-lut-epoch-offset-nights`, `--free-y` (autoscale each *drift* panel instead
of sharing one y-range; the DZ comparison panels always share one axis, since all three series are
in the same unit), `--science-lut-dir`, `--out-dir`, `--cache`, `--db-path`, `--consdb-url`,
`--fam-variant`, `--dz-col`, `--no-dz` (skip the DZ comparison pages). The DZ pages are skipped with
a note, rather than failing the run, when the named `fam_dz` variant is not registered.

Output goes to `output/fam_focus/` — the top level, outside any `param_set` or `mi_name`, because
the study consumes ConsDB and the value-added database rather than the FAM donut tables. The
optical-state variant is carried in the data instead.

| product | content |
|---|---|
| `fam_focus.pdf` | the document: 22 pages — opening description, selection validation, the per-set tables, 7 pages of 12-panel drift plots, the closing scatter comparison, the commanded-focus validation page, 6 pages of 12-panel plots showing the converted DZ(k=1, j=4), the `acq` response and the thermal prediction on one shared axis in µm of equivalent hexapod dz, and the DZ summary |
| `fam_focus_visits.parquet` | one row per selected `acq` visit: identity, `set_id`, band, pointing [deg], the v-mode-1 components, the response, the prediction and the corrected response [µm equiv hexapod dz], the five thermal features, and the matched FAM `dz` and `dz_err` [µm of wavefront] with the extra-focal `fam_seq_num` |
| `fam_focus_sets.parquet` | one row per set: `day_obs`, `seq_num` range, mean pointing [deg], band, `n`, and the within-set median, peak-to-peak, standard deviation and drift for both responses [µm equiv hexapod dz] |
| `fam_focus_dz_sets.parquet` | one row per set with a DZ fit: the within-set median, peak-to-peak and standard deviation of DZ(k=1, j=4) both as fitted [µm of wavefront] and converted [µm equiv hexapod dz], of the response, and of the thermal prediction [µm equiv hexapod dz], `n_dz`, and whether the set is complete |

## Relation to the other focus studies

| study | timescale | sample |
|---|---|---|
| [`science_lut`](science_lut.md) | night to night | all science visits |
| `fam_focus` | minutes, within one fixed-pointing block | FAM `acq` visits |
| [`lut`](lut.md) | static | dedicated FAM visits |

`fam_focus` consumes the `science_lut` model; the reverse dependence does not exist.
