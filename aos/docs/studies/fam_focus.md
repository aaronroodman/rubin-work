# Study: `fam_focus` — focus drift within a FAM block

> **Status:** current · **Last updated:** 2026-09-18 · **Kind:** reference (study)

Change in focus — v-mode 1 of the Active Optics System (AOS) sensitivity matrix, essentially
uniform defocus — against exposure sequence number within a single Full Array Mode (FAM) block. A
FAM block is a run of `acq, cwfs, cwfs` triplets taken at one fixed pointing over tens of minutes;
the in-focus `acq` visit of each triplet carries a Corner Wavefront Sensor (CWFS) optical state in
the Consolidated Database (ConsDB), so v-mode 1 can be read per triplet and followed across the
block.

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

The `seq_num` step of 3 validates the `acq, cwfs, cwfs` triplet structure; a 12-triplet set spans
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

## Code

| file | role |
|---|---|
| `code/fam_focus/run_fam_focus.py` | the whole study: selection, the applied correction, and the document |

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
`--max-seq-span`, `--keep-lut-epoch-offset-nights`, `--free-y` (autoscale each panel instead of
sharing one y-range), `--science-lut-dir`, `--out-dir`, `--cache`, `--db-path`, `--consdb-url`.

Output goes to `output/fam_focus/` — the top level, outside any `param_set` or `mi_name`, because
the study consumes ConsDB and the value-added database rather than the FAM donut tables. The
optical-state variant is carried in the data instead.

| product | content |
|---|---|
| `fam_focus.pdf` | the document: 14 pages, opening description, selection validation, the per-set tables, 7 pages of 12-panel drift plots, and the closing scatter comparison |
| `fam_focus_visits.parquet` | one row per selected `acq` visit: identity, `set_id`, band, pointing [deg], the v-mode-1 components, the response, the prediction and the corrected response [µm equiv hexapod dz], and the five thermal features |
| `fam_focus_sets.parquet` | one row per set: `day_obs`, `seq_num` range, mean pointing [deg], band, `n`, and the within-set median, peak-to-peak, standard deviation and drift for both responses [µm equiv hexapod dz] |

## Relation to the other focus studies

| study | timescale | sample |
|---|---|---|
| [`science_lut`](science_lut.md) | night to night | all science visits |
| `fam_focus` | minutes, within one fixed-pointing block | FAM `acq` visits |
| [`lut`](lut.md) | static | dedicated FAM visits |

`fam_focus` consumes the `science_lut` model; the reverse dependence does not exist.
