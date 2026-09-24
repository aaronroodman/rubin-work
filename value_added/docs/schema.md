# Value-added database schema

> **Status:** current · **Last updated:** 2026-09-18 · **Kind:** reference (schema)

The seven tables of the value-added database, what each holds, and where every column comes
from. Row counts were read from the live database on 2026-09-18 and are indicative of scale,
not fixed; for what is currently built and what is sparse, see
[`status/build_progress.md`](status/build_progress.md).

The database covers LSSTCam visits from `day_obs` 20250415 onward, all image types, and holds
exactly two kinds of quantity:

- **EFD-sourced quantities.** Engineering Facility Database (EFD) access is slow, so each
  quantity is fetched once and the database becomes the single place that access happens.
- **Value-added derived quantities** — those expensive or intricate to *compute* rather than
  merely to fetch: the M1M3 spatial thermal gradients, the hexapod motion history, the
  into-the-wind angle, and the degree-of-freedom (DOF) and v-mode decompositions of the
  measured optical state.

**It is not a Consolidated Database (ConsDB) mirror.** ConsDB and its transformed-EFD tables
are fast, so those columns are queried live at analysis time by `efd_db.join_consdb()`, which
merges them onto a database read and computes `truss_temp_mean_c` [°C] in one place. Copying
them in would only create a second, staler copy of something already cheap to read.

## Contents

- [The seven tables](#the-seven-tables)
- [Two schema idioms](#two-schema-idioms)
- [Both intrinsic routes, side by side](#both-intrinsic-routes-side-by-side)
- [`fam_dz` — the Full Array Mode Double Zernike fits](#fam_dz--the-full-array-mode-double-zernike-fits)
- [The registries](#the-registries)
- [`column_coverage` and `fetch_log`](#column_coverage-and-fetch_log)
- [Units](#units)

## The seven tables

| table | rows | columns | shape |
|---|---|---|---|
| `visit_telemetry` | 129,242 | 194 | wide — one row per exposure |
| `optical_state` | 90,695 | 10 | long — keyed `(visit_id, variant_id)` |
| `fam_dz` | 2,528 | 16 | long — keyed `(visit_id, fam_variant_id)` |
| `state_variant` | 3 | 11 | registry for `optical_state` |
| `fam_variant` | 1 | 14 | registry for `fam_dz` |
| `column_coverage` | 193 | 8 | units and provenance registry |
| `fetch_log` | 1,696 | 6 | build bookkeeping |

`visit_telemetry` covers `day_obs` 20251102 to 20260714; `optical_state` 20251102 to 20260713;
`fam_dz` 20250415 to 20260713. Note the earlier FAM start: a join of `fam_dz` to
`visit_telemetry` on `visit_id` drops the 2025 FAM visits, which have no telemetry rows.

## Two schema idioms

**`visit_telemetry` is wide** — one row per exposure, keyed `visit_id`, 194 columns. The EFD
groups genuinely are fixed: each is one value per visit per quantity, they arrive together
from the same per-night fetch, and none has variants.

| group | columns | kind | source |
|---|---|---|---|
| `trim` | 50 | EFD | `MTAOS.logevent_degreeOfFreedom`, as of `obs_start` [µm, arcsec] |
| `lut` | 10 | EFD | `MTHexapod.logevent_compensationOffset` [µm, deg] |
| `camera` | 25 | EFD | camera body, housing and lens temperatures [°C] |
| `turbulence` | 44 | EFD | sonic anemometer speeds [m/s] and temperature [°C] |
| `gradients` | 4 | value-added | M1M3 spatial thermal gradients [°C/m] |
| `tweak` | 50 | value-added | per-iteration DOF correction, differenced from `trim` |
| `wind_derived` | 4 | value-added | `into_wind_deg` [deg] and its inputs, kept for provenance |
| `hexhist` | 3 | value-added | hexapod motion history [µm, count] |

**`optical_state` is long** — keyed `(visit_id, variant_id)`, with DuckDB `DOUBLE[]` list
columns for `v_modes` and `dof`. The recovered optical state is not one column set but a
family of variants along three independent axes, each of which will gain members: the DOF
scheme (`22_12`, `50_34`), the intrinsic-wavefront route (`batoid`, `miw`), and the optical
path difference (OPD) version. A new variant is therefore **rows, not schema**, and comparing
two variants is a self-join on `visit_id`. `state_variant` is the registry saying what each
variant is; a reprocessing of the measured Zernikes arrives as a new `opd_version` rather
than overwriting existing numbers.

The one failure mode this introduces is a forgotten variant filter, which would silently
multiply the sample by the variant count. So `efd_db.optical_state(variant, ...)` requires
`variant` with no default and raises `KeyError` if it is not registered.

## Both intrinsic routes, side by side

The intrinsic wavefront is what defines the optical state, since the corner sensors measure
total OPD and the recovery consumes the deviation. Two routes are therefore kept
simultaneously, as two variants over the same visits rather than two columns or two files:

| variant | intrinsic | `intrinsic_ref` |
|---|---|---|
| `v50_34__batoid__consdb_v1` | batoid ray-trace prediction from `lsst.ts.ofc` | `ofc_v13` |
| `v50_34__miw__consdb_v1` | Measured Intrinsic Wavefront from FAM data | `pathA_50_34_i_5rot` |

Neither overwrites the other, `state_variant.intrinsic_ref` records which intrinsic produced
each, and comparing them is a self-join on `visit_id`:

```sql
SELECT a.visit_id, a.v_modes[1] AS v1_batoid, b.v_modes[1] AS v1_miw
FROM optical_state a JOIN optical_state b USING (visit_id)
WHERE a.variant_id = 'v50_34__batoid__consdb_v1'
  AND b.variant_id = 'v50_34__miw__consdb_v1' AND a.ok AND b.ok
```

or `efd_db.compare_variants(a, b)`. A further MIW build is a new `intrinsic_ref` and so a new
variant, which is why the build name is registered rather than assumed.

Both routes use the same 21-Zernike basis (`aos_state.ZK_NOLL`, Noll 4 to 26 excluding 20 and
21) and the same `ts_ofc` corner sample points, so the only thing that differs between them is
the intrinsic itself. Two asymmetries are properties of the intrinsics, not of the code: the
batoid prediction is band-dependent and identical across the four corners, while the MIW is
field-dependent per corner, varies with camera rotator angle, and carries no band dependence
because it is measured in one band.

The MIW is stored as a telescope-fixed component in the Observatory Coordinate System (OCS)
plus a camera-fixed component in the Camera Coordinate System (CCS), combined at each rotator
angle by `intrinsic_split.reconstruct_at`. The detector heights are camera-fixed and so live
in the CCS component, which means evaluating the combination at the corner field points
already accounts for the corner sensors' heights — no separate height term is added, and
adding one would double-count them.

## `fam_dz` — the Full Array Mode Double Zernike fits

**`fam_dz` is long** on the same pattern, keyed `(visit_id, fam_variant_id)` with
`fam_variant` as its registry. One row per Full Array Mode (FAM) extra/intra-focal pair holds
the Double Zernike (DZ) coefficients of that pair's wavefront fit in `dz_coeff` [µm of
wavefront], their formal errors in `dz_coeff_err`, and the `v_modes` and `dof` projected from
them — so a follow-up never re-projects and risks a different basis. The variant family runs
along four axes: the `param_set` (Butler collection plus processing variant), the
`intrinsic_route` (`batoid` or `miw`), the column `prefix` (`z1toz6` for focal orders k=1..6,
`z1toz3` for k=1..3), and the `scheme` (`50_34` or `22_12`).

`dz_coeff` is stored in the `kj_grid` order of `ofc_svd.build_ofc_svd` — `(k, j)` with pupil
Noll index `j` fastest within each focal order `k` — and that grid is recoverable from
`fam_variant`'s `k_min`, `k_max` and `pupil_j`, so `efd_db.fam_dz(variant, wide=True)` expands
the array to named `dz_k1_j4`-style scalars and nothing indexes it by hand. The pupil Noll set
is read from the `param_set`'s `visits.parquet` `nollIndices` column, never hardcoded: the
canonical set is the 21 indices 4–19 and 22–26, so a contiguous `range(4, 23)` would silently
carry Noll 20 and 21 as all-NaN and drop 23–26.

The FAM triplet is **intra-focal cwfs, extra-focal cwfs, in-focus acq** in ascending
`seq_num`. The DZ fit belongs to the extra-focal member, which is what `fits.parquet` is keyed
on, so each row also stores `intra_seq_num = seq_num - 1`, `acq_seq_num = seq_num + 1` and the
matching `acq_visit_id`. That makes the join to the in-focus visit's corner-sensor
`optical_state` a key lookup rather than a search — the comparison
[`../../thermal_focus/docs/thermal_focus.md`](../../thermal_focus/docs/thermal_focus.md) draws.

```bash
python common/scripts/build_fam_dz.py \
    --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x
python common/scripts/build_fam_dz.py --list
```

```python
fam = efd_db.fam_dz('fam__fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x__batoid__z1toz6__50_34',
                    wide=True)           # dz_k1_j4 .. dz_k6_j26, v1..v34, dof0..dof49
```

`build_fam_dz.py` needs `lsst.ts.ofc` and `$TS_CONFIG_MTTCS_DIR` for the sensitivity-matrix
singular value decomposition, the same constraint `build_optical_state.py` has, but no Butler
and no ConsDB: `fits.parquet` and `visits.parquet` are local. `good_only=True` on the reader
drops rows flagged `bad_fit` while keeping rows whose flag is NULL.

Two bookkeeping tables: **`column_coverage`** gives each column's first and last `day_obs`
and non-null count, so a reader can tell "never deployed at that epoch" from "fetch failed"
from "genuinely NaN"; **`fetch_log`** records one row per `(day_obs, group)`, which makes an
interrupted backfill resumable.

## The registries

Each long table has a registry describing what its variants mean.

`state_variant` (3 rows) — `variant_id`, `scheme` (e.g. `50_34`), `n_dof`, `n_modes`,
`intrinsic_route` (`batoid` or `miw`), `intrinsic_ref` (for the MIW route, the MIW build name,
e.g. `pathA_50_34_i_5rot`), `opd_source`, `opd_version`, `ofc_config_version`.

`fam_variant` (1 row) — `fam_variant_id`, `param_set`, `intrinsic_route`, `intrinsic_ref`,
`prefix`, `k_min`, `k_max`, `pupil_j`, `scheme`, `n_dof`, `n_modes`, `fits_path`.

**These registries embed path and `param_set` names as data**, which matters when renaming:
`fam_variant.fam_variant_id`, `fam_variant.param_set`, `fam_variant.fits_path` (an absolute
path) and `state_variant.intrinsic_ref` all contain names that also appear in the filesystem.
The single registered `fam_variant_id` is
`fam__fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x__batoid__z1toz6__50_34` — the `param_set` name
in full. Renaming a `param_set` or a MIW build without an `UPDATE` pass over these rows
silently breaks the join between the database and the files.

## `column_coverage` and `fetch_log`

**`column_coverage`** holds 193 rows, one per column of `visit_telemetry`, and is the single
source of truth for units. Columns: `column_name`, `group_name`, `source`, `units`,
`first_day_obs`, `last_day_obs`, `n_non_null`, `updated_at`. It lets a reader tell "never
deployed at that epoch" from "fetch failed" from "genuinely NaN".

It is generated from the column inventory in `code/efd_db.py`, which is also what builds the
`CREATE TABLE`, so the schema and its documentation cannot drift apart. `units` is free text
but is **never empty**: a bare number in this database is a bug.

```sql
SELECT column_name, units, source, n_non_null
FROM column_coverage WHERE group_name = 'gradients';
```

**`fetch_log`** holds 1,696 rows, one per `(day_obs, group)` outcome, which makes an
interrupted backfill resumable.

Nights are independent and every `(day_obs, group)` outcome is logged, so `--resume` skips
what already succeeded and the backfill restarts at any point. A group that fails or returns
nothing is logged and leaves NULLs rather than aborting the night.

Two constraints in the fetchers, both learned from earlier work: the M1M3 gradients are
fetched **one night at a time**, because a multi-night thermocouple query times out; and the
camera-body EFD client is constructed **inside** the coroutine, because aiohttp binds its
session to the running event loop.

## Units

Every column carries its units in `column_coverage`, and the convention is the repo-wide one:
a value is either given a physical unit or explicitly marked dimensionless with the ratio
named. The cases worth knowing:

- **DOF columns** (`trim`, `tweak`) are mixed-unit by nature: µm for the z/x/y displacements
  and bending modes, deg or arcsec for the u/v rotations. One column, one unit; the group is
  mixed.
- **Temperatures** are °C throughout; the M1M3 spatial gradients are °C/m.
- **`day_obs`** is dimensionless, formatted YYYYMMDD; `seq_num` is a dimensionless sequence
  number; `obs_start` is a TAI ISO-8601 timestamp, not a float.
- **DZ coefficients** are µm of wavefront.
- **Wind direction** exists both as recorded and as `into_wind_deg` [deg] derived relative to
  azimuth — check which one an analysis wants.
