# common

Shared utility code used across the topic directories, plus the repository's value-added
telemetry database. The repository is not an installed package, so these modules are
imported by inserting the repository root on `sys.path` — `parents[2]` from
`<topic>/code/x.py`, `parents[3]` from `<topic>/code/<study>/x.py`, and
`common.utils.repo_root()` from a notebook, which has no `__file__`.

## Modules

| module | content |
|---|---|
| `utils.py` | `nmad` (normalized median absolute deviation), `alt_to_deg`, `repo_root`, `setup_plotting` |
| `efd_db.py` | schema, upsert and read helpers for the value-added telemetry database |
| `telemetry_clients.py` | Engineering Facility Database (EFD) and Consolidated Database (ConsDB) client construction, with the per-topic time-window padding each quantity needs |
| `FocalPlaneInterpolator.py` | focal-plane interpolation of a quantity sampled per detector |
| `psf_moments_consdb.py` | Point Spread Function (PSF) moments read from ConsDB |
| `notebook_template.ipynb` | the starting point for a new notebook |
| `scripts/` | command-line builders, including the database passes below |

## The value-added telemetry database

`output/value_added/aos_efd.duckdb` — one DuckDB file covering LSSTCam visits from
`day_obs` 20250415 onward, all image types.

It holds exactly two kinds of quantity:

- **EFD-sourced quantities.** EFD access is slow, so each quantity is fetched once and the
  database becomes the single place that access happens.
- **Value-added derived quantities** — those that are expensive or intricate to compute
  rather than merely to fetch: the M1M3 spatial thermal gradients, the hexapod motion
  history, the into-the-wind angle, and the degree-of-freedom (DOF) and v-mode
  decompositions of the measured optical state.

**It is not a ConsDB mirror.** ConsDB and its transformed-EFD tables are fast, so those
columns are queried live at analysis time by `efd_db.join_consdb()`, which merges them onto a
database read and computes `truss_temp_mean_c` [°C] in one place. Copying them in would only
create a second, staler copy of something already cheap to read.

### Two schema idioms

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

Two bookkeeping tables: **`column_coverage`** gives each column's first and last `day_obs`
and non-null count, so a reader can tell "never deployed at that epoch" from "fetch failed"
from "genuinely NaN"; **`fetch_log`** records one row per `(day_obs, group)`, which makes an
interrupted backfill resumable.

### Reading it

```python
from common import efd_db

df = efd_db.visits(day_obs_range=(20260419, 20260713))   # the wide EFD/value-added columns
df = efd_db.join_consdb(df)                              # live ConsDB metadata merged on visit_id
st = efd_db.optical_state('v50_34__batoid__consdb_v1', wide=True)   # v1..v34 per visit

efd_db.variants()          # the variant registry
efd_db.coverage()          # column_coverage
```

Arbitrary SQL stays available through `efd_db.open_db()`. Note that DuckDB's file lock is
process-wide and excludes readers as well as writers, so a read-only open fails while a build
is running; follow a backfill through the builder's log rather than by querying `fetch_log`.

### Building it

Two passes, because the EFD groups need no LSST stack while the optical state needs both the
stack and a variant specification.

```bash
# EFD and value-added groups, one night and one group at a time
python common/scripts/build_efd_db.py --day-obs 20260419-20260713 --groups all --resume

# the recovered optical state, one variant at a time
python common/scripts/build_optical_state.py --variant v50_34__batoid__consdb_v1 \
    --day-obs 20260419-20260713 --resume
```

Nights are independent and every `(day_obs, group)` outcome is logged, so `--resume` skips
what already succeeded and the backfill restarts at any point. A group that fails or returns
nothing is logged and leaves NULLs rather than aborting the night.

Two constraints in the fetchers, both learned from earlier work: the M1M3 gradients are
fetched **one night at a time**, because a multi-night thermocouple query times out; and the
camera-body EFD client is constructed **inside** the coroutine, because aiohttp binds its
session to the running event loop.
