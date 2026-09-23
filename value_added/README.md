# value_added

A curated database of per-exposure quantities for Vera C. Rubin Observatory analysis, and
the builders that maintain it. It assembles telemetry that is slow to fetch or expensive to
compute into one DuckDB file that any topic can read quickly.

Three kinds of quantity are held. The first is **engineering telemetry** from the
Engineering Facility Database (EFD) that is slow to query per visit: the commanded Trim and
Tweak degrees of freedom (DOF), the hexapod Look-Up-Table (LUT) compensation offsets,
camera-body temperatures, air turbulence, and hexapod motion history. The second is
**derived quantities** that cost real computation, such as the M1M3 bulk thermal gradients,
which the EFD will only serve one night at a time. The third is **recovered optical state**:
the DOF and v-modes inferred per visit, together with the Double Zernike (DZ) coefficients
fitted to Full Array Mode (FAM) data.

Quantities already in the Consolidated Database (ConsDB) are deliberately **not** copied
here. ConsDB is fast, so it is joined live instead, keeping one authoritative source for
anything it already holds.

## Reading the database

`code/efd_db.py` is the read and write interface; nothing else should open the database file
directly. From a script in a topic's `code/` directory:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'value_added' / 'code'))
import efd_db

df = efd_db.visits(day_obs_range=(20260601, 20260714))
state = efd_db.optical_state('v50_34__batoid__consdb_v1', day_obs_range=(20260601, 20260714))
df = efd_db.join_consdb(df)          # band, image type, pointing, weather — read live
```

The readers open a read-only connection themselves and resolve the default database path,
so an analysis neither hardcodes a location nor manages a connection. Pass an explicit
`con=` only to reuse one across many reads, and keep it read-only: DuckDB's file lock is
process-wide and excludes readers as well as writers, so a stray read-write connection
blocks every other process, including a running build.

Two things to get right when reading:

- **Filter `optical_state` and `fam_dz` on their variant id.** Both are keyed by visit *and*
  variant. A forgotten filter silently multiplies the sample rather than raising, which is
  why `efd_db.optical_state()` requires the variant argument.
- **Check column coverage before conditioning on a column.** Coverage is not uniform; the
  turbulence columns in particular are sparse enough to drop a large fraction of the sample
  without any error. `column_coverage` is the table to consult, and it is also the single
  source of truth for every column's units.

## Building

The builders are per-night and independent, so a long backfill is split into shards that
each write their own database file and are merged afterwards. `code/run_build.sh` plans the
shards and launches them; `code/merge_db_shards.py` combines the results.

The telemetry builder is EFD-bound, and the EFD resolves from the Rubin Science Platform
(RSP) and from s3df interactive nodes but not from batch compute nodes, so it runs locally
only — the script rejects batch mode for it. The optical-state builder is ConsDB-bound and
does run in batch. **Submitting a batch job is a must-ask** — see the root `CLAUDE.md`.

## Contents

| path | what it holds |
|---|---|
| `code/efd_db.py` | the read/write interface, the column inventory and the schema definitions |
| `code/build_efd_db.py` | the telemetry builder — EFD fetches and derived columns |
| `code/build_optical_state.py` | the optical-state builder — per-visit DOF and v-mode recovery |
| `code/build_fam_dz.py` | loads fitted FAM Double Zernike coefficients and their v-modes |
| `code/backfill_commanded_vmodes.py` | fills commanded v-modes for visits built before they were stored |
| `code/merge_db_shards.py` | merges shard databases into the main one |
| `code/run_build.sh` | shard planner and launcher, local or Slurm batch |
| `notebooks/value_added_db_validation.ipynb` | validation plots and worked examples, readable by anyone with the stack |
| `output/` | the database, its archives, and the shard directory (not in git) |

## Notebooks

[`notebooks/value_added_db_validation.ipynb`](notebooks/value_added_db_validation.ipynb)
validates every table and demonstrates how to use the database. It deliberately imports no
`rubin-work` code — the LSST Science Pipelines stack plus `duckdb` (`pip install --user
duckdb`) is all it needs — so it can be shared with anyone who has read access to the
database file. It plots each quantity against time and as a histogram, checks
`into_wind_deg` against a recomputation from `wind_dir_deg` and `azimuth_deg`, and works
through which DOF each sensitivity-matrix block perturbed by combining the block identity
from ConsDB with the commanded DOF (Trim) held here.

That last example turns on the structure of the FAM observing pattern. A **triplet** is
three exposures — two defocused, one in focus — in which camera hexapod dz is offset by
±1500 micron about the in-focus position, and a **ladder** is five triplets stepping one
other DOF through `-Delta, -Delta/2, 0, +Delta/2, +Delta`. The perturbed DOF is then the
one with the largest Trim range across a ladder, and the recovered index is cross-checked
against the intent ConsDB records in `observation_reason`.

## Docs

- [`docs/schema.md`](docs/schema.md) — the seven tables, every column group, its source and
  its units
- [`docs/status/build_progress.md`](docs/status/build_progress.md) — what has been built,
  which columns are sparse, and the nights that failed

On the USDF RSP `output/` is a symlink to
`/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/value_added/output/` for disk quota,
which `efd_db.py` follows transparently.
