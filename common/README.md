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
| `telemetry_clients.py` | Engineering Facility Database (EFD) and Consolidated Database (ConsDB) client construction, with the per-topic time-window padding each quantity needs |
| `ess_telemetry.py` | Per-visit Environmental Sensor System (ESS) telemetry from the EFD: air temperatures and their differences, Telescope Mount Assembly (TMA) truss temperatures, the four M1M3 bulk thermal gradients in degrees Celsius per metre, and inside- and outside-dome wind |
| `FocalPlaneInterpolator.py` | focal-plane interpolation of a quantity sampled per detector |
| `psf_moments_consdb.py` | Point Spread Function (PSF) moments read from ConsDB |
| `psf_render.py` | PSF rendering helpers |
| `notebook_template.ipynb` | the starting point for a new notebook |
| `scripts/` | repository maintenance shell utilities — output archiving and relinking, package updates, undefined-name checks |


## The value-added telemetry database

The database and its builders live in the **`value_added/`** topic, not here. Read it through
`value_added/code/efd_db.py`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'value_added' / 'code'))
import efd_db

df = efd_db.visits(day_obs_range=(20260419, 20260713))
df = efd_db.join_consdb(df)
```

- `value_added/README.md` — scope, how to read it, how to build it
- `value_added/docs/schema.md` — the seven tables, every column group, its source and units
- `value_added/docs/status/build_progress.md` — what is built, what is sparse, what failed
