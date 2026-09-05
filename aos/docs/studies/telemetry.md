# Study: `telemetry` — the telescope's commanded and thermal state

> **Status:** current · **Last updated:** 2026-09-05 · **Kind:** reference (study)

**Question.** What was the telescope actually doing, per visit — what did the AOS loop
command, and what were the temperatures?

Every other study that correlates a wavefront against *conditions* depends on this one.
It is infrastructure, not an analysis: it fetches, caches, and normalizes per-visit state
from the EFD and ConsDB.

**No Snakefile rules** — these are libraries plus backfill scripts.

## Code and its reach beyond `aos/`

Three of these are **imported by four other topics** and therefore **stay flat at
`aos/code/`** (see [`../studies.md`](../studies.md#cross-cutting-code)):

| file | role | external references |
|---|---|---|
| `aos_trim.py` | AOS Trim/Offset from the EFD: the MTAOS closed loop's accumulated per-DOF corrections. Also the generic `make_consdb_client` | **21** — `blocks/`, `olr/`, `optatmo/`, `guider/` |
| `aos_state.py` | shared per-visit state helpers; single source of truth for the normalization-sensitive pieces (`make_state_estimator`, `vmodes_from_dofs`, `recover_optical_state`, `ZK_NOLL`) | **15** — `blocks/`, `olr/`, `optatmo/` |
| `aos_consdb_efd.py` | ConsDB *transformed*-EFD telemetry — the fast path, per-exposure means, no per-visit raw EFD | **3** — `blocks/` |
| `run_backfill_thermal.py` | backfill 13 core ESS-temperature / M1M3-gradient / TMA-truss columns onto a `visits.parquet` without re-running `mktable` | — |
| `run_backfill_camera_telemetry.py` | backfill camera-**body** temperatures as a per-chunk visit-keyed parquet, then optionally merge | — |
| `test_m1m3.py` | quick command-line test of M1M3 thermal-gradient retrieval | — |

**Do not move the first three into a subdirectory.** Sibling topics import them by bare
module name via a hardcoded `sys.path.insert(.../aos/code)`; moving them breaks
`blocks/`, `olr/`, `optatmo/`, and `guider/` silently, with no static-import warning.

## What is where

- **ConsDB** has hexapod LUT/Trim, wind, and thermal per visit (`efd_lsstcam`), but **no**
  mirror modes and no M1M3 gradients.
- **Camera-body temperatures are EFD-only** (`MTCamera.utiltrunk_body`, `AverageTemp`);
  ConsDB carries only ESS air/hexapod temps. That is why there are two backfill scripts.
- A per-visit AOS telemetry **cache** was designed and deferred — see
  `../../../notes/claude-memory/aos-private-consdb-cache.md`.

## Terminology — do not interchange

`optical_state` vs **Tweak** vs **Trim** are different quantities. See
`../../../notes/claude-memory/aos-dof-terminology.md` before using any of them, and the
22-DOF reduced set has **specific indices** — do not infer them
(`aos-22dof-reduced-set.md`).

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/run_backfill_thermal.py --help
python code/run_backfill_camera_telemetry.py --help
```

Needs ConsDB/EFD access — RSP or slaciana. `$TS_CONFIG_MTTCS_DIR` is preferred for OFC
config where the stack sets it; the hardcoded `/sdf/group/rubin/...` path is the fallback.

## See also

- [`correlations.md`](correlations.md) — the main consumer
- `../../../notes/claude-memory/transformed-efd-consdb.md`, `camera-temperature-sources.md`
