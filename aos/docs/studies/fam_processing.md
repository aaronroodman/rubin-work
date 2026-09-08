# Study: `fam_processing` — auditing the FAM chunk build

> **Status:** current · **Last updated:** 2026-09-08 · **Kind:** reference (study)

Tools for checking the Full Array Mode (FAM) chunk tables: what a chunk contains before it
is built, whether the Butler provenance is consistent across chunks, how the visits cover
elevation and rotator angle, and whether the expected telemetry columns are actually
populated.

The build itself is **not** here — `mktable`, `fit` and the combines run from the external
`ts_intrinsic_wavefront` package, as described in the
[Processing](../../README.md#processing) section of the topic README. This study is the
review layer over that output.

## Code

| file | role | pipeline rule |
|---|---|---|
| `run_attach_telemetry.py` | attach **all** per-visit telemetry in one pass: thermal, gradients, wind, camera, mirror LUT, Trim, derived Tweak | no |
| `run_chunk_status.py` | all-chunks status roll-up in one PDF: counts, coverage, telemetry completeness, DOF presence | no |
| `check_chunk.py` | **pre-flight**, before `mktable`: visits in ConsDB but missing from the Butler collection, and heterogeneous `nollIndices` across a chunk | no |
| `inspect_visit_provenance.py` | Butler provenance consistency across a `param_set`'s chunks | no |
| `plot_visits_summary.py` | elevation × rotator-angle coverage, one panel per filter | no |
| `compare_to_archive.py` | new combined tables against the archived pre-reorganization ones | no |

None of these is a Snakefile rule; they are run by hand when a chunk is added or a build is
questioned.

### `run_attach_telemetry.py` — one pass, sidecars as the source of truth

It replaces the split between `run_backfill_thermal.py` (rewrote each per-chunk
`visits.parquet` in place) and `run_backfill_camera_telemetry.py` (wrote per-chunk sidecars
but merged only into the *combined* table). That split is why re-running `combine_visits`
silently dropped the 28 `cam_*` columns.

Now every quantity lands in one per-chunk `telemetry.parquet`, and `--merge` joins it into
both the per-chunk and combined `visits.parquet`. A re-combine is always repairable with
`--merge --skip-fetch`, and a failed fetch never damages an expensive `mktable` output.

Sources are ConsDB-first, EFD only where ConsDB cannot answer — see
[`../telemetry.md`](../telemetry.md) for the measurements behind each choice. Verified on
two chunks: Trim anchors **54/54** and **24/24** visits via ConsDB `obs_start` with no MJD
fallback; wind returns 8 columns at 87.5% finite; the mirror LUT returns 228 axial-force
columns (156 M1M3 + 72 M2); and `Trim_i = Trim_0 + cumsum(Tweak)` reconstructs to 1e-6 in
DOF units.

Tweak is **0.0** where the AOS applied no new correction — a real measurement — and NaN
only where genuinely unknown (the first visit of a chunk, or an unresolved Trim or event
id). On `20260514_20260731`: 53 of 54 visits known, 8 with a non-zero correction, 45 with
none applied.

### `check_chunk.py` catches two failure modes early

Both are silent otherwise:

1. Visits listed in ConsDB but absent from the Butler collection, which would raise
   `DatasetNotFoundError` during `mktable`.
2. Heterogeneous `nollIndices` within a chunk. `mktable` locks the first visit's
   `nollIndices` and **skips** visits with a different set, so a mixed chunk quietly loses
   data rather than failing.

### `run_chunk_status.py` — the all-chunks view

Per-chunk PDFs (`chunks/<chunk>/*_visit_quality.pdf`) already exist and are kept. This
script sits above them, reading only the parquet tables so it needs no Butler, EFD or
ConsDB access:

1. **Chunk inventory** — visits, donuts and fits per chunk, date span, and whether the
   three per-chunk tables and the combined tables agree on row counts.
2. **Coverage** — `day_obs` timeline, band and science-program mix, elevation × rotator
   histogram over all chunks.
3. **Telemetry completeness** — the fraction of finite values per telemetry column per
   chunk, which is how a chunk built with `--no-thermal` is spotted.
4. **DOF presence** — whether Trim, Tweak and LUT degree-of-freedom (DOF) columns exist,
   and which fetcher supplies each. `mktable` writes none of them; they arrive via
   `run_attach_telemetry.py`. See
   [`../status/dof_telemetry_availability.md`](../status/dof_telemetry_availability.md).
5. **`nollIndices` consistency** — the pupil-Zernike set per chunk, flagging any variation.

## Output

`output/<param_set>/fam_processing/chunk_status.pdf`, plus a machine-readable
`chunk_status.parquet` with one row per chunk. Under `<param_set>/` rather than
`<mi_name>/`, because none of it depends on which Measured Intrinsic Wavefront build was
used — this is about the tables that precede any MIW.

## Running

```bash
cd ~/notebooks/rubin-work/aos
python code/fam_processing/run_attach_telemetry.py --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x --all-chunks --merge
python code/fam_processing/run_chunk_status.py --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x
python code/fam_processing/plot_visits_summary.py --param-set all
python code/fam_processing/inspect_visit_provenance.py --help
python code/fam_processing/check_chunk.py --help          # needs ConsDB + Butler
```

`run_chunk_status.py`, `plot_visits_summary.py` and `compare_to_archive.py` are
parquet-only. `check_chunk.py` and `inspect_visit_provenance.py` need the Butler and
ConsDB, and `run_attach_telemetry.py` needs both the EFD and ConsDB to resolve — so those
three are RSP or slaciana/slacrd only, **not** a batch compute node.

## State and open questions

- The mirror LUT is stored as axial **forces**; converting to bending amplitudes assumes
  the EFD force arrays share the actuator order of the ts_ofc influence matrix, which
  `aos_trim.fetch_mirror_lut_for_visits` flags as unverified.
- `run_backfill_thermal.py` and `run_backfill_camera_telemetry.py` are superseded by
  `run_attach_telemetry.py` but still in place; retiring them is outstanding.
- `compare_to_archive.py` compares against the pre-reorganization archive and will lose its
  purpose once that archive is dropped.

## See also

- [Processing](../../README.md#processing) — the build chain this study audits
- [`../status/dof_telemetry_availability.md`](../status/dof_telemetry_availability.md) — Trim/Tweak/LUT availability
- [`../miw_pipeline.md`](../miw_pipeline.md) — every rule, config file and output path
- [`dzfit.md`](dzfit.md) — validation of the DZ *fit*, as opposed to the chunk build
