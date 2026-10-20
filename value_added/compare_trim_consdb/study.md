---
study: compare_trim_consdb
topic: value_added
status: active
question: Where and why does the ConsDB transformed-EFD Trim (mt_logevent_aggregated_dof) differ from the value-added database Trim, now that ConsDB is backfilled for all nights?
answer: 'Every difference is explained by the two selection rules. Where both have a value they agree on 74,000 of 121,514 exposures; the other 47,514 each had an MTAOS degreeOfFreedom event during the exposure, which ConsDB reports and the value-added database (as of obs_start) does not. That is 85.4% of science exposures with both before 2026-04 and 21.8% from 2026-04. ConsDB still lacks a row on 38,845 exposures that have an event in its window (backfill incomplete, mostly 2025).'
uses:
  value_added: v1@snapshots/20261016
started: 2026-10-19
---

# `compare_trim_consdb`

> **Status:** active · **Last updated:** 2026-10-19 · **Kind:** study

## Question

Where and why does the ConsDB transformed-EFD Trim (mt_logevent_aggregated_dof) differ from the value-added database Trim, now that ConsDB is backfilled for all nights?

## Current answer

The two databases apply different rules to the same EFD events,
`lsst.sal.MTAOS.logevent_degreeOfFreedom` `aggregatedDoF0..49`:

| source | rule |
|---|---|
| value-added DB `visit_telemetry.dof0..49` | last event at or before `obs_start` |
| ConsDB `efd_lsstcam.exposure_efd_unpivoted`, property `mt_logevent_aggregated_dof` | `most_recent_value` over `[obs_start - 1 h, obs_end]` (`start_offset: -1`, `config_lsstcam.yaml`) |

Rebuilding both rules from the raw events reproduces **213,680 of 213,680** DB rows and
**121,515 of 121,515** ConsDB rows exactly (counts; agreement to rtol 1e-6). There are no
unexplained differences.

![status by img_type](figures/status_by_img_type.png)

Outcomes over 213,704 exposures, `day_obs` 20250415 to 20260714 (counts):

| outcome | cause | count |
|---|---|---|
| agree | no event during the exposure | 74,000 |
| differ | an event during the exposure; ConsDB takes it | 47,514 |
| DB only | no event in `[start - 1 h, end]`, so ConsDB has none by design | 53,321 |
| DB only | an event in the window but no ConsDB row: backfill gap | 38,845 |
| ConsDB only | DB row has no Trim | 1 |
| neither | — | 23 |

- **Differences are a science-exposure effect.** 47,322 of the 47,514 `differ` exposures
  are `science`; `acq` has 178, the others 14 together. The AOS publishes a new Trim during
  the exposure. The event falls either in the first 10 s or near the end of a 30 s exposure
  (report page 6). Only 4 exposures had more than one event.
- **It changed in 2026-04.** Of science exposures with a value in both, 85.4% differ before
  2026-04 (37,990 of 44,465) and 21.8% from 2026-04 (9,332 of 42,800); the AOS event moved
  to before most exposures.

  ![science by month](figures/status_by_month_science.png)

- **The difference is a full AOS step**, not noise: the median |ConsDB − DB| on camera
  hexapod dz (`dof5`) is about 10 µm (log10 |Δ| [log10 µm] peaks near 1).
- **The 1 h window drops calibrations by design.** Most `flat`, `dark`, `bias` and `cbp`
  exposures have no AOS event within 1 h, so ConsDB is NULL there; the DB carries the last
  Trim, however old.
- **ConsDB is still incomplete.** 67 of 366 nights have no ConsDB Trim at all, and 91 more
  are partial, mostly 2025-04 to 2025-09. These 38,845 exposures cover every img_type,
  including 2,961 `cwfs` and 18,404 `acq`. The science-only gate from the 2026-09
  20260513 notebook is gone: `cwfs` and `acq` now have rows.

Which value to use depends on the question. The DB value is the Trim in effect when the
shutter opened. The ConsDB value is the last Trim commanded before readout ended, which is
the Trim for the next exposure if the event came mid-exposure.

The time history on report page 9 is science exposure 2026042100703 (30 s): one event 28 s into the
exposure moves camera hexapod dx by about −65 µm and dy by about −67 µm. The DB keeps the
pre-exposure value and ConsDB the new one.

**Time-frame note.** The EFD influx index is already UTC. It equals `private_sndStamp`
(TAI → UTC) and `private_efdStamp`, checked on 2025-04-19. `select_time_series(...,
convert_influx_index=True)` shifts it 37 s early. With that shift neither database could
be reproduced (143,767 and 61,630 rows matched); without it both match exactly.
`rubinwork.common.dof_telemetry._asof_rows_by_night` (the hexapod LUT) and
`ess_telemetry` use `convert_influx_index=True`, which needs checking.

## Log

### 2026-10-19 — first full comparison

`code/fetch_inputs.py` (run `20261019_inputs`) reads the DB Trim from
`value_added=v1@snapshots/20261016`, the ConsDB exposure table and unpivoted Trim month by
month, and all 100,289 `degreeOfFreedom` events. `code/compare.py` (run `20261019_compare`)
classifies each exposure and explains it from the rebuilt rules. It writes
`comparison.parquet` and the CSV tables. Since the rerun below, it writes every figure
into one 10-page report, `20261019_report/compare_trim_consdb.pdf`:

| page | content |
|---|---|
| 1 | rules, reproduction counts, status and cause tables by img_type |
| 2 | status fraction by img_type |
| 3 | mismatch causes by img_type |
| 4 | science status fraction by month |
| 5 | DB vs ConsDB scatter and log10 abs(ConsDB − DB) for dof5, dof0, dof8, dof10 |
| 6 | when the ConsDB-chosen event fell within the exposure |
| 7–10 | EFD time histories: two science mismatches before 2026-04, two after | Both ran on sdfiana from a copy of `code/`
at commit `c00870c`.

The first pass used `convert_influx_index=True` and left 47k "unexplained" mismatches. A
−37 s shift made both rules match 100%, which traced to the index conversion (see the
time-frame note).

Rerun with the label `report`: one PDF in place of the scattered PNGs, per the
one-PDF-per-study rule. `20261019_compare` holds the earlier PNG output.

Next: confirm the 37 s index issue on the LUT and ESS readers; rerun once ConsDB reports
the backfill is done.
