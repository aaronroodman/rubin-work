# Handoff: intra-night prediction quality, and twilight time in the value-added DB

> **Status:** paused · **Last updated:** 2026-10-08 · **Kind:** status (handoff)

Covers the session of 2026-10-08. Everything described here is committed and pushed; the
working tree was clean at `9914a06`. For the study itself read
[`../thermal_focus.md`](../thermal_focus.md) — this file records only decisions and state.

## Done and committed

All on `main`, pushed to `origin/main`.

| SHA | what |
|---|---|
| `49b3b99` | `twilight` group on `visit_telemetry`: `min_after_twilight`, `min_after_twilight_18deg` [min], `sun_alt_deg` [deg], plus `common.utils` helpers |
| `36fcc38` | six new report pages; prediction now evaluated at the first science visit; report 20 -> 26 pages |
| `41ce749` | calculator coefficients to `trim_coefficients.yaml`, tests to `test_trim_calculator.py` (predated this session) |
| `b64da2d` | `aos/` 22/12 MIW arm, `n_iter` 15 -> 3 (predated this session) |
| `a90497b` | EFD/ConsDB notebook (predated this session) |
| `aa921be` | recovered 9 stranded cells into `aos/notebooks/cwfs/aos_miw_cwfs_intrinsic_check.ipynb` |
| `9914a06` | downgraded the `vt_day_obs` index issue to low priority |

Three findings, all written up in `../thermal_focus.md`:

- **Intra-night: the prediction is worse early, as a step not a drift.** Residual nMAD
  **84.9 µm of equivalent hexapod dz before 3 h after sunset against 54.9 µm after** (ratio
  1.55, dimensionless), with a **+23.3 µm** median bias early against −2.4 µm late. Flat from
  3 h to dawn. Reads as thermal lag.
- **Start-of-night comparison, delay removed.** Both sides now evaluated at the first science
  visit after `BLOCK-T539`: residual nMAD **103.4 µm** over **145 nights**, Huber slope
  **+0.814 ± 0.016** (dimensionless, predicted per actual), against 153.8 µm and +0.578 ± 0.025
  for run-first to run-last.
- **Long alignment blocks are not the alignment struggling.** 42 of 105 post-2026-02-01 nights
  fall outside 6–9 min, but minutes per exposure settles it: nominal is ~0.9 min/exposure while
  20260602 takes 274.0 min over 24 exposures and 20260515 takes 87.3 min over 6. The delay is
  *between* exposures — an operations artifact, so block length is not a proxy for alignment
  difficulty.

Verified at the end: 26 PDF pages, 23 tests pass, no new flake8 warnings, working tree clean,
`main` level with `origin/main`, no stashes.

## In progress

Nothing. No modified or untracked files.

## Next concrete action

Nothing is blocked. Two optional items, in priority order:

1. **Try a truss-temperature rate as a sixth feature** — `d(truss_temp_mean_c)/dt` over a
   trailing window. This is the physical candidate for the thermal lag above, and it is the
   most promising remaining improvement to the model. Cheaper than it sounds: the rate can be
   differenced from `visit_telemetry` with no new EFD fetch. A time-since-sunset term is the
   alternative and is now trivial (`min_after_twilight` is a stored column) but is the less
   physical fix — it would absorb the bias without explaining it, and would not transfer to a
   night with an unusual thermal history. **Adding either changes `DELIVERABLE_GROUPS` and the
   online calculator's channel count, so it is Aaron's call, not a fit-quality question.**
2. **Rebuild the `vt_day_obs` index** when the database is free. Command in
   [`../../../value_added/docs/status/build_progress.md`](../../../value_added/docs/status/build_progress.md).
   Low priority — see below.

## Tried and rejected, and why

**The 5-minute gap cut on the start-of-night comparison — removed, despite giving a better
number.** The earlier version predicted at the block's first visit and kept only nights whose
block-to-science gap was under `T539_SCI1_MAX_GAP_MIN = 5.0` min. That reached a residual nMAD
of 91.1 µm on 33 nights, better than the 103.4 µm now reported on 145. It was still the wrong
test: the cut selects thermally still nights (truss drift nMAD 0.131 °C against 0.205 °C
uncut), which is the very condition being tested, so 91.1 µm was a best case rather than an
expectation. Predicting at the science visit itself gets the delay to exactly zero on 4.4× the
nights. **Do not reintroduce the cut because the number looks better.** The constant is retired;
`gap_to_sci1_min` survives as a diagnostic only.

**A linear fit to the intra-night trend — reported as a step instead.** A Huber line through
the whole night gives −0.98 ± 0.47 µm of equivalent hexapod dz per h on the binned median, only
2.1 standard errors, and would read as marginal. The effect is a transient confined to the
first 3 h with a long flat tail that drags the line down. The split at
`EARLY_NIGHT_SPLIT_H = 3.0` h is unambiguous where the slope is not. Both are computed; the
split is what the page and the doc lead with.

**Deleting `day_obs` 20260620 from the value-added table — rejected.** The night is all
calibration (1,766 cbp, 10 dark, 3 bias, 3 flat, no science/acq/cwfs), which prompted the
question of whether it belongs at all. It does: `visit_spine` applies no `img_type` filter by
design, **54 of 366 nights are calibration-only**, calibration is ~32% of the table, and the
telemetry is real — this night's truss sits at 6.68–8.65 °C, which the report's database-wide
truss page plots. Removing it would put a hole in that page and the principle would remove 54
nights, not one.

**The `vt_day_obs` index issue is latent, not active.** Initially written up as more serious
than it is. `_day_obs_clause` in `efd_db.py` emits range predicates (`day_obs >= ? AND
day_obs <= ?`), so `efd_db.visits()` and everything downstream already return 20260620
correctly — which is why `thermal_focus_truss_all.parquet` carries all 1,782 rows. Only
hand-written `WHERE day_obs = 20260620` silently returns 0. And the twilight columns it is
missing are meaningless for a CBP night, so **treat twilight coverage as complete at 365 of
366 nights.**

**A twilight window opening at 18:00 local — a real bug, fixed.** `evening_twilight_mjd`
bisects for the evening solar-altitude crossing. An 18:00 local window start sits *below*
geometric sunset in June and July at Cerro Pachon (sunset ~17:46) and silently returned the
window edge rather than failing. The window is 15:00–03:00 local now, and the margin is
documented as load-bearing. Verified against all three altitudes (0, −12, −18 deg) across the
season to ~1e-9 deg.

**Three `sync.sh` auto-stashes — inspected, then dropped.** 17 notebooks across
`aos/ guider/ blocks/ wfs/ nightlyiq/ psf/ astrometry/`. Compared **code cells and markdown
cells only**, ignoring outputs and execution counts: 14 were output-only or had HEAD strictly
newer (mostly the path-resolution fix from the `<topic>/notebooks/<study>/` reorganization, so
the old flat paths read as "absent from HEAD" until matched by basename). One had real content
— `aos_miw_cwfs_intrinsic_check.ipynb`, 439 code lines present nowhere else — recovered in
`aa921be`. Two notebooks in the stashes were deliberately deleted in `d24aa22` as "superseded
by pipeline steps"; their absence is intentional. SHAs, reachable until git gc:
`3918b100b081e40da277b7c26991b9a3aa2fc0d3`, `ab8cd0ca51cb1f43c735af63c36e0d0b61dda961`,
`84f1a8bdb32aded07f1b7134c274e1ac0aba30dd`.

## Non-obvious constraints discovered

- **`visit_telemetry` does not store `img_type`.** It arrives on the ConsDB join. A question
  about exposure types has to go to `cdb_lsstcam.exposure`, not the value-added table.
- **The spine's `mjd` is TAI; the solar ephemeris is UTC.** The twilight columns therefore carry
  the 37 s offset. That is 0.6% of a minute and they are reported in minutes, so it is left
  uncorrected — but do not read sub-minute meaning into them.
- **`make_efd_client()` was unconditional in `build_efd_db.py`.** Now created only when a
  requested group needs it, so a derive-only backfill (`--groups twilight`) runs where the EFD
  does not resolve, including batch compute nodes.
- **Two Claude sessions appear to share this checkout.** The push needed rebasing twice onto
  `optics/` commits (Fresnel donut, Debye-Huygens PSF) that arrived mid-session from elsewhere.
  No file overlap either time, verified before rebasing. This is the likely origin of the three
  auto-stashes, and a reason to check `git status` before assuming the tree is yours.
- **The findings here came from one dropped assumption.** The thermal model is band-independent
  and carries only instantaneous temperatures; the intra-night result is what shows that second
  property, not the first, is where it costs something.
