# rubin-work — open work

> **Status:** current · **Last updated:** 2026-09-20 · **Kind:** working state (todo index)

One place to see what is open across the whole repository. This file is an **index**, not
a specification: each row points at the document that states the work and holds the
reasoning. Detail belongs there, never duplicated here.

Sources indexed: [`reorg_review_plan_2026-09.md`](reorg_review_plan_2026-09.md) (Parts
A/B/C, labelled `A1`…`C4`), [`step1_structure_decisions.md`](step1_structure_decisions.md)
(the numbered step-1 queue), [`memory_cleanup_plan.md`](memory_cleanup_plan.md), and the
per-topic `<topic>/docs/status/` docs.

## Blocked on a decision from Aaron

Nothing below can start until the question is answered. These are the bottleneck.

| # | decision | scope | where it's specified |
|---|---|---|---|
| 1 | Item 7: the joined `<param_set>_<mi_name>` is **59 chars**, not the ~30 D2 assumed — flatten anyway, shorten, or keep nested? | output tree + 54 `mi_name` call sites + 7 study docs | [step1][s1] item 7, D9 |
| 2 | A1: delete the 27 GB `aos/output/archive/`, delete only the ~12 GB of archived `donuts.parquet`, or keep it? (a deletion — must ask) | 27 GB | [reorg][rp] §A1 |
| 3 | A3: `FocalPlaneInterpolator.py` — delete, or keep and demonstrate? | 1 module | [reorg][rp] §A3 |
| 4 | A4: approve or amend the proposed study splits for `smatrix/`, `guider/`, `optatmo/`, `filters/` | 4 topics | [reorg][rp] §A4 |
| 5 | A5: how should `miw` / `coadd` / `static_optics` divide in `aos/`? | `aos/` | [reorg][rp] §A5 |
| 6 | A7: retire `camera/`, `des/`, `starcolor/`, `survey/`, `wcs/`, `alerts/`, `scratch/`, or give each a README? | 7 empty topics | [reorg][rp] §A7 |

## Ready to start

Disjoint from the paths another session was modifying, so safe to pick up now.

| # | item | topic | where it's specified |
|---|---|---|---|
| 7 | B1: rewrite the root `README.md` — highest priority in Part B | repo | [reorg][rp] §B1 |
| 8 | B2: add the required status header to 12 topic `README.md`s | repo | [reorg][rp] §B2 |
| 9 | B4: document the undocumented cross-topic couplings | `CLAUDE.md` | [reorg][rp] §B4 |
| 10 | A2 for `wfs/`, `psf/`, `nightlyiq/`, `astrometry/` — give each a `notebooks/` dir | 14 notebooks | [reorg][rp] §A2 |
| 11 | B3: refresh the two stale topic READMEs (`blocks/`, `smatrix/`) | 2 topics | [reorg][rp] §B3 |
| 12 | C2 item 3: review `run_wfs_dof_compare.py` | `wfs` | [reorg][rp] §C2 |
| 13 | C4: add characterization tests in a new `aos/tests/` | `aos` | [reorg][rp] §C4 |
| 14 | A1: the output-provenance helper in `common/utils.py` | `common` | [reorg][rp] §A1 |
| 15 | Add the missing row-count guard in `analyze_miw_field_order.py` | `aos` | [aos backlog][ab] |

## Deferred until the concurrent session's work lands

Touches paths that session had modified; the standing rule in [reorg][rp] §Sequencing is
that this plan does not touch them until that work is committed.

| # | item | blocked by |
|---|---|---|
| 16 | A3 — moves `common/efd_db.py` | `common/efd_db.py` modified |
| 17 | A4 and A2 for `guider/` | `guider/` work in flight |
| 18 | C2 items 1–2 | `science_lut/` adjacent to `build_optical_state.py` |

## Also open

| # | item | where |
|---|---|---|
| 19 | Outputs that predate a code change and need regenerating | [aos rerun_needed][rn] |
| 19a | `blocks/` rerun: rules `build_table`, `night_table`, `plots`, stale since `9387106` promoted the Environmental Sensor System (ESS) telemetry and again since the degree-of-freedom (DOF) telemetry moved to `common/` | `cd blocks && ./run_snake.sh` |
| 20 | B5: establish a lightweight per-study logbook | [reorg][rp] §B5 |
| 21 | B6: keep the memory snapshot fresh; fix the small doc defects | [reorg][rp] §B6 |

## Done

Newest first. Kept short — the reasoning stays in the source document.

| item | closed | ref |
|---|---|---|
| Step-1 queue items 1–6, 8–10 (9 of 10; only item 7 remains) | 2026-09-19 | [step1][s1] |
| Item 9 — retire the 3 superseded param_sets, short-name convention | 2026-09-19 | `9958662` |
| Item 6 — resolve the output collision (D8), 5 files | 2026-09-19 | `0c6c0c8` |
| Item 10 — `guider`/`optatmo` output to group space; `/sdf/home` 87% → 66% | 2026-09-19 | `92aef81` |
| Item 5 — `smatrix_vmode` → `smatrix/code/vmode/` | 2026-09-19 | `8e5f76c` |
| Items 3, 4 — `value_added/` topic created, 7 files moved | 2026-09-19 | `8f7ab33` |
| Items 1, 2, 8 — structure rule into `CLAUDE.md`, D4 path headers, prune empty dirs | 2026-09-18 | `3bc0134` |
| `aos/` study reorganization (8 of 8) | 2026-09-08 | [memory_cleanup][mc] |

## Keeping this current

- An item is **one line** here plus a link. If it needs a paragraph, it needs a doc.
- Closing an item: move its row to **Done** with a date and the commit, and update the
  source doc in the same commit — this index is never the only record.
- Numbers are stable handles for conversation ("start item 8"); don't renumber on close.
- Update the `Last updated:` date in the header when you touch this file.

[rp]: reorg_review_plan_2026-09.md
[s1]: step1_structure_decisions.md
[mc]: memory_cleanup_plan.md
[ab]: ../../aos/docs/status/code_review_backlog.md
[rn]: ../../aos/docs/status/rerun_needed.md
