# Phase 5 handoff: converting the studies

> **Status:** in progress · **Last updated:** 2026-10-18 · **Kind:** working state
> (handoff)

Phase 5 of `notes/status/organization_plan.md` (section 8): each study becomes
`<topic>/<study>/` with `study.md`. The per-study statuses and Aaron's answers are in
`notes/status/phase5_study_triage.md` ("your call" column and "Decisions taken"). The
layout is plan section 3; `study.md` is section 7; the dependency rules are section 2.

## Decisions taken

Aaron, 2026-10-18 (full text in the triage, "Decisions taken"):

1. The triage "your call" column applies per row.
2. No topic is retired; `guider_iq` and `zernike_psf_moments` stay where they are.
3. smatrix mode datasets stay smatrix-internal data.
4. One parked MIW-origin study under `aos` (aos `coadd`, `static_optics`, the MIW
   handoff, smatrix `miw_astig_coma` with its 3 sub-tests).
5. `trim_calculator` is a study, not a product.
6. One intra/extra study, `aos/intra_extra` (`wfs/giant_donuts`,
   `optics/intra_extra_phase_strip`, aos `blur_intra_extra`, the Z11/Z14 part of aos
   `cwfs`); `aos/unpaired_intra_vs_extra` links to it.
7. The RBR studies stay separate.

Delete no file without asking, of any kind. Leave Aaron's uncommitted notebooks alone:
`aos/notebooks/fam_processing/blur_intra_extra_offset.ipynb` and
`wfs/notebooks/giant_donuts/giant_donut_radial_profiles.ipynb`.

## Decisions for Aaron

Provisional choices, taken so the work continues; each is easy to reverse.

- (none yet)

## Step list

1. **Convert the studies, topic by topic** (this file). Per study:
   `python -m rubinwork.studies new <topic> <study>` makes the directory, `output@` and
   the data directory; `git mv` its notebooks, code and study doc in; write `study.md`
   by the `/wrap` backfill recipe (Question, Current answer with units, one Log entry
   `backfilled from docs and history` dated the last real work). Fix the topic's doc
   defects from the triage list. Rerun `/review`, commit per topic, push.
2. **Section 2 violations**: fix the wrong-way imports and reads, emptying `ALLOWED`.
3. **Topic READMEs**: merge the hand-written ones into generated indexes.
4. **Shims**: update the four callers of the `smatrix` shims (triage, last paragraph),
   then remove every shim.

Steps 2 to 4 are not started; step 1 does not touch them.

## Topic order

Small topics first, then `smatrix`, `thermal_focus` and `olr`, then `aos` last (the
largest; 169 tracked files). The small topics, smallest first: `astrometry`, `psf`,
`nightlyiq`, `filters`, `optics`, `wfs`, `blocks`, `optatmo`, `guider`.

Merges into `aos` wait for the `aos` step: `wfs/giant_donuts` and
`optics/intra_extra_phase_strip` (into `aos/intra_extra`), and smatrix `miw_astig_coma`
(into the MIW-origin study).

Topics with no study in the triage stay as they are, with no `study.md` and no moves:
`camera` (already holds `itl_dip`), `des`, `starcolor`, `survey`, `wcs`, `alerts`,
`scratch`. `value_added` is a service topic for a product and is not converted here.

## Done

| topic | commit | studies |
|---|---|---|

## Tried and rejected, and non-obvious findings

- `wrap-gather` on a directory with no `study.md` finds commits only under that
  directory, so before the move it finds none. Read `git log --follow` on the old paths
  instead.

## Next

Start with `astrometry`.
