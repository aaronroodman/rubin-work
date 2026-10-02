---
name: rubin-new-study
description: Adding new code or a new script anywhere in rubin-work, or deciding which study/tier a module belongs in; also moving or promoting modules between study, topic and common/.
---

# Where new code goes in rubin-work

## Studies — where new code goes
A **study** is a separable project inside a topic: the MIW pipeline, the FAM↔CWFS
comparison, the coadd-vs-MIW investigation, the sensitivity-matrix/v-mode work, and so
on. Topics are the top level (`aos/`, `guider/`); studies are the level below.

**Before writing any new code, agree which study it belongs to.** This applies
everywhere in `rubin-work`, not just `aos/`. Ask rather than guess — the answer decides
where the code, its docs, and its outputs land.

If the work does not fit an existing study, it is a **new study**, and it needs all
three of:

1. a short description in the topic's `README.md` (a few lines, linking to 2.);
2. a detail doc at `<topic>/docs/studies/<study>.md`, with the standard status header;
3. usually a `<topic>/code/<study>/` subdirectory, a `<topic>/notebooks/<study>/` for
   its notebooks, and an output directory placed by the rule in
   the `rubin-output-layout` skill — which is **not** a mirror of the code
   layout.

Do not add a script to a topic's `code/` root "for now" — that is how a flat 56-file
directory happens. `aos/docs/studies.md` is the worked example of the inventory.

**Which tier a module belongs to.** Code is placed by who uses it, and each tier has a
test that can be applied to one file:

| tier | location | test |
|---|---|---|
| study | `<topic>/code/<study>/` | answers one question about one data set |
| topic-common | `<topic>/code/` (flat) | encodes something true of the **instrument or a convention**, not of one analysis — or is used by more than one study |
| repo-common | `common/` | true across **topics**: used by two or more |
| service | its own topic | maintains **state or a product** other topics consume |

Promote a module only when it *already* has the second caller — not in anticipation of
one. The flat modules in `aos/code/` are the worked example of the topic-common tier, and
`aos/CLAUDE.md` lists which they are and why three of them cannot move.

The word is **study**, not "thread" — `aos/code/infra/check_threads.py` is about CPU threads,
and the repo already says study (`study_compare_donuts.ipynb`, `run_study_radialbins.py`).
