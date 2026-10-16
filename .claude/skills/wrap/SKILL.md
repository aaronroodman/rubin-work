---
name: wrap
description: Wrapping up a session of work on a rubin-work study — append a dated Log entry to <topic>/<study>/study.md, update the answer, status and products used, copy key figures, commit (never push). Also builds a first study.md for an old study from its history and docs. Use at the end of study work, or when Aaron says /wrap.
---

# /wrap [study] — log a study's progress and commit

Argument: `<topic>/<study>`. Without one, use the study the session worked in; ask if
that is unclear.

## 1. Gather

```bash
python -m rubinwork.studies wrap-gather aos/miw_hull
```

JSON with: the commits touching the study after the commit that last wrote `study.md`;
uncommitted changes there; run directories under `output/` whose `run.json` the file does
not yet name, and the product builds they read (`uses_from_runs`); the newest PNGs not
yet in `figures/`; and `invalid_yaml`, front-matter keys that `wrap-apply` re-quotes.

Read the commits (`git show --stat`, the diffs where needed) and the session's results.
If `anything_new` is false and the session produced no result, write no entry; still run
`wrap-apply` with no `--title` if `invalid_yaml` is not empty.

## 2. Write the entry

A Markdown body, saved to a temp file. In the study's existing register:

- what was done and what it showed, every number with its quantity name and units;
- the commits, as backticked short hashes, and the run directories by name;
- links to `figures/<name>.png` for the figures copied;
- what is next, if anything, as a short list.

Title: a few words, lowercase, saying what happened (`analysis 2: continued versus fitted
rings`). Keep it to the decisions and the results; it is not a transcript.

Update, when the work changed them:

- **answer** — one or two sentences with numbers and units, kept true; and edit the
  `## Current answer` section to match, by hand, before applying;
- **status** — `active | answered | parked | done | dropped`; `answered` means the
  question is answered but work may continue;
- **uses** — add `uses_from_runs`; pass builds read outside a run with `--uses`.

Copy one or two key PNGs, not more.

Show Aaron the entry and changes in a short message before applying, unless he asked for
a straight wrap.

## 3. Apply

```bash
python -m rubinwork.studies wrap-apply aos/miw_hull \
    --title "analysis 3: rim ring against the two builds" \
    --body-file /tmp/wrap_entry.md \
    --status answered \
    --answer "No MIW value inside 1.725 deg depends on a continued ring." \
    --uses miw=d12-A_50_34_i_5rot@20261014 \
    --figure output/20261016_rim/rim_ring.png
```

It inserts the entry at the top of `## Log`, updates the front matter, copies figures,
rewrites `STUDIES.md` and commits `study.md`, `STUDIES.md` and the figures. **It never
pushes.** Commit other changed study files (code, notebooks, the hand-edited Current
answer) first, in their own commit, so the gather is honest; `wrap-apply` includes the
Current answer edit because it is in `study.md`.

## Backfill: a study with no `study.md`

`wrap-gather` on a directory with no `study.md` returns `has_study_md: false`, all
commits touching the directory and `docs`: the tracked `.md` files in the topic and
`notes/status/` that name the study, with `<topic>/docs/studies/<study>.md` first.

1. Read that doc and the handoffs among `docs`; skim the commit subjects.
2. Make the directory with `python -m rubinwork.studies new` if it does not exist, or
   write `study.md` from the `new` template if it does (the CLI refuses a non-empty
   directory).
3. Fill the front matter and sections from the docs: Question; Current answer with its
   numbers and units; one Log entry dated the last real work, titled `backfilled from
   docs and history`, citing the source docs and the key commits. Handoffs become Log
   entries. Leave `uses:` empty unless a doc names a build.
4. Commit. Moving files into the study layout is phase 5 work; do not do it as part of
   a wrap unless Aaron asked.
