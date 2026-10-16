---
name: review
description: Reviewing the state of rubin-work's studies and products — regenerate STUDIES.md and PRODUCTS.md and report stale studies, missing answers, superseded builds, code no study covers and dependency-rule violations. Use when Aaron says /review, optionally with a topic.
---

# /review [topic] — regenerate the indexes and report

```bash
python -m rubinwork.studies review          # whole repo
python -m rubinwork.studies review aos      # study checks for one topic
```

It rewrites `STUDIES.md` and `PRODUCTS.md` (a file is touched only if its text changes),
then prints a Markdown report:

- **Schema** — `study.md` front matter that breaks the schema (errors) or needs care
  (warnings, such as a value that is not valid YAML on its own);
- **Active, untouched for more than 14 days** — counted from the later of the last Log
  entry and the last commit touching the study directory;
- **Answered or done with no answer**;
- **Pinned to a superseded build** — a `uses:` build that is no longer `current`;
- **Code not covered by any study** — tracked `.py` and `.ipynb` files per topic outside
  every `<topic>/<study>/`. Large until phase 5 converts the topics;
- **Rule violations** of `notes/status/organization_plan.md` section 2, found by parsing
  the code (docstrings and comments do not count): a library importing a product, a
  study's module imported from outside it, a read of another topic's or study's output,
  and a product path built by hand. **New** ones are listed in full; **known** ones,
  allow-listed in `rubinwork/studies/review.py` (`ALLOWED`) until phase 5 fixes them,
  are summarized.

Report only: nothing fails. Show Aaron the report as printed, or its non-empty sections
if it is long, and then:

- commit `STUDIES.md` and `PRODUCTS.md` if they changed (`/review: regenerate
  indexes`); do not push;
- for a stale study, ask whether to `/wrap` it, park it or leave it;
- for a new rule violation, say which rule and the fix the plan's section 2 implies;
  do not fix it unasked. Add to `ALLOWED` only with Aaron's OK;
- an allow-list entry that matched nothing is a fix already made: say so, and remove
  it from `ALLOWED` with Aaron's OK.
