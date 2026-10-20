---
name: start-study
description: Starting a new study in rubin-work, or deciding where new analysis code goes (a study, a library or a product change). Creates <topic>/<study>/ with study.md, code/, notebooks/, figures/ and the output symlink. Use before writing any new analysis code.
---

# /start-study — create a study

A **study** answers one question, usually by plotting product data. It lives in
`<topic>/<study>/` and ends with an answer and a status. A **library** (`rubinwork/`) is
imported and writes no data. A **product** (`rubinwork/products/<name>/`) writes data
other work reads, through the catalog. Rules: `notes/status/organization_plan.md`
section 2.

## 1. Decide what the request is

From Aaron's short description:

- **A product change** — new collection, programs, rotator bins, correction scheme,
  or other configuration of something that already writes data. Not a study. Run
  `python -m rubinwork.studies variants`, name the product's `variants.yaml` and the
  closest existing variant, and stop. A new variant is a new entry there; a new build is
  a rerun of the product.
- **Code another study or product will import** — a library, in `rubinwork/`. Not a
  study either. Say so and stop.
- **A question** — a study. Go on.
- **More work on an existing study** — check `STUDIES.md` first. If one fits, work in
  it and use `/wrap` when done; do not make a second study.

## 2. Propose, then wait

Propose in one short message and wait for Aaron's OK:

- **topic** — an existing top-level topic (`aos`, `camera`, `guider`, ...);
- **name** — lowercase snake_case, short, unique across all topics, naming the content
  (`coadd_vs_miw`, `itl_dip`), not a question;
- **question** — one sentence;
- **products used** — `product=variant@build`, from `PRODUCTS.md`. Pin the current
  build unless Aaron says otherwise; `@build` omitted means `current`.

## 3. Create

With the OK:

```bash
python -m rubinwork.studies new aos coadd_vs_miw \
    --question "Why do per-block FAM coadds disagree with the pooled MIW?" \
    --uses fam_tables=danish_1_2@20261014 \
    --uses miw=d12-A_50_34_i_5rot@20261014
```

It refuses a name already used in any topic, an unknown topic (`--new-topic` overrides,
only if Aaron asked for a new topic) and a `uses` entry the catalog cannot resolve. It
makes:

- `<topic>/<study>/study.md` — front matter, Question, Current answer, Log;
- `code/`, `notebooks/`, `figures/`;
- `output` → `<data root>/studies/<topic>/<study>/`, gitignored, and that directory;
- an updated `STUDIES.md`.

Then commit `study.md` and `STUDIES.md` (empty directories do not commit; they appear in
git with their first file).

## Inside a study

- Scripts in `code/`, run as `python <topic>/<study>/code/x.py`; notebooks in
  `notebooks/` (`rubin-notebooks` skill). Import libraries and products by package
  name (`from rubinwork.products import catalog`); no `sys.path` work.
- Read product data only through `rubinwork.products.catalog` (or a product's own
  `load`), never by building a path, and never another study's output.
- Each run writes into a dated directory, made by
  `run = rubinwork.common.study_run("<topic>/<study>", "<label>")`; save to
  `run.path`. Its `run.json` records the git commit, the command and the product builds
  read, which `/wrap` copies into `uses:`.
- **One multi-page PDF per study.** Every figure a run makes goes into a single PDF in
  its run directory (`matplotlib.backends.backend_pdf.PdfPages`), with a summary page
  first — not a directory of PNGs. A study's default output is one notebook (or script),
  one PDF and one `study.md`.
- `figures/` holds at most two PNGs, copies of PDF pages the Log cites; everything else
  stays under `output/`.
- Longer derivations go in their own `.md` beside `study.md`.
