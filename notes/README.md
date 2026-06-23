# Notes → tech notes

Working notes for Slack posts and Summit-Operations tech-note material. Each note is a
self-contained, dated directory. **Drafting is plain Markdown** (great for Slack);
**the tech note is `lsstdoc` LaTeX** (full power — equations, cross-refs, citations).
Markdown → LaTeX is a clean one-step `pandoc` conversion, *as long as the Markdown stays
plain* (see Conventions).

## Index

| Note | Date | Status | Slack | Target tech note |
|------|------|--------|-------|------------------|
| [aos-measured-intrinsics](aos-measured-intrinsics/) | 2026-06 | draft | — | [SOTN-006](https://github.com/lsst-so/sotn-006) (Measured Intrinsic Wavefront) |

*Status:* draft → posted (Slack) → in-technote. Fill the Slack column with the message
permalink once posted.

## Layout per note

```
<YYYYMMDD>-<topic-slug>/
  note.md             # the draft (plain Markdown — Slack post + pandoc source)
  figures/            # committed PNG (Slack/Markdown) AND PDF (vector, for LaTeX)
  make_figures.py     # regenerates every figure (PNG+PDF) from pipeline outputs
  provenance.md       # param_set / mi_name, git SHA, day_obs range, stack, caveats
  convert_to_latex.sh # note.md -> <name>_body.tex (pandoc + figure/Unicode fixups)
  two_epochs.tex      # lsstdoc wrapper (title/author/abstract + \input body)
  two_epochs_body.tex # pandoc-generated body (regenerable; don't hand-edit heavily)
```

`notes/` lives at the **repo top level** on purpose: it's outside the `*/output/*`
gitignore rule, so figures are versioned (unlike everything under `<topic>/output/`).

## Conventions

- **Draft in plain Markdown.** Do NOT use MyST directives (` ```{figure} `, `{cite}`,
  `{numref}`): `pandoc` turns those into literal junk. Plain Markdown (headings, tables,
  `![cap](figures/x.png)`, `$math$`) converts to clean LaTeX. (If you ever target Sphinx
  instead, *then* MyST is fine — but we're going LaTeX.)
- **Figures: PNG + PDF.** `make_figures.py` writes both. Markdown/Slack use the PNG; the
  LaTeX `\includegraphics{figures/x}` (no extension) picks the vector PDF.
- Commit figures here (small, and they *are* the deliverable). Never stage them under any
  `<topic>/output/` — that path is gitignored.
- Record provenance while it's fresh.

## Converting a note to `lsstdoc` LaTeX

1. Regenerate figures as PDF: `python make_figures.py`.
2. Body: `./convert_to_latex.sh` → `<name>_body.tex` (runs pandoc, drops the H1 — it
   becomes `\title` — strips `.png` so pdflatex uses the vector PDFs, drops pandoc's
   caption-less-longtable counter, and maps the Unicode glyphs to LaTeX).
3. Wrapper: an `lsstdoc` skeleton (`\documentclass[…]{lsstdoc}`, `\title`/`\author`/
   `\setDocRef`/`\setDocAbstract`, `\input{<name>_body.tex}`). The wrapper preamble
   defines the pandoc shims (`\pandocbounded`, `\tightlist`, `\real`, …) so the fragment
   compiles — see `two_epochs.tex`. Body compiles clean (tested with an `article`
   stand-in; `lsstdoc` is a superset).
4. Build with **`lsst-texmf`** ([github.com/lsst/lsst-texmf](https://github.com/lsst/lsst-texmf))
   on `TEXMFHOME`, or from a technote repo created from the `documenteer`/`lsst-texmf`
   template.

**Which series / format:** Summit-Ops/Camera tech notes come in both flavors — LaTeX
(e.g. [ctn-001](https://ctn-001.lsst.io), [ctn-002](https://ctn-002.lsst.io)) and Sphinx
([ctn-003](https://ctn-003.lsst.io)). We're using LaTeX. Established SIT-Com series is
`sitcomtn-NNN` under [github.com/lsst-sitcom](https://github.com/lsst-sitcom); operations
docs collect at [obs-ops.lsst.io](https://obs-ops.lsst.io). **Confirm the exact
operations-era handle** (LSST technote index / docs team) before reserving a number, and
copy the preamble from a known-good note in that series to match house style.
