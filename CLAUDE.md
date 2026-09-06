# CLAUDE.md — Instructions for Claude Code

## Project Overview
This repository contains Jupyter notebooks and Python scripts for Vera C. Rubin Observatory work, organized by topic. It is used on both the Rubin Science Platform (Summit and USDF) and locally.

## Repository Structure
Each topic directory has `code/` and `output/` subdirectories, and `docs/` where it has
prose documentation; notebooks (`.ipynb`) live directly in the topic directory. Most
topics also carry a `README.md` describing their scope — read it before working in an
unfamiliar topic. See [Markdown docs](#markdown-docs) for where each kind of `.md` lives.

AOS / wavefront:
- `aos/` — the main AOS topic: measured intrinsic wavefront (MIW), FAM coadds, DZ
  fitting, sensitivity/v-mode studies. Largest directory. Key docs:
  `docs/miw_coadd_equations.md` (derivations), `docs/status/miw_investigation_handoff.md` (portable
  state of the MIW investigation), `docs/camera_gravity.md`
- `smatrix/` — batoid_rubin DZ sensitivity-matrix construction and convention
  choices (`docs/conventions.md`, `docs/miw_astig_coma_investigation.md`)
- `wfs/` — corner wavefront sensor (CWFS) studies: sky foreground, ISR-processed
  image inspection, Danish pupil-mask findings, ts_wep dataflow
- `olr/` — Open Loop Reproduction: a Snakemake pipeline reproducing the open-loop
  WFS wavefront for a night of AOS operations
- `optatmo/` — standalone Optics+Atmosphere PSF moment tools; a differentiable
  rebuild of the old PIFF `optatmo3` ideas

Image quality / PSF / instrument:
- `psf/`, `guider/`, `camera/`, `nightlyiq/` (image quality image-by-image for a
  given `day_obs`, combining science-image PSF metrics with guider/AOS/pointing
  diagnostics), `blocks/`

Optical design / prescription (no data needed):
- `optics/` — batoid ray-trace studies: telecentricity, pupil geometry
- `filters/` — interference-filter design study for future narrow/medium-band
  imaging (multilayer dielectric stacks via `tmm_fast`, autograd on layer
  thicknesses); the f/1.23 beam is what limits narrow bands

Survey / astrometry / other:
- `survey/`, `wcs/`, `astrometry/`, `starcolor/`, `des/`, `alerts/`

Support:
- `common/` — shared utility functions used across topics
- `notes/` — working notes for Slack posts and Summit-Operations tech notes; each
  note is a self-contained dated directory, drafted in plain Markdown.
  `notes/claude-memory/` is a snapshot of Claude's own working memory (see its
  README — notes-to-self, not documentation)
- `scratch/` — work-in-progress, not yet organized

Note: `alerts/` and the top-level `output/` currently have no git-tracked content
(local/gitignored output only).

### Topic independence and shared code

The topic directories are **mostly independent lines of work** that happen to share one
git history. A session working in one topic should stay in it: do not refactor, rename,
or "fix" files in a sibling topic as a side effect of a task, and do not assume a
convention found in one topic applies in another.

Launch Claude from the **repo root**, not a subdirectory, so `common/` and git context
stay visible. Per-topic scoping comes from nested `CLAUDE.md` files, which load when
files in that subtree are touched:

- `aos/CLAUDE.md` — MIW, FAM coadds, DZ fitting, sensitivity/v-modes
- `guider/CLAUDE.md` — guider pipeline, `summit_utils` fork state, bias/streak work

Shared code lives in **`common/`** (`utils.py`, `FocalPlaneInterpolator.py`,
`psf_moments_consdb.py`, plus `common/scripts/`), imported by inserting the repo root on
`sys.path`. Genuinely shared helpers belong there rather than being copied between
topics.

Two cross-topic couplings are real and intentional — know them before refactoring:

- `guider/code/` imports `moments_hsm.measure_hsm_moments` from **`optatmo/code`**, so
  that both sides of the guider-vs-science-CCD moment comparison use the same
  galsim-HSM estimator. Changing that estimator changes guider results.
- `guider/code/check_rotator_field.py` imports `aos_trim.make_consdb_client` from
  **`aos/code`**.

Both go through `sys.path.insert`, not real packages, so the coupling is invisible to
static import checks.

Beware one naming collision: in `aos/code/`, `common` in an import almost always means
`lsst.ts.intrinsic.wavefront.common` — an **external** package — not this repo's
`common/`. Read the full import path before acting.

## Working with Aaron

Standing rules that apply to every session in this repo, on every machine. The
detailed reasoning behind each lives in the matching memory file under
`notes/claude-memory/` (named in parentheses) — this section carries the rule.

### Autonomy and hard stops
Default to **acting without asking** on routine coding and analysis work that is
already well-scoped; only ask when genuinely unsure what to do. Three exceptions
are hard MUST-ASK rules, no exceptions (`work-autonomy-and-guardrails`):

1. **Deleting any files** — ask first, always.
2. **Submitting any batch job** — Slurm `sbatch` / `condor_submit` on S3DF, in
   practice `run_snake.sh --mode batch`. **Ask Aaron to start it**, and give him two
   commands: the full submit command *and* the command to monitor its progress. Batch
   must be submitted from an s3df node (`slacrd`), never from an RSP pod (no Slurm
   there) (`s3df-batch-job-rules`). See [Batch jobs](#batch-jobs) for the exact forms.
3. **Connecting to SLAC / USDF from the laptop** — `ssh slacrd`, USDF,
   `/repo/main` Butler. Ask before opening the connection, or hand Aaron a
   runnable snippet to execute himself (his established preference). Local work in
   `rubin-work/` needs no permission (`ask-before-slac`, `usdf-access-slacrd`).

### Batch jobs
Never submit one. Hand Aaron the submit command **plus a monitoring command**, both
copy-paste-ready. The log path depends on the topic:

- **`aos/`** — one job per invocation, log `aos/logs/batch_<timestamp>.out`. The script
  does *not* print the path, so monitor the newest log:
  ```bash
  cd ~/notebooks/rubin-work/aos && ./run_snake.sh --mode batch
  tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/batch_*.out | head -1)"
  ```
- **`guider/`** — **one job per night**, so `--day-obs A,B,C` is three submissions.
  Logs are `guider/logs/batch_<night>_<timestamp>.out`, and the script prints each
  path as it submits:
  ```bash
  cd ~/notebooks/rubin-work/guider && ./run_snake.sh --day-obs 20260706 --mode batch
  tail -f "$(ls -t ~/notebooks/rubin-work/guider/logs/batch_20260706_*.out | head -1)"
  ```

Also useful alongside the tail: `squeue -u roodman` for queue state. For a local
(non-batch) detached run, the log is `logs/run_<timestamp>.log` (`aos/`) or
`logs/run_<tag>_<timestamp>.log` (`guider/`, which prints it).

### Commands handed to Aaron
Give the **full copy-paste-ready command — no `...`, no `<placeholder>`, no
abbreviation**. Spell out the entire seq_num list, the full collection name, every
argument. If a value is genuinely unknown, ask for it rather than leaving an
ellipsis (`full-commands-no-ellipsis`).

Two things to get right in those commands:
- **Repo sync:** say "run your gitpull script", not `git pull` — Aaron uses his own
  `gitpull` wrapper (`gitpull-script`). Claude's own commits/pushes still use git
  directly.
- **Python interpreter:** MacPorts `/opt/local/bin/python3` on the laptop; bare
  `python` on the RSP / USDF, where the stack interpreter is on PATH and the
  MacPorts binary does not exist (`python-interpreter-by-env`).
- **Aaron's interactive shells already set up the LSST stack via `.bashrc`** — do
  not prepend `source .../loadLSST.bash && setup lsst_distrib` to snippets he runs
  himself (`usdf-access-slacrd`).

### Paths across environments
Use the `/sdf/group/rubin/u/roodman/LSST/...` form in any code or config that might
run in batch — it resolves identically in the RSP notebook, the RSP terminal, and on
slaciana/sdfiana batch nodes. The `/home/r/roodman/u/LSST/...` form is **RSP-only**
and silently fails in a Slurm job (`usdf-mount-paths`). Where the stack sets
`$TS_CONFIG_MTTCS_DIR`, prefer it and keep the hardcoded path as fallback.

### Reporting numbers
**Every numerical value carries (a) the name of the quantity and (b) its units** —
or an explicit "dimensionless" with the ratio's numerator and denominator named. No
bare numbers, ever: not in prose, tables, captions, plot labels, print statements,
or commit messages (`reporting-units-standard`). In particular:

- Ratios: `sigma_emp/sigma_RLM = 8.5 (dimensionless; empirical over formal
  coefficient error)`, not "inflation 8.5".
- Correlations: give the statistic, both variables with units, and `n`.
- chi2: always as `chi2/dof`, with dof stated.
- Fractions: say *of what*, and say **power vs amplitude** explicitly — they differ
  by a square.
- Angles/field: give deg and the frame (OCS/CCS).
- Tables: units in the header row once, not per cell.

### Fitting AOS data
Prefer **robust** methods over plain OLS for Zernike/DZ correlations, corner
comparisons, and calibration lines — and **ask which robust method** before
implementing rather than defaulting to OLS (`robust-fits-aos`). Defaults if he
doesn't specify: **Huber** for linear fits (statsmodels `RLM(y, X, M=HuberT())`, as
in `aos/code/dz_fitting.py`); report **both** Pearson r and Spearman rho for
correlations; `nmad(residuals)` for robust scatter RMS.

## Conventions

### Notebook naming
Use descriptive snake_case names: `topic_description_version.ipynb`
Examples: `aos_wavefront_residuals_v2.ipynb`, `psf_ellipticity_focal_plane.ipynb`

### Writing style for READMEs and docs
A `README.md` opens with a **high-level description of the content**, in plain prose,
with every acronym defined on first use. It says what the work *is* — not how the code
got that way. Aaron's model for `aos/README.md`:

> Analysis and Development for the Rubin Observatory Active Optics System (AOS). This
> directory contains multiple studies of AOS engineering data and development of
> calibrations and methods for AOS operation. These include the construction of the
> Measured Intrinsic Wavefront (MIW) from Full Array Mode (FAM) data, study of
> correlations between Double Zernike (DZ), v-modes and Rubin telemetry, analysis of
> bounce test data for Look-Up-Tables (LUT), and comparisons between FAM and Corner
> Wavefront Sensor (CWFS) data.

Follow that register throughout. Specifically:

- **Describe content, not development history.** No lists of renamed or removed files,
  no "consolidates the former X and Y", no "ported from notebook Z". That is what git
  history is for.
- **Define acronyms on first use** — AOS, FAM, MIW, DZ, CWFS, LUT, OFC, DOF, PSF, FWHM,
  EFD, ConsDB. Assume a competent reader who does not know this project's shorthand.
- **No meta-commentary about the repository or the documentation itself.** Not "this is
  the largest topic", not "this file is the map", not "Phase 7 will fix this", not "do
  not re-derive these".
- **Describe a study by its content, not as a rhetorical question.** "Comparison of the
  optical state recovered from the CWFS against the FAM measurement", not "Does the
  corner WFS recover the same optical state as FAM?"
- Outstanding work is stated plainly as a fact about the current state ("splitting these
  per study is outstanding work"), not as a plan or a scolding.

Status headers, units, and the file-location rules below still apply.

### Markdown docs
Every `.md` file has exactly one home, by kind:

| kind | location | example |
|---|---|---|
| topic scope + index | `<topic>/README.md` | `aos/README.md` |
| Claude scoping rules | `<topic>/CLAUDE.md` | `aos/CLAUDE.md` |
| durable reference, conventions, investigation writeups | `<topic>/docs/` | `smatrix/docs/conventions.md` |
| transient working state — handoffs, todo lists, review backlogs, plans | `<topic>/docs/status/` | `aos/docs/status/code_review_findings.md` |
| repo-wide working state (belongs to no one topic) | `notes/status/` | `notes/status/memory_cleanup_plan.md` |
| outward-facing deliverable (Slack post → `lsstdoc` tech note) | `notes/<slug>/` | `notes/aos-measured-intrinsics/` |

Rules:
- **Filenames are snake_case**, like notebooks and Python modules. `README.md` and
  `CLAUDE.md` are the only uppercase names.
- **Nothing loose in a topic root** except `README.md` and `CLAUDE.md`.
- **Every doc opens with a status line** directly under its H1, so a reader knows
  immediately whether to trust it:
  `> **Status:** current · **Last updated:** YYYY-MM-DD · **Kind:** reference (conventions)`
  Use `Status: stale — verify before acting` once the content has been outrun by the
  code, and say what specifically went stale.
- **The `docs/` vs `docs/status/` split is the point**: a stale handoff must never be
  mistaken for a live convention. If a doc records *where the work is*, it is status; if
  it records *how something is defined or what was concluded*, it is reference.
- **Index every doc in the topic's `README.md`** — a doc nothing links to is a doc
  nobody finds.

### Notebook template
All new notebooks should follow the template in `common/notebook_template.ipynb`:
- Header markdown cell with title, author, date created, last modified, status, keywords, description, output, and references
- Change log section
- Table of Contents with anchor links
- Parameters section (all configurable values collected at top)
- Helper Functions section
- Numbered sections with markdown headers using anchor tags

### Studies — where new code goes
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
3. usually a `<topic>/code/<study>/` subdirectory, and an output directory that matches.

Do not add a script to a topic's `code/` root "for now" — that is how a flat 56-file
directory happens. `aos/docs/studies.md` is the worked example of the inventory.

The word is **study**, not "thread" — `aos/code/check_threads.py` is about CPU threads,
and the repo already says study (`study_compare_donuts.ipynb`, `run_study_radialbins.py`).

### Imports and `sys.path`
The repo is **not** an installed package, and scripts are run as `python code/x.py`
(script mode), so relative imports (`from ..other import x`) do not work — Python
leaves `__package__` unset and raises `ImportError`. Use exactly one idiom, at the top
of the file, before any repo import:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N]))  # repo root
from common.utils import nmad
```

`N` counts directories up to the repo root: **2** for `<topic>/code/x.py`, **3** for
`<topic>/code/<study>/x.py`. In a **notebook** there is no `__file__` — use
`common.utils.repo_root()`, which walks up from the working directory instead.

Rules:
- **Never hardcode `/home/r/roodman/...` or `/sdf/...` in import bootstrapping.** The
  `/home/...` form is RSP-only and fails silently in a Slurm job (`usdf-mount-paths`);
  `parents[N]` is correct in every environment.
- Genuinely shared helpers belong in `common/`, not copied between topics.
- Reaching into a sibling topic's `code/` is a real dependency — see
  [Topic independence](#topic-independence-and-shared-code) before adding one.

### Code style
- Python code should follow PEP 8
- Use descriptive variable names
- Shared utility functions go in `common/utils.py` or `common/` submodules
- Prefer `lsst.daf.butler` for data access on RSP
- Prefer `astropy` units and coordinates
- Use `matplotlib` for plotting (RSP standard)

### Docstrings — Rubin DM / numpydoc
Follow the [LSST DM docstring standard](https://developer.lsst.io/python/numpydoc.html):
numpydoc sections, with **types in backticks**. Section order is short summary,
extended summary, `Parameters`, `Returns`/`Yields`, `Raises`, `See Also`, `Notes`,
`Examples` — include only the ones that apply.

```python
def nmad(x, min_n=3):
    """Normalized median absolute deviation — a robust sigma estimate.

    Extended description if the one-liner is not enough.

    Parameters
    ----------
    x : `array_like`
        Values in any single unit; the result carries that same unit.
    min_n : `int`, optional
        Return NaN if fewer than this many finite values remain.

    Returns
    -------
    sigma : `float`
        Robust scatter in the units of `x`, or NaN if under-determined.
    """
```

Points to get right in this repo:

- **Every module opens with a docstring** giving a short description of what the code
  does. For a runnable script, include the invocation and its key arguments.
- **Units belong in the docstring**, on every physical parameter and return value — µm
  of wavefront, deg, arcsec, or an explicit "dimensionless" with the ratio named. This
  is the same rule as [Reporting numbers](#reporting-numbers); a docstring is where a
  reader looks first.
- Say which **frame** an angle or Zernike is in (OCS/CCS) when it matters.
- Note real **failure modes** in `Notes` rather than leaving them implicit — e.g.
  `common/utils.alt_to_deg` documents that degree values below 6.28 are misread as
  radians.
- Existing code is inconsistent (numpydoc appears in only a handful of files, and there
  is no Google-style `Args:` anywhere). Bring a file up to this standard when working in
  it; do not launch a repo-wide reformat as a side quest.

### Git workflow
- Commit messages should be descriptive: "Added M1M3 force analysis notebook" not "update"
- Notebook outputs are committed to git (no stripping) so plots and commentary are preserved
- Do NOT commit large data files (FITS, Parquet, HDF5)
- When creating new notebooks, always start from the template

### Output conventions
- Notebook outputs go in `<topic>/output/` — these are NOT in git (gitignored)
- On the USDF RSP, `output/` directories are symlinked to `/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/<topic>/output/` for disk quota
- Output is synced to the laptop via `~/bin/sync_rubin_work_output` (rsync)
- Large/ephemeral outputs (FITS, parquet, intermediate results) go in `~/notebooks/rubin-data/<topic>/` on RSP
- Notebooks should use a variable like `output_dir` in the Parameters cell to set the output path
- Name output files as `{topic}_{description}_{date_or_dayobs}.{ext}`

### RSP environment
- Code should work on the Rubin Science Platform (both Summit and USDF)
- The LSST Science Pipelines stack is available in RSP notebooks
- Butler repos are accessed via `/repo/main` (USDF) or site-specific paths
- EFD data is accessed via `lsst_efd_client`
