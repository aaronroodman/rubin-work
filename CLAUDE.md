# CLAUDE.md — Instructions for Claude Code

## Project Overview
This repository contains Jupyter notebooks and Python scripts for Vera C. Rubin Observatory work, organized by topic. It is used on both the Rubin Science Platform (Summit and USDF) and locally.

## Repository Structure
Each topic directory has `code/` and `output/` subdirectories; notebooks (`.ipynb`) live
directly in the topic directory. Most topics also carry a `README.md` describing their
scope — read it before working in an unfamiliar topic.

AOS / wavefront:
- `aos/` — the main AOS topic: measured intrinsic wavefront (MIW), FAM coadds, DZ
  fitting, sensitivity/v-mode studies. Largest directory. Key docs:
  `MIW_COADD_EQUATIONS.md` (derivations), `MIW_INVESTIGATION_HANDOFF.md` (portable
  state of the MIW investigation), `CAMERA_GRAVITY.md`
- `smatrix/` — batoid_rubin DZ sensitivity-matrix construction and convention
  choices (`CONVENTIONS.md`, `MIW_astig_coma_investigation.md`)
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
   practice `run_snake.sh --mode batch`. Hand over the command instead of running
   it. Batch must be submitted from an s3df node (`slacrd`), never from an RSP pod
   (no Slurm there) (`s3df-batch-job-rules`).
3. **Connecting to SLAC / USDF from the laptop** — `ssh slacrd`, USDF,
   `/repo/main` Butler. Ask before opening the connection, or hand Aaron a
   runnable snippet to execute himself (his established preference). Local work in
   `rubin-work/` needs no permission (`ask-before-slac`, `usdf-access-slacrd`).

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

### Notebook template
All new notebooks should follow the template in `common/notebook_template.ipynb`:
- Header markdown cell with title, author, date created, last modified, status, keywords, description, output, and references
- Change log section
- Table of Contents with anchor links
- Parameters section (all configurable values collected at top)
- Helper Functions section
- Numbered sections with markdown headers using anchor tags

### Code style
- Python code should follow PEP 8
- Use descriptive variable names
- Shared utility functions go in `common/utils.py` or `common/` submodules
- Prefer `lsst.daf.butler` for data access on RSP
- Prefer `astropy` units and coordinates
- Use `matplotlib` for plotting (RSP standard)

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
