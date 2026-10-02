# CLAUDE.md — Instructions for Claude Code

## Project Overview
This repository contains Jupyter notebooks and Python scripts for Vera C. Rubin Observatory work, organized by topic. It is used on both the Rubin Science Platform (Summit and USDF) and locally.

## Repository Structure
Each topic directory has `code/`, `notebooks/` and `output/` subdirectories, and `docs/`
where it has prose documentation. Most topics also carry a `README.md` indexing their
scope — skim its index when entering an unfamiliar topic rather than reading it whole.

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
- `thermal_focus/` — prediction of the uniform-defocus error from thermal telemetry, so
  focus can be set open-loop; includes a standalone numpy-only online calculator
  (`docs/thermal_focus.md`)

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
- `value_added/` — the value-added DuckDB database of per-exposure engineering telemetry,
  derived quantities and recovered optical state, plus the builders that maintain it. A
  **service topic**: it maintains a product other topics consume, read through
  `value_added/code/efd_db.py` (`docs/schema.md`, `docs/status/build_progress.md`)
- `common/` — shared utility functions used across topics
- `notes/` — working notes for Slack posts and Summit-Operations tech notes; each
  note is a self-contained dated directory, drafted in plain Markdown.
  `notes/claude-memory/` is an **archive** of old Claude notes-to-self. The rules in
  the `CLAUDE.md` files are complete; do not read the archive unless Aaron asks
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

Shared code lives in **`common/`** (`utils.py`, `telemetry_clients.py`,
`ess_telemetry.py`, `dof_telemetry.py`, `consdb_efd.py`, `visit_telemetry.py`,
`FocalPlaneInterpolator.py`, `psf_moments_consdb.py`, plus
`common/scripts/`), imported by inserting the repo root on
`sys.path`. Genuinely shared helpers belong there rather than being copied between
topics.

Two cross-topic couplings are real and intentional — know them before refactoring:

- `guider/code/` imports `moments_hsm.measure_hsm_moments` from **`optatmo/code`**, so
  that both sides of the guider-vs-science-CCD moment comparison use the same
  galsim-HSM estimator. Changing that estimator changes guider results.
- `blocks/`, `olr/`, `optatmo/`, `smatrix/`, `thermal_focus/` and `value_added/` import
  `aos_state` from
  **`aos/code`** for the v-modes, the DOF sets and the per-corner Zernike recovery. That is
  AOS physics and stays in `aos/`; the engineering telemetry those topics also need was
  moved to `common/` so `aos_state` is the only remaining reach into `aos/code`.

Both go through `sys.path.insert`, not real packages, so the coupling is invisible to
static import checks.

Beware one naming collision: in `aos/code/`, `common` in an import almost always means
`lsst.ts.intrinsic.wavefront.common` — an **external** package — not this repo's
`common/`. Read the full import path before acting.

## Working with Aaron

Standing rules that apply to every session in this repo, on every machine. They are
complete as written here; there is no need to consult `notes/claude-memory/` for them.

### Writing: plain, short, direct
Applies to everything you write: chat replies, code comments, docstrings, commit
messages, READMEs and docs. Aaron finds verbose, inflated prose a real problem.

- **Lead with the answer.** No preamble, no restating the question, no closing recap.
- Short sentences and common words, one idea per sentence. Cut every word that does
  not change the meaning.
- No filler or inflation: "it's worth noting", "importantly", "essentially", "in order
  to", "comprehensive", "leverage", "ensure", "seamlessly", "key insight", "delve".
- Explanations are a few plain sentences. Use headers and bullets only for content that
  is a list or a reference. Bold only for a hard rule or a warning.
- Default length: a chat answer is a few sentences unless Aaron asks for depth. Report a
  result as the number with units plus one line of interpretation, not a narrative of
  the steps you took.
- Code comments say *why*, and only where the code does not already say it. No comments
  that restate the line, no change-history comments ("now uses X", "fixed bug"), no
  banners around trivial code.
- Commit messages: one summary line of at most 72 characters; add a short body only
  when the reason is not obvious.
- Before sending, reread and cut by a third.

Bad: "This function is responsible for computing the robust scatter of the residuals,
which is an important quantity that allows us to effectively assess the fit quality."
Good: "Robust scatter (nMAD) of the fit residuals, in µm of wavefront."

### Autonomy and hard stops
Default to **acting without asking** on routine coding and analysis work that is
already well-scoped; only ask when genuinely unsure what to do. Three exceptions
are hard MUST-ASK rules, no exceptions:

1. **Deleting any files** — ask first, always.
2. **Submitting any batch job** — Slurm `sbatch` / `condor_submit` on S3DF, in
   practice `run_snake.sh --mode batch`. **Ask Aaron to start it**, and give him two
   commands: the full submit command *and* the command to monitor its progress. Batch
   must be submitted from an s3df node (`slacrd`), never from an RSP pod (no Slurm
   there). See Batch jobs below for the exact forms.
3. **Connecting to SLAC / USDF from the laptop** — `ssh slacrd`, USDF,
   `/repo/main` Butler. Ask before opening the connection, or hand Aaron a
   runnable snippet to execute himself (his established preference). Local work in
   `rubin-work/` needs no permission.

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
ellipsis.

Two things to get right in those commands:
- **Repo sync:** say "run your gitpull script", not `git pull` — Aaron uses his own
  `gitpull` wrapper. Claude's own commits/pushes still use git
  directly.
- **Python interpreter:** MacPorts `/opt/local/bin/python3` on the laptop; bare
  `python` on the RSP / USDF, where the stack interpreter is on PATH and the
  MacPorts binary does not exist.
- **Aaron's interactive shells already set up the LSST stack via `.bashrc`** — do
  not prepend `source .../loadLSST.bash && setup lsst_distrib` to snippets he runs
  himself.

### Paths across environments
Use the `/sdf/group/rubin/u/roodman/LSST/...` form in any code or config that might
run in batch — it resolves identically in the RSP notebook, the RSP terminal, and on
slaciana/sdfiana batch nodes. The `/home/r/roodman/u/LSST/...` form is **RSP-only**
and silently fails in a Slurm job. Where the stack sets
`$TS_CONFIG_MTTCS_DIR`, prefer it and keep the hardcoded path as fallback.

### Reporting numbers
**Every numerical value carries (a) the name of the quantity and (b) its units** —
or an explicit "dimensionless" with the ratio's numerator and denominator named. No
bare numbers, ever: not in prose, tables, captions, plot labels, print statements,
or commit messages. In particular:

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
implementing rather than defaulting to OLS. Defaults if he
doesn't specify: **Huber** for linear fits (statsmodels `RLM(y, X, M=HuberT())`, as
in `aos/code/dz_fitting.py`); report **both** Pearson r and Spearman rho for
correlations; `nmad(residuals)` for robust scatter RMS.

### Context hygiene (cost control)
Every turn re-reads the whole conversation, so tokens pulled into context are paid for
on every later call. Keep it small:

- Locate code with `rg`/`grep` first; Read with `offset`/`limit` for the section needed,
  not whole files. Do not re-read a file already read this session unless it changed.
- **Never Read a `.ipynb` whole** — outputs and embedded images are huge. Extract the
  code with `jupyter nbconvert --to script --stdout <nb>.ipynb` or
  `jq -r '.cells[] | select(.cell_type=="code") | .source | join("")' <nb>.ipynb`.
- Pipe long command output through `head`, `tail` or `grep`; never `cat` logs, large
  CSV/parquet dumps or notebook JSON. For a failing test or pipeline run, show only the
  tail of the log.
- Data products and logs (`output/`, `logs/`, `*.parquet`, `*.fits`): when a task needs
  them, look with `tail`/`grep` or a short Python summary (`df.shape`, `df.columns`,
  `df.describe()`, a few rows) rather than reading the whole file.
- Long reference docs (handoffs, equation docs): grep for the section the task needs.
- Batch independent tool calls; stop at natural checkpoints and summarize rather than
  iterating indefinitely. Before ending a multi-phase task, write status and next steps
  into its plan file so the next phase can start in a fresh session.

## Conventions

Detailed conventions are project skills in `.claude/skills/`, loaded only when the task
needs them:

| skill | use when |
|---|---|
| `rubin-new-study` | adding new code; deciding study vs topic-common vs `common/` |
| `rubin-output-layout` | choosing an output path or filename |
| `rubin-doc-style` | writing a README or any `.md`; where each kind of doc lives |
| `rubin-docstrings` | writing docstrings (Rubin DM numpydoc, units, frames) |
| `rubin-notebooks` | creating, naming or moving a notebook |

The short forms: before writing new code, **agree which study it belongs to** (ask
rather than guess); output goes under `<topic>/output/<study>/...`, never in git;
notebooks live in `<topic>/notebooks/<study>/`; docs describe content, define acronyms,
and carry a status line.

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
  `/home/...` form is RSP-only and fails silently in a Slurm job;
  `parents[N]` is correct in every environment.
- Genuinely shared helpers belong in `common/`, not copied between topics.
- Reaching into a sibling topic's `code/` is a real dependency — see
  Topic independence above before adding one.

### Code style
- Python code should follow PEP 8
- Use descriptive variable names
- Shared utility functions go in `common/utils.py` or `common/` submodules
- Docstrings: Rubin DM numpydoc with units on physical quantities (`rubin-docstrings` skill)
- Prefer `lsst.daf.butler` for data access on RSP
- Prefer `astropy` units and coordinates
- Use `matplotlib` for plotting (RSP standard)

### Git workflow
- Commit messages should be descriptive: "Added M1M3 force analysis notebook" not "update"
- Notebook outputs are committed to git (no stripping) so plots and commentary are preserved
- Do NOT commit large data files (FITS, Parquet, HDF5)
- When creating new notebooks, always start from `common/notebook_template.ipynb`

### RSP environment
- Code should work on the Rubin Science Platform (both Summit and USDF)
- The LSST Science Pipelines stack is available in RSP notebooks
- Butler repos are accessed via `/repo/main` (USDF) or site-specific paths
- EFD data is accessed via `lsst_efd_client`
