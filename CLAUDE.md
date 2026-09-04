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

Survey / astrometry / other:
- `survey/`, `wcs/`, `astrometry/`, `starcolor/`, `des/`, `filters/`, `alerts/`,
  `optics/`

Support:
- `common/` — shared utility functions used across topics
- `notes/` — working notes for Slack posts and Summit-Operations tech notes; each
  note is a self-contained dated directory, drafted in plain Markdown.
  `notes/claude-memory/` is a snapshot of Claude's own working memory (see its
  README — notes-to-self, not documentation)
- `scratch/` — work-in-progress, not yet organized

Note: `optics/`, `filters/`, `alerts/` and `output/` currently have no git-tracked
content (local/gitignored output only).

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
