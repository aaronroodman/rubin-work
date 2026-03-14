# CLAUDE.md — Instructions for Claude Code

## Project Overview
This repository contains Jupyter notebooks and Python scripts for Vera C. Rubin Observatory work, organized by topic. It is used on both the Rubin Science Platform (Summit and USDF) and locally.

## Repository Structure
- Topic directories: `aos/`, `camera/`, `psf/`, `guider/`, `starcolor/`, `des/`, `survey/`, `wcs/`, `blocks/`
- Each topic has `code/` and `output/` subdirectories; notebooks (`.ipynb`) live directly in the topic directory
- `common/` — shared utility functions used across topics
- `scratch/` — work-in-progress, not yet organized

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
- Small curated outputs (summary tables, key plots) go in `<topic>/output/` — these are git-tracked
- Large/ephemeral outputs (FITS, parquet, intermediate results) go in `~/notebooks/rubin-data/<topic>/` on RSP — NOT in git
- Notebooks should use a variable like `output_dir` in the Parameters cell to set the output path
- Name output files as `{topic}_{description}_{date_or_dayobs}.{ext}`

### RSP environment
- Code should work on the Rubin Science Platform (both Summit and USDF)
- The LSST Science Pipelines stack is available in RSP notebooks
- Butler repos are accessed via `/repo/main` (USDF) or site-specific paths
- EFD data is accessed via `lsst_efd_client`
