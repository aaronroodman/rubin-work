# rubin-work

Analysis and development for the Vera C. Rubin Observatory, organized by topic: the Active
Optics System (AOS) and wavefront sensing, point spread function (PSF) and image quality,
the guider, the camera, optical design, survey and astrometry studies, and the
value-added telemetry database that supports them. The work runs on the Rubin Science
Platform (RSP) at both the Summit and the United States Data Facility (USDF), and locally.

Each topic directory holds `code/`, `notebooks/` and `output/`, plus `docs/` where it has
prose documentation. Most carry a `README.md` describing their scope — read it before
working in an unfamiliar topic.

## Topics

AOS and wavefront sensing:

| topic | content |
|---|---|
| [`aos/`](aos/README.md) | The main AOS topic: construction of the Measured Intrinsic Wavefront (MIW) from Full Array Mode (FAM) donut data, FAM coadds, Double Zernike (DZ) fitting, look-up table (LUT) development, and FAM-versus-corner-wavefront-sensor comparisons |
| [`smatrix/`](smatrix/README.md) | The AOS double-Zernike sensitivity matrix computed with `batoid_rubin`, compared against the matrix shipped in `ts_ofc`, and the v-mode structure of its singular value decomposition |
| [`wfs/`](wfs/README.md) | Corner wavefront sensor (CWFS) studies: sky-foreground shape, inspection of instrument-signature-removed images, and the `ts_wep` dataflow |
| [`olr/`](olr/README.md) | Open Loop Reproduction — a Snakemake pipeline that reproduces the open-loop wavefront from a night of AOS operations |
| [`optatmo/`](optatmo/README.md) | Standalone optics-plus-atmosphere PSF moment tools; a differentiable rebuild of the ideas in the PIFF `optatmo3` branch |

Image quality, PSF and instrument:

| topic | content |
|---|---|
| [`psf/`](psf/README.md) | PSF simulation, measurement and analysis |
| [`nightlyiq/`](nightlyiq/README.md) | Image quality image-by-image for a given `day_obs`, combining science-image PSF metrics with guider, AOS and pointing diagnostics |
| [`guider/`](guider/README.md) | The guider system: region-of-interest placement, star catalog matching and pointing diagnostics |
| `camera/` | LSST Camera analysis |
| [`blocks/`](blocks/README.md) | Identifying and tabulating observing blocks and test programs, and trending image-quality metrics across them |

Optical design and prescription, needing no data:

| topic | content |
|---|---|
| [`optics/`](optics/README.md) | Batoid ray-trace studies of the optical system: telecentricity and pupil geometry |
| [`filters/`](filters/README.md) | Feasibility of multilayer dielectric interference filters for future narrow- and medium-band imaging, with automatic differentiation on the layer thicknesses |

Astrometry:

| topic | content |
|---|---|
| `astrometry/` | Per-visit World Coordinate System (WCS) astrometric residual fields as a probe of the atmospheric contribution, compared against the PSF ellipticity pattern across the focal plane |

Reserved, holding no work yet — `camera/`, `des/`, `starcolor/`, `survey/`, `wcs/` and
`alerts/` are scaffolding only.

Support:

| directory | content |
|---|---|
| [`value_added/`](value_added/README.md) | A curated DuckDB database of per-exposure engineering telemetry, derived quantities and recovered optical state, plus the builders that maintain it. Other topics read it rather than refetching |
| [`common/`](common/README.md) | Shared utility code used across topics, imported by inserting the repository root on `sys.path` |
| [`notes/`](notes/README.md) | Working notes for Slack posts and Summit-Operations tech notes; each note is a self-contained dated directory drafted in plain Markdown |
| `scratch/` | Work in progress, not yet organized |

`CLAUDE.md` at the repository root carries the conventions Claude Code follows, with
per-topic `CLAUDE.md` files adding scoping rules for `aos/` and `guider/`.

## Quick start

### First time setup

```bash
git clone git@github.com:aaronroodman/rubin-work.git
cd rubin-work
./setup_env.sh
```

`setup_env.sh` adds the `gitpull` and `gitpush` aliases and configures credential caching
for 24 hours.

### Daily workflow on the RSP

```bash
cd ~/notebooks/rubin-work
gitpull                                 # get latest changes
# ... do your work ...
gitpush "description of changes"        # commit and push
```

`gitpull` stashes any local changes, rebases on the remote and restores the stash, showing
the affected files and resolution steps on a conflict. `gitpush` stages, commits and pushes,
including any previously committed but unpushed commits. `./sync.sh pull` and
`./sync.sh push` do the same thing directly.

### Authentication on the RSP

Use a GitHub fine-grained personal access token scoped to this repository: GitHub →
Settings → Developer Settings → Personal Access Tokens → Fine-grained tokens, with
read/write access to the repository. On the first push, give your GitHub username and the
token as the password.

## Conventions

The root `CLAUDE.md` is the full reference; the essentials are below.

### Code and notebook layout

Code is organized by **study** — a separable project inside a topic — at
`<topic>/code/<study>/`, with its notebooks alongside at `<topic>/notebooks/<study>/`. A
topic with one study can use a flat `<topic>/notebooks/`. Nothing but `README.md` and
`CLAUDE.md` belongs loose in a topic root.

Modules shared across studies within a topic sit flat at `<topic>/code/`; modules shared
across topics go in `common/`. The repository is not an installed package and scripts run
in script mode, so imports bootstrap by inserting the repository root on `sys.path` via
`pathlib.Path(__file__).resolve().parents[N]` — never a hardcoded absolute path, which is
RSP-only and fails silently in a batch job.

Notebooks use descriptive snake_case names and follow the template in
`common/notebook_template.ipynb`: a header cell with title, author, dates, status,
keywords and description; a change log; a table of contents; a parameters section
collecting every configurable value at the top; and a helper-functions section.

### Markdown documentation

Durable reference — conventions, derivations, investigation writeups — lives in
`<topic>/docs/`. Transient working state such as handoffs, todo lists and review backlogs
lives in `<topic>/docs/status/`, and repository-wide working state in `notes/status/`.
Keeping the two apart is the point: a stale handoff must never be mistaken for a live
convention. Every document opens with a status line stating whether it can be trusted, and
is indexed in its topic's `README.md`.

### Output conventions

Output is **not** in git — `.gitignore` excludes the contents of every `<topic>/output/`.
On the USDF RSP several of those directories are symlinks to
`/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/<topic>/output/` for disk quota, and
output is pulled down to the laptop with the `sync_rubin_work_output` rsync wrapper there.
Large or ephemeral intermediates go to `~/notebooks/rubin-data/<topic>/` instead.

Within a topic's `output/`, products are laid out **study outermost with exactly one data
level**, and where a product depends on more than one data axis those axes are joined into
a single directory name rather than nested:

```
output/<study>/<axis1>_<axis2>/    # depends on two data axes
output/<study>/<axis1>/            # depends on one
output/<study>/                    # depends on neither
```

A product depending on nothing but the optical prescription, a design matrix or a database
sits at `output/<study>/` with no data level. Joining rather than nesting keeps the study
outermost, so a study directory lists exactly the data sets it was run against — which is
what distinguishes "not run" from "not applicable". In `aos/` the two axes are the
`param_set` (a Butler collection paired with a processing variant) and the `mi_name`
(which MIW build was used); most topics have none.

Notebook outputs are committed to git without stripping, so plots and commentary are
preserved across machines. Large data files — FITS, Parquet, HDF5 — are excluded.

`./list_notebooks.sh` inventories the `.ipynb` files in an RSP home directory, for triage.
