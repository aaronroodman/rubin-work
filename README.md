# rubin-work

Personal Jupyter notebooks and scripts for Vera C. Rubin Observatory work, covering Active Optics, PSF analysis, Camera, Guider, and related topics.

## Repository Structure

```
rubin-work/
├── README.md
├── .gitignore
├── .gitattributes
├── setup_env.sh          # Run once after cloning (adds gitpull/gitpush aliases)
├── sync.sh               # Pull/push helper (auto-stash, conflict guidance)
├── requirements.txt      # Python deps beyond what RSP provides
├── CLAUDE.md             # Instructions for Claude Code
│
├── aos/                  # Active Optics System
│   ├── *.ipynb           # Notebooks live directly in topic dir
│   ├── code/             # Saved Python code
│   └── output/           # Small curated outputs (git-tracked)
│
├── camera/               # Camera analysis
│   ├── code/
│   └── output/
│
├── psf/                  # Point Spread Function
│   ├── code/
│   └── output/
│
├── guider/               # Guider
│   ├── code/
│   └── output/
│
├── starcolor/            # Star color / photometry
│   ├── code/
│   └── output/
│
├── des/                  # Dark Energy Survey related
│   ├── code/
│   └── output/
│
├── survey/               # Survey strategy / operations
│   ├── code/
│   └── output/
│
├── wcs/                  # World Coordinate System
│   ├── code/
│   └── output/
│
├── blocks/               # Observing blocks
│   ├── code/
│   └── output/
│
├── common/               # Shared utilities across all topics
│   ├── __init__.py
│   └── utils.py
│
└── scratch/              # Work-in-progress, not yet organized
```

## Quick Start

### First time setup (laptop or RSP)

```bash
git clone git@github.com:aaronroodman/rubin-work.git
cd rubin-work
./setup_env.sh
```

### Daily workflow on RSP (Summit or USDF)

```bash
cd ~/notebooks/rubin-work
gitpull                                 # Get latest changes
# ... do your work ...
gitpush "description of changes"        # Commit and push
```

The `gitpull` and `gitpush` aliases are set up by `setup_env.sh` (see below). You can also use `./sync.sh pull` and `./sync.sh push` directly.

`gitpull` automatically stashes any local changes, rebases on the remote, and restores the stash. If there are merge conflicts, it shows the affected files and resolution steps.

`gitpush` stages all changes, commits, and pushes. It also pushes any previously committed but unpushed commits.

### Working with Claude Code (laptop)

Open the Claude Desktop app Code tab, point it at this repo folder, and ask it to create or edit notebooks and scripts. It can commit and push directly.

### Authentication on RSP

Use a GitHub Personal Access Token (fine-grained, scoped to this repo):

1. GitHub → Settings → Developer Settings → Personal Access Tokens → Fine-grained tokens
2. Create token with read/write access to this repository
3. On first git push, enter your GitHub username and the token as password
4. The setup_env.sh script configures credential caching for 24 hours

### Notebook Template

All notebooks should follow the standard template in `common/notebook_template.ipynb`, which includes:

* Header cell with title, author, date, status, keywords, description, and references
* Change log
* Table of Contents with anchor links
* Parameters section (all configurable values at the top)
* Helper Functions section
* Standard sections for Data Access, Analysis, Results

### Output conventions

There are two places for notebook outputs:

* **`<topic>/output/` (in git)** — Small, curated outputs worth preserving: summary CSV tables, key plots for papers/presentations. These are git-tracked but large binary formats are still excluded by `.gitignore`.

* **`~/notebooks/rubin-data/<topic>/` (on RSP, NOT in git)** — Large or ephemeral outputs: FITS files, big parquet tables, intermediate results. Create this directory structure on each RSP instance. Notebooks should write large outputs here. Run `./list_notebooks.sh` on each RSP to see what you have.

The `list_notebooks.sh` script inventories all `.ipynb` files in your RSP home directory to help with triage and organization.

### Notes

* Notebook outputs are committed to git (no stripping) so plots and commentary are preserved across machines
* Large data files (FITS, Parquet, HDF5) are excluded via `.gitignore`
* The `scratch/` directory is for work-in-progress
