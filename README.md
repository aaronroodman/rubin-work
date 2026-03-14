# rubin-work

Personal Jupyter notebooks and scripts for Vera C. Rubin Observatory work, covering Active Optics, PSF analysis, Camera, Guider, and related topics.

## Repository Structure

```
rubin-work/
├── README.md
├── .gitignore
├── .gitattributes
├── setup_env.sh          # Run once after cloning on a new RSP instance
├── sync.sh               # Quick pull/push helper
├── requirements.txt      # Python deps beyond what RSP provides
├── CLAUDE.md             # Instructions for Claude Code
│
├── aos/                  # Active Optics System
│   ├── notebooks/
│   └── scripts/
│
├── camera/               # Camera analysis
│   ├── notebooks/
│   └── scripts/
│
├── psf/                  # Point Spread Function
│   ├── notebooks/
│   └── scripts/
│
├── guider/               # Guider
│   ├── notebooks/
│   └── scripts/
│
├── starcolor/            # Star color / photometry
│   ├── notebooks/
│   └── scripts/
│
├── des/                  # Dark Energy Survey related
│   ├── notebooks/
│   └── scripts/
│
├── survey/               # Survey strategy / operations
│   ├── notebooks/
│   └── scripts/
│
├── wcs/                  # World Coordinate System
│   ├── notebooks/
│   └── scripts/
│
├── blocks/               # Observing blocks
│   ├── notebooks/
│   └── scripts/
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
git clone https://github.com/aaronroodman/rubin-work.git
cd rubin-work
./setup_env.sh
```

### Daily workflow on RSP (Summit or USDF)

```bash
cd ~/notebooks/rubin-work
./sync.sh pull                          # Get latest changes
# ... do your work ...
./sync.sh push "description of changes" # Commit and push
```

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

### Notes

* `nbstripout` is configured to automatically strip notebook outputs on commit
* Large data files (FITS, Parquet, HDF5) are excluded via `.gitignore`
* The `scratch/` directory is for work-in-progress
