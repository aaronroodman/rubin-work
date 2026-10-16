---
name: rubin-notebooks
description: Creating, naming, moving or bootstrapping a Jupyter notebook in rubin-work (location, template, imports, where its output goes).
---

# Notebook conventions for rubin-work

## Location and name
A study's notebooks live in **`<topic>/<study>/notebooks/`** (`/start-study` makes the
directory). A notebook that answers a new question starts a new study; do not add a
loose notebook to a topic root.

Topics not yet converted to the study layout (phase 5) keep `<topic>/notebooks/<study>/`;
a new notebook there goes beside its study's others.

Use descriptive snake_case names: `<description>_v<N>.ipynb`, such as
`itl_column_dip_v1.ipynb`. The study directory already names the topic and study.

## Imports
Import the installed package; no `sys.path` work:

```python
from rubinwork.common import study_run
from rubinwork.common.utils import nmad
from rubinwork.products import catalog
```

To import the study's own `code/` modules, add only that directory:

```python
import sys
from rubinwork.common.utils import repo_root
sys.path.insert(0, str(repo_root() / "aos" / "miw_hull" / "code"))
```

A notebook has no `__file__`; `repo_root()` walks up from the working directory.
Never import another study's code; shared code becomes a library in `rubinwork/`.

## Output
Read product data only through `rubinwork.products.catalog` (or a product's `load`).
Write output into a dated run directory:

```python
run = study_run("aos/miw_hull", "ring_coverage")   # in the Parameters cell
fig.savefig(run.path / "ring_coverage.pdf")
```

`run.path` is `<topic>/<study>/output/<YYYYMMDD>_<label>/`, through the study's
`output` symlink to the data root. Its `run.json` records the git commit, the notebook
and every product build read, which `/wrap` copies into `uses:`. Copy the one or two
PNGs the Log cites into `figures/`; `/wrap` does that.

## Template
New notebooks start from `common/notebook_template.ipynb`:
- header markdown cell with title, author, date created, last modified, status,
  keywords, description, output, and references;
- change log section;
- table of contents with anchor links;
- Parameters section (every configurable value at the top, including the `study_run`
  call);
- Helper Functions section;
- numbered sections with markdown headers using anchor tags.
