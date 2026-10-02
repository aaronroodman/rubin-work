---
name: rubin-notebooks
description: Creating, naming, moving or bootstrapping a Jupyter notebook in rubin-work (location, template, sys.path idiom for notebooks).
---

# Notebook conventions for rubin-work

## Notebook naming and location
Use descriptive snake_case names: `topic_description_version.ipynb`
Examples: `aos_wavefront_residuals_v2.ipynb`, `psf_ellipticity_focal_plane.ipynb`

Notebooks live in **`<topic>/notebooks/<study>/`**, mirroring `<topic>/code/<study>/`, so
a study's notebooks sit next to nothing but its own. A topic with only one study can use
a flat `<topic>/notebooks/` and add the `<study>/` level when a second study appears.
Nothing but `README.md` and `CLAUDE.md` belongs loose in a topic root.

A notebook has no `__file__`, so it cannot use the `parents[N]` idiom from
the Imports section of the root `CLAUDE.md`. Walk up to the topic directory instead, which works at
any depth:

```python
import sys
from pathlib import Path
_TOPIC = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'code').is_dir())
sys.path.insert(0, str(_TOPIC.parent))        # repo root -> common/
sys.path.insert(0, str(_TOPIC / 'code'))      # flat cross-study modules
for _d in sorted((_TOPIC / 'code').glob('*/')):
    if _d.is_dir() and not _d.name.startswith(('_', '.')):
        sys.path.insert(0, str(_d))
```

Do **not** write `Path.cwd() if (Path.cwd() / 'code').is_dir() else Path.cwd().parent` —
that guesses one level and breaks as soon as the notebook moves into a `<study>/`
subdirectory.

## Notebook template
All new notebooks should follow the template in `common/notebook_template.ipynb`:
- Header markdown cell with title, author, date created, last modified, status, keywords, description, output, and references
- Change log section
- Table of Contents with anchor links
- Parameters section (all configurable values collected at top)
- Helper Functions section
- Numbered sections with markdown headers using anchor tags
