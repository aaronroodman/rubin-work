---
name: rubin-output-layout
description: Choosing or creating an output path, output directory or output filename for any rubin-work product (notebook or pipeline output, parquet, FITS, plots).
---

# Output layout conventions for rubin-work

## Output conventions
- Notebook outputs go in `<topic>/output/` — these are NOT in git (gitignored)
- On the USDF RSP, `output/` directories are symlinked to `/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/<topic>/output/` for disk quota
- Output is synced to the laptop via `~/bin/sync_rubin_work_output` (rsync)
- Large/ephemeral outputs (FITS, parquet, intermediate results) go in `~/notebooks/rubin-data/<topic>/` on RSP
- Notebooks should use a variable like `output_dir` in the Parameters cell to set the output path
- Name output files as `{topic}_{description}_{date_or_dayobs}.{ext}`

**Where a product goes: the study first, then the data it depends on.** Code is organized
by *what question is being asked* (the study); output is organized by the study **and the
data the question was asked of**. There is exactly one data level, and when a product
depends on more than one data axis those axes are **joined into one directory name**
rather than nested:

```
output/<study>/<axis1>_<axis2>/    # depends on two data axes
output/<study>/<axis1>/           # depends on one
output/<study>/                   # depends on neither
```

A product that depends on nothing but the optical prescription, a design matrix, or a
database sits at `output/<study>/` with no data level at all.

Joining rather than nesting is what keeps the study outermost: no study has to carry a
nested subtree, so a study directory lists exactly the data sets it was actually run
against — which is what tells you "not run" from "not applicable". The directory name
still states the full dependence.

In `aos/` the two axes are `param_set` (a Butler collection paired with a processing
variant) and `mi_name` (which MIW build was used). A topic with one data axis uses one
level; most topics outside `aos/` have none.

**When a joined name would be unreadable, shorten the axis, do not nest the study.** The
`aos/` keys are long enough that joining them directly gives a 59-character directory, so
`param_sets.yaml` and `mi_config.yaml` each carry a `dir_name` giving a short form used in
paths only — `danish_1_2`, `A_50_34_i_5rot`. The long key remains the identity that
`--param-set`, the `value_added` database rows and the frozen provenance resolve against.
The Snakefile owns the translation and tells each script the directory to write into, so no
script derives a path from a key.

One carve-out, in `aos/` only: the **corner wavefront sensor (CWFS) variant nests** one
level below the data directory (`wfs_ingest/<P>/<cwfs>/`) rather than joining as a third
axis, which would reach 36 characters even with the short names. Nesting a *variant* under
the data level is not the same as nesting the study — the study stays outermost.

Two rules that follow, and that past work got wrong:

- **One product, one path.** Never write the same filename under two different levels —
  a reader cannot tell which is live, and the older copy silently becomes a trap. If a
  product turns out not to depend on a data axis, move it out and delete the copy (asking
  first, per the hard stops in the root `CLAUDE.md`). The same *filename*
  under two different **studies** is not a violation: `aos/` has a phase-1
  `fam_processing/<P>/fits.parquet` and a MIW-referenced `miw/<P>_<M>/fits.parquet`, which
  are different products that the study directory distinguishes.
- **Do not create a study's output directory until it has output.** An empty directory
  makes the tree claim a result exists. The study doc, not the tree, is what records that
  a study exists.
