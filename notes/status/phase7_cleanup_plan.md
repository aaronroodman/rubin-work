# Phase 7 plan: back up and remove the old output trees

> **Status:** readers moved to the catalog; nothing zipped or removed; Aaron runs it · **Last updated:** 2026-10-24 · **Kind:** plan

The old output trees are the pre-reorg `<topic>/output` directories. The repo's
`<topic>/output` symlinks point at them, through `/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/`,
which resolves to `/sdf/data/rubin/user/roodman/LSST/notebooks/rubin-work/`. The products
and study outputs live elsewhere (`.../LSST/rubin-work/products/` and `.../studies/`), so
removing these trees does not touch them.

## Inventory (2026-10-24)

`T` = `/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work`.

| tree | size [GB] | files [count] | largest parts [GB] | still read by tracked code | safe to remove |
|---|---|---|---|---|---|
| `T/aos/output` | 72.7 | 1073 | `fam_processing` 56.6, `archive` 9.9, `miw` 3.8, `coadd` 0.8, `wfs_fam_compare` 0.7, `dzfit` 0.7 | yes, below | **no** |
| `T/aos/output-archive-2026-06-11` | 35.9 | 2233 | old FAM table formats | only `fam_tables/qa/compare_to_archive.py`, by argument | yes |
| `T/blocks/output` | 2.5 | 94 | `ccdvisits_20251026-20260513.pq` 1.1 | no | yes |
| `T/guider/output` | 4.5 | 23354 | `night_*` (moments copied to the product 2026-10-24; QA plots, movies), `old` 0.7 | QA rules only, below | **no** |
| `T/optatmo/output` | 2.2 | 5242 | `runs` 2.1 | yes, below | **no** |
| `T/value_added/output` | 6.0 | 57 | `aos_efd.duckdb` 2.8, `shards` 2.7, `aos_efd_archive_20260916_pre_vmode_rebuild.duckdb` 0.5 | fallback only | yes |

Total 123.8 GB; 47.4 GB of it removable now. `/sdf/data/rubin` has 618 GB free.

The topic outputs inside the home repo (`psf/output`, `thermal_focus/output`, `wfs/output`
and others) hold 0.16 GB together and are not part of this phase.

## Readers, per tree

Checked by scanning tracked code for each tree's path, then reading each hit. Docstrings,
help text, provenance YAML and test fixtures with fake paths are left out (about 30 files,
count).

**`aos/output`: no longer preferred over the catalog (2026-10-24).**
- `aos/Snakefile` `_resolve()` takes the catalog's `current` build first and the old tree
  only for a variant with no catalog build. The study rules write to each study's
  `output@` (`dzfit`, `correlations`, `bounce`, `wfs_mimic`, `cwfs/output/wfs_dof_compare`,
  `miw_origin/output/{fam_coadd_miw,study_radialbins}`); their existing outputs were copied
  there. `snakemake -n` plans 31 jobs (count), the 35 before less the 4
  `wfs_corner_compare` jobs, which left with the product.
- `build_fam_dz.py` defaults to `--prefer catalog`. `build_optical_state.resolve_miw_decomp`
  maps an old-tree name in `intrinsic_ref` to its `miw` variant's catalog build, so
  `miw_corner.decomp_path` (a library, which may not import the catalog) is reached only for
  a name no `miw` variant claims.
- `cwfs_lut` and the `miw_origin` scripts try the catalog first; `cwfs_lut` and
  `analyze_miw_field_order` write to their study `output@`.
- `wfs_corner_compare` is a product; `extract_cwfs_sigma.py` reads it through the catalog,
  and `ALLOWED` is empty.

**What is still only in `aos/output`** (2026-10-24, after the copies below): nothing read
by tracked code. Copied, sizes checked file by file, old tree untouched:
- the 10 Danish 1.3 MIW builds (counts) into the `miw` product as `old-tree-<date>` builds,
  set `current`; `fits_july.parquet` as `miw/d12-A_50_34_i_5rot/old-tree-fits-july`; the
  location-only `bounce_tables` `_july` and `coadds` `_v2` registrations now hold their files;
- `wfs_fam_compare` and `miw_cwfs_intrinsic_check.parquet` to `cwfs/output`;
  `processing_compare` and `danish_tarts_compare_20260713` to `processing_compare/output`;
  `psf` to `psf_fp_maps/output`; `closed_loop`, `camera_gravity` (to `miw_origin/output`),
  `lut`; `bending_mode_test_meta.parquet` and `dof_normalization_split.parquet`; the two
  sparse-fit PDFs to `smatrix/sparse_fit/output`; the four `miw/` comparison directories to
  `miw_recovery_constraint/output` and `miw_version_comparisons/output`.

Left in the tree, and not needed elsewhere: `fam_processing/` (55 GB; the three variants
are in the catalog, the donut tables equal in size), `wfs_ingest/` (the four `d12` CWFS
variants are in the catalog), `coadd/` and `bounce/` (in the catalog), the other old-tree
`miw/` builds (in the catalog), and `archive/` (9.2 GB, frozen provenance, kept in the
backup). `run_wfs_fam_compare.py`, `run_closed_loop.py` and `run_psf_fp_maps.py` (parked)
still default to `--output-root output` and read a pre-September layout that no longer
exists; fix them when they are next run.

**`guider/output`: the moments are now a product; the rest is study output.** On
2026-10-24 the nine nights (count) 20260512-20260712 were copied, md5-checked, to
`products/guider_moments/v4/nights/<d>/` and registered there. They are schema v4, not v6,
so they are a variant of their own; the builders write only v6. `guider/Snakefile`
`momentsDir()` now reads the catalog first (v6, then v4), and `psfmoments` writes to the
`guider_vs_ccd_moments` study's `output@`, where the nine nights' `guider_psfmoments`
files were copied too. The `guider_ellipticity_ensemble` notebook reads the product
through the reader and the PSF moments from that study.

Still in this tree: the per-night QA PDFs, movies and rtvplots, which the Snakefile still
writes under `output/night_<d>/` because no study owns them; and the top-level saturation,
streak and `atmo_sim` files of the parked `guide_star_saturation`, `bias_streak` and
`guider_atmo_sim` studies, plus `old/` (0.7 GB) and `review_748/`. None is read by tracked
code except through those studies' own scripts, run by hand.

**`optatmo/output`: the live campaign root.** `optatmo/code/campaign.py:30`
`RUNS_ROOT = 'output/runs'`, relative to `optatmo/`; `fit_optatmo.py` and
`ensemble_corners.py` write and read fits there. It is current output, not an old copy.

**`blocks/output`, `value_added/output`, `aos/output-archive-2026-06-11`: not read.**
`blocks/code/find_long_exposures.py` writes there only when `--out` is passed.
`rubinwork.products.value_added.default_db_path()` resolves the catalog build
(`value_added/v1/snapshots/20261016/aos_efd.duckdb`) and falls back to the old file only
if none exists. `compare_to_archive.py` takes the archive path as an argument.

## What has to happen before the other three can go

1. `aos/output`: done (2026-10-24). It can go once backed up.
2. `guider/output`: done for the moments and PSF moments (2026-10-24). Left: give the QA
   PDFs, movies and rtvplots an owner (a study `output@`) and repoint those three rules.
3. `optatmo/output`: kept as the `psf_vs_cwfs` study's data (Aaron, 2026-10-24); a product
   later. Dropped from this phase.

## Commands (Aaron runs these; nothing here has been run)

Backups go to `/sdf/data/rubin/user/roodman/LSST/backup/` (618 GB free). `tar` with `zstd`
keeps symlinks and is faster than `zip` on parquet, which does not compress further; use
`zip -r` instead if you want zip archives. Each block checks the archive before removing.

**Safe now:**

```bash
B=/sdf/data/rubin/user/roodman/LSST/backup/pre_reorg_outputs_20261024
T=/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work
mkdir -p $B
tar -C $T -cf - aos/output-archive-2026-06-11 | zstd -T8 -o $B/aos_output-archive-2026-06-11.tar.zst
tar -C $T -cf - blocks/output | zstd -T8 -o $B/blocks_output.tar.zst
tar -C $T -cf - value_added/output | zstd -T8 -o $B/value_added_output.tar.zst
```

Check each archive lists the right number of files, then remove:

```bash
B=/sdf/data/rubin/user/roodman/LSST/backup/pre_reorg_outputs_20261024
T=/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work
zstd -dc $B/aos_output-archive-2026-06-11.tar.zst | tar -tf - | grep -vc '/$'
zstd -dc $B/blocks_output.tar.zst | tar -tf - | grep -vc '/$'
zstd -dc $B/value_added_output.tar.zst | tar -tf - | grep -vc '/$'
```

Expect 2233, 94 and 57 files (count). Then:

```bash
T=/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work
rm -rf $T/aos/output-archive-2026-06-11 $T/blocks/output $T/value_added/output
```

What breaks: nothing tracked. `blocks/output` and `value_added/output` become dangling
symlinks in the repo; `blocks/code/find_long_exposures.py --out blocks/output/x.csv` would
fail until the directory is recreated. `value_added_db_validation.ipynb` hardcodes the old
duckdb path and would need repointing if rerun.

**After the steps above, not before** (`optatmo/output` stays; it is study data):

```bash
B=/sdf/data/rubin/user/roodman/LSST/backup/pre_reorg_outputs_20261024
T=/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work
tar -C $T -cf - aos/output | zstd -T8 -o $B/aos_output.tar.zst
tar -C $T -cf - guider/output | zstd -T8 -o $B/guider_output.tar.zst
zstd -dc $B/aos_output.tar.zst | tar -tf - | grep -vc '/$'
zstd -dc $B/guider_output.tar.zst | tar -tf - | grep -vc '/$'
```

Recount the files first (`find $T/aos/output $T/guider/output -type f | wc -l`); the 1073
and 23354 above (count) were taken before 2026-10-24's copies. Then:

```bash
T=/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work
rm -rf $T/aos/output $T/guider/output
```

What breaks if run today: the guider QA, movie and rtvplot rules, until they get a study
output; `build_fam_manifest.py` as documented; and the three parked scripts above.

The laptop's synced output copies (about 60 GB) are not on s3df; remove them on the laptop
when the s3df trees are settled.
