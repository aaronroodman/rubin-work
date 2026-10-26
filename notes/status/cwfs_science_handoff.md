# cwfs_science product and the CWFS blur study: handoff

> **Status:** current · **Last updated:** 2026-10-26 · **Kind:** handoff

## Done and committed

- `22f5d2f`: new study `aos/cwfs_blur_intra_extra`, about the intra − extra blur difference
  in the corner wavefront sensors (CWFS) over science visits, and its link to thermal
  telemetry. `uses:` still lists `cwfs_tables=d13-blitz_unpaired@20261024`; switch it to
  `cwfs_science`.
- `b074bf6`: new product `rubinwork/products/cwfs_science`, variant `d13-blitz_rso1071`.
  Two tables:
  - `ccd_visits`: one row per visit and corner CCD;
  - `raft_visits`: one row per visit and corner raft.

  Schema is in the package docstring.
- `73447d5`: the HTML site moved to `site/` at the repo root (gitignored).
- Outside the repo:
  - `~/u/LSST/setup.sh` now uses w_2026_43 (old copy at `/tmp/setup.sh.w42`);
  - `ts_wep`, `donut_viz`, `ts_xml`, `ts_utils` and `ts_observatory_control` were
    fast-forwarded;
  - `~/.local/bin/snakemake` was reinstalled (9.22.0) so it runs on the w_2026_43 Python.

## Done since: the full build

- Build `20261026` is current (Slurm job 42019444, 17 min):
  - 92,388 visits over 191 nights;
  - 738,154 CCD rows and 367,255 raft rows;
  - 4 visits unreadable (20251120 seq 17-20, corrupt parquet);
  - 247 visits have no `zernikes` table.
- The MTAOS match is exact:
  - 0 unmatched rows;
  - largest `mtaos_check_um` 6.7e-4 µm of wavefront.
- First look, from raft-visits with at least 3 quality-cut donuts on each half:
  - median intra − extra blur is +0.2604 arcsec (n 298,003 raft-visits);
  - intra is larger in 0.9231 of them (fraction of raft-visits).
- The study's `uses:` now pins `cwfs_science=d13-blitz_rso1071@20261026`.

## Next

1. Blur analysis in `aos/cwfs_blur_intra_extra`: intra (SW1) against extra (SW0) blur
   per visit, joined to the `value_added` thermal columns. The FAM baseline is
   `aos/intra_extra` (+0.344 arcsec median).
2. Later: compare `raft_visits` with the ConsDB per-corner values, joined on `visit` and
   `raft`.

## Tried and rejected

- **A `cwfs_tables` variant**, rejected because that product walks the FAM visits (1,845
  visits for d13) and its schema is per donut.
- **Copying the ts_wep sigma clip ourselves**, rejected after Bryce's note that the Main
  Telescope AOS (MTAOS) reads the `zernikes` table. The MTAOS selection is that table's
  `used` flag, matched to the blitz donuts on detector and donut id.
- **Things that differ from what one might assume:**
  - the `zernikes` table holds CCS deviation Zernikes in nm, one table per raft with both
    halves;
  - this run's clip used `stdMin` 0.005 µm, not the 0.1 µm of RSO-250, which is
    unmerged.
- **Gotcha:** `zernikes` stores `intra_field` as a structured (x, y) type. Decide the
  side from the masked `*_donut_id` columns instead.
