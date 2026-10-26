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

## In progress

- Slurm job 42019444 is the full `cwfs_science` build: 191 nights, build `20261026`, an
  estimated 1 hour. Log: `aos/logs/cwfs_science_42019444.log`. It does not set `current`.

## Next

1. When the job ends, check it:
   - `grep cwfs_science <build>/_work/logs/night_*.log` for unreadable visits and visits
     with no `zernikes` table;
   - `raft_visits.n_mtaos_unmatched` should total 0;
   - `mtaos_check_um` should be about 1e-5 µm.
2. Make it current:
   `python -m rubinwork.products.catalog set-current cwfs_science d13-blitz_rso1071 20261026`.
3. Update the study's `uses:` to the new build, then start the blur analysis: intra
   (SW1) against extra (SW0) blur per visit, joined to the `value_added` thermal columns.
   The FAM baseline is `aos/intra_extra` (+0.344 arcsec median).
4. Later: compare `raft_visits` with the ConsDB per-corner values, joined on `visit`
   and `raft`.

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
