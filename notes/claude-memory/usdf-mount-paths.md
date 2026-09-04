---
name: usdf-mount-paths
description: "Use /sdf/group/rubin/u/roodman/LSST/... for cross-environment paths — /home/r/roodman/u/... is RSP-only and breaks on slaciana batch"
metadata: 
  node_type: memory
  type: reference
  originSessionId: d6ed390e-ee48-44ee-8c0a-a6efca6a5fdf
  modified: 2026-07-27T20:54:39.268Z
---

roodman's LSST/packages (and notebooks) area is reachable by three different
mount prefixes depending on where code runs:
- `/home/r/roodman/u/LSST/...`      — RSP (Nublado pod) ONLY; does NOT resolve on
  S3DF/sdfiana/slacrd.
- `/sdf/home/r/roodman/u/LSST/...`  — slaciana view; not the RSP view.
- `/sdf/group/rubin/u/roodman/LSST/...` — **resolves identically everywhere**:
  RSP notebook, RSP terminal, AND slaciana/sdfiana batch nodes.

**Always use the `/sdf/group/rubin/u/roodman/LSST/...` form** in any code/config
that may run in batch (see [[usdf-access-slacrd]], [[ask-before-slac]]). Using
the `/home/...` form silently works in the RSP but fails a Slurm job with e.g.
`RuntimeError: Provided data path (...ts_config_mttcs/MTAOS/v13/ofc) does not
exists.` (hit 2026-07-27 in the blocks batch pipeline: OFC config for the v-mode
StateEstimator + mirror LUT).

Fixed across rubin-work 2026-07-27 (blocks/config.yaml ofc_config_dir,
common/utils.py 'usdf' base, aos_state.resolve_ofc_config_dir + nightly_table
OFC fallbacks, notebooks). Note aos_state/nightly_table prefer $TS_CONFIG_MTTCS_DIR
when the stack sets it; the hardcoded path is only the fallback. One un-migrated
example remains: compare_to_archive.py's --archive usage comment points at the
`/home/.../LSST/notebooks/...` form (docstring only, not runtime).
