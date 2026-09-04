---
name: desc-work-repo
description: desc-work GitHub repo for DESC/DP1 coadd analysis; layout and first notebook
metadata: 
  node_type: memory
  type: project
  originSessionId: 8003039d-15db-40c8-a557-1185065ec486
  modified: 2026-07-21T22:13:06.208Z
---

New GitHub repo **aaronroodman/desc-work** (private), local clone at
`/Users/roodman/Astrophysics/desc-work`. Created 2026-07-21 for DESC analysis
on Rubin data, organized by topic. Follows the [[rubin-work]] conventions:
`common/utils.py` (setup_plotting), topic dirs with gitignored `output/`,
notebooks built from the rubin-work `common/notebook_template.ipynb`.

First notebook: `coadd/coadd_object_focal_plane_v1.ipynb` — traces DP1 coadd
Object catalog entries back to individual per-detector input images via
`deep_coadd.coaddInputs` (CoaddInputs.ccds), then maps each to LSSTComCam
focal-plane mm via cameraGeom. Key optimization: cache input-CCD catalog per
(band, tract, patch) since objects in a coadd cell share it; read only the
`coaddInputs` component (no pixel I/O). Meant to run on the USDF RSP with the
LSST kernel, Butler repo "dp1" (or /repo/main), collection LSSTComCam/DP1,
skymap lsst_cells_v1. Git identity is repo-local (roodman@stanford.edu).
