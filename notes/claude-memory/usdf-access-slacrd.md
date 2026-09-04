---
name: usdf-access-slacrd
description: How to reach USDF/S3DF from the laptop and set up the LSST stack + Butler for Rubin data queries
metadata: 
  node_type: memory
  type: reference
  originSessionId: cc746a2f-9746-41cd-90db-f7d33ae6f183
---

Laptop has no LSST stack/Butler. Reach USDF over ssh: `ssh slacrd` (alias in
~/.ssh/config → rubin-devl via ProxyJump s3dflogin; key auth, non-interactive
works). Home (`~`, e.g. `/sdf/home/r/roodman/`) is shared NFS across nodes;
`/tmp` is node-local (don't rely on it). Scratch used this project:
`~/optatmo_scratch/`.

LSST stack: **roodman's interactive shells already set up the stack via
`.bashrc`** — do NOT include the `source .../loadLSST.bash && setup lsst_distrib`
line in snippets he runs himself. Only when *I* ssh non-interactively (which
needs his OK first, see [[ask-before-slac]]) do I source it myself, since a
non-interactive ssh command may not run `.bashrc`:
`source /sdf/group/rubin/sw/tag/w_2026_27/loadLSST.bash && setup lsst_distrib`
(the default `/sdf/group/rubin/sw/loadLSST.bash` points at a stale conda env —
use a `tag/w_2026_NN` instead; weeklies live in `/sdf/group/rubin/sw/tag/`).
Pipe scripts via stdin to avoid quoting: `ssh slacrd '... && python3 -' < script.py`.

**`lsst.ts.ofc` and `lsst.ts.intrinsic` are NOT in `lsst_distrib`** — they need
the user's AOS/CWFS environment (ask roodman which setup; he runs the CWFS
pipeline there). For anything needing the OFC sensitivity matrix / ts_wep,
either use that env or hand roodman a snippet to run.

Butler: `Butler('/repo/main')`, instrument `LSSTCam`. API notes for this stack:
`queryDatasets(..., limit=)` unsupported; `single_visit_star` etc. come back as
astropy Table (use `.to_pandas()`; beware `.kurtosis`/`.e` are Table *methods* —
index with `['col']`). Use `jax.jacfwd` not jacrev when n_params ≪ n_residuals.
