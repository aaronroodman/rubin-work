# Claude working-memory snapshot

A **copy** of Claude Code's private working memory, snapshotted **2026-09-04** so it
is versioned and shareable across machines and accounts. Not the live files — the
originals stay in `~/.claude/` and will drift from these; re-copy to refresh.

This directory is now the *single* memory location in the repo. It merges what used
to be two disjoint snapshots, taken from the two per-project memory dirs on Aaron's
laptop:

| source | topics |
|---|---|
| `~/.claude/projects/-Users-roodman/memory/` | AOS / MIW / wavefront, guider pipeline |
| `~/.claude/projects/-Users-roodman-Astrophysics-Claude/memory/` | OptAtmo, smatrix, filters, DESC, environment + standing instructions |

`MEMORY.md` is the merged index (34 entries) — the file Claude loads at session start,
one line per memory. Every line resolves to a file in this directory.

## What these are

Notes-to-self, not documentation. They are written in Claude's own shorthand, are
deliberately terse, and record things like "mistakes I made and must not repeat".
Useful for continuity and for auditing how a conclusion was reached; not intended as
a primary reference.

**For handing the MIW work to another person or another assistant, use
[`../../aos/MIW_INVESTIGATION_HANDOFF.md`](../../aos/MIW_INVESTIGATION_HANDOFF.md)
instead.** That document covers the same ground written for an outside reader, with
full context, units on every quantity, and an explicit list of retracted claims.

## To load on another account or machine

Copy the `*.md` files (including `MEMORY.md`) into that account's memory dir,
`~/.claude/projects/<project-slug>/memory/`.

## Contents

### Standing instructions (behavioral rules, not project facts)

These five are how-to-work rules rather than findings. They are also being promoted
into `CLAUDE.md` so they load on every machine and account; the memory files stay as
the record.

| file | what it holds |
|---|---|
| `reporting-units-standard.md` | every number carries its quantity name and units (or "dimensionless" with the ratio named) |
| `full-commands-no-ellipsis.md` | always give complete, copy-paste-ready commands |
| `python-interpreter-by-env.md` | MacPorts `python3` on the laptop, bare `python` on RSP/USDF |
| `gitpull-script.md` | sync repos with Aaron's `gitpull` script, not raw `git pull` |
| `ask-before-slac.md` | get an explicit OK before `ssh slacrd` / USDF work |

### Environment and data access

| file | what it holds |
|---|---|
| `usdf-access-slacrd.md` | ssh route, stack setup, Butler `/repo/main`, `ts_ofc` not in `lsst_distrib` |
| `usdf-mount-paths.md` | `/sdf/group/rubin/u/roodman/LSST/...` everywhere; the `/home/r/...` path breaks batch |
| `dp2-cloud-rsp-only.md` | early DP2 is on `data.lsst.cloud`, not USDF |
| `desc-work-repo.md` | the separate `aaronroodman/desc-work` repo for DESC/DP1 coadd analysis |
| `transformed-efd-consdb.md` | what `efd_lsstcam` carries per visit, and what it does not |
| `camera-temperature-sources.md` | ConsDB vs EFD camera thermal telemetry, `MTCamera.utiltrunk_body` |
| `aos-private-consdb-cache.md` | deferred plan for a per-visit AOS telemetry cache |

### AOS / wavefront / MIW

| file | what it holds |
|---|---|
| `aos-dof-terminology.md` | `optical_state` vs Tweak vs Trim, and which to use |
| `aos-22dof-reduced-set.md` | the 22-DOF reduced set with explicit indices |
| `aos-vmode-normalization.md` | geometric weights, StateEstimator vs `build_ofc_svd`, degeneracy caveat |
| `frame-conventions-ccs-ocs.md` | CCS vs OCS, rotator angle source, what moves with the mirrors |
| `smatrix-project.md` | the `smatrix/` DZ sensitivity matrix; agreement with OFC and the sign/unit gotchas |
| `apr-2026-50dof-lut.md` | the fixed 50-DOF LUT on sky Apr 24–28 2026 |
| `sparse-fit-sensitivity.md` | sparse donut fit: primary↔secondary degeneracy and DOF observability |
| `z11-intra-extra-mystery.md` | unexplained Z11/Z14 intra- vs extra-focal split — **open question, not a known effect** |
| `miw-focal-order-truncation.md` | 83% of MIW power sits above the k<=6 focal orders the build fits |
| `miw-coadd-retrieval-bias.md` | the coadd-vs-MIW investigation: retrieval-bias hypothesis, the L estimator, the corrections made along the way |
| `miw-products-and-m3-backprojection.md` | which MIW file to use, back-projection and joint static-fit results |
| `optatmo-moments-project.md` | the standalone Optics+Atmo PSF-moment fit in `optatmo/` |

### Guider

| file | what it holds |
|---|---|
| `guider-summit-utils-fixes.md` | the local `summit_utils` clone and the fixes being collected for PR |
| `guider-fixes-branch-state.md` | what `u/aaronroodman/guider-fixes` contains relative to summit |
| `deploy-summit-guider-branch.md` | the 8 summit-only fixes on `origin/deploy-summit` |
| `guider-bias-streak-plan.md` | planned per-column bias and ROIROW streak subtraction |
| `guider-centroid-timescale.md` | scale-free centroid motion, the 1.6 s slow/fast split |
| `guider-mosaic-layout.md` | `GuiderPlotter` MosaicLayout geometry and the circular pixel mask |
| `guider-vs-ccd-hsm-frames.md` | matched galsim-HSM comparison and the OCS rotation for Q1/Q2 |
| `guider-atmo-sim.md` | the imSim atmospheric PSF simulator for the 8 guide sensors |
| `guider-pipeline-commands.md` | how to run the guider pipeline (`run_snake.sh`, per-visit runners) |

### Other projects

| file | what it holds |
|---|---|
| `thinfilm-filters-project.md` | `filters/` multilayer interference filter design; `tmm_fast` is the engine |

## Cross-references

Every `[[link]]` between these memories resolves to a file here, with three
exceptions. Those are forward references — placeholders marking something worth
writing later, not broken links — and they do not exist upstream either:

- `[[danish-tarts-compare-notebook]]`, from `z11-intra-extra-mystery.md`
- `[[guider-moment-weighting]]`, from `guider-summit-utils-fixes.md`
- `[[rubin-work]]`, from `desc-work-repo.md`
