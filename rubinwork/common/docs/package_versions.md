# Tracking which package versions an analysis used

> **Status:** current · **Last updated:** 2026-09-29 · **Kind:** how-to

The `ts_XXX` and other packages under `/sdf/data/rubin/user/roodman/LSST/packages` are
not part of the DM stack. They are plain `git clone`s on `develop`/`main`, set up with
`eups setup -r` from `LSST/setup.sh`. That means **a `git pull` silently changes the code
under a finished analysis**, and nothing in the output records which code produced it.

This note describes the lockfile mechanism that fixes that, and what Rubin does online.

## Contents

- [The tool](#the-tool)
- [Recommended workflow](#recommended-workflow)
- [How Rubin does this online](#how-rubin-does-this-online)
- [The 2026-09-29 incident](#the-2026-09-29-incident)

## The tool

`rubinwork/common/scripts/package_versions.py` writes and compares *lockfiles*: one line per
repository giving branch, full commit, `git describe`, last-commit date and a count of
uncommitted files.

```bash
cd ~/notebooks/rubin-work

# Record the current state of every cloned package.
python rubinwork/common/scripts/package_versions.py snapshot -o common/output/packages.lock.txt

# What has moved since a snapshot, and what landed in between?
python rubinwork/common/scripts/package_versions.py diff common/output/packages.lock.txt --log

# Put the working trees back to what a lockfile names (dry run without --apply).
python rubinwork/common/scripts/package_versions.py checkout common/output/packages.lock.txt
```

`diff` exits non-zero when anything moved, so a build script can gate on it.
Set `LSST_PACKAGES_DIR` to point at a different clone tree (e.g. on the summit RSP).

Two lockfiles are checked in under `common/output/`:

- `packages.lock.txt` — current state, written 2026-09-29 after the pull.
- `packages.lock.prepull-20260929.txt` — reconstructed from the repos' reflogs, the
  state that actually built `value_added/output/aos_efd.duckdb`. Keep this: reflog
  entries expire (90 days by default for reachable commits), and after that the
  "what did I actually run?" question becomes unanswerable.

## Recommended workflow

1. **Snapshot at the start of every long build**, next to the output it describes:
   `value_added/output/aos_efd.versions.txt`. The builders already carry provenance
   tags (`opd_version`, `fits_path`); the package lockfile is the same idea one level
   down. Consider having `build_efd_db.py` call `snapshot_text()` and write the lockfile
   automatically, and stamp a `package_lock` value into `fetch_log`.
2. **`diff` before you pull**, not after. That is the cheap moment to notice that
   `ts_m1m3_utils` is about to move.
3. **Never `git pull` all packages at once as routine hygiene.** Pull one package when
   you actually need something from it, read its `doc/version_history.rst` or news
   fragments, and note it. The `ts_` repos use towncrier, so
   `doc/news/*.rst` on the new commits is a fast summary of intent.
4. **Prefer tags over branch tips** for anything whose numbers you will publish.
   `git checkout v0.6.2` is reproducible; `develop` is not.

## How Rubin does this online

Rubin does **not** run `develop` tips in production. The mechanism is
[`ts_cycle_build`](https://github.com/lsst-ts/ts_cycle_build):

- A **cycle** is "a set of well-defined software versions." `cycle/cycle.env` pins every
  package by exact version — `ts_salobj=8.2.9`, `ts_scriptqueue=2.14.3`, and where a
  branch is unavoidable it is named explicitly (`cwfs=master`).
- The file carries `CYCLE=c00NN` and `rev=.0NN`. Changing a **core** package
  (`ts_xml`, `ts_sal`, `ts_salobj`) bumps the cycle and resets the revision; changing
  anything else bumps only the revision. Only components that changed get rebuilt.
- Deployment is by container built from that env file, so the summit runs a named,
  immutable set.

The lockfile here is the single-user analogue of `cycle.env`. `ts_wep` is also now moving
toward recording this at the science level: `EstimateZernikesBaseTask._logMaskVersions`
(new as of 2026-09) logs the danish version, the resolved mask file and the Batoid model
name into task metadata, precisely so a wavefront run can be traced to a mask+optics
pair later.

## The 2026-09-29 incident

A `git pull` of most packages moved ten repositories, including `ts_m1m3_utils` from
`v0.6.2` to `v0.6.2-7-g57b153b`, after `aos_efd.duckdb` had been built with the old code.
See `value_added/docs/status/m1m3_gradient_code_change_20260929.md` for the assessment of
what that did and did not change.
