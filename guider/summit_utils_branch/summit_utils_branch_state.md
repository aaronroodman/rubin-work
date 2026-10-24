# summit_utils guider branch state

> **Status:** current · **Last updated:** 2026-10-24 · **Kind:** working state (branch state)

State of the `summit_utils` branch `u/aaronroodman/guider-fixes` against `main` and
`deploy-summit`, from read-only git in the S3DF clone
(`/sdf/group/rubin/u/roodman/LSST/packages/summit_utils`) as of fetch 2026-10-24.
(The laptop path `~/Astrophysics/Claude/packages/summit_utils` named in the 2026-10-11
version of this doc does not exist on S3DF; this is the same branch, same remote.)

## Result

`guider-fixes` contains every guider commit on `origin/deploy-summit`, the branch the
summit and RubinTV run, plus Aaron's changes. No cherry-pick is needed. `origin/main`
now **does** contain the `deploy-summit` guider commits, merged 2026-09-14 via PR #188
(`DM-54305`) — with one bug fix on top that `guider-fixes` and `deploy-summit` both
still lack. See "What changed since 2026-09-03" below.

## Refs

| ref | commit | note |
|---|---|---|
| `u/aaronroodman/guider-fixes` | `e57e0f4` (2026-09-01) | same as `origin/u/aaronroodman/guider-fixes`: pushed, unchanged since 2026-09-03 |
| fork point with `main` | `0959ea2` | local `main` is still `0959ea2` |
| `origin/main` | `b429732` (2026-10-05) | 32 commits (count) ahead of fork point `0959ea2`; now includes the `deploy-summit` guider fixes (PR #188) plus their sign-error follow-up |
| `origin/deploy-summit` | `c04ec66` | unchanged since the 2026-09-03 fetch; still has the sign-error bug that `origin/main` fixed |

## deploy-summit guider commits

`git log main..origin/deploy-summit -- python/lsst/summit/utils/guiders/` lists 8
commits by Johnny Esteves (2–4 Mar 2026) in `reading.py`, `detection.py`, `tracking.py`.
`git cherry` shows each has a patch-equivalent commit on `guider-fixes`:

| deploy-summit | guider-fixes | change |
|---|---|---|
| `0f9701d` | `9a6559a` | 10th-percentile instead of median column bias (`biasPercentile=10.0` in `GuiderReader.get()` and `processStamps()`) |
| `6f7a79b` | `c7cbf46` | per-candidate rejection diagnostics |
| `45ac07a`, `d7362ff` | `4553a7c`, `d9bda82` | flux-weighted centroid fallback, added then reverted |
| `928437f` | `279f707` | flux threshold in `isBlankImage` |
| `d4c0813` | `6aaffe7` | median absolute deviation (MAD) robust std in `isBlankImage` |
| `0e2af24` | `78b54da` | source-detection robustness |
| `f213728` | `ead1d37` | single-stamp fallback when coadd detection fails |

The other 7 commits on `deploy-summit` not on `guider-fixes` are not guider code
(Consolidated Database (ConsDB) client methods `c04ec66`, patch-equivalent to `ac8fc39`
on `guider-fixes`, and six non-guider commits: package-version data model, typing and
tests). `guider/summit_utils_branch/deploy_summit_merge_note.md` quotes the `guider-fixes`
hashes, which were the `deploy-summit` hashes before the rebase.

Correction to the 2026-09 notes: `0a211d2` (STDEVCLIP with sigma68 fallback for source
detection) is not an extra commit on `guider-fixes`; it is already on `main`.

**Update 2026-10-24:** these 8 commits (merged via PR #188, `DM-54305`, 2026-09-14) are
now on `origin/main` too, under new hashes (patch-equivalent to the `deploy-summit` and
`guider-fixes` copies). `main` also carries a follow-up fix, `a676c71` "Fix sign error in
column bias re-centering", not present on `guider-fixes` or `deploy-summit` — see below.

## Aaron's commits on guider-fixes

All committed and pushed:

- `9a09407` `e2` typo fix in the ellipticity quality cut (`applyQualityCuts` and
  `_diagnoseQualityCutRejections`).
- `03d94d3`, `8b78516` `metrics.py`: `DEFAULT_RESIDUAL_SCALE = 5.0` (trend outlier band
  in units of the residual scatter, dimensionless) and a `residualScale` parameter.
- `bcf6e47` `reading.py`: `getGuiderRawStamps` and `get` skip a missing guider detector
  and raise only if none is present. `deploy-summit` (RubinTV) still fails if any
  detector is missing.
- `2c7d1db` `detection.py`: `galsim.Image(imageArray, xmin=0, ymin=0)`, fixing a
  1-pixel centroid offset from galsim's 1-based default.
- `7d1251d`, `a12dcfd`, `9d0d1e1`, `54ed2f3`, `e57e0f4` `plotting.py`: corner-paired
  `MosaicLayout`, labels outside the circles, circular pixel mask, independent star and
  full-frame geometry (`docs/mosaic_layout.md`).
- `16a9a54`, `9f44cd9` `plotting.py`: `stripPlot` Gaussian slow-curve overlay,
  `smoothFwhmSec = 1.6` s (`guider/centroid_timescale/centroid_motion_timescale.md`).

## Not done

- `processStamps` still uses the 10th-percentile column bias; the blank-2-stamp bias and
  streak correction are not folded in (`guider/bias_streak/bias_streak_plan.md`).
- A pull request `guider-fixes` → `main` had not been opened as of 2026-09. `main` has
  since caught up to the guider fixes on its own (PR #188, see below), but the open
  items under "What changed since 2026-09-03" still need a PR from `guider-fixes`.

## What changed since 2026-09-03

Checked by `git fetch --all --prune` on 2026-10-24 in
`/sdf/group/rubin/u/roodman/LSST/packages/summit_utils`. Fetch succeeded (SSH, no
credential prompt); only remote-tracking refs and tags moved, nothing in the working
tree. `guider-fixes` itself is unchanged, still at `e57e0f4` and in sync with its
remote (0 commits (count) ahead or behind `origin/u/aaronroodman/guider-fixes`).

| ref | ahead of `origin/main` | behind `origin/main` | merged into `origin/main`? |
|---|---|---|---|
| `u/aaronroodman/guider-fixes` | 21 commits (count) | 32 commits (count) | no (tip `e57e0f4` is not an ancestor of `origin/main`) |
| `origin/deploy-summit` | 15 commits (count) | 32 commits (count) | no (tip `c04ec66` is not an ancestor of `origin/main`) |

`origin/main` moved `0959ea2` &rarr; `b429732`, 32 commits (count) ahead of the fork
point, over two merged pull requests:

- **PR #188** (`DM-54305`, merged 2026-09-14): the same 8 guider-detection commits
  already on `deploy-summit`/`guider-fixes` (percentile column bias, MAD `isBlankImage`,
  flux-weighted centroid tried and reverted, per-candidate rejection diagnostics,
  single-stamp fallback, source-detection robustness), under new hashes, patch-equivalent
  to the `guider-fixes` copies — plus one new commit not on either branch:
  - `a676c71` "Fix sign error in column bias re-centering" (2026-09-09): the column-bias
    re-centering term in `reading.py` was subtracted instead of added, doubling the
    residual instead of canceling it. `guider-fixes` and `deploy-summit` both still carry
    the pre-fix sign (`- (rPerc - medianValue)`); `origin/main` has the fix
    (`+ (rPerc - medianValue)`). This is a real bug still live on both `guider-fixes` and
    `deploy-summit`.
- **PR #193** (`DM-55094`, merged 2026-10-05) and other small PRs: ConsDB package-version
  data model, typing/mypy coverage, test fixes — not guider code, none touch
  `guiders/`, `reading.py`, `detection.py`, `tracking.py`, or `consdbClient.py` in a way
  that conflicts with `guider-fixes`.

Two commits on `origin/main` since the fork point touch files under `guiders/` but are
typing/formatting only, not logic (`4b2feea` "Annotate the last unannotated definitions,
reaching 100% mypy coverage", `c4d8d98` "Take the exposure id off DriftResult.__str__
and require it in summary") — neither is patch-equivalent to anything on `guider-fixes`,
and neither touches the bias or detection logic.

The `e2`-typo fix in `applyQualityCuts`/`_diagnoseQualityCutRejections`
(`tracking.py`, `guider-fixes` commit `9a09407`) is still open upstream: `origin/main`
and `origin/deploy-summit` both still read `(stars["e1"].abs() <= maxEllipticity) &
(stars["e1"].abs() <= maxEllipticity)`, testing `e1` twice.

**Net:** `guider-fixes` is still at or ahead of both `main` and `deploy-summit` for the
detection-robustness work, but it is now the only one of the three missing the sign-error
fix (`a676c71`). Worth pulling that fix into `guider-fixes` before any PR back to `main`.
