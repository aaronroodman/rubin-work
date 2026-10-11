# summit_utils guider branch state

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** working state (branch state)

State of the `summit_utils` branch `u/aaronroodman/guider-fixes` against `main` and
`deploy-summit`, from read-only git in the local clone
(`~/Astrophysics/Claude/packages/summit_utils`) on 2026-10-11. The clone's last fetch
was 2026-09-03, so remote-tracking refs are as of that date.

## Result

`guider-fixes` contains every guider commit on `origin/deploy-summit`, the branch the
summit and RubinTV run, plus Aaron's changes. No cherry-pick is needed. `origin/main`
contains none of the `deploy-summit` guider commits.

## Refs

| ref | commit | note |
|---|---|---|
| `u/aaronroodman/guider-fixes` | `e57e0f4` (2026-09-01) | same as `origin/u/aaronroodman/guider-fixes`: pushed |
| fork point with `main` | `0959ea2` | local `main` is still `0959ea2` |
| `origin/main` | `f50eea2` | 16 commits past `0959ea2`; none are the `deploy-summit` guider commits |
| `origin/deploy-summit` | `c04ec66` | force-updated (rebased) at the 2026-09-03 fetch |

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
tests). `docs/deploy_summit_merge_note.md` quotes the `guider-fixes`
hashes, which were the `deploy-summit` hashes before the rebase.

Correction to the 2026-09 notes: `0a211d2` (STDEVCLIP with sigma68 fallback for source
detection) is not an extra commit on `guider-fixes`; it is already on `main`.

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
  `smoothFwhmSec = 1.6` s (`docs/centroid_motion_timescale.md`).

## Not done

- `processStamps` still uses the 10th-percentile column bias; the blank-2-stamp bias and
  streak correction are not folded in (`docs/status/bias_streak_plan.md`).
- A pull request `guider-fixes` → `main` had not been opened as of 2026-09. One would bring `main` up to
  the summit's guider code plus these changes.
