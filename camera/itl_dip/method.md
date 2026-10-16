# `itl_dip` method: finding and measuring the ITL column dip

> **Status:** current · **Last updated:** 2026-10-16 · **Kind:** reference (method)

How to reproduce the measurement in [`study.md`](study.md): the data, the ISR
configuration that keeps the dip, and how to find dips and measure their depth.
`notebooks/itl_column_dip_v1.ipynb` carries all of it as runnable code.

## What the feature is

On Imaging Technology Laboratory (ITL) Charge-Coupled Devices (CCDs) in LSSTCam, the sky
background is depressed in the columns that contain a saturated star. The deficit
extends hundreds of rows away from the star, well beyond the bleed trail, and is tens of
pixels wide. The e2v CCDs that make up the rest of the focal plane do not show it.

Columns run along the y axis of a trimmed LSSTCam detector image, so the dip is a dark
vertical band through the star. It is best measured *across* columns, as a profile in x
at a fixed offset in y from the star.

The depth is small: of order a few tens of counts against a sky of order 1400 counts in
a 30 s r-band exposure, so a few percent of sky at most. It is invisible at any ordinary
image stretch.

## Where to look

Everything below refers to the `/repo/main` Butler repository on the United States Data
Facility (USDF).

| what | where |
|---|---|
| raw frames | `LSSTCam/raw/all` |
| calibrations | `LSSTCam/calib` |
| World Coordinate System (WCS) and photometric calibration | `preliminary_visit_image` in `LSSTCam/runs/nightlyValidation/68` |
| bright-star positions and magnitudes | `the_monster_20250219` in `refcats` |

The example exposure is **2026070800022**: `day_obs` 20260708, `seq_num` 22, a `lowdust`
field, i band, 30 s. The example detector is **14 (R02_S12)**, an ITL CCD; **39
(R11_S10)** is an e2v CCD used as the control.

**Use a sparse field.** This is the single most important choice. The crowded
galactic-plane programs (`bulgy`, `dusty_plane`) supply plenty of saturated stars, but
the confusion noise between them is as large as the dip itself — a cross-column profile
wanders by tens of counts — so the depth cannot be measured. In a `lowdust` field the
sky between stars is clean and the same dips measure 50–70 counts deep against a sky of
about 1650 counts, a few percent of sky. Of 1288 science visits on 20260708–09, 985 are
`lowdust`.

Two things about collections are easy to get wrong:

- Preliminary Visit Images (PVIs) for the 2026 nights are in the **bare-numbered**
  `LSSTCam/runs/nightlyValidation/<n>` collections. The collections named by `day_obs`
  stop in September 2025. Collection `68` covers 20260708 and 20260709.
- `post_isr_image` is **not** kept for science exposures, only for `bias`, `dark`,
  `flat` and `cwfs` frames. There is no stored post-ISR product for these images.

## Run ISR yourself, with the masking off

The Science Pipelines already know about this feature and mask it. In
`lsst.ip.isr.isrTaskLSST.IsrTaskLSSTConfig`:

```python
doITLDipMask = True                           # default
itlDipMaskPlanes = ["SUSPECT", "ITL_DIP"]     # default
```

The detection logic is `lsst.ip.isr.isrFunctions.maskITLDip`, and there is a dedicated
`ITL_DIP` mask plane. Masking is right for science and wrong for studying the feature.

Two more defaults get in the way. `doSaturation` masks the saturated core, but the
saturated star is what marks the dip columns in the first place, so it stays off.
`doInterpolate` would replace pixels with fitted values.

A minimal configuration that preserves everything:

```python
config = IsrTaskLSSTConfig()
# Keep the pixels the study is about.
config.doSaturation = False
config.doITLEdgeBleedMask = False      # must be off when doSaturation is False
config.doE2VEdgeBleedMask = False      # likewise; config.validate() enforces this
config.doITLDipMask = False
config.doITLSatSagMask = False
config.doInterpolate = False
# Drop the expensive corrections; neither affects the sky level.
config.doBrighterFatter = False
config.doDeferredCharge = False
config.doCrosstalk = False
config.doLinearize = False
config.doCorrectGains = False
config.doAmpOffset = False
```

Calibrations needed: `bias`, `dark`, `flat`, `defects` and `ptc`. The PTC is required
even with `doCorrectGains = False`, because gains are read from it by default. Note the
`run()` keyword names differ from the dataset type names, for example `bfk` is passed as
`bfKernel` and `cti` as `deferredChargeCalib`.

Two practical notes:

- **Speed.** The default configuration spends nearly all its time in brighter-fatter
  (`ASTIER23` electrostatic) and deferred-charge correction. Turning those off takes one
  detector from several minutes to about **7 seconds**.
- **The per-amp gate.** `itlDipBackgroundFraction` is `0.0` in the bare config defaults,
  which disables dip masking regardless of `doITLDipMask`. The production values are set
  per detector in `obs_lsst/config/lsstCam/isrLSST.py`. So a bare `IsrTaskLSSTConfig`
  never masks dips anyway; setting the flag makes the intent explicit.

## Which detectors show it

The production configuration (ticket DM-50343) assigns dip-correction rules to **25
detectors, all ITL**, in five groups by strength. No e2v detector has a rule, which is
the pipeline's own statement that the effect is ITL specific.

| `itlDipBackgroundFraction` | `itlDipMinWidth` (pixels) | detectors |
|---|---|---|
| 0.0030 | 8 | R01_S12 |
| 0.0025 | 6 | R02_S10, R02_S12, R10_S02, R41_S21, R42_S02, R43_S11, R43_S20 |
| 0.0015 | 10 | R01_S10, R01_S11, R01_S20, R10_S10, R41_S00 |
| 0.0015 | 13 | R01_S21, R03_S20, R10_S22 |
| 0.0010 | 15 | R01_S22, R02_S02, R02_S11, R02_S21, R03_S00, R03_S02, R10_S01, R41_S01, R43_S01 |

`itlDipBackgroundFraction` is dimensionless: the dip's flux deficit at its centre,
scaled by width, over the sky level. R01_S12 has the strongest rule and is the best
place to start.

## Finding the stars

Use the **reference catalog**, not the pipeline's source catalog. The pipeline drops
stars once they saturate: on the example exposure the brightest entry in
`single_visit_star_unstandardized` is 16.3 mag, so the saturated stars this study needs
are exactly the ones missing from it. `the_monster_20250219` keeps them — its brightest
star on detector 5 is 10.2 mag.

Stars of **13 to 15 mag** are the useful range in a 30 s exposure. They saturate, with
peaks of order 2×10⁵ counts, without the full-column bleed trails of the brightest
stars.

The catalog is for identifying *which* star produced a dip, not for finding the dips.
**Do not locate dips from star positions.** The bleed trail is itself a bright column,
the dip sits beside it rather than on it, and many bright stars produce no dip at all.

Two details when loading the catalog: query only the shards overlapping the detector and
pass `with_dimension_records=True`, because `ReferenceObjectLoader` needs the shard sky
regions; and the flux fields are named `monster_ComCam_<band>_flux`, in nJy, with no
`SynthLSST` variant. Convert to AB magnitude with a zero point of 3631 Jy = 3.631×10¹²
nJy. Take the WCS from the PVI to put the catalog positions on the pixel grid.

## Finding the dips

Let the pipeline find them. Run ISR a second time with `doSaturation` and
`doITLDipMask` **on**, with the production rule for that detector installed, and read the
`ITL_DIP` mask plane:

```python
rule = copy.copy(config.overscanCamera.defaultDetectorConfig)
rule.itlDipMinWidth, rule.itlDipBackgroundFraction = 6, 0.0025   # R02_S12
config.overscanCamera.detectorRules['R02_S12'] = rule
```

Group the flagged columns into contiguous runs; each run is one dip. On detector 14 of
the example exposure this gives 9 dips, 10 to 19 pixels wide.

**Prefer a partially filled bleed trail.** Record what fraction of the column the trail
saturates and keep the ones well below 1. A trail spanning the full column leaves no
clean sky to reference the dip against. Then take the profile from rows *beyond* the end
of the trail, which also keeps the trail's ~10⁵ counts out of the measurement entirely.

## Displaying it

The stretch is what makes or breaks this. Clip around the sky, asymmetrically — further
below than above, since the feature is a deficit. For the example exposure,
`vmin = sky − 80` and `vmax = sky + 40` counts works; much tighter and the image turns
into salt and pepper. A zscale stretch hides the dip completely.

**Always put a colorbar on the image**, labelled in counts, so the depth of the feature
can be read off the display.

## Measuring the depth

Take the median over rows beyond the trail as a function of x, set a reference level
from the ends of the profile, and take the deficit inside the flagged columns. On the
example exposure this gives, for three detectors:

| detector | dip columns (x) | trail fills | depth (counts) | depth / sky |
|---|---|---|---|---|
| R43_S11 | 3480–3504 | 16% | 50.2 | 0.031 |
| R02_S12 | 892–910 | 26% | 71.8 | 0.041 |
| R01_S12 | 598–614 | 23% | 61.7 | 0.038 |

`depth / sky` is dimensionless: deficit in counts over sky level in counts. The sky is
1630–1735 counts in these cases.

For comparison, the same measurement in a crowded `bulgy` field gave 4–43 counts with no
consistency between stars — the confusion noise, not the dip.

### Further work

1. **Stack many dips.** Align on the dip columns and median-combine over dips and
   detectors to beat down the remaining structure.
2. **Correlate with the star.** Depth and width against star magnitude, and against the
   `itlDipBackgroundFraction` the production configuration assigns each detector.
3. **Check the amp dependence.** ITL amps are 509 pixels wide in x; whether depth
   depends on where in the amp the star sits is unanswered.

## Running it

`notebooks/itl_column_dip_v1.ipynb` carries all of the above as runnable code. It imports only the LSST Science Pipelines (`lsst_distrib`) and standard
scientific Python (`numpy`, `pandas`, `matplotlib`, `astropy`, `requests`), so it runs
anywhere the stack is set up.

Visit selection uses the Consolidated Database (ConsDB) through its REST interface with
`requests`, which avoids a dependency on `lsst.summit.utils` — that package is not part
of `lsst_distrib`. ConsDB has no galactic-coordinate column, so galactic latitude is
computed from the boresight position with astropy. The survey's two galactic-plane
programs are `target_name` of `bulgy` and `dusty_plane`.

One environment note: if `loadLSST.sh` fails with `Could not find conda environment:
lsst-scipipe-13.1.0-exact`, a stale `LSST_CONDA_ENV_NAME` is set in the shell. Clear it
with `unset LSST_CONDA_ENV_NAME` before sourcing the stack.
