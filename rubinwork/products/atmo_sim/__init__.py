"""The ``atmo_sim`` product: simulated atmospheric-PSF pixels for real Rubin visits.

For one real visit, one frozen-flow atmosphere (psf-weather-station layers from ERA5,
seeing matched to the visit's median tracked guider FWHM) is drawn on two sets of
detectors:

- the 8 guide sensors, one star each, at the tracked star's CCD position and flux, in
  the real region of interest (ROI), with the real stamp count and stamp times. Each
  stamp is converted to the amplifier readout frame with `rubinwork.guider_roi` and
  written in the real ``guider_raw`` file layout with the real headers, so
  ``summit_utils`` `GuiderReader` and the ``guider_moments`` builder read it unchanged;
- the 189 science CCDs, one star at each CCD centre, integrated over the visit exposure,
  in the CCD pixel frame.

The real boresight and rotator angle enter through the initial `SkyWcs`, the same one
``summit_utils`` builds for the guiders. There are no optics and no telescope motion.
Method: ``guider/guider_atmo_sim/method.md``.

**An incremental product**, partitioned by visit::

    products/atmo_sim/v1/visits/<expId>/

The guider files are also ingested in place as ``guider_raw`` into the RUN collection
``u/roodman/atmo_sim/v1/<day_obs>`` of ``/repo/main``, chained with ``LSSTCam/raw/all``
as ``u/roodman/atmo_sim/v1`` (`collection`).

Build one visit::

    python -m rubinwork.products.atmo_sim.simulate --day-obs 20260706 --seq-num 688

Read::

    from rubinwork.products import atmo_sim
    truth = atmo_sim.load_truth(20260706, 688)
"""

from .reader import (  # noqa: F401
    PRODUCT,
    PRODUCT_VARIANT,
    VISITS_DIR,
    collection,
    exp_id,
    load_atmosphere,
    load_truth,
    partition,
    run_collection,
    science_stars,
    variant_config,
    variants,
    visit_dir,
    visits,
)
