"""The ``giant_donuts`` product: giant-donut sides, their stamps and nominal Zernike fits.

Giant donuts are exposures with M2 z and camera z both stepped ±4 mm in Trim from the
in-focus state, so a donut is about 340 pixels in outer radius and resolves the pupil
edges. One variant per danish mask and fit configuration (`variants.yaml`).

A build holds:

``exposures.parquet``
    one row per exposure: ``visit``, ``day_obs``, ``seq_num``, ``side`` (intra or
    extra), ``band``, ``step_mm`` (mm), ``rtp_deg`` (rotator angle the fits used, deg),
    and the hexapod Trim ``dof0``-``dof9`` from ``value_added`` (µm; deg for the tilts
    3, 4, 8, 9).
``sides.parquet``
    one row per donut side: ``key`` (``<visit>_<star>``), ``star`` (reference-catalog
    id), ``night``, ``visit``, ``detector``, ``sensor_class`` (science, SW0, SW1),
    ``side``, ``field_x_ccs_deg``,
    ``field_y_ccs_deg``, ``field_r_deg``, ``field_az_ocs_deg`` (deg), ``rtp_deg``,
    ``r_out_pix`` (outer radius used for the stamp, pixels), ``npix`` (stamp size),
    ``sky_e`` (electrons per pixel), the selection flags, and ``source_run`` and
    ``source_key`` (the ``aos/giant_donut_pupil`` run and key it came from).
``fits.parquet``
    fit c per side: ``flux_e`` (electrons), ``dx_pix``, ``dy_pix`` (pixels),
    ``z<j>_ccs_um`` (µm of wavefront, CCS, for the variant's ``fit_noll``), ``chi2_dof``
    with ``dof``, ``noise_e`` (electrons per pixel), and the fit-a donut center
    ``cx_a``, ``cy_a`` (stamp pixels).
``stamps/<key>.fits``
    primary header (key, visit, detector, side, field, rotator angle), ``IMAGE``
    (electrons per pixel, float32, CCS pixels, origin lower left) and ``MASK`` (uint8, 1
    masked).
``masks/``
    the danish mask files (``RubinObsc_<mask>_r_det.._cam.._m2...yaml``) of the build.

Read a build::

    from rubinwork.products import giant_donuts
    t = giant_donuts.load("v402s_fitc")
    img, msk, hdr = giant_donuts.stamp(t["sides"].key[0], "v402s_fitc")

An import build, from the repository root, takes the ``aos/giant_donut_pupil`` runs the
variant lists::

    python -m rubinwork.products.giant_donuts.builders.import_study_runs --variant v402s_fitc
"""

from .reader import (PRODUCT, TABLES, build_dir, load, mask_name, mask_params, stamp,
                     variant_config, variants)

__all__ = ["PRODUCT", "TABLES", "build_dir", "load", "mask_name", "mask_params", "stamp",
           "variant_config", "variants"]
