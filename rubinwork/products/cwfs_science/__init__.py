"""The ``cwfs_science`` product: per-CCD, per-visit summaries of the corner wavefront
sensor (CWFS) donuts in every visit of a CWFS processing, not tied to Full Array Mode
(FAM).

The corner rafts R00, R04, R40 and R44 each hold an intra-focal half, SW1, and an
extra-focal half, SW0. A build reads each visit's per-donut corner table
(``donutBlitzCornerResults``) and the online per-raft ``zernikes`` tables from the
variant's Butler collections, and writes:

``ccd_visits.parquet``
    One row per (visit, corner CCD). Keys and visit context: ``day_obs``, ``seq_num``,
    ``visit``, ``detector``, ``detector_id``, ``raft``, ``wfs_offset`` (-1 intra-focal,
    0 extra-focal), ``defocal_offset_mm``, ``band``, ``physical_filter``,
    ``science_program``, ``observation_reason``, ``mjd`` (TAI, day), ``exp_time`` (s),
    and ``alt``, ``az``, ``rot_tel_pos``, ``rot_sky`` (deg). Counts: ``n_donuts`` in the
    table, ``n_quality`` passing the quality cut, ``n_mtaos`` averaged by the MTAOS
    selection. Per frame ``F`` (OCS, CCS) and Noll ``j``, in µm of wavefront:
    ``zk_F_median_z{j}`` and ``zk_F_nmad_z{j}``, median and nMAD of the total Zernike
    (deviation plus intrinsic) over the quality cut; ``zk_F_mtaos_z{j}``, the mean over
    this CCD's MTAOS donuts; and the intrinsic part, ``zk_intrinsic_F_median_z{j}`` and
    ``zk_intrinsic_F_mtaos_z{j}``. Blur in arcsec: ``blur_median``, ``blur_nmad``,
    ``blur_mtaos_median``. ``snr_median`` (dimensionless); ``thx_F_median``,
    ``thy_F_median`` (deg).
``raft_visits.parquet``
    One row per (visit, corner raft), the unit the MTAOS uses: both halves together.
    ``zk_CCS_deviation_average_z{j}`` is the online table's own average (µm, CCS, the
    value MTAOS reads); ``zk_F_mtaos_z{j}`` and ``zk_intrinsic_F_mtaos_z{j}`` are the
    mean of the same donuts' total and intrinsic Zernikes in both frames (µm);
    ``mtaos_check_um`` is the largest difference between the two in CCS deviation (µm),
    zero when the donut match is exact. Counts ``n_mtaos``, ``n_mtaos_intra``,
    ``n_mtaos_extra``, ``n_mtaos_rows`` and ``n_mtaos_unmatched``; ``blur_mtaos_median``
    (arcsec).

**Quality cut** (our selection): a donut that passed the pipeline's selection
(``candidate``), whose group fit succeeded with outcome ``ok``, and whose SNR, blur and
Zernikes are all finite. **MTAOS selection**: the donuts the ``zernikes`` table marks
``used``, matched to the blitz rows on detector and donut id. That table is written by
the blitz corner task, whose combine step (sigma clip of the deviation, then a blur
clip) sets ``used``; its settings are in the run's ``donutBlitzCornerTask_config``.

The total Zernike is kept so a study can subtract a different intrinsic. Join the ConsDB
per-corner values on ``visit`` and ``raft``.

Read a build::

    from rubinwork.products import cwfs_science
    ccd, raft = cwfs_science.load("d13-blitz_rso1071")

Build one, from the repository root::

    snakemake -s rubinwork/products/cwfs_science/Snakefile -c16 \\
        --config variant=d13-blitz_rso1071 build=20261026
"""

from .reader import (PRODUCT, TABLE_KEYS, TABLES, build_dir, load, load_table,
                     variant_config, variants)

__all__ = ["PRODUCT", "TABLE_KEYS", "TABLES", "build_dir", "load", "load_table",
           "variant_config", "variants"]
