"""Read Danish 1.3 "blitz" unpaired Full Array Mode (FAM) wavefront results and
recast them into the paired Danish 1.2 donut-table schema.

The Danish 1.3 blitz pipeline writes one ``donutBlitzFamResults`` table per FAM visit,
holding **one row per donut per exposure** ("unpaired"): both the intra-focal and the
extra-focal exposure of the FAM pair appear in the same table, distinguished by the
``visit_id`` column. The Danish 1.2 analysis chain that follows — the Double Zernike (DZ)
fit in `lsst.ts.intrinsic.wavefront.dz_fitting`, the Measured Intrinsic Wavefront (MIW)
build, the coadds — expects the **paired** schema instead: one row per donut, carrying a
single wavefront estimate plus the per-side quantities in ``*_intra`` / ``*_extra``
columns. This module bridges the two.

Pairing
-------
The two exposures are joined on ``(det_name, donut_id)``, where ``donut_id`` is the Gaia
source identifier of the star. That key is exact and needs no tolerance: within one side
of focus it is unique (verified per visit by `pair_sides`), and a paired star has
**identical** ``coord_ra`` / ``coord_dec`` on the two sides, to 0.0 arcsec. Positional
matching on detector pixel coordinates is deliberately *not* used here — the same star
lands up to tens of pixels apart on the two sides of focus, because the two exposures
point slightly differently, so a pixel tolerance tight enough to be safe would reject
real pairs.

The paired row's wavefront is the **arithmetic mean of the two unpaired sides**, in
micrometres of wavefront. That is the like-for-like counterpart of a Danish 1.2 joint
intra+extra fit, established in `notebooks/fam_processing/blitz_vs_danish12_20260315.ipynb`
section 8.1: against Danish 1.2 the median Pearson r over the 21 fitted Noll indices rises
from 0.7556 (intra alone) and 0.7430 (extra alone) to 0.9082 for the mean, and the
normalized median absolute deviation (nMAD) of the difference falls from 0.0321 and
0.0291 to 0.0114 micrometres of wavefront, on n = 2339 stars. Every other paired scalar
is likewise the mean of its two sides, which is exactly what Danish 1.2 does: on the
Danish 1.2 table ``thx_OCS``, ``thy_OCS``, ``centroid_x``, ``centroid_y`` and ``snr`` all
reproduce ``0.5 * (intra + extra)`` to 0.0 in their own units.

So one output row means: **one star, observed on both sides of focus in one FAM pair, with
the wavefront estimate averaged over the two sides.**

An unpaired mode is also offered (`pair_sides` with ``mode='intra'`` or ``'extra'``),
which emits one row per donut from a single side and fills both the ``*_intra`` and
``*_extra`` columns from it. That keeps the schema and the DZ fitter working, at the cost
of the side-dependent bias documented in the notebook above — the intra and extra
estimates of Z11 Spherical and Z14 Tetrafoil_x sit on opposite sides of the Danish 1.2
value. Use it only to study that bias, not to build a calibration.

Zernike vectors
---------------
The blitz ``zk_deviation_*`` vector has 27 entries and ``zk_intrinsic_*`` has 67, both
0-indexed so that column ``j`` is Noll ``j``. The Danish 1.2 schema carries only the 21
fitted Noll indices given by the table metadata ``noll_indices`` — [4..19, 22..26] — so
both blitz vectors are subset down to those 21 columns, in that order. The differing
source lengths are therefore not a problem: the selection is by Noll index, not by
position in the source array, and the selected index set is asserted against the
metadata. Within the source arrays the unfitted entries use two encodings — identically
zero at Noll 0-3, all-NaN at Noll 20 and 21 — and neither survives the selection.

The blitz product supplies the deviation and the intrinsic but no total wavefront, while
the Danish 1.2 schema carries all three. The total is reconstructed as
``zk = zk_deviation + zk_intrinsic``, which holds on the Danish 1.2 table to
1.04e-07 micrometres of wavefront (checked by `check_zk_sum_relation`).

Images
------
The per-donut image columns are **not** written to the output parquet. On the corner
product these are ``stamp``, ``wf_img`` and ``model_img``, each an 83x83 or larger pixel
array per donut; carrying them would multiply the file size by orders of magnitude for
data the DZ fit never reads. The FAM product as delivered does not carry them at all, so
the exclusion is defensive rather than active.

Portability
-----------
This module is written to be liftable into the ``ts_intrinsic_wavefront`` package
(``lsst.ts.intrinsic.wavefront``) unchanged. It depends only on `numpy`, `pyarrow`,
`astropy` and `lsst.daf.butler`, takes every path and identifier as an argument, reads no
configuration file, and imports nothing from this repository. The
repository-specific parts — resolving a ``param_set`` from ``param_sets.yaml``, resolving
the output directory, querying the Consolidated Database (ConsDB) — live in the
`run_blitz_mktable.py` script beside it.

Notes
-----
Reading a blitz table **requires** ``parameters={'strip_astropy_meta_yaml': False}``. The
Butler formatter otherwise strips the astropy metadata and ``.meta`` comes back with 0
keys, so ``noll_indices``, ``intra_visit_id``, ``extra_visit_id``, ``rot_tel_pos`` and the
input provenance all silently vanish. `read_blitz_visit` always passes it.

``instrument='LSSTCam'`` must also be passed to ``butler.get``, or the required
``instrument`` dimension is unresolved and a ``DimensionNameError`` is raised.

The output parquet is written with **one row group per visit**, carrying min/max
statistics on ``day_obs`` and ``seq_num``. The streaming DZ fitter locates a visit purely
from those row-group statistics and **silently skips** any visit it cannot find, so a file
written without per-visit row groups yields an empty fit table with no error. See
`BlitzDonutWriter`.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Radians to arcseconds, matching the literal 206265 used by
# lsst.ts.intrinsic.wavefront.intrinsics_lib for the intra/extra offset columns, so the
# two pipelines' `intra_extra_offset_*_arcsec` values are directly comparable.
RAD_TO_ARCSEC = 206265.0

# The dataset type and the frames the blitz product provides.
BLITZ_FAM_DATASET = 'donutBlitzFamResults'

# Per-donut image columns that must never reach the output parquet; see the module
# docstring. Checked rather than assumed, so a future blitz version that adds them to the
# FAM product does not silently inflate the file.
IMAGE_COLUMNS = ('stamp', 'wf_img', 'model_img')

# Noll indices the Danish 1.2 schema carries, and the blitz metadata `noll_indices`
# should equal. Asserted per visit rather than trusted.
EXPECTED_NOLL_INDICES = tuple(list(range(4, 20)) + list(range(22, 27)))

# Column order of the output donut table: the Danish 1.2 `donuts.parquet` schema, in its
# own order, so a reader of either table sees the same layout.
DONUT_COLUMNS = (
    'zk_CCS', 'zk_intrinsic_CCS', 'zk_deviation_CCS',
    'used', 'detector', 'extra_donut_id', 'intra_donut_id',
    'zk_OCS', 'zk_intrinsic_OCS', 'zk_deviation_OCS',
    'coord_ra', 'coord_dec', 'centroid_x', 'centroid_y',
    'thx_CCS', 'thy_CCS', 'thx_OCS', 'thy_OCS', 'snr',
    'centroid_x_intra', 'centroid_x_extra', 'centroid_y_intra', 'centroid_y_extra',
    'thx_CCS_intra', 'thx_CCS_extra', 'thy_CCS_intra', 'thy_CCS_extra',
    'thx_OCS_intra', 'thx_OCS_extra', 'thy_OCS_intra', 'thy_OCS_extra',
    'snr_intra', 'snr_extra', 'donut_id_intra', 'donut_id_extra',
    'seq_num', 'day_obs', 'blur', 'chi2',
    'lstsq_cost', 'lstsq_optimality', 'lstsq_status',
    'model_dx', 'model_dy', 'model_flux',
    'matched_intra_extra',
    'intra_extra_offset_x_arcsec', 'intra_extra_offset_y_arcsec',
)

# Per-side columns this reader adds beyond the Danish 1.2 schema, so a question about one
# side of focus does not require another Butler pass over the blitz collection. The
# paired columns above keep their names and their meaning (the mean of the two sides), so
# a Danish 1.2 reader is unaffected; these are additions, not replacements.
#
# `blur_intra` / `blur_extra` are the per-side `group_fwhm` in arcsec, the quantity the
# paired `blur` averages away. The Zernike vectors are the per-side counterparts of
# `zk_*`: the same 21 fitted Noll terms in micrometres of wavefront, before the mean.
SIDE_COLUMNS = (
    'blur_intra', 'blur_extra',
    'lstsq_cost_intra', 'lstsq_cost_extra',
    'lstsq_optimality_intra', 'lstsq_optimality_extra',
    'zk_OCS_intra', 'zk_OCS_extra',
    'zk_intrinsic_OCS_intra', 'zk_intrinsic_OCS_extra',
    'zk_deviation_OCS_intra', 'zk_deviation_OCS_extra',
    'zk_CCS_intra', 'zk_CCS_extra',
    'zk_intrinsic_CCS_intra', 'zk_intrinsic_CCS_extra',
    'zk_deviation_CCS_intra', 'zk_deviation_CCS_extra',
)


def _bare(column):
    """Strip an `astropy.units.Quantity` down to a plain float array.

    Parameters
    ----------
    column : `astropy.table.Column` or `astropy.units.Quantity` or `array_like`
        Column whose values are wanted without their unit. No unit conversion is
        performed, so the result carries whatever unit the source had.

    Returns
    -------
    values : `numpy.ndarray`
        The values as `float`, in the source column's own unit.

    Notes
    -----
    Used on ``thx_ocs`` / ``thy_ocs``, which the blitz product stores as a `Quantity` in
    **radians**. The streaming DZ fitter applies `numpy.rad2deg` to whatever it reads, so
    the stored value must stay in radians — this must not be followed by a
    ``.to(u.deg)``, or the field angles are converted twice.
    """
    values = getattr(column, 'value', column)
    return np.asarray(values, dtype=float)


def read_blitz_visit(butler, visit, collections, dataset_type=BLITZ_FAM_DATASET):
    """Read one blitz FAM result table, metadata included.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler on the repository holding the blitz collection.
    visit : `int`
        Visit identifier of the FAM **extra-focal** exposure, which is the key the blitz
        FAM dataset is registered under.
    collections : `str` or `list` of `str`
        Butler collection(s) to search.
    dataset_type : `str`, optional
        Dataset type to read. Defaults to ``donutBlitzFamResults``.

    Returns
    -------
    table : `astropy.table.Table`
        The blitz table, with ``.meta`` populated.

    Notes
    -----
    Passes ``parameters={'strip_astropy_meta_yaml': False}``, without which ``.meta``
    comes back with 0 keys, and ``instrument='LSSTCam'``, without which the required
    ``instrument`` dimension is unresolved.
    """
    return butler.get(dataset_type, collections=collections, instrument='LSSTCam',
                      visit=int(visit),
                      parameters={'strip_astropy_meta_yaml': False})


def noll_selection(meta):
    """Column indices into a blitz Zernike vector for the fitted Noll indices.

    Parameters
    ----------
    meta : `dict`
        The blitz table's ``.meta``, which must carry ``noll_indices``.

    Returns
    -------
    noll : `numpy.ndarray`
        The fitted Noll indices, dimensionless integers, in the order the Danish 1.2
        schema uses them.
    select : `numpy.ndarray`
        Integer column indices into a blitz ``zk_deviation_*`` or ``zk_intrinsic_*``
        vector selecting those Noll terms. The blitz vectors are 0-indexed on Noll, so
        this equals `noll`; it is returned separately so the indexing convention is
        stated once rather than assumed at every call site.

    Raises
    ------
    KeyError
        If ``noll_indices`` is absent, which means the table was read without
        ``strip_astropy_meta_yaml=False``.
    ValueError
        If the metadata's index set differs from the 21 the Danish 1.2 schema carries, in
        which case the output would not be schema-compatible.
    """
    if 'noll_indices' not in meta:
        raise KeyError(
            "blitz metadata has no 'noll_indices' — the table was read without "
            "parameters={'strip_astropy_meta_yaml': False}, so .meta is empty")
    noll = np.asarray([int(j) for j in meta['noll_indices']], dtype=int)
    if tuple(noll.tolist()) != EXPECTED_NOLL_INDICES:
        raise ValueError(
            f'blitz noll_indices {noll.tolist()} differs from the '
            f'{len(EXPECTED_NOLL_INDICES)} indices the Danish 1.2 schema carries, '
            f'{list(EXPECTED_NOLL_INDICES)}; the '
            'output would not be schema-compatible')
    return noll, noll.copy()


def focus_side(table, meta):
    """Per-row side of focus, from the ``visit_id`` column.

    Parameters
    ----------
    table : `astropy.table.Table`
        A blitz table.
    meta : `dict`
        Its ``.meta``, which must carry ``intra_visit_id`` and ``extra_visit_id``.

    Returns
    -------
    is_intra : `numpy.ndarray`
        Boolean, True where the row comes from the intra-focal exposure.
    is_extra : `numpy.ndarray`
        Boolean, True where the row comes from the extra-focal exposure.

    Raises
    ------
    ValueError
        If any row's ``visit_id`` is neither of the two, or if the side implied by
        ``visit_id`` disagrees with the sign of the varying ``defocal_offsets`` element.

    Notes
    -----
    ``visit_id`` is the direct indicator and is used as the answer. The sign of
    ``defocal_offsets`` is an independent second route — negative offset is intra-focal —
    and the two are cross-checked, because ``defocal_offsets`` is a 3-vector whose element
    order is undocumented and whose *varying* element is not the same in the two blitz
    products (the FAM product varies element 1, the corner product element 0). Relying on
    a hardcoded element index would silently mislabel the side of focus on one of them.
    """
    visit_id = np.asarray(table['visit_id'], dtype=np.int64)
    intra_id = int(meta['intra_visit_id'])
    extra_id = int(meta['extra_visit_id'])
    is_intra = visit_id == intra_id
    is_extra = visit_id == extra_id
    n_other = int((~(is_intra | is_extra)).sum())
    if n_other:
        raise ValueError(
            f'{n_other} rows carry a visit_id that is neither intra_visit_id '
            f'{intra_id} nor extra_visit_id {extra_id}')

    offsets = np.asarray(_bare(table['defocal_offsets']), dtype=float)
    if offsets.ndim != 2:
        raise ValueError(f'defocal_offsets should be a per-row vector, got shape '
                         f'{offsets.shape}')
    varying = [k for k in range(offsets.shape[1])
               if len(np.unique(offsets[:, k])) > 1]
    if len(varying) != 1:
        raise ValueError(f'expected exactly one varying defocal_offsets element, '
                         f'found elements {varying}')
    offset = offsets[:, varying[0]]
    if not np.all(offset[is_intra] < 0.0):
        raise ValueError('rows labelled intra-focal by visit_id do not all have a '
                         'negative defocal offset')
    if not np.all(offset[is_extra] > 0.0):
        raise ValueError('rows labelled extra-focal by visit_id do not all have a '
                         'positive defocal offset')
    return is_intra, is_extra


def pair_sides(table, meta, mode='mean'):
    """Join the two sides of focus of one unpaired blitz table.

    Parameters
    ----------
    table : `astropy.table.Table`
        One visit's blitz FAM table, all rows.
    meta : `dict`
        Its ``.meta``.
    mode : {'mean', 'intra', 'extra'}, optional
        ``'mean'`` (default) keeps only stars fitted successfully on **both** sides and
        returns the index pair for each; ``'intra'`` or ``'extra'`` keeps every
        successfully fitted star on that one side and returns the same index twice, so
        the paired schema is filled from a single exposure.

    Returns
    -------
    idx_intra : `numpy.ndarray`
        Integer row indices into `table` for the intra-focal member of each pair.
    idx_extra : `numpy.ndarray`
        Integer row indices into `table` for the extra-focal member, parallel to
        `idx_intra`. In the single-side modes both arrays are the same rows.
    stats : `dict`
        Counts, all dimensionless row counts: ``n_rows``, ``n_fit_ok``, ``n_intra_ok``,
        ``n_extra_ok``, ``n_paired``, plus ``max_coord_sep_arcsec``, the largest on-sky
        separation between the two members of a pair, in arcsec.

    Raises
    ------
    ValueError
        If ``(det_name, donut_id)`` is not unique within a side of focus, which would
        make the join ambiguous, or if a paired star's two rows disagree on ``coord_ra``
        or ``coord_dec`` by more than 1e-6 arcsec.

    Notes
    -----
    ``group_fit_success`` is the selection flag; it is exactly the set of rows with a
    finite deviation vector on all 21 fitted Noll indices, which `donut_table` asserts.
    """
    if mode not in ('mean', 'intra', 'extra'):
        raise ValueError(f"mode must be 'mean', 'intra' or 'extra', got {mode!r}")

    is_intra, is_extra = focus_side(table, meta)
    fit_ok = np.asarray(table['group_fit_success'], dtype=bool)
    det = np.asarray(table['det_name'], dtype=str)
    donut_id = np.asarray(table['donut_id'], dtype=np.int64)

    rows_intra = np.where(is_intra & fit_ok)[0]
    rows_extra = np.where(is_extra & fit_ok)[0]
    stats = {
        'n_rows': int(len(table)),
        'n_fit_ok': int(fit_ok.sum()),
        'n_intra_ok': int(len(rows_intra)),
        'n_extra_ok': int(len(rows_extra)),
    }

    if mode != 'mean':
        rows = rows_intra if mode == 'intra' else rows_extra
        stats['n_paired'] = int(len(rows))
        stats['max_coord_sep_arcsec'] = 0.0
        return rows, rows, stats

    def _key_map(rows, label):
        keys = list(zip(det[rows].tolist(), donut_id[rows].tolist()))
        if len(set(keys)) != len(keys):
            raise ValueError(
                f'(det_name, donut_id) is not unique among the {len(rows)} fitted '
                f'{label}-focal rows, so the two sides cannot be joined on it')
        return dict(zip(keys, rows.tolist()))

    map_intra = _key_map(rows_intra, 'intra')
    map_extra = _key_map(rows_extra, 'extra')
    common = sorted(set(map_intra) & set(map_extra))
    idx_intra = np.array([map_intra[k] for k in common], dtype=int)
    idx_extra = np.array([map_extra[k] for k in common], dtype=int)
    stats['n_paired'] = int(len(common))

    # The pairing key is a catalogue source id, so a correct join puts the same sky
    # position on both sides.  Verify rather than assume.
    max_sep = 0.0
    if len(common):
        ra = _bare(table['coord_ra'])
        dec = _bare(table['coord_dec'])
        d_ra = np.abs(ra[idx_intra] - ra[idx_extra]) * 3600.0
        d_dec = np.abs(dec[idx_intra] - dec[idx_extra]) * 3600.0
        max_sep = float(max(d_ra.max(), d_dec.max()))
        if max_sep > 1.0e-6:
            raise ValueError(
                f'paired rows disagree on sky position by up to {max_sep:.3e} arcsec; '
                'the (det_name, donut_id) join is not matching the same star')
    stats['max_coord_sep_arcsec'] = max_sep
    return idx_intra, idx_extra, stats


def donut_table(table, meta, day_obs, seq_num, mode='mean'):
    """Build one visit's paired donut rows in the Danish 1.2 schema.

    Parameters
    ----------
    table : `astropy.table.Table`
        One visit's blitz FAM table, all rows.
    meta : `dict`
        Its ``.meta``.
    day_obs : `int`
        Observing night of the FAM extra-focal exposure, as ``YYYYMMDD``.
    seq_num : `int`
        Sequence number of the FAM extra-focal exposure, dimensionless.
    mode : {'mean', 'intra', 'extra'}, optional
        Pairing mode, passed to `pair_sides`.

    Returns
    -------
    arrow_table : `pyarrow.Table`
        One row per paired donut, columns exactly `DONUT_COLUMNS` followed by
        `SIDE_COLUMNS`. Zernike columns are ``list<double>`` of the 21 fitted Noll terms
        in micrometres of wavefront; field angles ``thx_*`` / ``thy_*`` are `double` in
        **radians**; centroids are `double` in detector pixels of the binned image;
        ``blur``, ``blur_intra`` and ``blur_extra`` are `double` in arcsec.
    stats : `dict`
        The counts from `pair_sides`, plus ``n_donuts`` (the output row count,
        dimensionless) and ``n_detectors`` (distinct detectors present, dimensionless).

    Raises
    ------
    ValueError
        If an image column is present in the source (see `IMAGE_COLUMNS`), if
        ``group_fit_success`` is not exactly the finite-deviation set, or if the reported
        Zernike units are not micrometres.

    Notes
    -----
    Beyond the Danish 1.2 schema the table also carries `SIDE_COLUMNS`: the per-side
    ``blur``, fit-cost and Zernike vectors, before the mean over the two sides. They
    exist so a side-of-focus question — the intra/extra blur difference, or the
    side-dependent Zernike bias — is answerable from the parquet alone.

    Columns the blitz product has no counterpart for are filled with NaN rather than
    guessed: ``chi2``, ``lstsq_status``, ``model_dx``, ``model_dy``, ``model_flux``. The
    blitz ``group_fit_cost`` and ``group_fit_optimality`` are per fit *group* rather than
    per donut, so they populate ``lstsq_cost`` and ``lstsq_optimality`` under their
    Danish 1.2 names but mean the group's value; on the visits examined every group is a
    singleton, which will stop being true once blended donuts are fitted jointly.
    """
    present = [c for c in IMAGE_COLUMNS if c in table.colnames]
    if present:
        raise ValueError(
            f'per-donut image columns {present} are present in the source table and must '
            'not be written to the donut parquet; drop them before calling donut_table')

    noll, select = noll_selection(meta)
    idx_i, idx_e, stats = pair_sides(table, meta, mode=mode)

    # Zernikes.  Both blitz vectors are 0-indexed on Noll, so `select` picks the fitted
    # terms out of either the 27-long deviation or the 67-long intrinsic.
    for name in ('zk_deviation_ocs', 'zk_intrinsic_ocs',
                 'zk_deviation_ccs', 'zk_intrinsic_ccs'):
        unit = getattr(table[name], 'unit', None)
        if unit is not None and str(unit) not in ('micron', 'um'):
            raise ValueError(f'{name} carries unit {unit!r}, expected micron of wavefront')

    dev_ocs_all = _bare(table['zk_deviation_ocs'])[:, select]
    int_ocs_all = _bare(table['zk_intrinsic_ocs'])[:, select]
    dev_ccs_all = _bare(table['zk_deviation_ccs'])[:, select]
    int_ccs_all = _bare(table['zk_intrinsic_ccs'])[:, select]

    fit_ok = np.asarray(table['group_fit_success'], dtype=bool)
    dev_finite = np.isfinite(dev_ocs_all).all(axis=1)
    n_disagree = int((fit_ok != dev_finite).sum())
    if n_disagree:
        raise ValueError(
            f'group_fit_success disagrees with the finite-deviation set on {n_disagree} '
            'rows, so it is not a safe selection flag for this visit')

    def _mean(values):
        """Mean of the two sides, which is the identity in the single-side modes."""
        return 0.5 * (values[idx_i] + values[idx_e])

    dev_ocs = _mean(dev_ocs_all)
    int_ocs = _mean(int_ocs_all)
    dev_ccs = _mean(dev_ccs_all)
    int_ccs = _mean(int_ccs_all)

    thx_ocs = _bare(table['thx_ocs'])
    thy_ocs = _bare(table['thy_ocs'])
    thx_ccs = _bare(table['thx_ccs'])
    thy_ccs = _bare(table['thy_ccs'])
    x_det = _bare(table['x_det'])
    y_det = _bare(table['y_det'])
    snr = _bare(table['snr'])
    det = np.asarray(table['det_name'], dtype=str)
    donut_id = np.asarray(table['donut_id'], dtype=np.int64).astype(str)

    n_out = len(idx_i)
    nan_scalar = np.full(n_out, np.nan)
    # Danish 1.2 stores intra_extra_offset_* as the ABSOLUTE field-angle difference in
    # arcsec, using the same 206265 rad-to-arcsec factor.
    off_x = np.abs(thx_ocs[idx_i] - thx_ocs[idx_e]) * RAD_TO_ARCSEC
    off_y = np.abs(thy_ocs[idx_i] - thy_ocs[idx_e]) * RAD_TO_ARCSEC

    def _rows(matrix):
        """Per-row list column for pyarrow, matching the Danish 1.2 list<double>."""
        return pa.array([np.ascontiguousarray(matrix[r]) for r in range(matrix.shape[0])],
                        type=pa.list_(pa.float64()))

    columns = {
        # The blitz product has no total wavefront; reconstruct it as the sum, the
        # relation the Danish 1.2 table satisfies to 1.04e-07 micrometres of wavefront.
        'zk_CCS': _rows(dev_ccs + int_ccs),
        'zk_intrinsic_CCS': _rows(int_ccs),
        'zk_deviation_CCS': _rows(dev_ccs),
        # Every output row passed group_fit_success, so `used` is True by construction —
        # the same meaning Danish 1.2 gives it after run_mktable's selection.
        'used': pa.array(np.ones(n_out, dtype=bool)),
        'detector': pa.array(det[idx_e]),
        'extra_donut_id': pa.array(donut_id[idx_e]),
        'intra_donut_id': pa.array(donut_id[idx_i]),
        'zk_OCS': _rows(dev_ocs + int_ocs),
        'zk_intrinsic_OCS': _rows(int_ocs),
        'zk_deviation_OCS': _rows(dev_ocs),
        'coord_ra': pa.array(_bare(table['coord_ra'])[idx_e]),
        'coord_dec': pa.array(_bare(table['coord_dec'])[idx_e]),
        'centroid_x': pa.array(_mean(x_det)),
        'centroid_y': pa.array(_mean(y_det)),
        'thx_CCS': pa.array(_mean(thx_ccs)),
        'thy_CCS': pa.array(_mean(thy_ccs)),
        'thx_OCS': pa.array(_mean(thx_ocs)),
        'thy_OCS': pa.array(_mean(thy_ocs)),
        'snr': pa.array(_mean(snr)),
        'centroid_x_intra': pa.array(x_det[idx_i]),
        'centroid_x_extra': pa.array(x_det[idx_e]),
        'centroid_y_intra': pa.array(y_det[idx_i]),
        'centroid_y_extra': pa.array(y_det[idx_e]),
        'thx_CCS_intra': pa.array(thx_ccs[idx_i]),
        'thx_CCS_extra': pa.array(thx_ccs[idx_e]),
        'thy_CCS_intra': pa.array(thy_ccs[idx_i]),
        'thy_CCS_extra': pa.array(thy_ccs[idx_e]),
        'thx_OCS_intra': pa.array(thx_ocs[idx_i]),
        'thx_OCS_extra': pa.array(thx_ocs[idx_e]),
        'thy_OCS_intra': pa.array(thy_ocs[idx_i]),
        'thy_OCS_extra': pa.array(thy_ocs[idx_e]),
        'snr_intra': pa.array(snr[idx_i]),
        'snr_extra': pa.array(snr[idx_e]),
        'donut_id_intra': pa.array(donut_id[idx_i]),
        'donut_id_extra': pa.array(donut_id[idx_e]),
        'seq_num': pa.array(np.full(n_out, int(seq_num), dtype=np.int64)),
        'day_obs': pa.array(np.full(n_out, int(day_obs), dtype=np.int64)),
        # group_fwhm is the blitz analogue of the Danish 1.2 per-donut `blur`, in arcsec.
        # The two do not track star by star (Pearson r = 0.208, Spearman rho = 0.235,
        # n = 2339 stars, for the mean of the two sides against the Danish 1.2 blur), so
        # it feeds the per-visit median_blur_arcsec quality metric but is not a
        # substitute for `blur` in any per-donut analysis.
        'blur': pa.array(_mean(_bare(table['group_fwhm']))),
        # No per-donut fit-quality scalar comparable to chi2 exists in the blitz product.
        'chi2': pa.array(nan_scalar),
        'lstsq_cost': pa.array(_mean(_bare(table['group_fit_cost']))),
        'lstsq_optimality': pa.array(_mean(_bare(table['group_fit_optimality']))),
        'lstsq_status': pa.array(np.full(n_out, -1, dtype=np.int64)),
        'model_dx': _rows(np.full((n_out, 2), np.nan)),
        'model_dy': _rows(np.full((n_out, 2), np.nan)),
        'model_flux': _rows(np.full((n_out, 2), np.nan)),
        # Every row here is a genuine intra+extra join in 'mean' mode, and a single
        # exposure reused in the single-side modes; either way it is matched by
        # construction, which is what Danish 1.2 records when its threshold is disabled.
        'matched_intra_extra': pa.array(np.ones(n_out, dtype=bool)),
        'intra_extra_offset_x_arcsec': pa.array(off_x),
        'intra_extra_offset_y_arcsec': pa.array(off_y),
    }

    # The per-side columns, kept so a side-of-focus question needs no second Butler pass.
    # In the single-side modes idx_i and idx_e are the same rows, so both sides of each
    # pair carry that one side's value, consistent with the paired columns above.
    fwhm_all = _bare(table['group_fwhm'])
    cost_all = _bare(table['group_fit_cost'])
    opt_all = _bare(table['group_fit_optimality'])
    columns.update({
        'blur_intra': pa.array(fwhm_all[idx_i]),
        'blur_extra': pa.array(fwhm_all[idx_e]),
        'lstsq_cost_intra': pa.array(cost_all[idx_i]),
        'lstsq_cost_extra': pa.array(cost_all[idx_e]),
        'lstsq_optimality_intra': pa.array(opt_all[idx_i]),
        'lstsq_optimality_extra': pa.array(opt_all[idx_e]),
        'zk_OCS_intra': _rows(dev_ocs_all[idx_i] + int_ocs_all[idx_i]),
        'zk_OCS_extra': _rows(dev_ocs_all[idx_e] + int_ocs_all[idx_e]),
        'zk_intrinsic_OCS_intra': _rows(int_ocs_all[idx_i]),
        'zk_intrinsic_OCS_extra': _rows(int_ocs_all[idx_e]),
        'zk_deviation_OCS_intra': _rows(dev_ocs_all[idx_i]),
        'zk_deviation_OCS_extra': _rows(dev_ocs_all[idx_e]),
        'zk_CCS_intra': _rows(dev_ccs_all[idx_i] + int_ccs_all[idx_i]),
        'zk_CCS_extra': _rows(dev_ccs_all[idx_e] + int_ccs_all[idx_e]),
        'zk_intrinsic_CCS_intra': _rows(int_ccs_all[idx_i]),
        'zk_intrinsic_CCS_extra': _rows(int_ccs_all[idx_e]),
        'zk_deviation_CCS_intra': _rows(dev_ccs_all[idx_i]),
        'zk_deviation_CCS_extra': _rows(dev_ccs_all[idx_e]),
    })

    all_columns = tuple(DONUT_COLUMNS) + tuple(SIDE_COLUMNS)
    missing = [c for c in all_columns if c not in columns]
    extra = [c for c in columns if c not in all_columns]
    if missing or extra:
        raise ValueError(f'donut column set mismatch: missing {missing}, extra {extra}')

    stats['n_donuts'] = n_out
    stats['n_detectors'] = int(len(np.unique(det[idx_e]))) if n_out else 0
    arrow_table = pa.Table.from_arrays([columns[c] for c in all_columns],
                                       names=list(all_columns))
    return arrow_table, stats


def visit_metrics(arrow_table, min_donuts_per_detector=3):
    """Per-visit quality metrics from one visit's donut rows.

    Parameters
    ----------
    arrow_table : `pyarrow.Table`
        One visit's output of `donut_table`.
    min_donuts_per_detector : `int`, optional
        Per-detector donut floor used to count "covered" detectors, dimensionless.
        Default 3, matching the ``run_mktable`` default.

    Returns
    -------
    metrics : `dict`
        ``n_donuts``, ``n_detectors``, ``n_detectors_with_min_donuts`` (all dimensionless
        counts), ``median_blur_arcsec`` (arcsec, NaN when no donut has a finite blur) and
        the per-side ``median_blur_intra_arcsec`` / ``median_blur_extra_arcsec`` (arcsec,
        same NaN convention).

    Notes
    -----
    These are the three columns the downstream ``quality_visit_mask`` cuts on, so they
    must be computed the same way ``run_mktable`` does: counts over the kept donuts, and
    the median over finite blur values only.
    """
    detector = np.asarray(arrow_table.column('detector').to_pylist(), dtype=str)
    names, counts = (np.unique(detector, return_counts=True) if len(detector)
                     else (np.array([]), np.array([])))

    def _median_blur(column):
        values = np.asarray(arrow_table.column(column).to_pylist(), dtype=float)
        with np.errstate(invalid='ignore'):
            return (float(np.nanmedian(values))
                    if np.isfinite(values).any() else float('nan'))

    return {
        'n_donuts': int(arrow_table.num_rows),
        'n_detectors': int(len(names)),
        'n_detectors_with_min_donuts': int((counts >= min_donuts_per_detector).sum()),
        'median_blur_arcsec': _median_blur('blur'),
        'median_blur_intra_arcsec': _median_blur('blur_intra'),
        'median_blur_extra_arcsec': _median_blur('blur_extra'),
    }


def visit_meta_record(meta, day_obs, seq_num, visit):
    """Per-visit fields the blitz table metadata can supply on its own.

    Parameters
    ----------
    meta : `dict`
        The blitz table's ``.meta``.
    day_obs : `int`
        Observing night as ``YYYYMMDD``.
    seq_num : `int`
        Sequence number of the FAM extra-focal exposure, dimensionless.
    visit : `int`
        Visit identifier of the FAM extra-focal exposure.

    Returns
    -------
    record : `dict`
        ``day_obs``, ``seq_num``, ``visit``; ``az``, ``alt``, ``skyAngle`` in **radians**
        (the Danish 1.2 visits-table convention, converted here from the blitz
        `Quantity` in deg); ``mjd`` in days (TAI, mid-exposure); ``nollIndices`` as a
        list of dimensionless integers; ``rot_tel_pos_deg``, ``boresight_rot_angle_deg``
        and ``boresight_par_angle_deg`` in deg; ``intra_visit`` and ``extra_visit`` as
        visit identifiers; and the version strings ``ts_wep_version``,
        ``danish_version``, ``batoid_version``.

    Notes
    -----
    The Danish 1.2 ``visits.parquet`` stores ``ra``, ``dec``, ``az``, ``alt`` and
    ``skyAngle`` as **bare floats in radians** with no unit attached, while the blitz
    metadata stores angles as `astropy.units.Quantity` in deg. Confirmed on
    ``day_obs`` 20260315 ``seq_num`` 122: the Danish 1.2 ``az`` of 4.710581 rad is
    269.89641441 deg, equal to the blitz ``boresight_az``, and the Danish 1.2
    ``skyAngle`` of 0.196330 rad is 11.24885987 deg, equal to the blitz
    ``boresight_rot_angle``. Getting this backwards would rotate every field angle.

    ``ra``, ``dec``, ``band``, ``science_program`` and ``rotator_angle`` are **not**
    available from the blitz metadata and are merged in by the caller from the
    Consolidated Database (ConsDB). In particular ``rot_tel_pos`` is *not* the rotator
    angle the Danish 1.2 tables carry — that is ConsDB ``physical_rotator_angle``, which
    reads -0.126907 deg on this visit against a ``rot_tel_pos`` of 0.176609 deg.
    """
    def _deg(key):
        value = meta.get(key)
        if value is None:
            return float('nan')
        return float(getattr(value, 'to_value', lambda _u: float(value))('deg')
                     if hasattr(value, 'to_value') else value)

    def _rad(key):
        return float(np.deg2rad(_deg(key)))

    date = meta.get('date')
    mjd = float(getattr(date, 'mjd', np.nan)) if date is not None else float('nan')
    noll, _ = noll_selection(meta)
    return {
        'day_obs': int(day_obs),
        'seq_num': int(seq_num),
        'visit': int(visit),
        'az': _rad('boresight_az'),
        'alt': _rad('boresight_alt'),
        'skyAngle': _rad('boresight_rot_angle'),
        'mjd': mjd,
        'nollIndices': noll.tolist(),
        'rot_tel_pos_deg': _deg('rot_tel_pos'),
        'boresight_rot_angle_deg': _deg('boresight_rot_angle'),
        'boresight_par_angle_deg': _deg('boresight_par_angle'),
        'intra_visit': int(meta['intra_visit_id']),
        'extra_visit': int(meta['extra_visit_id']),
        'ts_wep_version': str(meta.get('ts_wep_version', '')),
        'danish_version': str(meta.get('danish_version', '')),
        'batoid_version': str(meta.get('batoid_version', '')),
    }


def intrinsic_provenance(meta):
    """Butler run collections the blitz table's inputs came from, by dataset type.

    Parameters
    ----------
    meta : `dict`
        The blitz table's ``.meta``, carrying the ``LSST.BUTLER.*`` input provenance
        block: ``N_INPUTS`` followed by ``INPUT.<i>.ID`` / ``.RUN`` / ``.DATASETTYPE``
        triples.

    Returns
    -------
    provenance : `dict`
        Maps dataset type name to a sorted `list` of the run collection names that
        supplied it. ``intrinsicZernikes`` is the entry that names the default Batoid
        intrinsic wavefront calibration used by the fit.

    Notes
    -----
    Returns an empty `dict` when the provenance block is absent, which is what a table
    read without ``strip_astropy_meta_yaml=False`` looks like.
    """
    prefix = 'LSST.BUTLER.INPUT.'
    by_type = {}
    for key, value in meta.items():
        if not (key.startswith(prefix) and key.endswith('.DATASETTYPE')):
            continue
        index = key[len(prefix):-len('.DATASETTYPE')]
        run = meta.get(f'{prefix}{index}.RUN')
        if run is None:
            continue
        by_type.setdefault(str(value), set()).add(str(run))
    return {k: sorted(v) for k, v in sorted(by_type.items())}


class BlitzDonutWriter:
    """Write paired donut rows to parquet with **one row group per visit**.

    Parameters
    ----------
    output_file : `str` or `pathlib.Path`
        Parquet path to create, overwriting any existing file.
    compression : `str`, optional
        Parquet compression codec. Default ``'snappy'``, matching the Danish 1.2 tables.

    Notes
    -----
    The streaming Double Zernike fitter
    (`lsst.ts.intrinsic.wavefront.dz_fitting.fit_focal_zernikes_streaming`) builds a
    ``(day_obs, seq_num) -> row_group`` lookup from the **row-group statistics** of the
    ``day_obs`` and ``seq_num`` columns, and **silently skips** any visit it cannot find
    there. So each `write_visit` call must produce exactly one row group, and statistics
    must be on. `pyarrow.parquet.ParquetWriter.write_table` writes one row group per call
    as long as the table is smaller than ``row_group_size``, and statistics are on by
    default; both are asserted after the fact by `verify_row_groups`.

    Every visit must present the identical schema or the file is corrupt, so tables after
    the first are cast to the first one's schema and a visit that cannot be cast is
    skipped with a warning rather than written.
    """

    def __init__(self, output_file, compression='snappy'):
        self.output_file = str(output_file)
        self.compression = compression
        self._writer = None
        self._schema = None
        self.n_visits = 0
        self.n_rows = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def write_visit(self, arrow_table):
        """Append one visit's rows as a single row group.

        Parameters
        ----------
        arrow_table : `pyarrow.Table`
            One visit's output of `donut_table`. A table with 0 rows is skipped, since an
            empty row group would carry no usable ``day_obs`` statistics.

        Returns
        -------
        written : `bool`
            True when a row group was written.
        """
        if arrow_table.num_rows == 0:
            return False
        if self._writer is None:
            self._schema = arrow_table.schema
            self._writer = pq.ParquetWriter(self.output_file, self._schema,
                                            compression=self.compression)
        else:
            try:
                arrow_table = arrow_table.select(
                    self._schema.names).cast(self._schema)
            except Exception as exc:  # pragma: no cover - defensive
                print(f'WARNING: schema mismatch, skipping visit: {exc}')
                return False
        self._writer.write_table(arrow_table)
        self.n_visits += 1
        self.n_rows += arrow_table.num_rows
        return True

    def close(self):
        """Close the parquet writer, if one was opened."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def verify_row_groups(output_file, expected_visits=None):
    """Check the one-row-group-per-visit contract the DZ fitter depends on.

    Parameters
    ----------
    output_file : `str` or `pathlib.Path`
        Donut parquet to inspect.
    expected_visits : `set` of `tuple`, optional
        The ``(day_obs, seq_num)`` pairs that should be present, each a dimensionless
        integer pair. When given, the recovered set must equal it exactly.

    Returns
    -------
    report : `dict`
        ``n_row_groups`` and ``n_rows`` (dimensionless counts), ``keys`` (the sorted
        ``(day_obs, seq_num)`` pairs recovered from row-group statistics) and
        ``n_missing_statistics`` (row groups whose ``day_obs`` or ``seq_num`` statistics
        are absent, which the fitter would skip).

    Raises
    ------
    ValueError
        If any row group lacks the statistics, if a ``(day_obs, seq_num)`` pair appears in
        more than one row group, or if `expected_visits` is given and does not match.

    Notes
    -----
    This is the check that catches the single most damaging failure mode: a donut parquet
    written without per-visit row groups makes the streaming fitter skip every visit and
    return an **empty** fit table, with no error raised anywhere.
    """
    parquet_file = pq.ParquetFile(str(output_file))
    keys, n_missing = [], 0
    for i in range(parquet_file.num_row_groups):
        group = parquet_file.metadata.row_group(i)
        day_obs = seq_num = None
        for ci in range(group.num_columns):
            column = group.column(ci)
            if column.statistics is None:
                continue
            if column.path_in_schema == 'day_obs':
                day_obs = column.statistics.min
                if column.statistics.max != day_obs:
                    raise ValueError(
                        f'row group {i} spans day_obs {day_obs} to '
                        f'{column.statistics.max}; it must hold exactly one visit')
            elif column.path_in_schema == 'seq_num':
                seq_num = column.statistics.min
                if column.statistics.max != seq_num:
                    raise ValueError(
                        f'row group {i} spans seq_num {seq_num} to '
                        f'{column.statistics.max}; it must hold exactly one visit')
        if day_obs is None or seq_num is None:
            n_missing += 1
            continue
        keys.append((int(day_obs), int(seq_num)))

    if n_missing:
        raise ValueError(
            f'{n_missing} of {parquet_file.num_row_groups} row groups lack day_obs or '
            'seq_num statistics; the streaming DZ fitter would skip those visits')
    if len(set(keys)) != len(keys):
        raise ValueError('a (day_obs, seq_num) pair appears in more than one row group; '
                         'the fitter would see only one of them')
    if expected_visits is not None and set(keys) != set(expected_visits):
        only_file = sorted(set(keys) - set(expected_visits))
        only_expected = sorted(set(expected_visits) - set(keys))
        raise ValueError(
            f'row-group keys do not match the expected visits: '
            f'{len(only_file)} only in the file {only_file[:5]}, '
            f'{len(only_expected)} only expected {only_expected[:5]}')
    return {
        'n_row_groups': int(parquet_file.num_row_groups),
        'n_rows': int(parquet_file.metadata.num_rows),
        'keys': sorted(keys),
        'n_missing_statistics': n_missing,
    }


def check_zk_sum_relation(donuts_parquet, coord_sys='OCS', tolerance=1.0e-6,
                          max_row_groups=1):
    """Verify ``zk = zk_deviation + zk_intrinsic`` on an existing donut table.

    Parameters
    ----------
    donuts_parquet : `str` or `pathlib.Path`
        A donut parquet in the Danish 1.2 schema.
    coord_sys : `str`, optional
        ``'OCS'`` or ``'CCS'``. Default ``'OCS'``.
    tolerance : `float`, optional
        Largest acceptable ``max |zk - (deviation + intrinsic)|``, in micrometres of
        wavefront. Default 1e-6.
    max_row_groups : `int`, optional
        How many row groups (visits) to check, dimensionless. Default 1.

    Returns
    -------
    max_abs_diff : `float`
        The largest absolute residual found, in micrometres of wavefront.

    Raises
    ------
    ValueError
        If the residual exceeds `tolerance`, meaning the total wavefront cannot be
        reconstructed as the sum and the blitz recast would be wrong.

    Notes
    -----
    On the Danish 1.2 ``donuts.parquet`` this returns 1.04e-07 micrometres of wavefront
    for OCS, which is float32 rounding in the stored CCS columns rather than a real
    inconsistency.
    """
    parquet_file = pq.ParquetFile(str(donuts_parquet))
    max_abs_diff = 0.0
    for i in range(min(max_row_groups, parquet_file.num_row_groups)):
        frame = parquet_file.read_row_group(
            i, columns=[f'zk_{coord_sys}', f'zk_intrinsic_{coord_sys}',
                        f'zk_deviation_{coord_sys}']).to_pandas()
        total = np.stack(frame[f'zk_{coord_sys}'].values)
        intrinsic = np.stack(frame[f'zk_intrinsic_{coord_sys}'].values)
        deviation = np.stack(frame[f'zk_deviation_{coord_sys}'].values)
        residual = np.nanmax(np.abs(total - (intrinsic + deviation)))
        max_abs_diff = max(max_abs_diff, float(residual))
    if max_abs_diff > tolerance:
        raise ValueError(
            f'max |zk - (deviation + intrinsic)| = {max_abs_diff:.3e} micrometres of '
            f'wavefront in {coord_sys} exceeds the tolerance of {tolerance:.1e}; the '
            'total wavefront cannot be reconstructed as the sum')
    return max_abs_diff
