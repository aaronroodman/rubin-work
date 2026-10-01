"""Assemble the thermal-focus table: one row per visit, response plus thermal telemetry.

This is the only stage that needs the network. Everything it reads comes from the value-added
DuckDB except the Telescope Mount Assembly (TMA) truss temperature, which the Consolidated
Database (ConsDB) serves and `efd_db.join_consdb` derives on the fly, so the table is cached to
parquet and the analysis stage runs offline from it.

Four tables are written:

* ``thermal_focus.parquet`` -- ordinary science visits, the sample the focus model is fitted on.
* ``<fam_dir>/thermal_focus_fam.parquet`` -- the in-focus ``acq`` visit of each Full Array Mode
  (FAM) triplet, joined to the Double Zernike (DZ) fit of its own defocused pair. Keyed by the
  FAM variant because the DZ coefficients depend on which reduction produced them.
* ``thermal_focus_t539.parquet`` -- one row per night for the initial alignment block run at the
  start of the night, carrying the thermal telemetry at the run's first visit and the commanded
  Trim degrees of freedom at its last. These visits are ``acq``, so they are absent from the
  science table above, which keeps ``science`` exposures only.
* ``thermal_focus_truss_all.parquet`` -- the mean TMA truss temperature for **every** exposure in
  the value-added database, with no image-type, band, LUT-epoch or temperature cut, so the report
  can show where the fitted sample sits within the full range of conditions the telescope has
  seen. This is the most expensive stage, one ConsDB round trip per night, and is the reason
  ``--only-truss-all`` exists.

Invocation::

    python code/run_thermal_focus.py
    python code/run_thermal_focus.py --day-obs-range 20251103 20260713
    python code/run_thermal_focus.py --no-fam
    python code/run_thermal_focus.py --no-truss-all
    python code/run_thermal_focus.py --only-truss-all

Notes
-----
``truss_temp_mean_c`` is **not a stored column**. `efd_db.join_consdb` computes it as the mean of
the ``tma_truss_temp_pxpy`` and ``tma_truss_temp_mxmy`` thermometers and interpolates it within
the night, so there is no offline route to the study's headline regressor. That is why this stage
exists separately from the analysis, and why ``truss_temp_mean_c_interpolated`` is carried
through: a visit whose truss temperature was filled by interpolation rather than measured is
still usable, but the analysis must be able to cut on it.

The DuckDB file lock is process-wide and excludes readers as well as writers, so every
connection here is read-only. A stray read-write connection blocks every other process,
including a running build.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                                   # -> thermal_focus_lib
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))                                   # repo root -> common/
sys.path.insert(0, str(_ROOT / 'value_added' / 'code'))           # -> efd_db

import efd_db                                                    # noqa: E402
import thermal_focus_lib as L                                    # noqa: E402
from common.utils import nmad                                    # noqa: E402

#: Image types kept as "science visits". The focus model is fitted on ordinary survey exposures.
SCIENCE_IMG_TYPES = ('science',)

#: Bands kept. ``u`` and ``y`` are retained in the table and cut in the analysis if wanted, so
#: the funnel is visible rather than hidden here.
BANDS = ('u', 'g', 'r', 'i', 'z', 'y')

#: ConsDB groups fetched by `efd_db.join_consdb`. ``thermal`` carries the truss thermometers,
#: ``meta`` the band, image type and pointing.
CONSDB_GROUPS = ('meta', 'thermal', 'wind')

#: Telemetry columns pulled from ``visit_telemetry``, beyond the identity columns. The
#: turbulence group is deliberately excluded: it covers 90,814 of 213,704 rows, so conditioning
#: on it would silently cut more than half the sample.
TELEMETRY_COLS = (
    'm1m3_x_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
    'm1m3_z_gradient_c_per_m', 'm1m3_radial_gradient_c_per_m',
    'cam_AverageTemp', 'cam_AmbAirtemp', 'cam_n_samp',
    'wind_dir_deg', 'wind_speed_ms', 'azimuth_deg', 'into_wind_deg',
    'cum_hex_dz_um', 'recent_hex_dz_um', 'n_moves_night',
    # The commanded Trim in the four degrees of freedom v-mode 1 contains, as of obs_start.
    # ts_ofc ordering: dof0-4 are the M2 hexapod, dof5-9 the camera hexapod, dof10-29 the M1M3
    # bending modes and dof30-49 the M2 bending modes. Carried for the BLOCK-T539 comparison,
    # which needs the Trim the initial alignment actually settled on.
    'dof0', 'dof5', 'dof12', 'dof34',
)

#: Science programs of the initial alignment block. Two labels appear in the Consolidated
#: Database over the covered span and both are the same block: the bare one and a
#: ``_hexapods`` suffixed variant used on 12 nights in the 2025 November era.
T539_PROGRAM_PREFIX = 'BLOCK-T539'

#: Image types that count as "the observing night has started", for finding the initial
#: alignment run. Biases, darks and flats are taken before the block and must not consume the
#: start-of-night window; ``cwfs`` is excluded because a wavefront pair is not the alignment
#: exposure itself.
NIGHT_START_IMG_TYPES = ('science', 'acq')

#: How many exposures of `NIGHT_START_IMG_TYPES` count as "the start of the night". The initial
#: alignment block is looked for inside this window, so a later re-run of the same block on the
#: same night is not picked up.
NIGHT_START_WINDOW = 10


def _telemetry(day_obs_range):
    """Read the telemetry columns, keyed on ``visit_id`` alone.

    Parameters
    ----------
    day_obs_range : `tuple` [`int`] or `None`
        Inclusive night range as ``YYYYMMDD``.

    Returns
    -------
    vis : `pandas.DataFrame`
        ``visit_id`` plus `TELEMETRY_COLS`.

    Notes
    -----
    ``day_obs`` and ``seq_num`` are deliberately left out even though ``visit_telemetry`` holds
    them: the caller already carries them from ``optical_state``, and merging a second copy
    renames both to ``day_obs_x``/``day_obs_y``, after which `efd_db.join_consdb` cannot find the
    ``day_obs`` it needs to plan its per-night queries.
    """
    vis = efd_db.visits(day_obs_range=day_obs_range,
                        columns=['visit_id'] + list(TELEMETRY_COLS))
    return vis[['visit_id'] + [c for c in TELEMETRY_COLS if c in vis.columns]]


def _r2_terms(day_obs_range):
    """Read the quadratic-in-radius M1M3 thermal terms, keyed on ``visit_id`` alone.

    Parameters
    ----------
    day_obs_range : `tuple` [`int`] or `None`
        Inclusive night range as ``YYYYMMDD``.

    Returns
    -------
    r2 : `pandas.DataFrame`
        ``visit_id`` plus the three coefficient columns of `thermal_focus_lib.R2_COLS` [°C per
        unit normalized radius-squared amplitude] and the residual scatter of the whole-mirror
        fit ``m1m3_rms_c`` [°C]. Empty with the right columns if the table does not exist yet,
        so the build runs before the quadratic table is populated.

    Notes
    -----
    ``day_obs`` and ``seq_num`` are dropped for the same reason as in `_telemetry`: the caller
    already carries them, and a second copy renames both and breaks the ConsDB join.
    """
    cols = [c for c, _ in L.R2_COLS] + ['m1m3_rms_c']
    try:
        r2 = efd_db.m1m3_thermal_r2(day_obs_range=day_obs_range, columns=cols)
    except Exception as exc:
        print(f'  m1m3_thermal_r2 unavailable ({type(exc).__name__}: {exc}); the quadratic '
              'radial terms will be absent from the table')
        return pd.DataFrame(columns=['visit_id'] + cols)
    return r2[['visit_id'] + [c for c in cols if c in r2.columns]]


def load_science(variant, day_obs_range, v1_per_um_dz, verbose=True):
    """Build the per-visit science table: response, thermal features and pointing.

    Parameters
    ----------
    variant : `str`
        Recovered-optical-state variant id, e.g. ``'v50_34__batoid__consdb_v1'``.
    day_obs_range : `tuple` [`int`] or `None`
        Inclusive ``(lo, hi)`` night range as ``YYYYMMDD``, or `None` for everything.
    v1_per_um_dz : `float`
        Conversion from `thermal_focus_lib.v1_per_um_dz_value` [dimensionless v-mode-1
        amplitude per µm of total hexapod dz travel].
    verbose : `bool`, optional
        Print the selection funnel.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per science visit with ``y`` [µm of equivalent hexapod dz], the five thermal
        features, band, elevation and the identity columns.

    Notes
    -----
    The funnel is printed rather than summarised because every stage of it has cost a real
    misunderstanding at some point: a forgotten variant filter multiplies the sample, and the
    LUT-epoch nights change what the commanded baseline means.
    """
    st = efd_db.optical_state(variant, day_obs_range=day_obs_range, wide=True, ok_only=True)
    n_state = len(st)
    keep = ['visit_id', 'day_obs', 'seq_num', 'resid_rms_um', 'v1', 'v1_lut', 'v1_trim']
    st = st[[c for c in keep if c in st.columns]].copy()

    vis = _telemetry(day_obs_range)
    df = st.merge(vis, on='visit_id', how='left')
    n_joined = int(df['m1m3_z_gradient_c_per_m'].notna().sum())

    r2 = _r2_terms(day_obs_range)
    df = df.merge(r2, on='visit_id', how='left')
    n_r2 = int(df['m1m3_r2_coeff_c'].notna().sum()) if 'm1m3_r2_coeff_c' in df.columns else 0

    # The truss temperature is derived on the join, not stored, so this is the network step.
    df = efd_db.join_consdb(df, groups=CONSDB_GROUPS)

    n_all, nights_all = len(df), df['day_obs'].nunique()
    if 'img_type' in df.columns:
        df = df[df['img_type'].isin(SCIENCE_IMG_TYPES)]
    n_science = len(df)
    if 'band' in df.columns:
        df = df[df['band'].isin(BANDS)]
    n_band = len(df)

    drop = df['day_obs'].isin(L.LUT_EPOCH_OFFSET_NIGHTS)
    n_lut_nights = int(df.loc[drop, 'day_obs'].nunique())
    n_lut_visits = int(drop.sum())
    df = df[~drop]

    # An isolated warm population, detached from the sample by an empty 5.1792 deg C interval.
    hot = df['truss_temp_mean_c'] > L.TRUSS_TEMP_MAX_C
    n_hot_nights = int(df.loc[hot, 'day_obs'].nunique())
    n_hot_visits = int(hot.sum())
    df = df[~hot]

    df = L.attach_response(df, v1_per_um_dz)
    df = df[np.isfinite(df['y'])]
    n_resp = len(df)

    features = L.resolve_features(L.DELIVERABLE_GROUPS)
    have = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f'the deliverable model needs {", ".join(missing)}, absent from the '
                         f'assembled table; check the ConsDB groups fetched')
    n_feat = int(df[have].notna().all(axis=1).sum())

    if verbose:
        print(f'optical_state {variant}: {n_state} visits with a recovered state')
        print(f'  joined to visit_telemetry gradients : {n_joined} '
              f'({100 * n_joined / max(n_state, 1):.1f}%)')
        print(f'  joined to the quadratic radial terms: {n_r2} '
              f'({100 * n_r2 / max(n_state, 1):.1f}%)')
        print(f'  after ConsDB join                   : {n_all} visits, '
              f'{nights_all} nights')
        print(f'  img_type in {SCIENCE_IMG_TYPES}            : {n_science}')
        print(f'  band in {BANDS}   : {n_band}')
        print(f'  dropping {n_lut_nights} LUT-epoch nights        : '
              f'-{n_lut_visits} visits')
        print(f'  truss temperature above {L.TRUSS_TEMP_MAX_C:.0f} deg C     : '
              f'-{n_hot_visits} visits on {n_hot_nights} nights')
        print(f'  with a finite response              : {n_resp}')
        print(f'  with all five thermal features      : {n_feat}')
        print(f'  -> {len(df)} visits, {df["day_obs"].nunique()} nights, '
              f'day_obs {int(df["day_obs"].min())} to {int(df["day_obs"].max())}')
        print(f'response [um equiv hexapod dz]: median {df["y"].median():+.1f}, '
              f'nMAD {nmad(df["y"].to_numpy()):.1f}, n {len(df)}')
        if 'truss_temp_mean_c_interpolated' in df.columns:
            n_interp = int(df['truss_temp_mean_c_interpolated'].fillna(False).sum())
            print(f'truss temperature filled by within-night interpolation: {n_interp} of '
                  f'{len(df)} visits ({100 * n_interp / max(len(df), 1):.1f}%)')
    return df


def load_fam(fam_variant, variant, day_obs_range, v1_per_um_dz, verbose=True):
    """Build the FAM table: the in-focus response beside its own triplet's DZ defocus.

    Parameters
    ----------
    fam_variant : `str`
        FAM DZ variant id.
    variant : `str`
        Recovered-optical-state variant id, for the ``acq`` visit's own state.
    day_obs_range : `tuple` [`int`] or `None`
        Inclusive night range as ``YYYYMMDD``.
    v1_per_um_dz : `float`
        Conversion from `thermal_focus_lib.v1_per_um_dz_value`.
    verbose : `bool`, optional
        Print the join yield.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per FAM triplet whose ``acq`` visit has a recovered optical state, carrying the
        response ``y`` [µm of equivalent hexapod dz] and the triplet's own DZ defocus.

    Notes
    -----
    The DZ row is keyed on the extra-focal member of the triplet and carries ``acq_visit_id``,
    which is what lets the in-focus response be compared against the defocus measured by the
    same triplet rather than against a neighbouring visit. Of 2528 FAM rows, 1526 have a
    recovered state for their ``acq`` frame, so roughly 40% are lost here; the shortfall is
    reported rather than absorbed.
    """
    dz = efd_db.fam_dz(fam_variant, day_obs_range=day_obs_range, wide=True, good_only=True)
    n_dz = len(dz)
    # `fam_dz` wide mode emits its own v-mode columns for the DZ fit, so both tables offer a
    # column called `v1`. They are different quantities measured by different engines -- and
    # carry opposite v1 sign conventions -- so the FAM one is renamed rather than left to collide
    # into `v1_x`/`v1_y`, which would silently decide which the response is built from.
    dz = dz.rename(columns={c: f'dz_{c}' for c in dz.columns
                            if c == 'v1' or (c.startswith('v') and c[1:].isdigit())})
    st = efd_db.optical_state(variant, day_obs_range=day_obs_range, wide=True, ok_only=True)
    keep = ['visit_id', 'v1', 'v1_lut', 'v1_trim']
    st = st[[c for c in keep if c in st.columns]].rename(columns={'visit_id': 'acq_visit_id'})

    df = dz.merge(st, on='acq_visit_id', how='inner')
    n_state = len(df)

    vis = _telemetry(day_obs_range)
    df = df.merge(vis.rename(columns={'visit_id': 'acq_visit_id'}),
                  on='acq_visit_id', how='left')
    r2 = _r2_terms(day_obs_range)
    df = df.merge(r2.rename(columns={'visit_id': 'acq_visit_id'}),
                  on='acq_visit_id', how='left')
    df = efd_db.join_consdb(df, groups=CONSDB_GROUPS)
    df = df[~df['day_obs'].isin(L.LUT_EPOCH_OFFSET_NIGHTS)]
    if 'truss_temp_mean_c' in df.columns:
        df = df[~(df['truss_temp_mean_c'] > L.TRUSS_TEMP_MAX_C)]
    df = L.attach_response(df, v1_per_um_dz)
    df = df[np.isfinite(df['y'])]

    if verbose:
        print(f'fam_dz {fam_variant}:')
        print(f'  quality-passing DZ rows             : {n_dz}')
        print(f'  with a recovered state for the acq  : {n_state} '
              f'({100 * n_state / max(n_dz, 1):.1f}%)')
        print(f'  -> {len(df)} triplets, {df["day_obs"].nunique()} nights')
    return df


def load_truss_all(day_obs_range=None, verbose=True):
    """Mean TMA truss temperature for every exposure in the database, night by night.

    Parameters
    ----------
    day_obs_range : `tuple` [`int`] or `None`, optional
        Inclusive ``(lo, hi)`` night range as ``YYYYMMDD``, or `None` for the whole database.
    verbose : `bool`, optional
        Print per-night progress and the closing summary.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per exposure with ``visit_id``, ``day_obs``, ``seq_num``, ``obs_start_mjd`` [d],
        ``img_type``, ``truss_temp_mean_c`` [°C] and ``truss_temp_mean_c_interpolated``.

    Notes
    -----
    The loop over nights is a **correctness requirement, not an optimisation**.
    `efd_db.join_consdb` derives ``truss_temp_mean_c`` as the mean of the two truss thermometers
    and then interpolates it over the exposures it is given; handed the whole survey at once it
    would interpolate across night boundaries, filling a gap at the end of one night from the
    start of the next. One night per call confines the interpolation to where it is physically
    defensible. The ``meta`` group is fetched alongside ``thermal`` because it supplies
    ``obs_start_mjd``, the numerical axis that interpolation runs on.

    No cut of any kind is applied: this table is deliberately the whole population, including
    biases, darks and flats, because its purpose is to show what the fitted sample excluded.
    ``img_type`` is therefore carried, and any plot drawn from this table must say which
    population it shows.

    A night that fails -- a ConsDB hiccup, a night with no truss telemetry at all -- is reported
    and skipped rather than being allowed to lose a pass that costs several hundred queries.
    """
    keys = ['visit_id', 'day_obs', 'seq_num']
    nights = sorted(int(d) for d in
                    efd_db.visits(day_obs_range=day_obs_range, columns=keys)['day_obs'].unique())
    if verbose:
        print(f'  mean truss temperature over {len(nights)} nights, day_obs {nights[0]} to '
              f'{nights[-1]}, one ConsDB round trip per night')

    want = keys + ['obs_start_mjd', 'img_type', 'truss_temp_mean_c',
                   'truss_temp_mean_c_interpolated']
    frames, failed = [], []
    for i, d in enumerate(nights, start=1):
        try:
            vis = efd_db.visits(day_obs_range=(d, d), columns=keys)
            vis = efd_db.join_consdb(vis, groups=('meta', 'thermal'))
            frames.append(vis[[c for c in want if c in vis.columns]])
        except Exception as exc:                                  # noqa: BLE001 - see Notes
            failed.append(d)
            print(f'    day_obs {d}: skipped, {type(exc).__name__}: {exc}')
        if verbose and (i % 25 == 0 or i == len(nights)):
            print(f'    {i} of {len(nights)} nights, '
                  f'{sum(len(f) for f in frames)} exposures so far')

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=want)

    if verbose and len(df):
        t = df['truss_temp_mean_c']
        n_night_nan = int(df.groupby('day_obs')['truss_temp_mean_c']
                          .apply(lambda s: s.notna().sum() == 0).sum())
        n_interp = int(df['truss_temp_mean_c_interpolated'].fillna(False).sum()) \
            if 'truss_temp_mean_c_interpolated' in df.columns else 0
        print(f'  -> {len(df)} exposures over {df["day_obs"].nunique()} nights, '
              f'{len(failed)} nights skipped on error')
        print(f'     truss temperature resolved      : {int(t.notna().sum())} exposures '
              f'({100 * t.notna().mean():.1f}%)')
        print(f'     of those, filled by interpolation: {n_interp}')
        print(f'     nights with no truss sample at all: {n_night_nan}')
        print(f'     mean truss temperature [deg C]: median {t.median():+.2f}, '
              f'nMAD {nmad(t.to_numpy()):.2f}, range {t.min():+.2f} to {t.max():+.2f}, '
              f'n {int(t.notna().sum())}')
        if 'img_type' in df.columns:
            counts = df['img_type'].value_counts()
            print('     img_type: ' + ', '.join(f'{k} {v}' for k, v in counts.items()))
    return df


def _exposure_inventory(day_obs_range, cdb=None, instrument='lsstcam'):
    """Read ``science_program``, ``img_type`` and ``seq_num`` for every exposure in a night range.

    Parameters
    ----------
    day_obs_range : `tuple` [`int`]
        Inclusive ``(lo, hi)`` night range as ``YYYYMMDD``.
    cdb : `lsst.summit.utils.ConsDbClient`, optional
        Existing client; one is made if omitted.
    instrument : `str`, optional
        ConsDB instrument schema.

    Returns
    -------
    df : `pandas.DataFrame`
        ``visit_id``, ``day_obs``, ``seq_num``, ``img_type``, ``science_program``.

    Notes
    -----
    The query is issued **one calendar month at a time**. A single request spanning the whole
    survey returns HTTP 500 from the ConsDB server rather than a result, and so does any
    ``SELECT DISTINCT`` combined with a ``LIKE`` filter, so the program filter is applied here in
    pandas rather than in SQL.
    """
    sys.path.insert(0, str(_ROOT))
    from common.telemetry_clients import make_consdb_client       # noqa: E402
    cdb = cdb or make_consdb_client()

    lo, hi = int(day_obs_range[0]), int(day_obs_range[1])
    # Month boundaries as YYYYMM01, inclusive of the month hi falls in.
    edges, y, m = [], lo // 10000, (lo // 100) % 100
    while y * 10000 + m * 100 + 1 <= hi:
        edges.append(y * 10000 + m * 100 + 1)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    edges = [lo] + [e for e in edges if e > lo] + [hi + 1]

    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        q = (f'SELECT exposure_id AS visit_id, day_obs, seq_num, img_type, science_program '
             f'FROM cdb_{instrument}.exposure '
             f'WHERE day_obs >= {a} AND day_obs < {b}')
        parts.append(cdb.query(q).to_pandas())
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(['day_obs', 'seq_num']).reset_index(drop=True)


def _t539_runs(inv, verbose=True):
    """Find the start-of-night initial alignment run on each night.

    Parameters
    ----------
    inv : `pandas.DataFrame`
        Output of `_exposure_inventory`.
    verbose : `bool`, optional
        Print the run-length distribution.

    Returns
    -------
    runs : `pandas.DataFrame`
        One row per night: ``day_obs``, ``science_program``, ``seq_num_first``, ``seq_num_last``,
        ``n_run`` (exposures in the run), ``n_night`` (all alignment-block exposures that night),
        ``visit_id_first`` and ``visit_id_last``.

    Notes
    -----
    The rule is: among the first `NIGHT_START_WINDOW` exposures of `NIGHT_START_IMG_TYPES` on the
    night — so biases, darks and flats taken before the block do not consume the window — keep
    those belonging to the alignment block, then extend from the lowest such ``seq_num`` through
    the **contiguous** ``seq_num`` run.

    The run length is *not* fixed. Over the covered span the modal length is 10 exposures but
    runs of 20 and 24 are common and the longest reaches 45, all of them a single program label
    running consecutively rather than two blocks chained. So the run is taken as contiguous and
    its length reported per night rather than assumed.

    Most nights also have further alignment-block exposures later on, which is why the
    start-of-night window exists: ``n_night`` against ``n_run`` shows how many were set aside.
    """
    inv = inv.assign(
        _is_block=inv['science_program'].astype(str).str.startswith(T539_PROGRAM_PREFIX),
        _started=inv['img_type'].isin(NIGHT_START_IMG_TYPES))

    rows = []
    for night, g in inv.groupby('day_obs'):
        night_start = g[g['_started']].sort_values('seq_num').head(NIGHT_START_WINDOW)
        opening = night_start[night_start['_is_block']]
        if not len(opening):
            continue
        block = g[g['_is_block'] & g['_started']].sort_values('seq_num')
        seq = block['seq_num'].to_numpy()
        i0 = int(np.flatnonzero(seq == int(opening['seq_num'].min()))[0])
        j = i0
        while j + 1 < len(seq) and seq[j + 1] == seq[j] + 1:
            j += 1
        rows.append({'day_obs': int(night),
                     'science_program': block['science_program'].iloc[i0],
                     'seq_num_first': int(seq[i0]), 'seq_num_last': int(seq[j]),
                     'n_run': j - i0 + 1, 'n_night': len(block),
                     'visit_id_first': int(block['visit_id'].iloc[i0]),
                     'visit_id_last': int(block['visit_id'].iloc[j])})
    runs = pd.DataFrame(rows)
    if verbose and len(runs):
        n_extra = int((runs['n_night'] > runs['n_run']).sum())
        print(f'  nights with a start-of-night {T539_PROGRAM_PREFIX} run: {len(runs)}')
        print(f'    run length [exposures]: median {runs["n_run"].median():.0f}, '
              f'min {runs["n_run"].min()}, max {runs["n_run"].max()}')
        print('    program labels: '
              + ', '.join(f'{k} {v}' for k, v in
                          runs['science_program'].value_counts().items()))
        print(f'    nights with further block exposures later in the night: {n_extra} '
              f'of {len(runs)}')
    return runs


def load_t539(day_obs_range, verbose=True):
    """Build the initial-alignment comparison table: prediction at the start, Trim at the end.

    One row per night. The thermal telemetry is taken at the **first** visit of the start-of-night
    initial alignment run, which is what an open-loop prediction would have had available, and the
    commanded Trim at the **last** visit of the same run, which is what the alignment converged
    to. Comparing them tests the prediction against an independent measurement rather than against
    the fit's own residual.

    Parameters
    ----------
    day_obs_range : `tuple` [`int`] or `None`
        Inclusive night range as ``YYYYMMDD``. Defaults to the span over which both the Trim and
        the M1M3 gradients exist.
    verbose : `bool`, optional
        Print the selection funnel.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per usable night: the run identification from `_t539_runs`, the five thermal
        features suffixed ``_first``, and the four Trim degrees of freedom suffixed ``_last``
        [µm].

    Notes
    -----
    The two epochs are deliberately given distinct column suffixes. They are separated by the
    whole alignment run — typically 10 exposures but up to 45 — so they are not simultaneous, and
    a column name that did not say which epoch it came from would invite exactly that confusion.

    The same cuts as `load_science` are applied, so the comparison sample is a subset of the
    fitted one: the look-up-table epoch nights and nights above
    `thermal_focus_lib.TRUSS_TEMP_MAX_C`.
    """
    if day_obs_range is None:
        day_obs_range = (20251102, 20260714)

    if verbose:
        print(f'  reading the ConsDB exposure inventory, day_obs {day_obs_range[0]} to '
              f'{day_obs_range[1]}, one month per query')
    inv = _exposure_inventory(day_obs_range)
    if verbose:
        print(f'    {len(inv)} exposures over {inv["day_obs"].nunique()} nights')
    runs = _t539_runs(inv, verbose=verbose)
    if not len(runs):
        return runs
    n_selected = len(runs)

    tel_cols = [c for c in L.resolve_features(L.DELIVERABLE_GROUPS) if c != 'truss_temp_mean_c']
    trim_cols = ['dof5', 'dof0', 'dof12', 'dof34']

    # The truss temperature is derived on the ConsDB join and interpolated within the night, so
    # the whole night must be passed through `join_consdb`, not just the two visits of interest.
    vis = efd_db.visits(day_obs_range=day_obs_range,
                        columns=['visit_id', 'day_obs', 'seq_num'] + tel_cols + trim_cols)
    vis = vis[vis['day_obs'].isin(runs['day_obs'])]
    vis = efd_db.join_consdb(vis, groups=('meta', 'thermal'))

    first = vis[['visit_id'] + tel_cols + ['truss_temp_mean_c',
                                           'truss_temp_mean_c_interpolated']].copy()
    first = first.rename(columns={c: f'{c}_first' for c in first.columns if c != 'visit_id'})
    last = vis[['visit_id'] + trim_cols].copy()
    last = last.rename(columns={c: f'{c}_last' for c in last.columns if c != 'visit_id'})

    df = runs.merge(first, left_on='visit_id_first', right_on='visit_id', how='left') \
             .drop(columns='visit_id')
    df = df.merge(last, left_on='visit_id_last', right_on='visit_id', how='left') \
           .drop(columns='visit_id')

    feat_first = [f'{c}_first' for c in tel_cols] + ['truss_temp_mean_c_first']
    n_feat = int(df[feat_first].notna().all(axis=1).sum())
    n_trim = int(df[[f'{c}_last' for c in trim_cols]].notna().all(axis=1).sum())

    drop = df['day_obs'].isin(L.LUT_EPOCH_OFFSET_NIGHTS)
    n_lut = int(drop.sum())
    df = df[~drop]
    hot = df['truss_temp_mean_c_first'] > L.TRUSS_TEMP_MAX_C
    n_hot = int(hot.sum())
    df = df[~hot]
    df = df[df[feat_first].notna().all(axis=1)
            & df[[f'{c}_last' for c in trim_cols]].notna().all(axis=1)]

    if verbose:
        print(f'    with all five thermal features at the first visit: {n_feat} of {n_selected}')
        print(f'    with all four Trim DOF at the last visit          : {n_trim} of {n_selected}')
        print(f'    dropping LUT-epoch nights                        : -{n_lut}')
        print(f'    truss temperature above {L.TRUSS_TEMP_MAX_C:.0f} deg C              : -{n_hot}')
        print(f'  -> {len(df)} nights, day_obs {int(df["day_obs"].min())} to '
              f'{int(df["day_obs"].max())}')
        n_interp = int(df['truss_temp_mean_c_interpolated_first'].fillna(False).sum())
        print(f'     truss temperature filled by interpolation: {n_interp} of {len(df)} nights')
        print(f'     camera hexapod dz Trim at the run end [um]: median '
              f'{df["dof5_last"].median():+.1f}, nMAD {nmad(df["dof5_last"].to_numpy()):.1f}')
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', default=L.DEFAULT_VARIANT,
                    help='recovered-optical-state variant id')
    ap.add_argument('--fam-variant', default=L.DEFAULT_FAM_VARIANT,
                    help='FAM Double Zernike variant id')
    ap.add_argument('--day-obs-range', nargs=2, type=int, default=None,
                    metavar=('LO', 'HI'), help='inclusive night range as YYYYMMDD')
    ap.add_argument('--dof-set', default='all_50',
                    help='ts_ofc DOF set for the v1 to dz conversion')
    ap.add_argument('--n-modes', type=int, default=34,
                    help='v-modes retained in the conversion')
    ap.add_argument('--output-dir', default=None,
                    help='where to write; default thermal_focus/output')
    ap.add_argument('--fam-dir-name', default='fam_danish_1_2',
                    help='short directory name for the FAM variant')
    ap.add_argument('--no-fam', action='store_true', help='skip the FAM table')
    ap.add_argument('--no-t539', action='store_true',
                    help='skip the initial-alignment comparison table')
    ap.add_argument('--no-truss-all', action='store_true',
                    help='skip the database-wide mean truss temperature table')
    ap.add_argument('--only-truss-all', action='store_true',
                    help='write only the database-wide truss table, skipping the other three; '
                         'this stage is one ConsDB query per night and is worth running alone')
    args = ap.parse_args()

    out_dir = (pathlib.Path(args.output_dir) if args.output_dir
               else _ROOT / 'thermal_focus' / 'output')
    out_dir.mkdir(parents=True, exist_ok=True)

    day_obs_range = tuple(args.day_obs_range) if args.day_obs_range else None

    if not args.only_truss_all:
        print('=== v1 to equivalent hexapod dz conversion ===')
        v1_per_um_dz = L.v1_per_um_dz_value(dof_set=args.dof_set, n_modes=args.n_modes)

        print('\n=== science visits ===')
        sci = load_science(args.variant, day_obs_range, v1_per_um_dz)
        sci_path = out_dir / 'thermal_focus.parquet'
        sci.to_parquet(sci_path, index=False)
        print(f'wrote {sci_path} ({len(sci)} rows)')

    if not args.no_fam and not args.only_truss_all:
        print('\n=== FAM triplets ===')
        fam = load_fam(args.fam_variant, args.variant, day_obs_range, v1_per_um_dz)
        fam_dir = out_dir / args.fam_dir_name
        fam_dir.mkdir(parents=True, exist_ok=True)
        fam_path = fam_dir / 'thermal_focus_fam.parquet'
        fam.to_parquet(fam_path, index=False)
        print(f'wrote {fam_path} ({len(fam)} rows)')

    if not args.no_t539 and not args.only_truss_all:
        print('\n=== initial alignment block, start-of-night runs ===')
        t539 = load_t539(day_obs_range)
        if len(t539):
            t539_path = out_dir / 'thermal_focus_t539.parquet'
            t539.to_parquet(t539_path, index=False)
            print(f'wrote {t539_path} ({len(t539)} rows)')
        else:
            print('no start-of-night alignment runs found; no table written')

    if not args.no_truss_all:
        print('\n=== mean truss temperature, every exposure in the database ===')
        truss = load_truss_all(day_obs_range)
        if len(truss):
            truss_path = out_dir / 'thermal_focus_truss_all.parquet'
            truss.to_parquet(truss_path, index=False)
            print(f'wrote {truss_path} ({len(truss)} rows)')
        else:
            print('no exposures returned; no table written')


if __name__ == '__main__':
    main()
