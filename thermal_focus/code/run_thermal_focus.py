"""Assemble the thermal-focus table: one row per visit, response plus thermal telemetry.

This is the only stage that needs the network. Everything it reads comes from the value-added
DuckDB except the Telescope Mount Assembly (TMA) truss temperature, which the Consolidated
Database (ConsDB) serves and `efd_db.join_consdb` derives on the fly, so the table is cached to
parquet and the analysis stage runs offline from it.

Two tables are written:

* ``thermal_focus.parquet`` -- ordinary science visits, the sample the focus model is fitted on.
* ``<fam_dir>/thermal_focus_fam.parquet`` -- the in-focus ``acq`` visit of each Full Array Mode
  (FAM) triplet, joined to the Double Zernike (DZ) fit of its own defocused pair. Keyed by the
  FAM variant because the DZ coefficients depend on which reduction produced them.

Invocation::

    python code/run_thermal_focus.py
    python code/run_thermal_focus.py --day-obs-range 20251103 20260713
    python code/run_thermal_focus.py --no-fam

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
)


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
    args = ap.parse_args()

    out_dir = (pathlib.Path(args.output_dir) if args.output_dir
               else _ROOT / 'thermal_focus' / 'output')
    out_dir.mkdir(parents=True, exist_ok=True)

    day_obs_range = tuple(args.day_obs_range) if args.day_obs_range else None

    print('=== v1 to equivalent hexapod dz conversion ===')
    v1_per_um_dz = L.v1_per_um_dz_value(dof_set=args.dof_set, n_modes=args.n_modes)

    print('\n=== science visits ===')
    sci = load_science(args.variant, day_obs_range, v1_per_um_dz)
    sci_path = out_dir / 'thermal_focus.parquet'
    sci.to_parquet(sci_path, index=False)
    print(f'wrote {sci_path} ({len(sci)} rows)')

    if not args.no_fam:
        print('\n=== FAM triplets ===')
        fam = load_fam(args.fam_variant, args.variant, day_obs_range, v1_per_um_dz)
        fam_dir = out_dir / args.fam_dir_name
        fam_dir.mkdir(parents=True, exist_ok=True)
        fam_path = fam_dir / 'thermal_focus_fam.parquet'
        fam.to_parquet(fam_path, index=False)
        print(f'wrote {fam_path} ({len(fam)} rows)')


if __name__ == '__main__':
    main()
