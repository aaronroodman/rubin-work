"""The thermal-focus science sample: one row per science visit, response plus telemetry.

Shared by the ``thermal_focus`` and ``thermal_vmodes`` studies, so both fit the same selection
funnel. Every reader takes the `rubinwork.products.value_added` module as its first argument,
since a library may not import a product.
"""
import numpy as np
import pandas as pd

from rubinwork.common.utils import nmad
from rubinwork.thermal_focus import lib as L

#: Image types kept as "science visits". The focus model is fitted on ordinary survey exposures.
SCIENCE_IMG_TYPES = ('science',)

#: Bands kept. ``u`` and ``y`` are retained in the table and cut in the analysis if wanted, so
#: the funnel is visible rather than hidden here.
BANDS = ('u', 'g', 'r', 'i', 'z', 'y')

#: ConsDB groups fetched by `value_added.join_consdb`. ``thermal`` carries the truss thermometers,
#: ``meta`` the band, image type and pointing.
CONSDB_GROUPS = ('meta', 'thermal', 'wind')

#: Telemetry columns pulled from ``visit_telemetry``, beyond the identity columns. The
#: turbulence group is deliberately excluded: it covers 90,814 of 213,704 rows, so conditioning
#: on it would silently cut more than half the sample.
TELEMETRY_COLS = (
    'm1m3_x_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
    'm1m3_z_gradient_c_per_m', 'm1m3_radial_gradient_c_per_m',
    'cam_AverageTemp', 'cam_AmbAirtemp', 'cam_n_samp',
    # Camera structure and lens temperatures, for the per-channel v-mode screen in
    # `thermal_vmodes`. The air-handling channels (shroud ring, plenum, shutter, charger) are
    # left out: they track cam_AverageTemp and carry no separate optical structure.
    'cam_CamBodyXPlusTemp', 'cam_CamBodyYPlusTemp', 'cam_CamBodyYMinusTemp',
    'cam_CamHousXPlusTemp', 'cam_CamHousXMinusTemp',
    'cam_CamHousYPlusTemp', 'cam_CamHousYMinusTemp',
    'cam_L1XMinusTemp', 'cam_L1YMinusTemp',
    'cam_L2XPlusTemp', 'cam_L2XMinusTemp', 'cam_L2YPlusTemp',
    'wind_dir_deg', 'wind_speed_ms', 'azimuth_deg', 'into_wind_deg',
    'cum_hex_dz_um', 'recent_hex_dz_um', 'n_moves_night',
    # The commanded Trim in the four degrees of freedom v-mode 1 contains, as of obs_start.
    # ts_ofc ordering: dof0-4 are the M2 hexapod, dof5-9 the camera hexapod, dof10-29 the M1M3
    # bending modes and dof30-49 the M2 bending modes. Carried for the BLOCK-T539 comparison,
    # which needs the Trim the initial alignment actually settled on.
    'dof0', 'dof5', 'dof12', 'dof34',
)


def telemetry(reader, day_obs_range):
    """Read the telemetry columns, keyed on ``visit_id`` alone.

    Parameters
    ----------
    reader : `module`
        `rubinwork.products.value_added`, passed in because a library may not import a
        product.
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
    renames both to ``day_obs_x``/``day_obs_y``, after which `reader.join_consdb` cannot find the
    ``day_obs`` it needs to plan its per-night queries.
    """
    vis = reader.visits(day_obs_range=day_obs_range,
                        columns=['visit_id'] + list(TELEMETRY_COLS))
    return vis[['visit_id'] + [c for c in TELEMETRY_COLS if c in vis.columns]]


def r2_terms(reader, day_obs_range):
    """Read the quadratic-in-radius M1M3 thermal terms, keyed on ``visit_id`` alone.

    Parameters
    ----------
    reader : `module`
        `rubinwork.products.value_added`, passed in because a library may not import a
        product.
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
    ``day_obs`` and ``seq_num`` are dropped for the same reason as in `telemetry`: the caller
    already carries them, and a second copy renames both and breaks the ConsDB join.
    """
    cols = [c for c, _ in L.R2_COLS] + ['m1m3_rms_c']
    try:
        r2 = reader.m1m3_thermal_r2(day_obs_range=day_obs_range, columns=cols)
    except Exception as exc:
        print(f'  m1m3_thermal_r2 unavailable ({type(exc).__name__}: {exc}); the quadratic '
              'radial terms will be absent from the table')
        return pd.DataFrame(columns=['visit_id'] + cols)
    return r2[['visit_id'] + [c for c in cols if c in r2.columns]]


def load_science(reader, variant, day_obs_range, v1_per_um_dz, verbose=True, keep_extra=()):
    """Build the per-visit science table: response, thermal features and pointing.

    Parameters
    ----------
    reader : `module`
        `rubinwork.products.value_added`, passed in because a library may not import a
        product.
    variant : `str`
        Recovered-optical-state variant id, e.g. ``'v50_34__batoid__consdb_v1'``.
    day_obs_range : `tuple` [`int`] or `None`
        Inclusive ``(lo, hi)`` night range as ``YYYYMMDD``, or `None` for everything.
    v1_per_um_dz : `float`
        Conversion from `thermal_focus_lib.v1_per_um_dz_value` [dimensionless v-mode-1
        amplitude per µm of total hexapod dz travel].
    verbose : `bool`, optional
        Print the selection funnel.
    keep_extra : `iterable` [`str`], optional
        Further `optical_state` columns to carry through, for a caller modelling something
        other than v-mode 1 -- `thermal_vmodes.response_columns` for the all-mode study.
        Columns absent from the read are ignored. The ``y`` column is always v-mode 1, so a
        caller using these sets its own response afterwards.

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
    st = reader.optical_state(variant, day_obs_range=day_obs_range, wide=True, ok_only=True)
    n_state = len(st)
    keep = ['visit_id', 'day_obs', 'seq_num', 'resid_rms_um', 'v1', 'v1_lut', 'v1_trim']
    keep += [c for c in keep_extra if c not in keep]
    st = st[[c for c in keep if c in st.columns]].copy()

    vis = telemetry(reader, day_obs_range)
    df = st.merge(vis, on='visit_id', how='left')
    n_joined = int(df['m1m3_z_gradient_c_per_m'].notna().sum())

    r2 = r2_terms(reader, day_obs_range)
    df = df.merge(r2, on='visit_id', how='left')
    n_r2 = int(df['m1m3_r2_coeff_c'].notna().sum()) if 'm1m3_r2_coeff_c' in df.columns else 0

    # The truss temperature is derived on the join, not stored, so this is the network step.
    df = reader.join_consdb(df, groups=CONSDB_GROUPS)

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
