#!/usr/bin/env python3
"""Focus look-up table (LUT) from science exposures: truss temperature, then elevation.

Extracts the dependence of the Active Optics System (AOS) uniform-defocus state on Telescope
Mount Assembly (TMA) truss temperature from ordinary science exposures, then studies the
elevation dependence of what remains and tests four candidate modulating factors.

The v-mode-1 amplitude per visit is assembled as

    v1_total = v1(hexapod LUT) + v1(Trim) + MEASURED_SIGN * v1(measured)

with ``MEASURED_SIGN = -1``. The LUT is the elevation- and temperature-dependent hexapod
baseline and the Trim is the accumulated closed-loop offset; a physical hexapod position is
LUT + Trim, neither alone. The measured term is the v-mode-1 amplitude of the optical state
recovered at the four Corner Wavefront Sensors (CWFS), read from the value-added database
rather than recomputed here.

Four stages, all in one invocation:

A. assemble the per-visit table from the value-added database plus live Consolidated Database
   (ConsDB) reads, and print the ``science_program`` inventory;
B. read one or more ``optical_state`` variants and form ``v1_total``;
C. fit ``v1_total`` against mean truss temperature per band, Huber with a Theil-Sen check;
D. fit the residual against elevation, then test each candidate modulator as a second
   regressor.

Usage
-----
One variant, the whole range::

    python code/science_lut/run_science_lut.py --variant v22_12__batoid__consdb_v1 \\
        --day-obs 20251023-20260913

Two variants side by side, which is what makes the scheme and intrinsic-route comparisons a
single run::

    python code/science_lut/run_science_lut.py \\
        --variant v22_12__batoid__consdb_v1 --variant v50_34__batoid__consdb_v1 \\
        --day-obs 20251023-20260913 --bands g r i z

Notes
-----
Two unit traps in the commanded vectors. ``lut_dof3/4/8/9`` are the hexapod tilts in **deg**
as ``MTHexapod`` reports them, while the Trim ``dof3/4/8/9`` are in **arcsec** following the
Optical Feedback Control (OFC) convention, so the tilt axes are converted before the DOF
vector is built. The hexapod LUT covers only the 10 hexapod degrees of freedom (DOF), so the
mirror bending entries are set to zero rather than left NaN — otherwise
`aos_state.vmodes_from_dofs` rejects every row on the inactive indices.

``MEASURED_SIGN = -1`` is a convention fixed by observation, not by any fit here: on the
``BLOCK-T539`` ``infocus_initial_alignment`` sequence the AOS answered a +3.87 µm of wavefront
focus error with −119.7 µm of camera hexapod dz, so the commanded motion opposes the measured
defocus and a surviving measured residual enters the sum with the sign that cancels it.

Fits are per band because filter thickness changes the camera-hexapod dz look-up table; a
pooled fit would leave a band-to-band focus offset in the residual as four offset clusters.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))                       # repo root -> common/
sys.path.insert(0, str(_ROOT / 'aos' / 'code'))      # flat cross-study modules

from common import efd_db                                        # noqa: E402
from common.utils import nmad                                    # noqa: E402

#: Overall sign of the measured term in v1_total. A convention, not a fitted quantity.
MEASURED_SIGN = -1.0

#: Hexapod tilt DOF indices: deg in the hexapod LUT, arcsec in the Trim.
HEX_TILT_LUT = [3, 4, 8, 9]
DEG_TO_ARCSEC = 3600.0

#: Commanded truss slope from the FAM Double Zernike fits, `code/correlations/run_dz14_truss.py`
#: [dimensionless v-mode-1 amplitude per deg C]. The science-image slope must land near this.
FAM_TRUSS_SLOPE = 0.09634

#: Danish wavefront-estimation v1.2 changeover, as a day_obs. Used as an epoch factor only.
DANISH_V12_DAY_OBS = 20260419

BAND_COLORS = {'u': 'tab:purple', 'g': 'tab:blue', 'r': 'tab:green',
               'i': 'tab:orange', 'z': 'tab:red', 'y': 'tab:brown'}


# --------------------------------------------------------------------------- fitting

def huber_fit(x, y, min_n=10):
    """Huber M-estimator slope with Pearson r and Spearman rho on the finite pairs.

    Parameters
    ----------
    x, y : `array_like`
        Paired values, each in its own single unit. The slope carries y-units per x-unit.
    min_n : `int`, optional
        Return None if fewer than this many finite pairs remain.

    Returns
    -------
    res : `dict` or `None`
        ``n``, ``slope``, ``slope_err``, ``intercept``, ``pearson_r``, ``spearman_rho``,
        ``resid_nmad``, ``chi2_dof``, ``dof``. `None` when under-determined.

    Notes
    -----
    ``chi2_dof`` uses the robust scatter as the per-point error, so it is a goodness-of-fit
    relative to the observed spread rather than to a propagated measurement error: it is near
    1 by construction for a well-behaved fit and inflates when the residual is heavy-tailed.
    """
    import statsmodels.api as sm
    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return None
    x, y = x[m], y[m]
    # A regressor that is constant over the finite pairs is rank-deficient against the
    # intercept: add_constant collapses to a single column and the slope does not exist.
    # Callers screen x over their whole selection, but a factor with contrast overall can
    # still be constant within one band, so the guard belongs here where the pairs are final.
    if np.ptp(x) == 0.0:
        return None
    X = sm.add_constant(x)
    r = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    resid = y - r.predict(X)
    sig = nmad(resid)
    dof = int(x.size - 2)
    chi2_dof = float(np.sum((resid / sig) ** 2) / dof) if (sig > 0 and dof > 0) else np.nan
    return dict(n=int(x.size), intercept=float(r.params[0]), slope=float(r.params[1]),
                slope_err=float(r.bse[1]),
                pearson_r=float(pearsonr(x, y)[0]),
                spearman_rho=float(spearmanr(x, y)[0]),
                resid_nmad=float(sig), chi2_dof=chi2_dof, dof=dof)


def theilsen_fit(x, y, min_n=10):
    """Theil-Sen slope on the same pairs, as a cross-check on the Huber fit.

    Parameters
    ----------
    x, y : `array_like`
        Paired values, each in its own single unit.
    min_n : `int`, optional

    Returns
    -------
    res : `dict` or `None`
        ``n``, ``slope``, ``slope_err``, ``intercept``, ``resid_nmad``. `slope_err` is half
        the width of the 95% confidence interval `scipy.stats.theilslopes` returns, so it is
        comparable in size to the Huber standard error but is not the same statistic.

    Notes
    -----
    Theil-Sen is non-parametric in the slope — the median of all pairwise slopes — so it does
    not assume the residual distribution. A large Huber/Theil-Sen disagreement therefore flags
    leverage from the extreme points of the predictor rather than a real trend.
    """
    from scipy.stats import theilslopes

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return None
    x, y = x[m], y[m]
    if np.ptp(x) == 0.0:      # every pairwise slope would be a division by zero
        return None
    slope, inter, lo, hi = theilslopes(y, x, alpha=0.95)
    resid = y - (inter + slope * x)
    return dict(n=int(x.size), slope=float(slope), intercept=float(inter),
                slope_err=float(0.5 * (hi - lo)), resid_nmad=float(nmad(resid)))


def fit_pair(x, y, stage, band, variant, xunit, yunit, min_n=10):
    """Huber and Theil-Sen on one sample, as rows for the fit table.

    Parameters
    ----------
    x, y : `array_like`
    stage : `str`
        Label for the fit table, e.g. ``'truss'`` or ``'elevation_resid'``.
    band : `str`
    variant : `str`
    xunit, yunit : `str`
        Units of the predictor and the response; the slope's unit is ``yunit/xunit``.
    min_n : `int`, optional

    Returns
    -------
    rows : `list` [`dict`]
        Zero, one or two rows — Huber first, then Theil-Sen.
    huber : `dict` or `None`
        The Huber result, so a caller can subtract the fit.
    """
    rows = []
    h = huber_fit(x, y, min_n)
    if h is not None:
        rows.append(dict(variant=variant, band=band, stage=stage, method='huber',
                         x_unit=xunit, y_unit=yunit,
                         slope_unit=f'{yunit} per {xunit}', **h))
    t = theilsen_fit(x, y, min_n)
    if t is not None:
        rows.append(dict(variant=variant, band=band, stage=stage, method='theilsen',
                         x_unit=xunit, y_unit=yunit,
                         slope_unit=f'{yunit} per {xunit}', **t))
    return rows, h


def partial_correlation(y, x, z):
    """Pearson correlation of `y` with `x` after linearly removing `z` from both.

    Parameters
    ----------
    y, x, z : `array_like`
        Each in its own single unit.

    Returns
    -------
    r : `float`
        Partial Pearson r [dimensionless], or NaN if fewer than 10 finite triples remain.
    n : `int`
        Triples used.
    """
    from scipy.stats import pearsonr

    y, x, z = (np.asarray(v, float) for v in (y, x, z))
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    y, x, z = y[m], x[m], z[m]
    Z = np.column_stack([np.ones_like(z), z])
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    if not (np.std(ry) > 0 and np.std(rx) > 0):
        return np.nan, int(len(y))
    return float(pearsonr(rx, ry)[0]), int(len(y))


# --------------------------------------------------------------------------- stage A

def program_inventory(df):
    """Distinct ``science_program`` with visit counts and ``day_obs`` range.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Must carry ``science_program``, ``visit_id``, ``day_obs`` and ``band``.

    Returns
    -------
    inv : `pandas.DataFrame`
        One row per program, descending by visit count.

    Notes
    -----
    Printed before any selection so the feature-based-scheduler program strings are read off
    the data rather than hardcoded.
    """
    g = (df.groupby('science_program')
           .agg(n_visits=('visit_id', 'size'), n_nights=('day_obs', 'nunique'),
                day_obs_min=('day_obs', 'min'), day_obs_max=('day_obs', 'max'),
                bands=('band', lambda s: ''.join(sorted(set(s.dropna())))))
           .sort_values('n_visits', ascending=False)
           .reset_index())
    return g


def assemble(day_obs_range, bands, programs=None, db_path=None, consdb_url='auto',
             verbose=True):
    """Stage A — the per-visit table, from the database plus live ConsDB reads.

    Parameters
    ----------
    day_obs_range : `tuple` [`int`]
        Inclusive ``(first, last)``; either may be None.
    bands : `list` [`str`] or None
        Bands to keep. None keeps every band.
    programs : `list` [`str`], optional
        ``science_program`` values to keep. None keeps all.
    db_path : `str`, optional
    consdb_url : `str`, optional
    verbose : `bool`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        One row per science visit, with the EFD Trim and hexapod LUT, the value-added
        columns, and the live ConsDB metadata, temperatures, wind and image quality.
    inv : `pandas.DataFrame`
        The program inventory, before the program cut.
    """
    cols = ([f'dof{k}' for k in range(50)] + [f'lut_dof{k}' for k in range(10)]
            + ['m1m3_x_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
               'm1m3_z_gradient_c_per_m', 'm1m3_radial_gradient_c_per_m',
               'into_wind_deg', 'wind_dir_deg', 'wind_speed_ms', 'azimuth_deg',
               'cum_hex_dz_um', 'recent_hex_dz_um', 'n_moves_night',
               'cam_AverageTemp', 'turb126_speed_mag_ms'])
    # all_columns() yields (name, sql_type) pairs, so take the names.
    have = {c[0] for c in efd_db.all_columns()}
    missing = [c for c in cols if c not in have]
    if missing and verbose:
        print(f'not in the database schema, skipped: {", ".join(missing)}')
    cols = [c for c in cols if c in have]
    df = efd_db.visits(day_obs_range=day_obs_range, columns=cols, db_path=db_path)
    if verbose:
        print(f'database: n = {len(df)} visits, {df.day_obs.nunique()} nights, '
              f'{len(df.columns)} columns')
    if not len(df):
        return df, pd.DataFrame()

    df = efd_db.join_consdb(df, consdb_url=consdb_url)
    if verbose:
        print(f'after join_consdb: {len(df)} rows, {len(df.columns)} columns')

    # ConsDB image-quality and temperature columns arrive as object dtype.
    for c in df.columns:
        if c in ('band', 'img_type', 'science_program', 'obs_start'):
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df[df['img_type'] == 'science'].copy()
    if verbose:
        print(f"img_type = 'science': n = {len(df)}")
    inv = program_inventory(df) if 'science_program' in df.columns else pd.DataFrame()
    if verbose and len(inv):
        print('\nscience_program inventory (read off the data, not hardcoded):')
        print(inv.to_string(index=False))
        print()
    if programs:
        df = df[df['science_program'].isin(programs)].copy()
        if verbose:
            print(f'after --programs {",".join(programs)}: n = {len(df)}')
    if bands:
        df = df[df['band'].isin(bands)].copy()
        if verbose:
            print(f'after --bands {",".join(bands)}: n = {len(df)}')
    return df.reset_index(drop=True), inv


# --------------------------------------------------------------------------- stage B

def commanded_v1(df, dof_set='standard_22', n_modes=12, verbose=True):
    """v-mode-1 amplitude of the commanded state: the hexapod LUT and the Trim.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Must carry ``dof0..49`` (µm and arcsec) and ``lut_dof0..9`` (µm and deg).
    dof_set : `str`, optional
        ts_ofc DOF-set name for the DOF -> v-mode projection.
    n_modes : `int`, optional
    verbose : `bool`, optional

    Returns
    -------
    v1_lut : `numpy.ndarray`
        v-mode-1 amplitude of the hexapod LUT [dimensionless].
    v1_trim : `numpy.ndarray`
        v-mode-1 amplitude of the Trim [dimensionless].
    v1_per_um_dz : `float`
        v-mode-1 amplitude per µm of hexapod dz [per µm], the mean magnitude over the camera
        (DOF 5) and M2 (DOF 0) axes.

    Notes
    -----
    The tilt axes are converted from deg to arcsec, and the LUT's mirror bending entries are
    set to zero rather than NaN, before the projection.
    """
    import aos_state
    se = aos_state.make_state_estimator(dof_set=dof_set)

    lut_dof = np.zeros((len(df), 50))
    lut_cols = [f'lut_dof{k}' for k in range(10)]
    lut_dof[:, :10] = df[lut_cols].to_numpy(float)
    lut_dof[:, HEX_TILT_LUT] *= DEG_TO_ARCSEC
    trim_dof = df[[f'dof{k}' for k in range(50)]].to_numpy(float)

    v1_lut = aos_state.vmodes_from_dofs(lut_dof, se, n_modes=n_modes)[:, 0]
    v1_trim = aos_state.vmodes_from_dofs(trim_dof, se, n_modes=n_modes)[:, 0]

    c = {}
    for k in (0, 5):
        d = np.zeros(50)
        d[k] = 1.0
        c[k] = float(aos_state.vmodes_from_dofs(d, se, n_modes=n_modes)[0, 0])
    v1_per_um_dz = 0.5 * (abs(c[5]) + abs(c[0]))
    if verbose:
        print(f'v1 per um of camera-hexapod dz (DOF 5) = {c[5]:+.7e} per um')
        print(f'v1 per um of M2-hexapod dz     (DOF 0) = {c[0]:+.7e} per um')
        print(f'mean magnitude                         = {v1_per_um_dz:.5e} per um; '
              f'the two axes agree to '
              f'{100 * abs(c[5] - c[0]) / v1_per_um_dz:.1f}% (dimensionless)')
        print(f'v1(hexapod LUT): n finite = {int(np.isfinite(v1_lut).sum())}, '
              f'median = {np.nanmedian(v1_lut):+.4f}, nMAD = {nmad(v1_lut):.4f} '
              f'(dimensionless)')
        print(f'v1(Trim)       : n finite = {int(np.isfinite(v1_trim).sum())}, '
              f'median = {np.nanmedian(v1_trim):+.4f}, nMAD = {nmad(v1_trim):.4f} '
              f'(dimensionless)')
    return v1_lut, v1_trim, v1_per_um_dz


def attach_measured(df, variant, day_obs_range, db_path=None, verbose=True):
    """Read one ``optical_state`` variant and attach its v-mode-1 amplitude.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carries ``visit_id``.
    variant : `str`
        Registered ``variant_id``.
    day_obs_range : `tuple` [`int`]
    db_path : `str`, optional
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        `df` with ``v1_meas`` [dimensionless], ``resid_rms_um`` [µm of wavefront] and
        ``dz_meas_dof5_um`` [µm, the recovered camera-hexapod dz] merged on ``visit_id``.
    n_state : `int`
        Visits for which the variant supplied a recovered state.

    Notes
    -----
    The read goes through `efd_db.optical_state`, which requires the variant explicitly — in
    a long table a forgotten variant filter silently multiplies the sample by the variant
    count, and the row count is asserted against the visit count here for that reason.
    """
    st = efd_db.optical_state(variant, day_obs_range=day_obs_range, wide=True,
                             db_path=db_path)
    if not len(st):
        if verbose:
            print(f'variant {variant}: no rows in this day_obs range')
        return df.assign(v1_meas=np.nan, resid_rms_um=np.nan, dz_meas_dof5_um=np.nan), 0
    if st['visit_id'].duplicated().any():
        raise RuntimeError(f'variant {variant} has duplicate visit_id rows — the long-table '
                           f'read lost its variant filter')
    # `dof5` here is the RECOVERED camera-hexapod dz of the measured state; the assembled
    # table already carries the commanded Trim under the same name, so it is renamed on the
    # way in rather than allowed to collide into dof5_x / dof5_y.
    keep = ['visit_id', 'v1']
    for c in ('resid_rms_um', 'dof5'):
        if c in st.columns:
            keep.append(c)
    st = st[keep].rename(columns={'v1': 'v1_meas', 'dof5': 'dz_meas_dof5_um'})
    out = df.merge(st, on='visit_id', how='left')
    n_state = int(np.isfinite(out['v1_meas']).sum())
    if verbose:
        print(f'variant {variant}: {len(st)} rows available, {n_state} of {len(out)} '
              f'science visits matched')
    if n_state == 0:
        # A variant built over a different night or image-type set than the selection leaves
        # v1_total all-NaN, and every downstream fit then silently reports nothing. Say so
        # here instead, with the two night sets, since that is the actual cause.
        sel_days = sorted(int(d) for d in df['day_obs'].unique())
        st_days = sorted(int(d) for d in st['day_obs'].unique())
        print(f'  WARNING: variant {variant} matched no selected visit. The variant covers '
              f'day_obs {st_days[:6]}{" ..." if len(st_days) > 6 else ""} and the selection '
              f'covers {sel_days[:6]}{" ..." if len(sel_days) > 6 else ""}. Build the '
              f'variant over the selected nights with build_optical_state.py, or widen '
              f'--bands / --day-obs.')
    return out, n_state


def build_v1_total(df, v1_lut, v1_trim, v1_per_um_dz, verbose=True):
    """Form ``v1_total`` and report the size of the measured term against the commanded one.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carries ``v1_meas`` [dimensionless].
    v1_lut, v1_trim : `array_like`
        [dimensionless]
    v1_per_um_dz : `float`
        [per µm], used only to express scatters in µm of equivalent hexapod dz.
    verbose : `bool`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        With ``v1_lut``, ``v1_trim``, ``v1_lut_trim`` and ``v1_total`` added, all
        dimensionless v-mode-1 amplitudes.
    """
    df = df.copy()
    df['v1_lut'] = v1_lut
    df['v1_trim'] = v1_trim
    df['v1_lut_trim'] = df['v1_lut'] + df['v1_trim']
    df['v1_total'] = df['v1_lut_trim'] + MEASURED_SIGN * df['v1_meas']
    if verbose:
        nm_cmd = nmad(df['v1_lut_trim'].to_numpy())
        nm_meas = nmad((MEASURED_SIGN * df['v1_meas']).to_numpy())
        nm_tot = nmad(df['v1_total'].to_numpy())
        print(f'\nv1_lut_trim : nMAD = {nm_cmd:.4f} (dimensionless) = '
              f'{nm_cmd / v1_per_um_dz:.1f} um of equivalent hexapod dz')
        print(f'v1_meas     : nMAD = {nm_meas:.4f} (dimensionless) = '
              f'{nm_meas / v1_per_um_dz:.1f} um of equivalent hexapod dz')
        print(f'v1_total    : nMAD = {nm_tot:.4f} (dimensionless) = '
              f'{nm_tot / v1_per_um_dz:.1f} um of equivalent hexapod dz, '
              f'n finite = {int(np.isfinite(df.v1_total).sum())}')
        verdict = 'reduces' if nm_tot < nm_cmd else 'increases'
        print(f'  adding the measured term with MEASURED_SIGN = {MEASURED_SIGN:+.0f} '
              f'{verdict} the scatter of the commanded state '
              f'(ratio {nm_tot / nm_cmd:.3f} dimensionless, total over commanded)')
    return df


# --------------------------------------------------------------------------- stages C, D

def fit_per_band(df, xcol, ycol, stage, variant, bands, xunit, yunit, verbose=True):
    """Per-band Huber and Theil-Sen fits, plus the residual about the Huber fit.

    Parameters
    ----------
    df : `pandas.DataFrame`
    xcol, ycol : `str`
    stage : `str`
    variant : `str`
    bands : `list` [`str`]
    xunit, yunit : `str`
    verbose : `bool`, optional

    Returns
    -------
    rows : `list` [`dict`]
        Fit-table rows, per band and per method, plus one pooled 'all' row per method.
    resid : `numpy.ndarray`
        Residual of `ycol` about its own band's Huber fit, NaN where not fitted.
    """
    rows = []
    resid = np.full(len(df), np.nan)
    x_all = df[xcol].to_numpy(float)
    y_all = df[ycol].to_numpy(float)
    for b in bands:
        m = (df['band'] == b).to_numpy() & np.isfinite(x_all) & np.isfinite(y_all)
        r, h = fit_pair(x_all[m], y_all[m], stage, b, variant, xunit, yunit)
        rows += r
        if h is not None:
            resid[m] = y_all[m] - (h['intercept'] + h['slope'] * x_all[m])
    # The band-pooled row is reported for completeness but is NOT the physical slope: the
    # camera-hexapod dz look-up table is filter-dependent, so pooling leaves a band-to-band
    # focus offset that biases a single straight line. Read the per-band rows instead.
    rows += fit_pair(x_all, y_all, stage, 'all_pooled', variant, xunit, yunit)[0]
    if verbose:
        print(f'\n{stage}: {ycol} against {xcol}   [{yunit} per {xunit}]')
        print(f'{"band":>5s} {"method":>9s} {"n":>7s} {"slope":>12s} {"slope_err":>10s} '
              f'{"intercept":>11s} {"Pearson r":>10s} {"Spearman":>9s} {"resid nMAD":>11s} '
              f'{"chi2/dof":>9s}')
        for r in rows:
            pr = f'{r["pearson_r"]:+10.3f}' if 'pearson_r' in r else f'{"--":>10s}'
            sp = f'{r["spearman_rho"]:+9.3f}' if 'spearman_rho' in r else f'{"--":>9s}'
            c2 = f'{r["chi2_dof"]:9.3f}' if 'chi2_dof' in r else f'{"--":>9s}'
            print(f'{r["band"]:>5s} {r["method"]:>9s} {r["n"]:7d} {r["slope"]:+12.5f} '
                  f'{r["slope_err"]:10.5f} {r["intercept"]:+11.4f} {pr} {sp} '
                  f'{r["resid_nmad"]:11.4f} {c2}')
        _report_method_agreement(rows, verbose=True)
    return rows, resid


def _report_method_agreement(rows, verbose=True):
    """Compare the Huber and Theil-Sen slopes band by band.

    Parameters
    ----------
    rows : `list` [`dict`]
        Fit rows from one stage.
    verbose : `bool`, optional

    Returns
    -------
    worst : `float`
        Largest |Huber − Theil-Sen| slope difference in units of the Huber standard error
        [dimensionless], or NaN if no band has both.

    Notes
    -----
    Theil-Sen is non-parametric in the slope, so a difference of many Huber standard errors
    means the Huber fit is being levered by the extreme points of the predictor rather than
    that either slope is wrong.
    """
    by = {}
    for r in rows:
        if r['band'] == 'all_pooled':
            continue        # not a physical slope; see fit_per_band
        by.setdefault(r['band'], {})[r['method']] = r
    worst, worst_band = np.nan, None
    for b, d in by.items():
        if 'huber' in d and 'theilsen' in d and d['huber']['slope_err'] > 0:
            z = abs(d['huber']['slope'] - d['theilsen']['slope']) / d['huber']['slope_err']
            if not np.isfinite(worst) or z > worst:
                worst, worst_band = z, b
    if verbose and np.isfinite(worst):
        note = ('consistent' if worst < 3 else
                'DISAGREE — check for leverage from the predictor extremes')
        print(f'  Huber vs Theil-Sen: largest slope difference {worst:.1f} Huber standard '
              f'errors (dimensionless), in the {worst_band} band — {note}')
    return worst


def modulator_tests(df, ycol, variant, bands, modulators, verbose=True):
    """Test each candidate factor as a second regressor on a residual.

    Parameters
    ----------
    df : `pandas.DataFrame`
    ycol : `str`
        Residual column to explain [dimensionless].
    variant : `str`
    bands : `list` [`str`]
    modulators : `list` [`tuple`]
        ``(column, label, unit)`` per candidate.
    verbose : `bool`, optional

    Returns
    -------
    rows : `list` [`dict`]
        One row per (band, modulator): the univariate Huber slope against the modulator, the
        partial Pearson r of the residual with the modulator at fixed elevation, the
        incremental ``chi2/dof`` from adding it, and the ``day_obs`` range over which the
        column is present.

    Notes
    -----
    A modulator whose column is absent for part of the range is fitted only where it exists,
    and the sub-range and n are reported rather than the shorter sample being passed off as
    the whole.
    """
    rows = []
    for col, label, unit in modulators:
        if col not in df.columns:
            if verbose:
                print(f'  {label}: column {col} absent from the assembled table — skipped')
            continue
        x = pd.to_numeric(df[col], errors='coerce').to_numpy(float)
        if not np.isfinite(x).any():
            if verbose:
                print(f'  {label}: no finite values over this range — skipped')
            continue
        # A regressor that is constant over the selection is rank-deficient against the
        # intercept: statsmodels returns slope 0.0 with error 0.0 and a NaN correlation
        # rather than raising, which reads as a real null result. Report the absence of
        # contrast instead. The Danish-epoch factor hits this whenever the selected nights
        # all fall on one side of the changeover.
        xf = x[np.isfinite(x)]
        if np.ptp(xf) == 0.0:
            if verbose:
                print(f'  {label}: constant at {xf[0]:+.6g} {unit} over the whole '
                      f'selection (no contrast) — skipped')
            continue
        sub_days = df.loc[np.isfinite(x), 'day_obs']
        d0, d1 = int(sub_days.min()), int(sub_days.max())
        for b in bands + ['all']:
            m = np.ones(len(df), bool) if b == 'all' else (df['band'] == b).to_numpy()
            y = df[ycol].to_numpy(float)
            el = df['altitude_deg'].to_numpy(float)
            ok = m & np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                continue
            h = huber_fit(x[ok], y[ok])
            if h is None:
                # Either too few pairs or no contrast within this band; say which, since a
                # silently absent row reads the same as a null result.
                if verbose:
                    why = ('constant over this band'
                           if np.ptp(x[ok]) == 0.0 else f'only {int(ok.sum())} pairs')
                    print(f'  {label}, {b} band: {why} — skipped')
                continue
            pr, npart = partial_correlation(y[ok], x[ok], el[ok])
            base = huber_fit(el[ok], y[ok])
            chi2_base = base['chi2_dof'] if base else np.nan
            chi2_add = _chi2_dof_two(y[ok], el[ok], x[ok])
            rows.append(dict(
                variant=variant, band=b, stage='modulator', method='huber',
                modulator=col, modulator_label=label, x_unit=unit,
                y_unit='dimensionless v-mode-1 amplitude',
                slope_unit=f'dimensionless v-mode-1 amplitude per {unit}',
                partial_pearson_r=pr, n_partial=npart,
                chi2_dof_elevation_only=chi2_base, chi2_dof_with_modulator=chi2_add,
                delta_chi2_dof=(chi2_add - chi2_base
                                if np.isfinite(chi2_base) and np.isfinite(chi2_add)
                                else np.nan),
                day_obs_min=d0, day_obs_max=d1, **h))
        if verbose:
            sel = [r for r in rows if r['modulator'] == col]
            print(f'\n  {label}  [{unit}]  present over day_obs {d0}-{d1}')
            print(f'  {"band":>5s} {"n":>7s} {"slope":>13s} {"slope_err":>10s} '
                  f'{"Pearson r":>10s} {"partial r|el":>13s} {"chi2/dof el":>12s} '
                  f'{"+modulator":>11s} {"delta":>8s}')
            for r in sel:
                print(f'  {r["band"]:>5s} {r["n"]:7d} {r["slope"]:+13.6f} '
                      f'{r["slope_err"]:10.6f} {r["pearson_r"]:+10.3f} '
                      f'{r["partial_pearson_r"]:+13.3f} '
                      f'{r["chi2_dof_elevation_only"]:12.3f} '
                      f'{r["chi2_dof_with_modulator"]:11.3f} '
                      f'{r["delta_chi2_dof"]:+8.3f}')
    return rows


def _chi2_dof_two(y, x1, x2):
    """chi2/dof of a two-regressor robust fit, scaled by the robust residual scatter.

    Parameters
    ----------
    y : `array_like`
        Response.
    x1, x2 : `array_like`
        Regressors, each in its own unit.

    Returns
    -------
    chi2_dof : `float`
        Dimensionless, with dof = n − 3. NaN if under-determined.
    """
    import statsmodels.api as sm

    y, x1, x2 = (np.asarray(v, float) for v in (y, x1, x2))
    m = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    if m.sum() < 12:
        return np.nan
    # Either regressor constant over the finite triples makes the design rank-deficient, and
    # the fit would then report a two-regressor chi2/dof that only one regressor earned.
    if np.ptp(x1[m]) == 0.0 or np.ptp(x2[m]) == 0.0:
        return np.nan
    X = sm.add_constant(np.column_stack([x1[m], x2[m]]))
    r = sm.RLM(y[m], X, M=sm.robust.norms.HuberT()).fit()
    resid = y[m] - r.predict(X)
    sig = nmad(resid)
    dof = int(m.sum() - 3)
    if not (sig > 0 and dof > 0):
        return np.nan
    return float(np.sum((resid / sig) ** 2) / dof)


# --------------------------------------------------------------------------- plotting

def plot_fit_page(pdf, df, xcol, ycol, bands, xlabel, ylabel, title, fits_by_band,
                  v1_per_um_dz=None):
    """One page of per-band scatter with the Huber line overlaid.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    xcol, ycol : `str`
    bands : `list` [`str`]
    xlabel, ylabel, title : `str`
    fits_by_band : `dict`
        Band -> Huber result dict, used for the overlaid line and the panel title.
    v1_per_um_dz : `float`, optional
        When given, adds a right-hand axis in µm of equivalent hexapod dz [per µm].
    """
    import matplotlib.pyplot as plt

    n = max(len(bands), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    y_all = df[ycol].to_numpy(float)
    y_all = y_all[np.isfinite(y_all)]
    lim = 5.0 * nmad(y_all) if len(y_all) else 1.0
    med = float(np.median(y_all)) if len(y_all) else 0.0
    for j, b in enumerate(bands):
        ax = axes[0][j]
        sub = df[df['band'] == b]
        x = sub[xcol].to_numpy(float)
        y = sub[ycol].to_numpy(float)
        ax.plot(x, y, '.', ms=2.0, alpha=0.3, color=BAND_COLORS.get(b, 'k'))
        r = fits_by_band.get(b)
        if r is not None and np.isfinite(x).any():
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 20)
            ax.plot(xs, r['intercept'] + r['slope'] * xs, '-', color='k', lw=1.3)
            ax.set_title(f'{b} band   slope {r["slope"]:+.5f} +/- {r["slope_err"]:.5f}\n'
                         f'Pearson r {r["pearson_r"]:+.3f}   '
                         f'Spearman rho {r["spearman_rho"]:+.3f}   n = {r["n"]}',
                         fontsize=8)
        else:
            ax.set_title(f'{b} band   too few finite pairs', fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        if j == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(med - lim, med + lim)
        if v1_per_um_dz:
            sec = ax.secondary_yaxis('right',
                                     functions=(lambda v: v / v1_per_um_dz,
                                                lambda v: v * v1_per_um_dz))
            sec.set_ylabel('equivalent hexapod dz [um]', fontsize=6)
            sec.tick_params(labelsize=6)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def plot_table_page(pdf, text, title, fontsize=7):
    """A monospaced text page, for the inventories and coverage tables.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    text : `str`
    title : `str`
    fontsize : `int`, optional
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.02, 0.97, title, fontsize=11, va='top')
    fig.text(0.02, 0.92, text, fontsize=fontsize, va='top', family='monospace')
    pdf.savefig(fig)
    plt.close(fig)


def plot_modulator_page(pdf, df, rows, col, label, unit, bands, ycol):
    """One page per candidate modulator: residual against the factor, per band.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
    df : `pandas.DataFrame`
    rows : `list` [`dict`]
        The modulator fit rows for this column.
    col, label, unit : `str`
    bands : `list` [`str`]
    ycol : `str`
    """
    fits = {r['band']: r for r in rows if r['modulator'] == col}
    if not fits:
        return
    d0 = min(r['day_obs_min'] for r in fits.values())
    d1 = max(r['day_obs_max'] for r in fits.values())
    plot_fit_page(pdf, df, col, ycol, bands, f'{label} [{unit}]',
                  'elevation-corrected residual\n[dimensionless v-mode-1 amplitude]',
                  f'Candidate modulator: {label} — present over day_obs {d0} to {d1}',
                  fits)


# --------------------------------------------------------------------------- driver

def run_variant(df0, variant, bands, day_obs_range, v1_lut, v1_trim, v1_per_um_dz,
                pdf=None, verbose=True):
    """Stages B to D for one ``optical_state`` variant.

    Parameters
    ----------
    df0 : `pandas.DataFrame`
        The assembled Stage A table.
    variant : `str`
    bands : `list` [`str`]
    day_obs_range : `tuple` [`int`]
    v1_lut, v1_trim : `array_like`
        Commanded v-mode-1 amplitudes for the rows of `df0` [dimensionless].
    v1_per_um_dz : `float`
        [per µm]
    pdf : `matplotlib.backends.backend_pdf.PdfPages`, optional
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per-visit rows for this variant, with the fit residuals.
    fit_rows : `list` [`dict`]
        Fit-table rows for this variant.
    """
    if verbose:
        print(f'\n{"=" * 78}\nvariant {variant}\n{"=" * 78}')
    df, _n_state = attach_measured(df0, variant, day_obs_range, verbose=verbose)
    df = build_v1_total(df, v1_lut, v1_trim, v1_per_um_dz, verbose=verbose)
    df['variant'] = variant

    fit_rows = []

    # Stage C — truss temperature.
    rows_c, resid_c = fit_per_band(df, 'truss_temp_mean_c', 'v1_total', 'truss', variant,
                                   bands, 'deg C', 'dimensionless v-mode-1 amplitude',
                                   verbose=verbose)
    fit_rows += rows_c
    df['v1_resid_truss'] = resid_c
    fits_c = {r['band']: r for r in rows_c if r['method'] == 'huber'}

    # The same fit on the commanded state alone, which is what the FAM slope compares with.
    rows_cmd, _ = fit_per_band(df, 'truss_temp_mean_c', 'v1_lut_trim', 'truss_commanded',
                               variant, bands, 'deg C',
                               'dimensionless v-mode-1 amplitude', verbose=verbose)
    fit_rows += rows_cmd
    if verbose:
        _compare_fam(rows_cmd)

    # Stage D — elevation in the truss residual.
    rows_d, resid_d = fit_per_band(df, 'altitude_deg', 'v1_resid_truss', 'elevation',
                                   variant, bands, 'deg elevation',
                                   'dimensionless v-mode-1 amplitude', verbose=verbose)
    fit_rows += rows_d
    df['v1_resid_truss_elev'] = resid_d
    fits_d = {r['band']: r for r in rows_d if r['method'] == 'huber'}

    # Stage D — the candidate modulators of the elevation-corrected residual.
    df['danish_epoch'] = (df['day_obs'] >= DANISH_V12_DAY_OBS).astype(float)
    modulators = [
        ('cum_hex_dz_um', 'cumulative |hexapod dz| since night start', 'um'),
        ('recent_hex_dz_um', 'trailing-window |hexapod dz|', 'um'),
        ('n_moves_night', 'commanded hexapod moves so far this night', 'count'),
        ('m1m3_radial_gradient_c_per_m', 'M1M3 radial thermal gradient', 'deg C per m'),
        ('m1m3_x_gradient_c_per_m', 'M1M3 x thermal gradient', 'deg C per m'),
        ('m1m3_y_gradient_c_per_m', 'M1M3 y thermal gradient', 'deg C per m'),
        ('m1m3_z_gradient_c_per_m', 'M1M3 z thermal gradient', 'deg C per m'),
        ('into_wind_deg', 'wind direction relative to azimuth (0 = into the wind)', 'deg'),
        ('wind_speed_ms', 'weather-station wind speed', 'm per s'),
        ('wind_inside_maxmag', 'salIndex 110 sonic anemometer maximum speed', 'm per s'),
        ('turb126_speed_mag_ms', 'salIndex 126 sonic anemometer speed magnitude', 'm per s'),
        ('danish_epoch', 'Danish v1.2 epoch (0 before, 1 from '
                         f'{DANISH_V12_DAY_OBS})', 'dimensionless'),
    ]
    if verbose:
        print(f'\ncandidate modulators of the elevation-corrected residual '
              f'(variant {variant}):')
    rows_m = modulator_tests(df, 'v1_resid_truss_elev', variant, bands, modulators,
                             verbose=verbose)
    fit_rows += rows_m

    if pdf is not None:
        plot_fit_page(pdf, df, 'truss_temp_mean_c', 'v1_total', bands,
                      'mean TMA truss temperature [deg C]',
                      'v1_total [dimensionless v-mode-1 amplitude]',
                      f'{variant}: v1_total against mean TMA truss temperature',
                      fits_c, v1_per_um_dz)
        plot_fit_page(pdf, df, 'altitude_deg', 'v1_resid_truss', bands,
                      'elevation [deg]',
                      'truss-corrected residual\n[dimensionless v-mode-1 amplitude]',
                      f'{variant}: elevation dependence of the truss residual',
                      fits_d, v1_per_um_dz)
        for col, label, unit in modulators:
            plot_modulator_page(pdf, df, rows_m, col, label, unit, bands,
                                'v1_resid_truss_elev')
    return df, fit_rows


def _compare_fam(rows_cmd, verbose=True):
    """Compare the commanded truss slopes with the FAM value.

    Parameters
    ----------
    rows_cmd : `list` [`dict`]
        Rows from the ``truss_commanded`` stage.
    verbose : `bool`, optional

    Returns
    -------
    worst : `float`
        Largest |science − FAM| commanded slope difference [dimensionless v-mode-1 amplitude
        per °C], or NaN.
    """
    hub = [r for r in rows_cmd if r['method'] == 'huber' and r['band'] != 'all']
    if not hub:
        return np.nan
    print(f'\n  commanded truss slope against the FAM value '
          f'{FAM_TRUSS_SLOPE:+.5f} dimensionless v-mode-1 amplitude per deg C '
          f'(run_dz14_truss.py):')
    worst = 0.0
    for r in hub:
        d = r['slope'] - FAM_TRUSS_SLOPE
        z = abs(d) / r['slope_err'] if r['slope_err'] > 0 else np.nan
        worst = max(worst, abs(d))
        print(f'  {r["band"]:>5s} {r["slope"]:+.5f} +/- {r["slope_err"]:.5f} per deg C, '
              f'difference {d:+.5f} per deg C ({z:.1f} standard errors)')
    tag = ('consistent with FAM' if worst < 0.03 else
           'DEVIATES from FAM by more than 0.03 per deg C — check units and DOF indexing')
    print(f'  largest difference {worst:.5f} dimensionless v-mode-1 amplitude per deg C '
          f'— {tag}')
    return worst


def parse_range(s):
    """Parse a ``day_obs`` argument into an inclusive range.

    Parameters
    ----------
    s : `str` or None
        ``'20251023'``, ``'20251023-20260913'``, or ``'20251023-'`` for open-ended.

    Returns
    -------
    rng : `tuple` [`int`] or None
        ``(first, last)``, either element possibly None.
    """
    if not s:
        return None
    if '-' in s:
        a, b = s.split('-', 1)
        return (int(a) if a.strip() else None, int(b) if b.strip() else None)
    d = int(s)
    return (d, d)


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', action='append', default=None,
                   help='registered optical_state variant id; repeatable')
    p.add_argument('--day-obs', default='20251023-',
                   help="inclusive day_obs range, e.g. '20251023-20260913'")
    p.add_argument('--bands', nargs='+', default=['g', 'r', 'i', 'z'])
    p.add_argument('--programs', nargs='+', default=None,
                   help='science_program values to keep; default all')
    p.add_argument('--out-dir', default=None,
                   help='output directory; default aos/output/science_lut')
    p.add_argument('--db', default=None)
    p.add_argument('--consdb-url', default='auto')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args(argv)
    verbose = not a.quiet

    if not a.variant:
        av = efd_db.variants(db_path=a.db)
        print('--variant is required. Registered variants:')
        print(av.to_string(index=False) if len(av) else '  none registered')
        return 2

    rng = parse_range(a.day_obs)
    out_dir = pathlib.Path(a.out_dir) if a.out_dir else (
        _ROOT / 'aos' / 'output' / 'science_lut')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage A.
    df0, inv = assemble(rng, a.bands, a.programs, db_path=a.db,
                        consdb_url=a.consdb_url, verbose=verbose)
    if not len(df0):
        print('no science visits in this selection — nothing to do')
        return 1

    v1_lut, v1_trim, v1_per_um_dz = commanded_v1(df0, verbose=verbose)

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = out_dir / 'science_lut.pdf'
    per_visit, fit_rows = [], []
    with PdfPages(pdf_path) as pdf:
        if len(inv):
            plot_table_page(pdf, inv.to_string(index=False),
                            'science_program inventory over the selected day_obs range')
        for vid in a.variant:
            d, fr = run_variant(df0, vid, a.bands, rng, v1_lut, v1_trim, v1_per_um_dz,
                                pdf=pdf, verbose=verbose)
            per_visit.append(d)
            fit_rows += fr
        cov = efd_db.coverage(db_path=a.db)
        if len(cov):
            keep = cov[cov['n_non_null'] > 0][
                ['column_name', 'group_name', 'units', 'first_day_obs', 'last_day_obs',
                 'n_non_null']]
            plot_table_page(pdf, keep.to_string(index=False),
                            'column_coverage: where each database column exists',
                            fontsize=5)

    pv = pd.concat(per_visit, ignore_index=True)
    fits = pd.DataFrame(fit_rows)
    pv_path = out_dir / 'science_lut.parquet'
    ft_path = out_dir / 'science_lut_fits.parquet'
    pv.to_parquet(pv_path)
    fits.to_parquet(ft_path)

    print(f'\nwrote {len(pv)} per-visit rows over {pv.variant.nunique()} variant(s) '
          f'-> {pv_path}')
    print(f'wrote {len(fits)} fit rows -> {ft_path}')
    print(f'wrote {pdf_path}')

    if len(a.variant) > 1:
        _compare_variants(pv, a.bands)
    return 0


def _compare_variants(pv, bands):
    """Report v1_meas agreement between the variants of one run.

    Parameters
    ----------
    pv : `pandas.DataFrame`
        Per-visit rows carrying ``variant`` and ``v1_meas``.
    bands : `list` [`str`]

    Notes
    -----
    v-mode 1 is non-degenerate, so it is unique across DOF truncations: two schemes at the
    same intrinsic route must agree on ``v1_meas`` to floating-point noise. Two intrinsic
    routes differ by the intrinsic wavefront, so an offset there is expected and a slope
    change is not.
    """
    from scipy.stats import pearsonr

    vs = sorted(pv['variant'].unique())
    print(f'\nvariant comparison on v1_meas [dimensionless v-mode-1 amplitude]:')
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a = pv[pv.variant == vs[i]].set_index('visit_id')['v1_meas']
            b = pv[pv.variant == vs[j]].set_index('visit_id')['v1_meas']
            k = a.index.intersection(b.index)
            x, y = a.loc[k].to_numpy(float), b.loc[k].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 10:
                print(f'  {vs[i]} vs {vs[j]}: fewer than 10 shared finite visits')
                continue
            d = y[m] - x[m]
            r = float(pearsonr(x[m], y[m])[0])
            print(f'  {vs[i]} vs {vs[j]}: n = {int(m.sum())}, '
                  f'max |difference| = {np.max(np.abs(d)):.6f}, '
                  f'median difference = {np.median(d):+.6f}, Pearson r = {r:.6f} '
                  f'(all dimensionless)')
    return None


if __name__ == '__main__':
    sys.exit(main())
