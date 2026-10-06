"""Shared definitions for the thermal-focus study: the response, the units and the features.

The quantity the whole study predicts is the telescope's uniform-defocus error — the focus the
Active Optics System (AOS) closed loop had accumulated but not yet corrected. It is formed from
v-mode 1, the amplitude of the first singular vector of the AOS sensitivity matrix, which is
essentially uniform defocus:

    response [µm of equivalent hexapod dz] = (v1_trim + MEASURED_SIGN * v1) / v1_per_um_dz

where ``v1_trim`` is v-mode 1 of the commanded Trim degrees of freedom (DOF) and ``v1`` is
v-mode 1 of the measured optical state. Both are dimensionless v-mode amplitudes; the division
puts the response into µm of hexapod dz travel, which is a length an observer can act on.

This module exists so that every consumer — the science-visit analysis, the Full Array Mode
(FAM) block analysis and the standalone online calculator — shares one definition. The response
was previously written out in three places, which is how a sign convention drifts.

Notes
-----
The hexapod Look-Up-Table (LUT) baseline is **deliberately excluded** from the response. A
physical hexapod position is LUT + Trim, but the LUT is a known commanded function of elevation
and temperature carrying essentially the whole elevation dependence and about 37x the measured
term's scatter. Including it would put a large elevation dependence into the response that is
not a focus error at all. What remains is what the closed loop and the wavefront sensors do on
their own.

Two unit traps live in the commanded vectors, and matter to anything that rebuilds a DOF vector
from telemetry rather than reading it from the value-added database. The hexapod tilts
``lut_dof3/4/8/9`` and the Trim ``dof3/4/8/9`` are **both in deg**, which is what
`aos_state.vmodes_from_dofs` expects: one unit of DOF 3 moves the v-modes by 22.74
(dimensionless v-mode norm per unit DOF 3) against an allowed range of 0.12, so the unit is a
degree. ``lsst.ts.intrinsic.wavefront.ofc_svd.DOF_UNITS_50`` labels those four arcsec, so a
comparison against results built on that convention needs 3600 arcsec/deg. The hexapod LUT
covers only the 10 hexapod DOF, so the mirror-bending entries must be set to zero rather than
left NaN, or `aos_state.vmodes_from_dofs` rejects every row on the inactive indices.
"""
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))                       # repo root -> common/
sys.path.insert(0, str(_ROOT / 'aos' / 'code'))      # -> aos_state, the v-mode engine
sys.path.insert(0, str(_ROOT / 'value_added' / 'code'))  # -> efd_db

#: Sign with which a surviving measured v-mode-1 residual enters the response
#: [dimensionless]. Fixed by observation, not by any fit: on the ``BLOCK-T539``
#: ``infocus_initial_alignment`` sequence of ``day_obs`` 20260513 the AOS answered a +3.87 µm of
#: wavefront focus error with −119.7 µm of camera hexapod dz, so the commanded motion opposes
#: the measured defocus and a surviving measured residual enters the sum with the sign that
#: cancels it. The same sequence confirms ``Trim = Trim_previous + Tweak`` exactly, an n -> n+2
#: visit latency in the commanded response, and a proportional gain of 0.75 (dimensionless) for
#: that block against the usual 0.3.
MEASURED_SIGN = -1.0

#: Nights on which the hexapod LUT epoch was offset, so the commanded baseline does not mean the
#: same thing as on neighbouring nights. Dropped from every fit.
LUT_EPOCH_OFFSET_NIGHTS = (20251102, 20251210, 20251211, 20251212,
                           20251218, 20251219, 20260115, 20260116)

#: Upper limit on the mean TMA truss temperature admitted to any fit [°C]. Two nights,
#: ``day_obs`` 20251118 and 20251119, carry 217 visits between +22.88 and +25.07 °C, detached
#: from the rest of the sample by an empty interval of 5.1792 °C — the largest gap anywhere above
#: +14 °C runs from +17.7016 to +22.8808 °C. A 20 °C cut sits in the middle of that gap and
#: removes exactly those 217 visits, with no boundary sensitivity: nothing else in the sample
#: lies within 2.3 °C of the threshold.
TRUSS_TEMP_MAX_C = 20.0

#: Equivalent hexapod dz per µm of wavefront defocus [µm of equivalent hexapod dz per µm of
#: wavefront]. Derived in ``smatrix/notebooks/vmode/ofc_conversion_constants.ipynb`` by the
#: full 50-DOF pseudo-inverse. Negative: positive hexapod dz produces negative defocus. The
#: camera-only inverse gives −62.8389 and the singular-value-decomposition minimum-norm solution
#: −63.2195, so the three routes agree to 1.2% (dimensionless, spread over mean).
DZ_UM_PER_UM_WF = -63.9902

#: Per-axis v-mode-1 response, filled by `v1_per_um_dz_value` and keyed by DOF index: 5 is the
#: camera-hexapod dz axis, 0 the M2-hexapod dz axis [dimensionless v-mode-1 amplitude per µm].
#: Kept reachable so a document can state what the mean of the two means physically without
#: hardcoding numbers that would drift with the scheme.
V1_PER_UM_DZ_AXES = {}

#: Telemetry feature groups, resolved by `resolve_features`. The deliverable model is
#: ``('truss', 'grads')`` — the Telescope Mount Assembly (TMA) truss temperature and the four
#: M1M3 bulk thermal gradients.
FEATURE_GROUPS = {
    'truss': ['truss_temp_mean_c'],
    'grads': ['m1m3_z_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
              'm1m3_radial_gradient_c_per_m', 'm1m3_x_gradient_c_per_m'],
    'zgrad': ['m1m3_z_gradient_c_per_m'],
    'r2grads': ['m1m3_r2_coeff_c', 'm1_r2_coeff_c', 'm3_r2_coeff_c'],
    'r2all': ['m1m3_r2_coeff_c'],
    'r2split': ['m1_r2_coeff_c', 'm3_r2_coeff_c'],
    'camtemp': ['cam_AverageTemp'],
    'wind': ['wind_speed_ms', 'into_wind_deg'],
    'hexhist': ['cum_hex_dz_um', 'recent_hex_dz_um', 'n_moves_night'],
    'elev': ['altitude_deg'],
}

#: The deliverable feature set: five thermal channels, band-independent.
DELIVERABLE_GROUPS = ('truss', 'grads')

#: M1M3 thermal-gradient columns and their labels [°C/m].
GRAD_COLS = (('m1m3_z_gradient_c_per_m', 'M1M3 z thermal gradient'),
             ('m1m3_radial_gradient_c_per_m', 'M1M3 radial thermal gradient'),
             ('m1m3_x_gradient_c_per_m', 'M1M3 x thermal gradient'),
             ('m1m3_y_gradient_c_per_m', 'M1M3 y thermal gradient'))

#: Quadratic-in-radius M1M3 thermal columns and their labels [°C per unit normalized radius-
#: squared amplitude]. Built by ``value_added/code/m1m3_thermal_r2.py`` over three thermocouple
#: populations — the whole mirror, the M1 annulus and the M3 inner disc. The quadratic shape is
#: Gram-Schmidt orthogonalized against the constant, linear-radius and depth terms over each
#: population's own sensor positions and scaled to unit root-mean-square, so the coefficient is
#: the radial curvature those terms cannot express and is not a °C/m² curvature; see
#: ``m1m3_thermal_r2.R2_SHAPE_RMS_M2`` for the conversion back.
R2_COLS = (('m1m3_r2_coeff_c', 'M1M3 quadratic radial thermal term'),
           ('m1_r2_coeff_c', 'M1 quadratic radial thermal term'),
           ('m3_r2_coeff_c', 'M3 quadratic radial thermal term'))

#: Default recovered-optical-state variant, kept for continuity with the published v-mode-1
#: result. All four registered variants now carry rows, including the range-bounded recovery and
#: the measured-intrinsic-wavefront route; ``thermal_vmodes.PRIMARY_VARIANT`` prefers the
#: range-bounded one, which is physically realizable.
DEFAULT_VARIANT = 'v50_34__batoid__consdb_v1'

#: Default FAM Double Zernike (DZ) variant, the only one populated.
DEFAULT_FAM_VARIANT = ('fam__fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x__batoid__z1toz6__50_34')


def v1_per_um_dz_value(dof_set='all_50', n_modes=34, verbose=True):
    """v-mode-1 amplitude per µm of hexapod dz [per µm].

    Parameters
    ----------
    dof_set : `str`, optional
        ts_ofc degree-of-freedom (DOF) set name for the DOF to v-mode projection.
    n_modes : `int`, optional
        Number of v-modes retained in the projection.
    verbose : `bool`, optional
        Print the two per-axis coefficients and what their mean means.

    Returns
    -------
    v1_per_um_dz : `float`
        Mean magnitude over the camera-hexapod (DOF 5) and M2-hexapod (DOF 0) dz axes
        [dimensionless v-mode-1 amplitude per µm of total dz travel].

    Notes
    -----
    Derived rather than copied, so the conversion factor cannot drift from the stored v-modes.
    The projection is `aos_state.vmodes_from_dofs` through `aos_state.make_state_estimator` --
    the single sanctioned v-mode engine, the basis the measured state's v-modes are reported in
    and the commanded terms are stored in.

    **The return is a magnitude; the sign is carried by `MEASURED_SIGN`.** Both axes are
    genuinely negative — −8.9144254e-04 on the camera hexapod (DOF 5) and −9.1026032e-04 on M2
    (DOF 0), dimensionless per µm — so taking magnitudes here is a deliberate choice, not a lost
    sign.

    What the mean magnitude means physically: the two coefficients carry the **same sign**, so
    the two axes add rather than cancel, and their sum over their mean is 2.00000 (dimensionless)
    to five decimal places. Moving 0.5 µm on each hexapod -- 1 µm of **total** dz travel --
    therefore produces exactly this factor's worth of v1. So a value reported in these units is
    µm of total defocus travel split evenly between the camera and M2 hexapods, and **not** µm of
    camera-hexapod motion with M2 held still. Per unit v-mode-1 amplitude that is 1110.1 µm of
    total travel shared, 555.0 µm on each hexapod, against 1121.8 µm if the camera hexapod moves
    alone -- the two differ by only 1.1% (dimensionless), because the two coefficients agree to
    2.1%.

    v1 is the camera-hexapod dz and M2-hexapod dz combination plus small mirror-bending terms,
    and is the same mode in both schemes, so this factor comes out equal to five decimal places
    at ``standard_22``/12 and ``all_50``/34. That equality is **not** safe to assume at the
    10-DOF/1-mode projection the online system would use, where the two routes differ; see
    `v1_per_um_dz_table`.
    """
    import aos_state
    se = aos_state.make_state_estimator(dof_set=dof_set, n_modes=n_modes)
    c = {}
    for k in (0, 5):
        d = np.zeros(50)
        d[k] = 1.0
        c[k] = float(aos_state.vmodes_from_dofs(d, se, n_modes=n_modes)[0, 0])
    val = 0.5 * (abs(c[5]) + abs(c[0]))
    V1_PER_UM_DZ_AXES.update({5: c[5], 0: c[0]})
    if verbose:
        print(f'v1 per um camera-hexapod dz (DOF 5) = {c[5]:+.7e} per um')
        print(f'v1 per um M2-hexapod dz     (DOF 0) = {c[0]:+.7e} per um')
        print(f'mean magnitude = {val:.5e} per um; axes agree to '
              f'{100 * abs(c[5] - c[0]) / val:.1f}% (dimensionless)')
        print(f'  the two axes share a sign, so their sum over their mean is '
              f'{abs(c[5] + c[0]) / val:.5f} (dimensionless): this factor is 1 um of TOTAL '
              f'dz travel, 0.5 um on each hexapod')
    return val


def v1_per_um_dz_table(schemes=(('all_50', 34), ('standard_22', 12), ('hexapod_10', 1)),
                       verbose=True):
    """The dz conversion across projection schemes, including the online 10-DOF/1-mode case.

    Parameters
    ----------
    schemes : `tuple` [`tuple`], optional
        Each entry ``(dof_set, n_modes)``: a ts_ofc DOF set name and the number of v-modes.
    verbose : `bool`, optional
        Print the table as it is built.

    Returns
    -------
    rows : `list` [`dict`]
        One entry per scheme with ``dof_set``, ``n_modes``, ``c5`` and ``c0`` (dimensionless
        v-mode-1 amplitude per µm on the camera and M2 hexapod dz axes), ``mean_mag``
        (dimensionless per µm of total travel), ``um_per_v1_shared`` and ``um_per_v1_camera``
        (µm of hexapod dz per unit v-mode-1 amplitude, sharing the motion between the two
        hexapods and moving the camera hexapod alone), and ``axes_agree_pct``.

    Notes
    -----
    Why this exists: the conversion is equal to five decimal places at ``all_50``/34 and
    ``standard_22``/12, which invites the assumption that it is scheme-independent. It is not
    safe to assume that at the 10-DOF/1-mode projection, where only the two hexapods and one
    mode are retained, so the online calculator's constant is read from this table rather than
    inherited. A scheme whose DOF set omits an axis returns NaN for that axis rather than
    raising, since the mode is then not the same physical combination.
    """
    import aos_state
    rows = []
    for dof_set, n_modes in schemes:
        se = aos_state.make_state_estimator(dof_set=dof_set, n_modes=n_modes)
        c = {}
        for k in (0, 5):
            d = np.zeros(50)
            d[k] = 1.0
            try:
                c[k] = float(aos_state.vmodes_from_dofs(d, se, n_modes=n_modes)[0, 0])
            except (IndexError, ValueError):
                c[k] = float('nan')
        mean_mag = 0.5 * (abs(c[5]) + abs(c[0]))
        row = {'dof_set': dof_set, 'n_modes': n_modes, 'c5': c[5], 'c0': c[0],
               'mean_mag': mean_mag,
               'um_per_v1_shared': 1.0 / mean_mag if mean_mag else float('nan'),
               'um_per_v1_camera': 1.0 / abs(c[5]) if c[5] else float('nan'),
               'axes_agree_pct': (100 * abs(c[5] - c[0]) / mean_mag
                                  if mean_mag else float('nan'))}
        rows.append(row)
        if verbose:
            print(f'{dof_set:>12s}/{n_modes:<3d} '
                  f'c5 {c[5]:+.7e}  c0 {c[0]:+.7e} per um  '
                  f'mean {mean_mag:.5e} per um  '
                  f'{row["um_per_v1_shared"]:8.1f} um shared / '
                  f'{row["um_per_v1_camera"]:8.1f} um camera-alone per unit v1  '
                  f'axes agree {row["axes_agree_pct"]:.1f}%')
    return rows


def resolve_features(groups):
    """Expand feature-group names into telemetry column names.

    Parameters
    ----------
    groups : `iterable` [`str`]
        Keys of `FEATURE_GROUPS`.

    Returns
    -------
    features : `list` [`str`]
        Column names, de-duplicated, in the order the groups were given.

    Raises
    ------
    ValueError
        If a group name is not in `FEATURE_GROUPS`.
    """
    features = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise ValueError(f'unknown feature group {g!r}; choose from '
                             f'{", ".join(sorted(FEATURE_GROUPS))}')
        for c in FEATURE_GROUPS[g]:
            if c not in features:
                features.append(c)
    return features


def attach_response(df, v1_per_um_dz, trim_col='v1_trim', meas_col='v1', out_col='y'):
    """Add the response column to a per-visit table.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying the commanded and measured v-mode-1 amplitudes [dimensionless].
    v1_per_um_dz : `float`
        Conversion from `v1_per_um_dz_value` [dimensionless v-mode-1 amplitude per µm of total
        hexapod dz travel].
    trim_col, meas_col : `str`, optional
        Columns holding v-mode 1 of the commanded Trim and of the measured state.
    out_col : `str`, optional
        Column to write.

    Returns
    -------
    df : `pandas.DataFrame`
        A copy with `out_col` added [µm of equivalent hexapod dz].

    Notes
    -----
    The hexapod LUT term is not part of this sum; see the module docstring.
    """
    out = df.copy()
    out[out_col] = (out[trim_col] + MEASURED_SIGN * out[meas_col]) / v1_per_um_dz
    return out
