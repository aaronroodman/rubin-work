"""Value-added telemetry database — slow Engineering Facility Database (EFD) reads and
derived quantities, fetched once and stored per visit.

The Consolidated Database (ConsDB), including its transformed-EFD tables, is fast, so
**nothing that lives there is copied here**. This database holds exactly two kinds of
thing:

* **EFD quantities.** Direct EFD access is slow and was scattered across several topics'
  scripts. Each quantity is fetched once into this database, which then becomes the single
  place EFD access happens.
* **Value-added quantities** — expensive or intricate to *compute* rather than to fetch.
  The M1M3 bulk thermal gradients (derived from raw thermocouple telemetry, and fetchable
  only one night at a time) are one; the recovered optical state, needing an
  intrinsic-wavefront subtraction and a sensitivity-matrix singular value decomposition
  (SVD), is another.

Two storage shapes, for two different kinds of quantity:

* `visit_telemetry` is **wide**, one row per exposure. Its groups (Trim, hexapod look-up
  table (LUT), Tweak, gradients, camera-body temperatures, air turbulence, hexapod motion
  history) are each one value per visit per quantity, arrive together from the same
  per-night fetch, and have no variants.
* `optical_state` is **long**, keyed ``(visit_id, variant_id)``. The recovered optical
  state is a *family* of variants along three independent axes — reduction scheme
  (22/12, 50/34), intrinsic-wavefront route (batoid, measured intrinsic wavefront (MIW)),
  and optical path difference (OPD) source version — so a new variant must be rows rather
  than schema. `state_variant` is the registry describing each one.

Import from the repo root::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N]))
    from common import efd_db

Typical read, EFD columns joined to live ConsDB metadata::

    df = efd_db.visits(day_obs_range=(20251023, None))
    df = efd_db.join_consdb(df)                      # adds band, pointing, temperatures
    st = efd_db.optical_state('v50_34__batoid__consdb_v1')

Notes
-----
`optical_state` requires an explicit `variant` on every read. A long table with a
forgotten variant filter silently multiplies the sample by the variant count, which is the
one failure mode this shape introduces.

Sonic-anemometer and turbulence sensor locations are recorded in ``ts_config_ocs``.
"""
import os
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DEFAULT_DB = 'output/value_added/aos_efd.duckdb'

# ---------------------------------------------------------------------------
# Column inventory.  Each group is (column, sql_type, units, source) and is the
# single source of truth for both the CREATE TABLE and the column_coverage rows.
# 'units' is free text but must always be filled: a bare number in this database is
# a bug.
# ---------------------------------------------------------------------------
IDENTITY = [
    ('visit_id', 'BIGINT', 'dimensionless (exposure id)', 'consdb_exposure'),
    ('day_obs', 'INTEGER', 'dimensionless (YYYYMMDD)', 'consdb_exposure'),
    ('seq_num', 'INTEGER', 'dimensionless (sequence number)', 'consdb_exposure'),
    ('obs_start', 'VARCHAR', 'TAI ISO-8601 timestamp', 'consdb_exposure'),
]

N_DOF = 50
N_HEX_LUT = 10

#: Camera-body temperature fields, mirroring
#: ``aos/code/fam_processing/run_attach_telemetry.py:CAM_FIELDS`` so both products carry the
#: same column names. ``DomeXMinusTemp`` is deliberately absent — the sensor read 0 of 3385
#: visits finite while its ``DomeYMinusTemp`` sibling read 98.6%.
CAM_FIELDS = [
    'AverageTemp',
    'CamBodyXPlusTemp', 'CamBodyYPlusTemp', 'CamBodyYMinusTemp',
    'CamHousXPlusTemp', 'CamHousXMinusTemp', 'CamHousYPlusTemp', 'CamHousYMinusTemp',
    'BackFlngXMinusTemp', 'BackFlngYMinusTemp',
    'ShrdRngXPlusTemp', 'ShrdRngXMinusTemp', 'ShrdRngYPlusTemp',
    'L1XMinusTemp', 'L1YMinusTemp', 'L2XPlusTemp', 'L2XMinusTemp', 'L2YPlusTemp',
    'DomeYMinusTemp', 'VPPlenumInTemp',
    'ShtrEboxRtnAirTemp', 'ShtrMtrRtnAirTemp', 'ChgrYMinusRtnAirTemp', 'AmbAirtemp',
]

#: 3D sonic anemometer ESS ``salIndex`` values on the ``lsst.sal.ESS.airTurbulence``
#: topic. 110 is the deployable platform at the top of the stairs and *is* in the
#: transformed EFD of the ConsDB (as ``mt_salindex110_*``, a fast live read); 123–126 are
#: the Telescope Mount Assembly (TMA) top-ring quadrants (-x-y / +x-y / +x+y / -x+y) and
#: are **not**, which is why they are fetched here. Locations are recorded in
#: ``ts_config_ocs`` ``ESS/v8/_init.yaml``.
#:
#: Two epochs bound what these columns can contain, both measured against the EFD:
#:
#: * **salIndex 123–126 first report 2026-01-28/29** (``TMA-3D-ANEM-01..04``). Earlier
#:   nights are legitimately NULL, and `column_coverage` reports that as their true
#:   ``first_day_obs``.
#: * **The whole topic goes silent between 2026-07-20 and 2026-07-22**, salIndex 110
#:   included, having reported continuously since April 2025. Nights after that carry no
#:   ``turb*`` values from any sensor.
#:
#: salIndex 201 also appears on this topic from about 2026-01-10; it is ``AuxTel-ESS04``,
#: not a TMA sensor, and is deliberately excluded.
TURB_SALINDEX = (123, 124, 125, 126)

#: ``airTurbulence`` field -> (column suffix, units). The per-axis components are the
#: wind-frame speeds; ``speedMagnitude`` is the 3D magnitude.
TURB_FIELDS = {
    'speed0': ('speed_0_ms', 'm/s'),
    'speed1': ('speed_1_ms', 'm/s'),
    'speed2': ('speed_2_ms', 'm/s'),
    'speedMagnitude': ('speed_mag_ms', 'm/s'),
    'speedMaxMagnitude': ('speed_max_ms', 'm/s'),
    'speedStdDev0': ('speed_std_0_ms', 'm/s'),
    'speedStdDev1': ('speed_std_1_ms', 'm/s'),
    'speedStdDev2': ('speed_std_2_ms', 'm/s'),
    'sonicTemperature': ('sonic_temp_c', 'deg C'),
    'sonicTemperatureStdDev': ('sonic_temp_std_c', 'deg C'),
}

GROUPS = {
    # --- raw EFD -----------------------------------------------------------
    'trim': (
        [(f'dof{i}', 'DOUBLE',
          'um (z/x/y, bending) or deg (u/v)', 'efd_MTAOS_degreeOfFreedom')
         for i in range(N_DOF)]),
    # No event-id column: aos_trim.fetch_hexapod_lut_for_visits returns only a count, so
    # unlike Trim there is no per-visit source event to record.
    'lut': (
        [(f'lut_dof{i}', 'DOUBLE',
          'um (z/x/y) or deg (u/v)', 'efd_MTHexapod_compensationOffset')
         for i in range(N_HEX_LUT)]),
    'camera': (
        [(f'cam_{f}', 'DOUBLE', 'deg C', 'efd_MTCamera_utiltrunk_body')
         for f in CAM_FIELDS]
        + [('cam_n_samp', 'INTEGER', 'dimensionless (samples averaged)',
            'efd_MTCamera_utiltrunk_body')]),
    'turbulence': (
        [(f'turb{idx}_{suffix}', 'DOUBLE', units,
          f'efd_ESS_airTurbulence_idx{idx}')
         for idx in TURB_SALINDEX for suffix, units in TURB_FIELDS.values()]
        + [(f'turb{idx}_n_samp', 'INTEGER', 'dimensionless (samples averaged)',
            f'efd_ESS_airTurbulence_idx{idx}') for idx in TURB_SALINDEX]),
    # --- value-added -------------------------------------------------------
    'gradients': (
        [(f'm1m3_{n}_gradient_c_per_m', 'DOUBLE', 'deg C/m',
          'derived_from_efd_M1M3_thermocouples')
         for n in ('x', 'y', 'z', 'radial')]),
    'tweak': (
        [(f'tweak_dof{i}', 'DOUBLE',
          'um (z/x/y, bending) or deg (u/v)', 'derived_from_trim')
         for i in range(N_DOF)]),
    'hexhist': [
        ('cum_hex_dz_um', 'DOUBLE', 'um (cumulative |delta dz| since night start)',
         'derived_from_lut_trim'),
        ('recent_hex_dz_um', 'DOUBLE', 'um (|delta dz| over trailing 30 min)',
         'derived_from_lut_trim'),
        ('n_moves_night', 'INTEGER', 'dimensionless (count of commanded moves)',
         'derived_from_lut_trim'),
    ],
    'wind_derived': [
        ('wind_dir_deg', 'DOUBLE', 'deg (direction wind comes FROM)',
         'consdb_exposure_weather'),
        ('wind_speed_ms', 'DOUBLE', 'm/s', 'consdb_exposure_weather'),
        ('azimuth_deg', 'DOUBLE', 'deg', 'consdb_exposure'),
        ('into_wind_deg', 'DOUBLE', 'deg (0 = pointing into the wind)',
         'derived_wrap180_winddir_minus_az'),
    ],
}

#: Groups whose columns live in ``visit_telemetry``, in build order. ``hexhist``
#: depends on ``lut``/``trim`` already being present, so it is last.
GROUP_ORDER = ('trim', 'lut', 'camera', 'turbulence', 'gradients', 'tweak',
               'wind_derived', 'hexhist')


def group_columns(group):
    """Column specifications for one group.

    Parameters
    ----------
    group : `str`
        A key of `GROUPS`.

    Returns
    -------
    cols : `list` [`tuple`]
        ``(name, sql_type, units, source)`` per column.
    """
    if group not in GROUPS:
        raise KeyError(f'unknown group {group!r}; choose from {sorted(GROUPS)}')
    return list(GROUPS[group])


def all_columns():
    """Every `visit_telemetry` column specification, identity block first."""
    out = list(IDENTITY)
    seen = {c[0] for c in out}
    for g in GROUP_ORDER:
        for spec in GROUPS[g]:
            if spec[0] not in seen:       # azimuth_deg is shared with the identity read
                out.append(spec)
                seen.add(spec[0])
    return out


# ---------------------------------------------------------------------------
# Connection and schema
# ---------------------------------------------------------------------------
def default_db_path():
    """Absolute path to the default database file, resolved from the repo root.

    Returns
    -------
    path : `pathlib.Path`
        ``<repo_root>/output/value_added/aos_efd.duckdb``, overridable with the
        ``AOS_EFD_DB`` environment variable.

    Notes
    -----
    Resolved via ``parents[1]`` of this file rather than a hardcoded ``/sdf`` or
    ``/home`` path, so it is correct in the RSP notebook, an RSP terminal, and a Slurm
    job alike.
    """
    env = os.environ.get('AOS_EFD_DB')
    if env:
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[1] / DEFAULT_DB


def open_db(path=None, readonly=True, create=False):
    """Open the database.

    Parameters
    ----------
    path : `str` or `pathlib.Path`, optional
        Database file; defaults to `default_db_path`.
    readonly : `bool`, optional
        Open read-only (the default). Many readers may share a read-only handle; a
        writer needs exclusive access.
    create : `bool`, optional
        Create the file and schema if absent. Implies ``readonly=False``.

    Returns
    -------
    con : `duckdb.DuckDBPyConnection`

    Raises
    ------
    FileNotFoundError
        If the file does not exist and `create` is false.
    `duckdb.IOException`
        If a writer holds the file. DuckDB's lock is process-wide and excludes readers
        too, so a read-only open fails while a build is running -- track a backfill
        through the builder's own log rather than by querying `fetch_log`.

    Notes
    -----
    Many readers may share the file once no writer is attached.
    """
    import duckdb

    path = pathlib.Path(path) if path is not None else default_db_path()
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        readonly = False
    elif not path.exists():
        raise FileNotFoundError(
            f'no value-added database at {path}\n'
            f'build it with: python common/scripts/build_efd_db.py --day-obs <range>')
    con = duckdb.connect(str(path), read_only=readonly)
    if create:
        create_schema(con)
    return con


def create_schema(con):
    """Create every table and index if not already present.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        A writable connection.

    Notes
    -----
    Idempotent, and additive: a column added to `GROUPS` later appears here on the next
    call via ``ALTER TABLE ... ADD COLUMN``, so extending the inventory never requires a
    rebuild.
    """
    cols = all_columns()
    ddl = ',\n  '.join(f'{n} {t}' for n, t, _u, _s in cols)
    con.execute(f'CREATE TABLE IF NOT EXISTS visit_telemetry (\n  {ddl},\n'
                '  PRIMARY KEY (visit_id)\n)')
    # Additive migration for columns introduced after the table was first created.
    have = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'visit_telemetry'").fetchall()}
    for n, t, _u, _s in cols:
        if n not in have:
            con.execute(f'ALTER TABLE visit_telemetry ADD COLUMN {n} {t}')

    con.execute("""
        CREATE TABLE IF NOT EXISTS state_variant (
          variant_id          VARCHAR PRIMARY KEY,
          scheme              VARCHAR,     -- '22_12' | '50_34'
          n_dof               INTEGER,
          n_modes             INTEGER,
          intrinsic_route     VARCHAR,     -- 'batoid' | 'miw'
          intrinsic_ref       VARCHAR,     -- MIW build name, or OFC config version
          opd_source          VARCHAR,
          opd_version         VARCHAR,
          ofc_config_version  VARCHAR,
          created_at          TIMESTAMP,
          notes               VARCHAR
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS optical_state (
          visit_id      BIGINT,
          variant_id    VARCHAR,
          v_modes       DOUBLE[],   -- length n_modes, um of wavefront
          dof           DOUBLE[],   -- length 50, um / deg
          n_modes       INTEGER,
          resid_rms_um  DOUBLE,     -- um of wavefront
          ok            BOOLEAN,
          computed_at   TIMESTAMP,
          PRIMARY KEY (visit_id, variant_id)
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS column_coverage (
          column_name    VARCHAR PRIMARY KEY,
          group_name     VARCHAR,
          source         VARCHAR,
          units          VARCHAR,
          first_day_obs  INTEGER,
          last_day_obs   INTEGER,
          n_non_null     BIGINT,
          updated_at     TIMESTAMP
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
          day_obs      INTEGER,
          group_name   VARCHAR,
          status       VARCHAR,     -- 'ok' | 'empty' | 'error'
          n_rows       INTEGER,
          error        VARCHAR,
          attempted_at TIMESTAMP,
          PRIMARY KEY (day_obs, group_name)
        )""")
    con.execute('CREATE INDEX IF NOT EXISTS vt_day_obs ON visit_telemetry (day_obs)')
    con.execute('CREATE INDEX IF NOT EXISTS os_variant ON optical_state (variant_id)')


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def upsert_visits(con, df, group=None):
    """Insert or update one group's columns for a set of visits.

    Only the identity block and `group`'s own columns are written, so a later group's
    pass over the same visits leaves this group's values intact.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Writable connection.
    df : `pandas.DataFrame`
        Must carry `visit_id` and the identity columns; any of `group`'s columns that are
        absent are written as NULL.
    group : `str`, optional
        A key of `GROUPS`. Omit to write the identity block alone, which establishes the
        night's rows without touching any group's columns — use this before the group
        passes so a night exists even if every fetch fails.

    Returns
    -------
    n : `int`
        Rows written.
    """
    ident = [c[0] for c in IDENTITY]
    gcols = [] if group is None else [c[0] for c in group_columns(group)]
    cols = ident + [c for c in gcols if c not in ident]
    out = pd.DataFrame({c: (df[c] if c in df.columns else np.nan) for c in cols})
    if out.empty:
        return 0
    # Object columns of all-NaN confuse the DuckDB type inference; force the declared type.
    types = {n: t for n, t, _u, _s in all_columns()}
    for c in cols:
        if types.get(c) in ('DOUBLE', 'INTEGER', 'BIGINT'):
            out[c] = pd.to_numeric(out[c], errors='coerce')
        elif types.get(c) == 'VARCHAR':
            out[c] = out[c].astype('object').where(out[c].notna(), None)
    updates = ', '.join(f'{c} = excluded.{c}' for c in cols if c != 'visit_id')
    con.register('_upsert_src', out)
    con.execute(f'INSERT INTO visit_telemetry ({", ".join(cols)}) '
                f'SELECT {", ".join(cols)} FROM _upsert_src '
                f'ON CONFLICT (visit_id) DO UPDATE SET {updates}')
    con.unregister('_upsert_src')
    return len(out)


def log_fetch(con, day_obs, group, status, n_rows=0, error=None):
    """Record the outcome of one ``(day_obs, group)`` fetch in `fetch_log`."""
    con.execute(
        'INSERT INTO fetch_log (day_obs, group_name, status, n_rows, error, attempted_at) '
        'VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (day_obs, group_name) DO UPDATE SET '
        'status = excluded.status, n_rows = excluded.n_rows, error = excluded.error, '
        'attempted_at = excluded.attempted_at',
        [int(day_obs), group, status, int(n_rows), error,
         datetime.now(timezone.utc)])


def done_pairs(con, status=('ok', 'empty')):
    """``(day_obs, group)`` pairs already recorded with one of `status`.

    Returns
    -------
    pairs : `set` [`tuple`]
        Used by the builder's ``--resume`` to skip work already done.
    """
    q = ', '.join('?' for _ in status)
    rows = con.execute(f'SELECT day_obs, group_name FROM fetch_log '
                       f'WHERE status IN ({q})', list(status)).fetchall()
    return {(int(d), g) for d, g in rows}


def refresh_coverage(con):
    """Rebuild `column_coverage` from the data now in `visit_telemetry`.

    For every column this records the first and last `day_obs` carrying a non-NULL value
    and the non-NULL count, so a reader can tell a quantity that was not yet deployed at
    a given epoch from one whose fetch failed from one that is genuinely NaN.

    Returns
    -------
    n : `int`
        Columns described.
    """
    owner = {}
    for g in GROUP_ORDER:
        for n, _t, _u, _s in GROUPS[g]:
            owner.setdefault(n, g)
    specs = all_columns()
    rows = []
    now = datetime.now(timezone.utc)
    for name, _t, units, source in specs:
        if name == 'visit_id':
            continue
        r = con.execute(
            f'SELECT MIN(day_obs), MAX(day_obs), COUNT({name}) '
            f'FROM visit_telemetry WHERE {name} IS NOT NULL').fetchone()
        rows.append((name, owner.get(name, 'identity'), source, units,
                     None if r[0] is None else int(r[0]),
                     None if r[1] is None else int(r[1]),
                     int(r[2] or 0), now))
    con.execute('DELETE FROM column_coverage')
    con.executemany(
        'INSERT INTO column_coverage (column_name, group_name, source, units, '
        'first_day_obs, last_day_obs, n_non_null, updated_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)', rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Optical-state variants
# ---------------------------------------------------------------------------
def variant_id(scheme, intrinsic_route, opd_version):
    """Canonical variant name, e.g. ``v50_34__batoid__consdb_v1``.

    Parameters
    ----------
    scheme : `str`
        ``'22_12'`` or ``'50_34'`` — the degree-of-freedom (DOF) count and the number of
        retained v-modes.
    intrinsic_route : `str`
        ``'batoid'`` or ``'miw'``.
    opd_version : `str`
        Tag for the measured-OPD provenance, e.g. ``'consdb_v1'``. A reprocessing of the
        measured Zernikes becomes a new value here, so old and new coexist.

    Returns
    -------
    vid : `str`
    """
    return f'v{scheme}__{intrinsic_route}__{opd_version}'


def register_variant(con, scheme, intrinsic_route, opd_version, n_dof, n_modes,
                     intrinsic_ref=None, opd_source='consdb_ccdvisit1_quicklook',
                     ofc_config_version='v13', notes=None):
    """Register a variant in `state_variant`, returning its `variant_id`.

    Idempotent: re-registering an existing variant updates its descriptive fields and
    leaves its `optical_state` rows untouched.
    """
    vid = variant_id(scheme, intrinsic_route, opd_version)
    con.execute(
        'INSERT INTO state_variant (variant_id, scheme, n_dof, n_modes, '
        'intrinsic_route, intrinsic_ref, opd_source, opd_version, '
        'ofc_config_version, created_at, notes) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) '
        'ON CONFLICT (variant_id) DO UPDATE SET scheme = excluded.scheme, '
        'n_dof = excluded.n_dof, n_modes = excluded.n_modes, '
        'intrinsic_route = excluded.intrinsic_route, '
        'intrinsic_ref = excluded.intrinsic_ref, opd_source = excluded.opd_source, '
        'opd_version = excluded.opd_version, '
        'ofc_config_version = excluded.ofc_config_version, notes = excluded.notes',
        [vid, scheme, int(n_dof), int(n_modes), intrinsic_route, intrinsic_ref,
         opd_source, opd_version, ofc_config_version,
         datetime.now(timezone.utc), notes])
    return vid


def upsert_optical_state(con, vid, visit_ids, v_modes, dof, resid_rms_um=None, ok=None):
    """Write one variant's recovered state for a set of visits.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Writable connection.
    vid : `str`
        Variant id, already registered via `register_variant`.
    visit_ids : `array_like` [`int`]
        Length n.
    v_modes : `numpy.ndarray`
        Shape (n, n_modes), µm of wavefront.
    dof : `numpy.ndarray`
        Shape (n, 50), µm and deg.
    resid_rms_um : `array_like` [`float`], optional
        Per-visit recovery residual, µm of wavefront.
    ok : `array_like` [`bool`], optional
        False where the recovery was under-determined or NaN-poisoned. Defaults to rows
        whose v-modes are all finite.

    Returns
    -------
    n : `int`
        Rows written.
    """
    v_modes = np.atleast_2d(np.asarray(v_modes, float))
    dof = np.atleast_2d(np.asarray(dof, float))
    visit_ids = np.asarray(visit_ids).astype('int64')
    n, n_modes = v_modes.shape
    if len(visit_ids) != n or len(dof) != n:
        raise ValueError(f'length mismatch: {len(visit_ids)} visit_ids, {n} v_modes '
                         f'rows, {len(dof)} dof rows')
    if ok is None:
        ok = np.isfinite(v_modes).all(axis=1)
    if resid_rms_um is None:
        resid_rms_um = np.full(n, np.nan)
    now = datetime.now(timezone.utc)
    rows = [(int(visit_ids[i]), vid, v_modes[i].tolist(), dof[i].tolist(),
             int(n_modes), float(resid_rms_um[i]), bool(ok[i]), now)
            for i in range(n)]
    con.executemany(
        'INSERT INTO optical_state (visit_id, variant_id, v_modes, dof, n_modes, '
        'resid_rms_um, ok, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) '
        'ON CONFLICT (visit_id, variant_id) DO UPDATE SET v_modes = excluded.v_modes, '
        'dof = excluded.dof, n_modes = excluded.n_modes, '
        'resid_rms_um = excluded.resid_rms_um, ok = excluded.ok, '
        'computed_at = excluded.computed_at', rows)
    return n


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _day_obs_clause(day_obs_range, col='day_obs'):
    if day_obs_range is None:
        return '', []
    lo, hi = day_obs_range
    parts, params = [], []
    if lo is not None:
        parts.append(f'{col} >= ?')
        params.append(int(lo))
    if hi is not None:
        parts.append(f'{col} <= ?')
        params.append(int(hi))
    return (' WHERE ' + ' AND '.join(parts)) if parts else '', params


def visits(day_obs_range=None, columns=None, con=None, db_path=None):
    """Read `visit_telemetry` — the EFD and value-added columns.

    Parameters
    ----------
    day_obs_range : `tuple`, optional
        ``(first, last)`` inclusive; either may be None for an open end.
    columns : `iterable` [`str`], optional
        Columns to read. Defaults to all. The identity block is always included.
    con : `duckdb.DuckDBPyConnection`, optional
        Existing connection; one is opened read-only if omitted.
    db_path : `str` or `pathlib.Path`, optional
        Database file, when `con` is omitted.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per exposure, ordered by `day_obs`, `seq_num`.

    Notes
    -----
    ConsDB metadata — band, image type, science program, pointing, transformed-EFD
    temperatures and wind — is deliberately **not** stored here. Attach it with
    `join_consdb`.
    """
    own = con is None
    con = con or open_db(db_path, readonly=True)
    try:
        if columns is None:
            sel = '*'
        else:
            want = [c[0] for c in IDENTITY] + [c for c in columns
                                               if c not in {i[0] for i in IDENTITY}]
            sel = ', '.join(want)
        where, params = _day_obs_clause(day_obs_range)
        return con.execute(f'SELECT {sel} FROM visit_telemetry{where} '
                           f'ORDER BY day_obs, seq_num', params).df()
    finally:
        if own:
            con.close()


def variants(con=None, db_path=None):
    """The `state_variant` registry as a DataFrame."""
    own = con is None
    con = con or open_db(db_path, readonly=True)
    try:
        return con.execute('SELECT * FROM state_variant ORDER BY variant_id').df()
    finally:
        if own:
            con.close()


def optical_state(variant, day_obs_range=None, wide=True, ok_only=True,
                  con=None, db_path=None):
    """Read one optical-state variant.

    Parameters
    ----------
    variant : `str`
        Variant id — **required**. There is deliberately no default: `optical_state` is a
        long table, and a forgotten variant filter would silently multiply the sample by
        the number of variants.
    day_obs_range : `tuple`, optional
        ``(first, last)`` inclusive.
    wide : `bool`, optional
        Expand the `v_modes` and `dof` list columns into ``v1..vN`` and ``dof0..dof49``
        columns (the default), so analysis code sees ordinary scalars.
    ok_only : `bool`, optional
        Keep only rows whose recovery succeeded (the default).
    con, db_path
        As in `visits`.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per visit, with `visit_id`, `day_obs`, `seq_num`, the v-modes in µm of
        wavefront and the recovered DOF in µm and deg.

    Raises
    ------
    KeyError
        If `variant` is not registered in `state_variant`.
    """
    own = con is None
    con = con or open_db(db_path, readonly=True)
    try:
        known = [r[0] for r in con.execute(
            'SELECT variant_id FROM state_variant').fetchall()]
        if variant not in known:
            raise KeyError(f'variant {variant!r} is not registered; '
                           f'known variants: {known or "(none)"}')
        where, params = _day_obs_clause(day_obs_range, 'v.day_obs')
        clause = where or ' WHERE TRUE'
        if ok_only:
            clause += ' AND s.ok'
        df = con.execute(
            'SELECT s.visit_id, v.day_obs, v.seq_num, s.v_modes, s.dof, s.n_modes, '
            's.resid_rms_um, s.ok FROM optical_state s '
            'JOIN visit_telemetry v USING (visit_id) '
            f'{clause} AND s.variant_id = ? ORDER BY v.day_obs, v.seq_num',
            params + [variant]).df()
    finally:
        if own:
            con.close()
    if not wide or df.empty:
        return df
    nm = int(df['n_modes'].iloc[0])
    vm = np.vstack([np.asarray(x, float) for x in df['v_modes']])
    dd = np.vstack([np.asarray(x, float) for x in df['dof']])
    for j in range(nm):
        df[f'v{j + 1}'] = vm[:, j]
    for j in range(dd.shape[1]):
        df[f'dof{j}'] = dd[:, j]
    return df.drop(columns=['v_modes', 'dof'])


def compare_variants(variant_a, variant_b, day_obs_range=None, n_modes=1,
                     con=None, db_path=None):
    """Self-join two variants on `visit_id` for a variant-vs-variant comparison.

    Parameters
    ----------
    variant_a, variant_b : `str`
        Variant ids.
    day_obs_range : `tuple`, optional
    n_modes : `int`, optional
        How many leading v-modes to return per variant, as ``v1_a``/``v1_b``, ... in µm
        of wavefront.
    con, db_path
        As in `visits`.

    Returns
    -------
    df : `pandas.DataFrame`
        Visits present in **both** variants, one row each.
    """
    own = con is None
    con = con or open_db(db_path, readonly=True)
    try:
        sel = ', '.join(f'a.v_modes[{j + 1}] AS v{j + 1}_a, '
                        f'b.v_modes[{j + 1}] AS v{j + 1}_b' for j in range(n_modes))
        where, params = _day_obs_clause(day_obs_range, 'v.day_obs')
        clause = where or ' WHERE TRUE'
        return con.execute(
            f'SELECT a.visit_id, v.day_obs, v.seq_num, {sel} '
            'FROM optical_state a JOIN optical_state b USING (visit_id) '
            'JOIN visit_telemetry v USING (visit_id) '
            f'{clause} AND a.variant_id = ? AND b.variant_id = ? '
            'ORDER BY v.day_obs, v.seq_num',
            params + [variant_a, variant_b]).df()
    finally:
        if own:
            con.close()


def coverage(con=None, db_path=None):
    """The `column_coverage` table as a DataFrame.

    Returns
    -------
    df : `pandas.DataFrame`
        Per column: owning group, source, units, first and last `day_obs` with a
        non-NULL value, and the non-NULL count. A column whose quantity post-dates the
        start of the build shows its true `first_day_obs`, which is what distinguishes
        "not yet deployed" from "fetch failed".
    """
    own = con is None
    con = con or open_db(db_path, readonly=True)
    try:
        return con.execute('SELECT * FROM column_coverage '
                           'ORDER BY group_name, column_name').df()
    finally:
        if own:
            con.close()


def fetch_status(day_obs_range=None, con=None, db_path=None):
    """The `fetch_log` as a DataFrame, for auditing a backfill."""
    own = con is None
    con = con or open_db(db_path, readonly=True)
    try:
        where, params = _day_obs_clause(day_obs_range)
        return con.execute(f'SELECT * FROM fetch_log{where} '
                           f'ORDER BY day_obs, group_name', params).df()
    finally:
        if own:
            con.close()


# ---------------------------------------------------------------------------
# Live ConsDB join
# ---------------------------------------------------------------------------
#: ConsDB column groups available to `join_consdb`. ConsDB is fast, so these are read
#: live at analysis time rather than copied into the database.
CONSDB_GROUPS = ('meta', 'thermal', 'wind', 'iq')


def join_consdb(df, groups=CONSDB_GROUPS, cdb=None, consdb_url='auto',
                instrument='lsstcam'):
    """Attach live ConsDB columns to a `visits` result, merging on `visit_id`.

    Parameters
    ----------
    df : `pandas.DataFrame`
        A `visits` result, carrying `visit_id`, `day_obs` and `seq_num`.
    groups : `iterable` [`str`], optional
        Which ConsDB column groups to attach, from `CONSDB_GROUPS`: exposure metadata
        (band, image type, science program, pointing), transformed-EFD temperatures,
        transformed-EFD wind, and image quality.
    cdb : `lsst.summit.utils.ConsDbClient`, optional
        Existing client; one is made if omitted.
    consdb_url : `str`, optional
        Passed to `common.telemetry_clients.make_consdb_client`.
    instrument : `str`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        `df` with the requested ConsDB columns added, plus the derived
        ``truss_temp_mean_c`` [°C] where its two inputs are present.

    Notes
    -----
    ``truss_temp_mean_c`` is computed here rather than stored, so the definition — the
    mean of the +X+Y and -X-Y Telescope Mount Assembly (TMA) truss resistance
    thermometers — lives in one place and stays identical to the Full Array Mode
    analysis it is compared against.

    Image-quality columns arrive from ConsDB as object dtype and are coerced with
    `pandas.to_numeric`; a non-numeric entry becomes NaN rather than propagating as a
    string.
    """
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'aos' / 'code'))
    import aos_consdb_efd as ace                                       # noqa: E402
    from common.telemetry_clients import make_consdb_client            # noqa: E402

    groups = tuple(groups)
    bad = [g for g in groups if g not in CONSDB_GROUPS]
    if bad:
        raise ValueError(f'unknown ConsDB group(s) {bad}; choose from {CONSDB_GROUPS}')
    if df.empty:
        return df
    cdb = cdb or make_consdb_client(consdb_url)
    visit_ids = df['visit_id'].astype('int64').tolist()
    out = df

    if 'meta' in groups:
        days = sorted(set(int(d) for d in df['day_obs']))
        day_list = ', '.join(str(d) for d in days)
        # physical_rotator_angle lives on visit1_quicklook, not exposure, so this is a
        # join within cdb_<instrument> -- same-database, which the ConsDB server handles.
        q = (f'SELECT e.exposure_id AS visit_id, e.band, e.img_type, e.science_program, '
             f'e.exp_time, e.altitude, e.azimuth, e.wind_dir, e.wind_speed, '
             f'e.obs_start_mjd, q.physical_rotator_angle '
             f'FROM cdb_{instrument}.exposure e '
             f'LEFT JOIN cdb_{instrument}.visit1_quicklook q '
             f'  ON q.visit_id = e.exposure_id '
             f'WHERE e.day_obs IN ({day_list})')
        meta = cdb.query(q).to_pandas()
        meta = meta.rename(columns={
            'exp_time': 'exp_time_sec', 'altitude': 'altitude_deg',
            'azimuth': 'azimuth_deg_consdb',
            'physical_rotator_angle': 'rotator_angle_deg',
            'wind_dir': 'wind_dir_deg_consdb', 'wind_speed': 'wind_speed_ms_consdb'})
        out = out.merge(meta, on='visit_id', how='left')

    if 'thermal' in groups or 'wind' in groups:
        want = {}
        if 'thermal' in groups:
            want.update(ace.TEMP_COLS)
        if 'wind' in groups:
            want.update(ace.WIND_COLS)
        tel = ace.fetch_scalars_pivoted(cdb, visit_ids, hexapod=False)
        if tel is not None and len(tel):
            # fetch_scalars_pivoted returns raw ConsDB column names indexed by
            # exposure_id; rename to the TEMP_COLS/WIND_COLS short names here.
            tel = tel.rename(columns=want)
            if 'visit_id' not in tel.columns:
                tel = tel.reset_index().rename(columns={'exposure_id': 'visit_id',
                                                        'index': 'visit_id'})
            tel['visit_id'] = tel['visit_id'].astype('int64')
            keep = ['visit_id'] + [c for c in want.values() if c in tel.columns]
            out = out.merge(tel[keep], on='visit_id', how='left')

    if 'iq' in groups:
        days = sorted(set(int(d) for d in df['day_obs']))
        day_list = ', '.join(str(d) for d in days)
        q = (f'SELECT v.visit_id, v.psf_sigma_median, v.psf_area_median, '
             f'v.seeing_zenith_500nm_median '
             f'FROM cdb_{instrument}.visit1_quicklook v '
             f'JOIN cdb_{instrument}.visit1 e ON e.visit_id = v.visit_id '
             f'WHERE e.day_obs IN ({day_list})')
        try:
            iq = cdb.query(q).to_pandas()
            for c in iq.columns:
                if c != 'visit_id':
                    iq[c] = pd.to_numeric(iq[c], errors='coerce')
            out = out.merge(iq, on='visit_id', how='left')
        except Exception as e:
            print(f'(ConsDB image-quality columns unavailable '
                  f'[{type(e).__name__}: {e}])')

    a, b = 'tma_truss_temp_pxpy', 'tma_truss_temp_mxmy'
    if a in out.columns and b in out.columns:
        out['truss_temp_mean_c'] = 0.5 * (pd.to_numeric(out[a], errors='coerce')
                                          + pd.to_numeric(out[b], errors='coerce'))
    return out


def wrap180(angle_deg):
    """Wrap an angle to (-180, 180] degrees.

    Parameters
    ----------
    angle_deg : `array_like`
        Angle in degrees.

    Returns
    -------
    wrapped : `numpy.ndarray`
        Angle in degrees on (-180, 180].
    """
    return (np.asarray(angle_deg, float) + 180.0) % 360.0 - 180.0
