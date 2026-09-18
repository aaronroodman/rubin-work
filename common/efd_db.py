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
          v_modes       DOUBLE[],   -- measured, length n_modes, dimensionless amplitudes
          v_modes_lut   DOUBLE[],   -- hexapod LUT, length n_modes, dimensionless
          v_modes_trim  DOUBLE[],   -- Trim, length n_modes, dimensionless
          dof           DOUBLE[],   -- length 50, um / deg
          n_modes       INTEGER,
          resid_rms_um  DOUBLE,     -- um of wavefront
          ok            BOOLEAN,
          computed_at   TIMESTAMP,
          PRIMARY KEY (visit_id, variant_id)
        )""")
    # Additive migration, as for visit_telemetry: v_modes_lut and v_modes_trim were added
    # after the table was first created, so that all three v-mode terms are projected in the
    # variant's own scheme rather than the commanded pair being reprojected per analysis.
    have_os = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'optical_state'").fetchall()}
    for n in ('v_modes_lut', 'v_modes_trim'):
        if n not in have_os:
            con.execute(f'ALTER TABLE optical_state ADD COLUMN {n} DOUBLE[]')
    con.execute("""
        CREATE TABLE IF NOT EXISTS fam_variant (
          fam_variant_id   VARCHAR PRIMARY KEY,
          param_set        VARCHAR,     -- Butler collection + processing variant
          intrinsic_route  VARCHAR,     -- 'batoid' | 'miw'
          intrinsic_ref    VARCHAR,     -- MIW build name (mi_name), or the OFC config version
          prefix           VARCHAR,     -- DZ-fit column prefix: 'z1toz6' | 'z1toz3'
          k_min            INTEGER,     -- focal (field) Zernike order range
          k_max            INTEGER,
          pupil_j          INTEGER[],   -- canonical nollIndices, from the visits sidecar
          scheme           VARCHAR,     -- '50_34' | '22_12'
          n_dof            INTEGER,
          n_modes          INTEGER,
          fits_path        VARCHAR,     -- provenance: which fits.parquet was read
          created_at       TIMESTAMP,
          notes            VARCHAR
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS fam_dz (
          visit_id        BIGINT,      -- the EXTRA-focal member, i.e. fits.parquet's own visit
          fam_variant_id  VARCHAR,
          day_obs         INTEGER,
          seq_num         INTEGER,     -- extra-focal member
          intra_seq_num   INTEGER,     -- seq_num - 1
          acq_seq_num     INTEGER,     -- seq_num + 1, the in-focus member of the triplet
          acq_visit_id    BIGINT,      -- so the join to optical_state is a key lookup
          dz_coeff        DOUBLE[],    -- DZ(k,j) in kj_grid order, um of wavefront
          dz_coeff_err    DOUBLE[],    -- formal errors, um of wavefront
          v_modes         DOUBLE[],    -- projected from dz_coeff, length n_modes, dimensionless
          dof             DOUBLE[],    -- length n_dof, um / deg
          n_modes         INTEGER,
          n_donuts        INTEGER,
          bad_fit         BOOLEAN,
          quality_pass    BOOLEAN,
          computed_at     TIMESTAMP,
          PRIMARY KEY (visit_id, fam_variant_id)
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
    con.execute('CREATE INDEX IF NOT EXISTS fd_variant ON fam_dz (fam_variant_id)')
    con.execute('CREATE INDEX IF NOT EXISTS fd_day_obs ON fam_dz (day_obs)')
    con.execute('CREATE INDEX IF NOT EXISTS fd_acq ON fam_dz (acq_visit_id)')


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


def upsert_optical_state(con, vid, visit_ids, v_modes, dof, resid_rms_um=None, ok=None,
                         v_modes_lut=None, v_modes_trim=None):
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
        Measured optical state, shape (n, n_modes) [dimensionless v-mode amplitudes].
    dof : `numpy.ndarray`
        Shape (n, 50), µm and deg.
    resid_rms_um : `array_like` [`float`], optional
        Per-visit recovery residual, µm of wavefront.
    ok : `array_like` [`bool`], optional
        False where the recovery was under-determined or NaN-poisoned. Defaults to rows
        whose v-modes are all finite.
    v_modes_lut : `numpy.ndarray`, optional
        Hexapod look-up-table (LUT) commanded state projected in this variant's scheme,
        shape (n, n_modes) [dimensionless]. NULL where not supplied.
    v_modes_trim : `numpy.ndarray`, optional
        Trim commanded state in the same scheme and units.

    Returns
    -------
    n : `int`
        Rows written.

    Notes
    -----
    All three v-mode terms are stored in the variant's own scheme, so an analysis forming
    ``LUT + Trim - measured`` never mixes projection bases. `ok` describes the measured
    recovery only; the commanded terms are NULL rather than false when unavailable.
    """
    v_modes = np.atleast_2d(np.asarray(v_modes, float))
    dof = np.atleast_2d(np.asarray(dof, float))
    visit_ids = np.asarray(visit_ids).astype('int64')
    n, n_modes = v_modes.shape
    if len(visit_ids) != n or len(dof) != n:
        raise ValueError(f'length mismatch: {len(visit_ids)} visit_ids, {n} v_modes '
                         f'rows, {len(dof)} dof rows')

    def _opt(a, name):
        """Validate an optional (n, n_modes) block, returning a list of row values."""
        if a is None:
            return [None] * n
        a = np.atleast_2d(np.asarray(a, float))
        if a.shape != (n, n_modes):
            raise ValueError(f'{name} has shape {a.shape}, expected {(n, n_modes)}; the '
                             f'commanded terms must be projected in the same scheme as the '
                             f'measured state')
        return [a[i].tolist() for i in range(n)]

    lut_rows = _opt(v_modes_lut, 'v_modes_lut')
    trim_rows = _opt(v_modes_trim, 'v_modes_trim')
    if ok is None:
        ok = np.isfinite(v_modes).all(axis=1)
    if resid_rms_um is None:
        resid_rms_um = np.full(n, np.nan)
    now = datetime.now(timezone.utc)
    rows = [(int(visit_ids[i]), vid, v_modes[i].tolist(), lut_rows[i], trim_rows[i],
             dof[i].tolist(), int(n_modes), float(resid_rms_um[i]), bool(ok[i]), now)
            for i in range(n)]
    con.executemany(
        'INSERT INTO optical_state (visit_id, variant_id, v_modes, v_modes_lut, '
        'v_modes_trim, dof, n_modes, resid_rms_um, ok, computed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) '
        'ON CONFLICT (visit_id, variant_id) DO UPDATE SET v_modes = excluded.v_modes, '
        'v_modes_lut = excluded.v_modes_lut, v_modes_trim = excluded.v_modes_trim, '
        'dof = excluded.dof, n_modes = excluded.n_modes, '
        'resid_rms_um = excluded.resid_rms_um, ok = excluded.ok, '
        'computed_at = excluded.computed_at', rows)
    return n


def fam_variant_id(param_set, intrinsic_route, prefix, scheme):
    """Canonical FAM-fit variant name, e.g. ``fam__<param_set>__batoid__z1toz6__50_34``.

    Parameters
    ----------
    param_set : `str`
        Butler collection paired with a processing variant, as used in ``aos/output/``.
    intrinsic_route : `str`
        ``'batoid'`` for the design intrinsic, ``'miw'`` for a Measured Intrinsic Wavefront
        build.
    prefix : `str`
        Double Zernike (DZ) fit column prefix in ``fits.parquet`` — ``'z1toz6'`` or
        ``'z1toz3'``, which fixes the focal (field) Zernike orders the fit spans.
    scheme : `str`
        Degree-of-freedom (DOF) and v-mode counts, ``'50_34'`` or ``'22_12'``.

    Returns
    -------
    fvid : `str`
    """
    return f'fam__{param_set}__{intrinsic_route}__{prefix}__{scheme}'


def register_fam_variant(con, param_set, intrinsic_route, prefix, scheme, k_min, k_max,
                         pupil_j, n_dof, n_modes, intrinsic_ref=None, fits_path=None,
                         notes=None):
    """Register a FAM Double Zernike (DZ) fit variant in `fam_variant`.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Writable connection.
    param_set, intrinsic_route, prefix, scheme
        As in `fam_variant_id`.
    k_min, k_max : `int`
        Inclusive range of focal (field) Zernike orders the DZ fit spans.
    pupil_j : `array_like` [`int`]
        Canonical pupil Noll indices, read from ``visits.parquet``'s ``nollIndices``.
    n_dof, n_modes : `int`
        Optical Feedback Control (OFC) degrees of freedom and retained v-modes.
    intrinsic_ref : `str`, optional
        MIW build name for ``intrinsic_route='miw'``, or the OFC config version.
    fits_path : `str`, optional
        The ``fits.parquet`` that was read, kept as provenance.
    notes : `str`, optional

    Returns
    -------
    fvid : `str`
        The variant id.

    Notes
    -----
    Idempotent, as `register_variant`: re-registering updates the descriptive fields and
    leaves the `fam_dz` rows untouched.
    """
    fvid = fam_variant_id(param_set, intrinsic_route, prefix, scheme)
    con.execute(
        'INSERT INTO fam_variant (fam_variant_id, param_set, intrinsic_route, '
        'intrinsic_ref, prefix, k_min, k_max, pupil_j, scheme, n_dof, n_modes, '
        'fits_path, created_at, notes) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) '
        'ON CONFLICT (fam_variant_id) DO UPDATE SET param_set = excluded.param_set, '
        'intrinsic_route = excluded.intrinsic_route, '
        'intrinsic_ref = excluded.intrinsic_ref, prefix = excluded.prefix, '
        'k_min = excluded.k_min, k_max = excluded.k_max, pupil_j = excluded.pupil_j, '
        'scheme = excluded.scheme, n_dof = excluded.n_dof, n_modes = excluded.n_modes, '
        'fits_path = excluded.fits_path, notes = excluded.notes',
        [fvid, param_set, intrinsic_route, intrinsic_ref, prefix, int(k_min), int(k_max),
         [int(j) for j in pupil_j], scheme, int(n_dof), int(n_modes),
         str(fits_path) if fits_path is not None else None,
         datetime.now(timezone.utc), notes])
    return fvid


def upsert_fam_dz(con, fvid, df, dz_coeff, v_modes, dof, dz_coeff_err=None):
    """Write one FAM variant's per-pair Double Zernike (DZ) fit and its projection.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Writable connection.
    fvid : `str`
        FAM variant id, already registered via `register_fam_variant`.
    df : `pandas.DataFrame`
        One row per FAM extra/intra-focal pair, carrying `visit_id`, `day_obs`, `seq_num`
        and optionally `n_donuts`, `bad_fit` and `quality_pass`. The `seq_num` is the
        **extra-focal** member, so `intra_seq_num` and `acq_seq_num` are derived as
        ``seq_num - 1`` and ``seq_num + 1``.
    dz_coeff : `numpy.ndarray`
        Shape (n, n_kj), DZ coefficients in ``kj_grid`` order [µm of wavefront].
    v_modes : `numpy.ndarray`
        Shape (n, n_modes), projected from `dz_coeff` [dimensionless amplitudes].
    dof : `numpy.ndarray`
        Shape (n, n_dof) [µm and deg].
    dz_coeff_err : `numpy.ndarray`, optional
        Formal errors matching `dz_coeff` [µm of wavefront]. NULL where not supplied.

    Returns
    -------
    n : `int`
        Rows written.

    Notes
    -----
    `dz_coeff` column order is the variant's ``kj_grid`` — recoverable from `fam_variant`'s
    `k_min`, `k_max` and `pupil_j` — so the array is never indexed by hand; use the
    `fam_dz` reader's ``wide=True`` expansion instead.

    The triplet is ordered intra-focal, extra-focal, in-focus `acq` in ascending `seq_num`,
    so the in-focus member is ``seq_num + 1`` and `acq_visit_id` is formed from it by the
    ``day_obs * 100000 + seq_num`` visit-id convention.
    """
    dz_coeff = np.atleast_2d(np.asarray(dz_coeff, float))
    v_modes = np.atleast_2d(np.asarray(v_modes, float))
    dof = np.atleast_2d(np.asarray(dof, float))
    n, n_modes = v_modes.shape
    if not (len(df) == n == len(dz_coeff) == len(dof)):
        raise ValueError(f'length mismatch: {len(df)} df rows, {len(dz_coeff)} dz_coeff, '
                         f'{n} v_modes, {len(dof)} dof')
    if dz_coeff_err is None:
        err_rows = [None] * n
    else:
        e = np.atleast_2d(np.asarray(dz_coeff_err, float))
        if e.shape != dz_coeff.shape:
            raise ValueError(f'dz_coeff_err has shape {e.shape}, expected '
                             f'{dz_coeff.shape}')
        err_rows = [e[i].tolist() for i in range(n)]

    day_obs = np.asarray(df['day_obs']).astype('int64')
    seq = np.asarray(df['seq_num']).astype('int64')
    vid = np.asarray(df['visit_id']).astype('int64')
    acq_seq = seq + 1
    acq_vid = day_obs * 100000 + acq_seq

    def _col(name, default, cast):
        if name not in df.columns:
            return [default] * n
        return [cast(v) for v in np.asarray(df[name])]

    n_donuts = _col('n_donuts', None, lambda v: None if v is None else int(v))
    bad_fit = _col('bad_fit', None, lambda v: None if v is None else bool(v))
    q_pass = _col('quality_pass', None, lambda v: None if v is None else bool(v))

    now = datetime.now(timezone.utc)
    rows = [(int(vid[i]), fvid, int(day_obs[i]), int(seq[i]), int(seq[i] - 1),
             int(acq_seq[i]), int(acq_vid[i]), dz_coeff[i].tolist(), err_rows[i],
             v_modes[i].tolist(), dof[i].tolist(), int(n_modes), n_donuts[i],
             bad_fit[i], q_pass[i], now) for i in range(n)]
    con.executemany(
        'INSERT INTO fam_dz (visit_id, fam_variant_id, day_obs, seq_num, intra_seq_num, '
        'acq_seq_num, acq_visit_id, dz_coeff, dz_coeff_err, v_modes, dof, n_modes, '
        'n_donuts, bad_fit, quality_pass, computed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) '
        'ON CONFLICT (visit_id, fam_variant_id) DO UPDATE SET '
        'day_obs = excluded.day_obs, seq_num = excluded.seq_num, '
        'intra_seq_num = excluded.intra_seq_num, acq_seq_num = excluded.acq_seq_num, '
        'acq_visit_id = excluded.acq_visit_id, dz_coeff = excluded.dz_coeff, '
        'dz_coeff_err = excluded.dz_coeff_err, v_modes = excluded.v_modes, '
        'dof = excluded.dof, n_modes = excluded.n_modes, '
        'n_donuts = excluded.n_donuts, bad_fit = excluded.bad_fit, '
        'quality_pass = excluded.quality_pass, computed_at = excluded.computed_at', rows)
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
        Expand the list columns into scalars (the default): `v_modes` into ``v1..vN``,
        `dof` into ``dof0..dof49``, and — where they were stored — `v_modes_lut` and
        `v_modes_trim` into ``v1_lut..vN_lut`` and ``v1_trim..vN_trim``.
    ok_only : `bool`, optional
        Keep only rows whose recovery succeeded (the default).
    con, db_path
        As in `visits`.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per visit, with `visit_id`, `day_obs`, `seq_num`, the measured v-mode
        amplitudes (dimensionless), the commanded hexapod-LUT and Trim v-mode amplitudes
        (dimensionless) where present, and the recovered DOF in µm and deg.

    Raises
    ------
    KeyError
        If `variant` is not registered in `state_variant`.

    Notes
    -----
    The ``_lut`` and ``_trim`` columns are absent for variants built before the commanded
    terms were stored. When present they are projected in the variant's own scheme, so
    `v1 - v1_lut - v1_trim` mixes no bases; see `upsert_optical_state`.
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
            'SELECT s.visit_id, v.day_obs, v.seq_num, s.v_modes, s.v_modes_lut, '
            's.v_modes_trim, s.dof, s.n_modes, '
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
    # The commanded terms are NULL for variants built before they were stored, so expand them
    # only where present and leave the columns absent otherwise rather than fabricating NaNs
    # that would look like a failed projection.
    for src, tag in (('v_modes_lut', 'lut'), ('v_modes_trim', 'trim')):
        present = df[src].notna()
        if not present.any():
            continue
        arr = np.full((len(df), nm), np.nan)
        arr[present.to_numpy()] = np.vstack(
            [np.asarray(x, float) for x in df.loc[present, src]])
        for j in range(nm):
            df[f'v{j + 1}_{tag}'] = arr[:, j]
    return df.drop(columns=['v_modes', 'v_modes_lut', 'v_modes_trim', 'dof'])


def fam_variants(con=None, db_path=None):
    """Read the `fam_variant` registry.

    Parameters
    ----------
    con, db_path
        As in `visits`.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per registered FAM Double Zernike (DZ) fit variant.
    """
    own = con is None
    con = con or open_db(db_path, readonly=True)
    try:
        return con.execute('SELECT * FROM fam_variant ORDER BY fam_variant_id').df()
    finally:
        if own:
            con.close()


def fam_dz(variant, day_obs_range=None, wide=True, good_only=True,
           con=None, db_path=None):
    """Read one FAM Double Zernike (DZ) fit variant.

    Parameters
    ----------
    variant : `str`
        FAM variant id — **required**, as in `optical_state`: `fam_dz` is a long table and a
        forgotten variant filter would silently multiply the sample by the variant count.
    day_obs_range : `tuple`, optional
        ``(first, last)`` inclusive.
    wide : `bool`, optional
        Expand the list columns into scalars (the default): `dz_coeff` into
        ``dz_k<k>_j<j>`` in µm of wavefront, its errors into ``dz_k<k>_j<j>_err``,
        `v_modes` into ``v1..vN`` (dimensionless) and `dof` into ``dof0..dof<n-1>``
        (µm and deg). The ``(k, j)`` names come from the variant's own `k_min`, `k_max` and
        `pupil_j`, so no analysis indexes the stored array by position.
    good_only : `bool`, optional
        Drop rows flagged `bad_fit` (the default). Rows with a NULL flag are kept, since the
        flag is optional at write time.
    con, db_path
        As in `visits`.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per FAM extra/intra-focal pair, keyed by the **extra-focal** `visit_id`,
        carrying `intra_seq_num` and `acq_seq_num` for the other two members of the
        ``intra, extra, acq`` triplet and `acq_visit_id` for the join to `optical_state`.

    Raises
    ------
    KeyError
        If `variant` is not registered in `fam_variant`.

    Notes
    -----
    The DZ index convention is ``k`` for the focal (field) Zernike order and ``j`` for the
    pupil Noll index, matching the ``<prefix>_z<j>_c<k>`` columns of ``fits.parquet`` and
    the ``kj_grid`` column order the coefficients are stored in.
    """
    own = con is None
    con = con or open_db(db_path, readonly=True)
    try:
        reg = con.execute(
            'SELECT k_min, k_max, pupil_j, n_modes, n_dof FROM fam_variant '
            'WHERE fam_variant_id = ?', [variant]).fetchall()
        if not reg:
            known = [r[0] for r in con.execute(
                'SELECT fam_variant_id FROM fam_variant').fetchall()]
            raise KeyError(f'FAM variant {variant!r} is not registered; '
                           f'known variants: {known or "(none)"}')
        k_min, k_max, pupil_j, _n_modes, _n_dof = reg[0]
        where, params = _day_obs_clause(day_obs_range, 'day_obs')
        clause = where or ' WHERE TRUE'
        if good_only:
            clause += ' AND (bad_fit IS NULL OR NOT bad_fit)'
        df = con.execute(
            'SELECT visit_id, day_obs, seq_num, intra_seq_num, acq_seq_num, acq_visit_id, '
            'dz_coeff, dz_coeff_err, v_modes, dof, n_modes, n_donuts, bad_fit, '
            'quality_pass FROM fam_dz '
            f'{clause} AND fam_variant_id = ? ORDER BY day_obs, seq_num',
            params + [variant]).df()
    finally:
        if own:
            con.close()
    if not wide or df.empty:
        return df
    # kj_grid order, as build_ofc_svd lays it out: pupil j fastest within focal order k.
    kj = [(k, int(j)) for k in range(int(k_min), int(k_max) + 1)
          for j in [int(x) for x in pupil_j]]
    # Built as one block and concatenated once: expanding 126 DZ terms plus 34 v-modes plus
    # 50 DOF column by column fragments the frame badly enough for pandas to warn.
    wide_cols = {}
    for src, suffix in (('dz_coeff', ''), ('dz_coeff_err', '_err')):
        present = df[src].notna()
        if not present.any():
            continue
        arr = np.full((len(df), len(kj)), np.nan)
        arr[present.to_numpy()] = np.vstack(
            [np.asarray(x, float) for x in df.loc[present, src]])
        for ci, (k, j) in enumerate(kj):
            wide_cols[f'dz_k{k}_j{j}{suffix}'] = arr[:, ci]
    nm = int(df['n_modes'].iloc[0])
    vm = np.vstack([np.asarray(x, float) for x in df['v_modes']])
    for m in range(nm):
        wide_cols[f'v{m + 1}'] = vm[:, m]
    dd = np.vstack([np.asarray(x, float) for x in df['dof']])
    for m in range(dd.shape[1]):
        wide_cols[f'dof{m}'] = dd[:, m]
    df = df.drop(columns=['dz_coeff', 'dz_coeff_err', 'v_modes', 'dof'])
    return pd.concat([df, pd.DataFrame(wide_cols, index=df.index)], axis=1)


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
        ``truss_temp_mean_c`` [°C] where its two inputs are present, and
        ``truss_temp_mean_c_interpolated`` marking the visits where that value was
        filled by interpolation.

    Notes
    -----
    ``truss_temp_mean_c`` is computed here rather than stored, so the definition — the
    mean of the +X+Y and -X-Y Telescope Mount Assembly (TMA) truss resistance
    thermometers — lives in one place and stays identical to the Full Array Mode
    analysis it is compared against. Both thermometers drop out together on about 11% of
    science exposures, so the column is then filled by `interpolate_within_night`, which
    interpolates in time inside one ``day_obs`` and leaves a night with no valid sample
    entirely NaN.

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
        out = interpolate_within_night(out, 'truss_temp_mean_c')
    return out


def interpolate_within_night(df, col, time_col=None, flag=True):
    """Fill gaps in a per-visit telemetry column by interpolating in time within a night.

    Parameters
    ----------
    df : `pandas.DataFrame`
        One row per visit, carrying `day_obs`, `col`, and a time or sequence column.
    col : `str`
        Column to fill, in whatever unit it already carries; the filled values are in
        that same unit.
    time_col : `str`, optional
        Column to interpolate against. Defaults to the first of ``obs_start_mjd`` [day],
        ``obs_start`` or ``seq_num`` [dimensionless] present in `df`.
    flag : `bool`, optional
        Also write ``<col>_interpolated``, `True` where a value was filled.

    Returns
    -------
    df : `pandas.DataFrame`
        A copy of `df` with `col` filled where a night has valid samples on both sides,
        or on one side within that night, and the flag column when `flag` is set.

    Notes
    -----
    Interpolation is strictly **within** one ``day_obs`` and does not extrapolate past
    the night's first or last valid sample, so a night with no valid sample at all is
    left entirely NaN rather than being filled from a neighbouring night. That is the
    intended behaviour: the alternative — imputing a global median — substitutes a value
    from the wrong night and biases the whole night by the difference, whereas a NaN
    stays visible to any consumer that drops or flags it.

    For the Telescope Mount Assembly truss temperature the gaps are scattered single
    exposures where the Consolidated Database aggregation window caught no resistance-
    thermometer sample. The temperature drifts at about 0.22 °C per hour and the median
    gap to the nearest visit carrying a value is under a minute, so the interpolation
    error is a few thousandths of a °C.
    """
    if col not in df.columns or 'day_obs' not in df.columns:
        return df
    out = df.copy()
    if time_col is None:
        for c in ('obs_start_mjd', 'obs_start', 'seq_num'):
            if c in out.columns:
                time_col = c
                break
    if time_col is None:
        return out

    y = pd.to_numeric(out[col], errors='coerce')
    missing = y.isna()
    if flag:
        out[f'{col}_interpolated'] = False
    if not missing.any():
        return out

    t = out[time_col]
    # obs_start is a timestamp; np.interp needs a float abscissa.
    t = (pd.to_datetime(t, errors='coerce').astype('int64').astype(float)
         if not pd.api.types.is_numeric_dtype(t) else pd.to_numeric(t, errors='coerce'))

    filled = y.copy()
    for _, idx in out.groupby('day_obs').groups.items():
        sub_y, sub_t = y.loc[idx], t.loc[idx]
        good = sub_y.notna() & sub_t.notna()
        need = sub_y.isna() & sub_t.notna()
        if not good.any() or not need.any():
            continue
        order = sub_t[good].argsort()
        tg = sub_t[good].to_numpy()[order]
        yg = sub_y[good].to_numpy()[order]
        tn = sub_t[need].to_numpy()
        # np.interp clamps outside the range; mask those so no extrapolation survives.
        vals = np.interp(tn, tg, yg)
        inside = (tn >= tg[0]) & (tn <= tg[-1])
        vals = np.where(inside, vals, np.nan)
        filled.loc[sub_y.index[need]] = vals

    out[col] = filled
    if flag:
        out[f'{col}_interpolated'] = missing & filled.notna()
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
