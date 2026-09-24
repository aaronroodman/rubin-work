"""Build the quadratic-in-radius M1M3 thermal table, one night at a time.

Reduces the raw M1M3 thermocouple telemetry to three quadratic radial temperature terms --
over the whole mirror, over the M1 annulus and over the M3 inner disc -- interpolates them onto
each exposure's start time, and writes the ``m1m3_thermal_r2`` table of the value-added
database. The fit itself lives in `m1m3_thermal_r2.py`; this script is the Engineering Facility
Database (EFD) loop and the database write.

Run from the repository root, one night or a range::

    python value_added/code/build_m1m3_thermal_r2.py --day-obs 20260315
    python value_added/code/build_m1m3_thermal_r2.py --day-obs 20251103-20260714 --resume

Key arguments:

``--day-obs``
    A single night ``YYYYMMDD`` or an inclusive range ``YYYYMMDD-YYYYMMDD``.
``--resume``
    Skip nights that already have rows, so an interrupted run continues where it stopped.
``--refetch``
    Recompute and overwrite nights that already have rows.
``--time-bin``
    Thermocouple binning [s], default 30, matching the existing bulk-gradient build.
``--db``
    Database path; defaults to `efd_db.default_db_path`.

Notes
-----
Needs the Rubin Science Platform Active Optics System (AOS) stack for
`lsst.ts.m1m3.utils.ThermocoupleAnalysis` and a live EFD, so it is not a laptop job. It opens
the database **read-write**, and the DuckDB file lock is process-wide: no other process may
hold the database open, including a read-only reader, while this runs.

One night per EFD query is a hard constraint, not a tuning choice -- a multi-night thermocouple
span times out. Each night is committed as it completes, so an interrupted run loses at most
the night in flight.
"""
import argparse
import asyncio
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import efd_db                                                        # noqa: E402
import m1m3_thermal_r2 as r2lib                                      # noqa: E402
from common.telemetry_clients import make_efd_client                 # noqa: E402


def parse_day_obs(text):
    """Expand a ``--day-obs`` argument into a list of integer nights.

    Parameters
    ----------
    text : `str`
        ``YYYYMMDD`` or ``YYYYMMDD-YYYYMMDD`` (inclusive).

    Returns
    -------
    nights : `list` [`int`]
        The endpoints are calendar dates, so the range is expanded through the calendar
        rather than numerically -- 20251130 to 20251202 is three nights, not seventy-two.
    """
    text = str(text).strip()
    if '-' not in text:
        return [int(text)]
    first, last = (t.strip() for t in text.split('-', 1))
    a = pd.Timestamp(f'{first[:4]}-{first[4:6]}-{first[6:]}')
    b = pd.Timestamp(f'{last[:4]}-{last[4:6]}-{last[6:]}')
    if b < a:
        raise ValueError(f'--day-obs range runs backwards: {text}')
    return [int(d.strftime('%Y%m%d')) for d in pd.date_range(a, b, freq='D')]


def night_spine(con, day_obs):
    """Exposures of one night that need a quadratic-term row.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Connection to the value-added database.
    day_obs : `int`
        The night.

    Returns
    -------
    spine : `pandas.DataFrame`
        ``visit_id``, ``day_obs``, ``seq_num``, ``obs_start`` ordered by ``seq_num``, empty if
        the night has no exposures in `visit_telemetry`.
    """
    return con.execute(
        'SELECT visit_id, day_obs, seq_num, obs_start FROM visit_telemetry '
        'WHERE day_obs = ? AND obs_start IS NOT NULL ORDER BY seq_num', [day_obs]).df()


def interpolate_onto_visits(terms, spine):
    """Interpolate the time-indexed quadratic terms onto each exposure's start.

    Parameters
    ----------
    terms : `pandas.DataFrame`
        Time-indexed output of `m1m3_thermal_r2.fit_r2_terms`; the index is UTC.
    spine : `pandas.DataFrame`
        Exposures with an ``obs_start`` column of International Atomic Time (TAI) ISO strings.

    Returns
    -------
    out : `pandas.DataFrame`
        ``spine`` with one column per entry of `efd_db.R2_TABLE_COLS`.

    Notes
    -----
    ``obs_start`` is TAI while the thermocouple index is UTC -- a 37 s offset at present, which
    is more than the 30 s binning, so converting is not optional. The sensor-count columns are
    interpolated like everything else and then rounded, since a count that changes mid-night
    would otherwise land on a fraction.
    """
    out = spine.copy()
    cols = list(efd_db.R2_TABLE_COLS)
    if terms is None or not len(terms):
        for c in cols:
            out[c] = np.nan
        return out
    visit_utc = Time([str(v) for v in spine['obs_start'].values],
                     format='isot', scale='tai').utc.isot
    visit_ns = pd.to_datetime(visit_utc, format='ISO8601', utc=True).astype('int64')
    term_ns = pd.to_datetime(terms.index, utc=True).astype('int64')
    t0 = term_ns[0]
    term_x = (term_ns - t0) / 1e9
    visit_x = (visit_ns - t0) / 1e9
    for c in cols:
        if c not in terms.columns:
            out[c] = np.nan
            continue
        vals = pd.Series(terms[c].to_numpy(float)).interpolate().to_numpy()
        good = np.isfinite(vals)
        if not good.any():
            out[c] = np.nan
            continue
        # np.interp clamps outside the span rather than extrapolating, which is what is
        # wanted: an exposure a few seconds before the first bin takes the first value.
        out[c] = np.interp(visit_x, term_x[good], vals[good])
        if c.endswith('_n_sensors'):
            out[c] = np.rint(out[c]).astype('int64')
    return out


async def build_night(con, client, day_obs, time_bin=30, verbose=True):
    """Compute and store the quadratic terms for one night.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Writable connection.
    client : `lsst_efd_client.EfdClient`
        EFD client.
    day_obs : `int`
        The night.
    time_bin : `int`, optional
        Thermocouple binning [s].
    verbose : `bool`, optional
        Print a per-night line.

    Returns
    -------
    n : `int`
        Rows written; 0 where the night has no exposures or no thermocouple telemetry.
    """
    spine = night_spine(con, day_obs)
    if not len(spine):
        if verbose:
            print(f'  {day_obs}: no exposures in visit_telemetry, skipped')
        return 0
    start = Time(str(spine['obs_start'].iloc[0]), format='isot', scale='tai').utc
    end = Time(str(spine['obs_start'].iloc[-1]), format='isot', scale='tai').utc
    # Pad so an exposure at either end is interpolated rather than clamped.
    pad = TimeDelta(600.0, format='sec')
    start = start - pad
    end = end + pad
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        terms = await r2lib.r2_terms_for_span(client, start, end, time_bin=time_bin)
        gaps = [str(w.message) for w in caught]
    rows = interpolate_onto_visits(terms, spine)
    n = efd_db.upsert_m1m3_thermal_r2(con, rows)
    if verbose:
        if terms is None:
            why = gaps[0] if gaps else 'no thermocouple telemetry'
            print(f'  {day_obs}: {n} exposures written as NaN -- {why}')
        else:
            got = int(np.isfinite(rows['m1m3_r2_coeff_c']).sum())
            med = float(np.nanmedian(rows['m1m3_r2_coeff_c'])) if got else float('nan')
            print(f'  {day_obs}: {n} exposures, {got} with a fit, {len(terms)} time bins, '
                  f'median M1M3 quadratic term {med:+.5f} deg C '
                  '(per unit normalized r^2 amplitude)')
    return n


def nights_present(con):
    """Nights that already have at least one `m1m3_thermal_r2` row."""
    return {int(r[0]) for r in con.execute(
        'SELECT DISTINCT day_obs FROM m1m3_thermal_r2').fetchall()}


def main(argv=None):
    """Command-line entry point. See the module docstring for the arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--day-obs', required=True,
                   help='YYYYMMDD or YYYYMMDD-YYYYMMDD (inclusive)')
    p.add_argument('--resume', action='store_true',
                   help='skip nights that already have rows')
    p.add_argument('--refetch', action='store_true',
                   help='recompute nights that already have rows')
    p.add_argument('--time-bin', type=int, default=30,
                   help='thermocouple binning in seconds (default 30)')
    p.add_argument('--db', default=None, help='database path')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args(argv)

    if r2lib.ThermocoupleAnalysis is None:
        raise SystemExit('lsst.ts.m1m3.utils is unavailable; this build needs the AOS stack '
                         'environment, where ThermocoupleAnalysis can be imported')

    nights = parse_day_obs(a.day_obs)
    verbose = not a.quiet
    con = efd_db.open_db(a.db, readonly=False, create=True)
    try:
        efd_db.create_schema(con)
        have = nights_present(con)
        if a.resume and not a.refetch:
            skipped = [d for d in nights if d in have]
            nights = [d for d in nights if d not in have]
            if verbose and skipped:
                print(f'--resume: skipping {len(skipped)} night(s) already present')
        if verbose:
            print(f'building m1m3_thermal_r2 for {len(nights)} night(s), '
                  f'time_bin {a.time_bin} s')
        if not nights:
            return 0

        async def run():
            # The client is built inside the loop that will use it: `lsst_efd_client` holds an
            # aiohttp session bound to the running loop, and one created beforehand fails every
            # query with a TaskGroup error.
            client = make_efd_client()
            total = 0
            for day_obs in nights:
                try:
                    total += await build_night(con, client, day_obs,
                                              time_bin=a.time_bin, verbose=verbose)
                except Exception as exc:                       # one bad night must not stop
                    print(f'  {day_obs}: FAILED -- {type(exc).__name__}: {exc}')
            return total

        total = asyncio.run(run())
        if verbose:
            n_tot = con.execute('SELECT count(*) FROM m1m3_thermal_r2').fetchone()[0]
            n_fit = con.execute('SELECT count(m1m3_r2_coeff_c) FROM '
                                'm1m3_thermal_r2').fetchone()[0]
            print(f'wrote {total} rows; table now holds {n_tot} exposures, '
                  f'{n_fit} with a fit')
        return 0
    finally:
        con.close()


if __name__ == '__main__':
    raise SystemExit(main())
