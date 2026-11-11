"""Nightly update of the value-added database: new nights into ``v1/live``, then a snapshot.

One run does, in order:

1. Pick the nights. *New* nights are the Consolidated Database (ConsDB) nights with
   exposures after the last night whose ``trim`` group is recorded ``ok`` in
   ``fetch_log`` (or from ``--from``), up to the last finished night. *Redo* nights are the ``--redo`` nights
   with exposures just before the first new night; they are rebuilt for the data that
   lands late (ConsDB PSF metrics and corner wavefront-sensor quicklook).
2. ``build_efd_db --groups all`` on the new nights. Not ``--resume``: ``tweak`` and
   ``hexhist`` need ``trim`` fetched in the same pass, so skipping a done ``trim`` makes
   them fail.
3. ``build_efd_db --groups psf_shape --refetch`` on the redo nights.
4. ``build_m1m3_thermal_r2 --resume`` on the new nights.
5. ``build_optical_state`` for every registered variant over redo plus new nights. Skipped
   if step 2 failed, because the commanded v-modes read that step's Trim and LUT.
6. If every step succeeded: copy ``live`` to ``snapshots/<YYYYMMDD>`` (``b``, ``c`` ... on
   a second run the same day), write its manifest and move ``current`` to it.

Snapshots are never deleted here; the ones outside the retention rule (the last 7 plus
the last of each month) are listed for Aaron to prune by hand.

A builder that hits an EFD error on one group logs it in ``fetch_log`` and still exits 0,
so the run ends by listing every ``error`` row for its nights.

More than ``--max-new`` new nights is refused: a gap that size is a sharded backfill with
``run_build.sh``, not a serial nightly run.

Command line::

    python -u -m rubinwork.products.value_added.builders.update_nightly --dry-run
    python -u -m rubinwork.products.value_added.builders.update_nightly
"""

import argparse
import datetime as dt
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import time

from rubinwork.products.value_added import reader as efd_db

__all__ = ["last_finished_night", "select_nights", "snapshot_name",
           "prune_candidates", "main"]

PKG = "rubinwork.products.value_added.builders"
REDO_GROUPS = ("psf_shape",)


def last_finished_night(now=None):
    """The ``day_obs`` of the most recent night that has ended.

    ``day_obs`` rolls over at 12:00 UTC (it is the UTC date of the time minus 12 h), so
    the night in progress or about to start is ``day_obs(now)`` and the last finished
    one is the day before.

    Parameters
    ----------
    now : `datetime.datetime`, optional
        Timezone-aware time; the current time by default.

    Returns
    -------
    day_obs : `int`
        YYYYMMDD, dimensionless.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    d = (now.astimezone(dt.timezone.utc) - dt.timedelta(hours=12)).date()
    return int((d - dt.timedelta(days=1)).strftime("%Y%m%d"))


def _shift(day_obs, days):
    d = dt.datetime.strptime(str(day_obs), "%Y%m%d").date() + dt.timedelta(days=days)
    return int(d.strftime("%Y%m%d"))


def select_nights(consdb_nights, last_done, redo):
    """Split ConsDB nights into new and redo nights.

    Parameters
    ----------
    consdb_nights : `list` [`int`]
        Nights with exposures, ascending, ending at the last night to build.
    last_done : `int` or `None`
        Last night already built (``trim`` recorded ``ok``).
    redo : `int`
        Number of nights with exposures before the first new night to rebuild.

    Returns
    -------
    new, redo_nights : `list` [`int`]
    """
    nights = sorted(set(consdb_nights))
    new = [d for d in nights if last_done is None or d > last_done]
    before = [d for d in nights if d not in new]
    return new, (before[-redo:] if redo > 0 else [])


def snapshot_name(day, existing):
    """``YYYYMMDD``, or with the first free suffix ``b``, ``c`` ... if taken."""
    if day not in existing:
        return day
    for c in "bcdefghijklmnopqrstuvwxyz":
        if day + c not in existing:
            return day + c
    raise RuntimeError(f"no free snapshot name for {day}")


def prune_candidates(names, keep_last=7):
    """Snapshots outside the retention rule: keep the last `keep_last` and the last of
    each month. Names sort in time order (``20261104`` < ``20261104b`` < ``20261105``)."""
    names = sorted(names)
    keep = set(names[-keep_last:])
    by_month = {}
    for n in names:
        by_month[n[:6]] = n
    keep |= set(by_month.values())
    return [n for n in names if n not in keep]


def _run(label, args, dry_run):
    cmd = [sys.executable, "-u", "-m", f"{PKG}.{args[0]}"] + list(args[1:])
    print(f"\n=== {label}\n    {' '.join(cmd)}", flush=True)
    if dry_run:
        return True
    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f"=== {label}: exit {rc}, {time.time() - t0:.0f} s", flush=True)
    return rc == 0


def _copy_with_md5(src, dst):
    h = hashlib.md5()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while chunk := fi.read(64 << 20):
            h.update(chunk)
            fo.write(chunk)
    shutil.copystat(src, dst)
    return h.hexdigest()


def make_snapshot(live_db, snapshots_dir, today, nights, dry_run):
    """Copy the live database to a new dated snapshot, register it and make it current."""
    from rubinwork.products import catalog
    from rubinwork.products.value_added import write_manifest

    existing = {p.name for p in snapshots_dir.iterdir()} if snapshots_dir.exists() else set()
    name = snapshot_name(today, existing)
    snap = snapshots_dir / name
    print(f"\n=== snapshot {snap}", flush=True)
    if dry_run:
        return name, existing | {name}
    snap.mkdir(parents=True)
    t0 = time.time()
    md5 = _copy_with_md5(live_db, snap / efd_db.DB_FILE)
    man = write_manifest.write(build_dir=snap, build_options={
        "what": "snapshot_of_live", "source": "v1/live", "md5": md5,
        "added": f"nightly update, day_obs {','.join(map(str, nights))}"})
    os.chmod(snap / efd_db.DB_FILE, 0o440)
    os.chmod(man, 0o440)
    catalog.set_current(efd_db.PRODUCT, efd_db.PRODUCT_VARIANT, f"snapshots/{name}")
    print(f"    md5 {md5}, {(snap / efd_db.DB_FILE).stat().st_size} B, "
          f"{time.time() - t0:.0f} s; current -> snapshots/{name}", flush=True)
    return name, existing | {name}


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--through", type=int, default=None,
                   help="last night to build, YYYYMMDD (default: last finished night)")
    p.add_argument("--from", dest="from_day", type=int, default=None,
                   help="treat nights from this YYYYMMDD on as new, whatever fetch_log "
                        "says (default: after the last night with trim ok)")
    p.add_argument("--redo", type=int, default=3,
                   help="nights before the first new night to rebuild (default 3)")
    p.add_argument("--max-new", type=int, default=7,
                   help="refuse more new nights than this (default 7)")
    p.add_argument("--no-snapshot", action="store_true",
                   help="update live only; do not snapshot or move current")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and the commands; write nothing")
    p.add_argument("--consdb-url", default="auto")
    a = p.parse_args(argv)

    from rubinwork.common.telemetry_clients import make_consdb_client
    from rubinwork.products.value_added.builders.build_efd_db import parse_day_obs

    variant_dir = efd_db.build_dir(efd_db.PRODUCT_VARIANT)
    live_db = efd_db.build_dir(efd_db.PRODUCT_VARIANT, "live") / efd_db.DB_FILE
    (variant_dir / "_work").mkdir(exist_ok=True)
    lock = open(variant_dir / "_work" / "update_nightly.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another update_nightly run holds the lock; exiting")

    con = efd_db.open_db(live_db, readonly=True)
    last_done = con.execute("SELECT MAX(day_obs) FROM fetch_log "
                            "WHERE group_name = 'trim' AND status = 'ok'").fetchone()[0]
    variants = [r[0] for r in con.execute(
        "SELECT variant_id FROM state_variant ORDER BY variant_id").fetchall()]
    con.close()

    through = a.through or last_finished_night()
    if a.from_day:
        last_done = _shift(a.from_day, -1)
    start = _shift(last_done, -30) if last_done else 20250415
    cdb = make_consdb_client(a.consdb_url)
    nights = parse_day_obs(f"{start}-{through}", cdb=cdb)
    new, redo = select_nights(nights, last_done, a.redo)
    print(f"live {live_db}\nnew nights start after {last_done}; through {through}\n"
          f"new nights: {new or 'none'}\nredo nights: {redo or 'none'}\n"
          f"variants: {', '.join(variants)}", flush=True)
    if len(new) > a.max_new:
        sys.exit(f"{len(new)} new nights exceed --max-new {a.max_new}; fill the gap with a "
                 "sharded backfill (run_build.sh --mode batch) first")
    if not new:
        print("no new nights; nothing to do")
        return 0

    db = ["--db", str(live_db)]
    csv = lambda ds: ",".join(map(str, ds))  # noqa: E731
    ok = {}
    if new:
        ok["telemetry"] = _run("telemetry, new nights", ["build_efd_db", "--day-obs",
                               csv(new), "--groups", "all"] + db, a.dry_run)
    if redo:
        ok["psf_redo"] = _run("psf_shape, redo nights", ["build_efd_db", "--day-obs",
                              csv(redo), "--groups", ",".join(REDO_GROUPS),
                              "--refetch"] + db, a.dry_run)
    if new:
        ok["thermal"] = _run("m1m3_thermal_r2, new nights", ["build_m1m3_thermal_r2",
                             "--day-obs", csv(new), "--resume"] + db, a.dry_run)
    if ok.get("telemetry", True):
        for v in variants:
            ok[f"state {v}"] = _run(f"optical_state {v}, redo and new nights",
                                    ["build_optical_state", "--variant", v, "--day-obs",
                                     csv(redo + new)] + db, a.dry_run)
    else:
        print("\ntelemetry failed: optical_state skipped (its commanded v-modes need Trim)")
        ok["state"] = False

    if not a.dry_run:
        con = efd_db.open_db(live_db, readonly=True)
        errs = con.execute(
            "SELECT day_obs, group_name, status FROM fetch_log WHERE status NOT IN "
            f"('ok', 'empty') AND day_obs IN ({csv(redo + new)}) ORDER BY 1, 2").fetchall()
        con.close()
        if errs:
            print("\nfetch_log rows not ok for this run's nights (day_obs, group, status):")
            for r in errs:
                print(f"    {r}")

    failed = [k for k, v in ok.items() if not v]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}; no snapshot, current unchanged")
        return 1
    if a.no_snapshot:
        print("\n--no-snapshot: live updated, current unchanged")
        return 0
    today = dt.date.today().strftime("%Y%m%d")
    name, names = make_snapshot(live_db, variant_dir / "snapshots", today, redo + new,
                                a.dry_run)
    prune = prune_candidates(names)
    if prune:
        print(f"\nsnapshots outside retention (last 7 + last of each month), to prune by "
              f"hand: {', '.join(prune)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
