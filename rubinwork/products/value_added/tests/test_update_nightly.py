"""Tests for the pure helpers of the nightly updater. Nothing here opens a database."""

import datetime as dt

from rubinwork.products.value_added.builders import update_nightly as un


def test_last_finished_night_rolls_at_noon_utc():
    utc = dt.timezone.utc
    # 18:17 UTC on 11-11 is day_obs 20261111 (not yet observed), so 20261110 is done.
    assert un.last_finished_night(dt.datetime(2026, 11, 11, 18, 17, tzinfo=utc)) == 20261110
    # 06:00 UTC on 11-11 is still night 20261110, in progress.
    assert un.last_finished_night(dt.datetime(2026, 11, 11, 6, 0, tzinfo=utc)) == 20261109


def test_select_nights():
    nights = [20261101, 20261102, 20261104, 20261105, 20261106]
    new, redo = un.select_nights(nights, 20261104, redo=2)
    assert new == [20261105, 20261106]
    assert redo == [20261102, 20261104]
    assert un.select_nights(nights, 20261106, redo=3) == ([], [20261104, 20261105, 20261106])
    assert un.select_nights(nights, 20261104, redo=0)[1] == []


def test_snapshot_name():
    assert un.snapshot_name("20261112", {"20261111"}) == "20261112"
    assert un.snapshot_name("20261112", {"20261112", "20261112b"}) == "20261112c"


def test_prune_candidates():
    names = ["20261016", "20261025", "20261025b", "20261104", "20261104b", "20261105",
             "20261106", "20261107", "20261108", "20261109"]
    # The last 7 are kept, and 20261025b as October's last; the other two go.
    assert un.prune_candidates(names) == ["20261016", "20261025"]
