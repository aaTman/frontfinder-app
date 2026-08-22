import os
from datetime import datetime, timezone

from frontfinder.scheduler.cli import (
    _cycles_after,
    _most_recent_output_cycle,
    most_recent_completed_cycle,
)


def test_most_recent_completed_cycle_picks_previous_synoptic_hour():
    # 14:00 UTC minus 7h publish lag = 07:00 -> most recent synoptic hour <= 7 is 06Z
    now = datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)
    cycle_date, run_hour = most_recent_completed_cycle(now, publish_lag_hours=7)
    assert cycle_date == "2026-08-19"
    assert run_hour == 6


def test_most_recent_completed_cycle_rolls_back_across_midnight():
    # 02:00 UTC minus 7h = 19:00 the previous day -> most recent synoptic hour is 18Z
    now = datetime(2026, 8, 19, 2, 0, tzinfo=timezone.utc)
    cycle_date, run_hour = most_recent_completed_cycle(now, publish_lag_hours=7)
    assert cycle_date == "2026-08-18"
    assert run_hour == 18


def test_most_recent_completed_cycle_exact_synoptic_boundary():
    # 19:00 UTC minus 7h publish lag = 12:00 exactly -> most recent synoptic hour is 12Z
    now = datetime(2026, 8, 19, 19, 0, tzinfo=timezone.utc)
    cycle_date, run_hour = most_recent_completed_cycle(now, publish_lag_hours=7)
    assert cycle_date == "2026-08-19"
    assert run_hour == 12


def _touch_store(output_root: str, model_name: str, cycle_date: str, run_hour: int, step: int = 0) -> None:
    model_dir = os.path.join(output_root, model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, f"{cycle_date}T{run_hour:02d}Z_f{step:03d}.zarr"))


def test_most_recent_output_cycle_none_when_model_dir_missing(tmp_path):
    assert _most_recent_output_cycle(str(tmp_path), "theta-e_uv_q") is None


def test_most_recent_output_cycle_picks_latest_across_steps_and_dates(tmp_path):
    output_root = str(tmp_path)
    _touch_store(output_root, "theta-e_uv_q", "2026-08-21", 18, step=0)
    _touch_store(output_root, "theta-e_uv_q", "2026-08-21", 18, step=240)
    _touch_store(output_root, "theta-e_uv_q", "2026-08-20", 12, step=0)
    assert _most_recent_output_cycle(output_root, "theta-e_uv_q") == ("2026-08-21", 18)


def test_most_recent_output_cycle_ignores_unrelated_entries(tmp_path):
    output_root = str(tmp_path)
    model_dir = os.path.join(output_root, "theta-e_uv_q")
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, "latest.json"), "w") as f:
        f.write("{}")
    assert _most_recent_output_cycle(output_root, "theta-e_uv_q") is None


def test_cycles_after_none_returns_only_current():
    # No output on disk anywhere -- nothing to backfill from, so only the
    # current candidate cycle is pending (matches pre-backfill behavior).
    assert _cycles_after(None, ("2026-08-22", 12)) == [("2026-08-22", 12)]


def test_cycles_after_same_cycle_returns_empty():
    assert _cycles_after(("2026-08-22", 12), ("2026-08-22", 12)) == []


def test_cycles_after_fills_missed_synoptic_boundaries():
    # 18Z on the 21st is the last real output; 12Z on the 22nd is current.
    # 00Z and 06Z on the 22nd were both missed in between.
    pending = _cycles_after(("2026-08-21", 18), ("2026-08-22", 12))
    assert pending == [
        ("2026-08-22", 0),
        ("2026-08-22", 6),
        ("2026-08-22", 12),
    ]
