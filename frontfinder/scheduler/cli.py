"""CLI entrypoint invoked by the systemd timer on the Proxmox VM. Determines
the most recent completed IFS synoptic cycle from the current UTC time,
loads the active Keras model(s), and runs `run_cycle` for each -- which, as
of 2026-08-21, means every published forecast step for that cycle (every 6h
out to 240h, capped at 144h for 06Z/18Z cycles -- see
ecmwf_ifs.target_steps_for_cycle), not just a single step=0 run.

Not covered by unit tests (loads real Keras models + hits real network) --
`run_cycle.run_cycle`/`run_one_model` carry the tested logic; this module is
just wiring. Smoke-test on the Proxmox VM before enabling the timer.

model_1702 is currently DISABLED (see build_run_configs below): its trained
pressure levels [1000, 950, 900, 850] include 950hPa and 900hPa, which IFS
open-data's 0.25deg feed simply does not publish (confirmed live against
the feed's actual index, 2026-08-20 -- available pl levels are [10, 50,
100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]). Feeding it
interpolated/substituted levels instead of its real training data is a
real accuracy tradeoff, not a mechanical fix -- Taylor chose to leave it
off rather than guess at that tradeoff. Re-enabling it means deciding how
to handle the missing levels first (see the two commented-out lines below
for where it plugs back in once that's resolved).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_MANIFEST  # MODEL_1702_MANIFEST -- see module docstring
from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource, IFSCycle, target_steps_for_cycle
from frontfinder.inference.engine import KerasPredictor
from frontfinder.scheduler.retention import prune_old_cache_files, prune_old_output_stores
from frontfinder.scheduler.run_cycle import ModelRunConfig, run_cycle

SYNOPTIC_HOURS = (0, 6, 12, 18)


def most_recent_completed_cycle(now: datetime, publish_lag_hours: int = 7) -> tuple[str, int]:
    """The most recent synoptic cycle (date, run_hour) whose IFS open-data
    files should already be published, given ECMWF's typical ~6-8h publish
    lag after the nominal cycle time. `publish_lag_hours` is a first-pass
    estimate -- tune it against actual observed availability on the Proxmox
    VM; if runs start failing with "file not found" on the ECMWF side,
    increase it rather than assume the pipeline is broken.

    2026-08-22: the deployed timer no longer relies on this guess at all --
    it runs `--poll` mode (see `main`), which calls this with
    `publish_lag_hours=0` just to name the current candidate cycle, then
    HEAD-checks real availability via `EcmwfOpenDataSource.is_cycle_available`
    instead of trusting elapsed time. `publish_lag_hours` remains here for
    manual/non-polling invocations.

    2026-08-21: used to return a full `IFSCycle` (always step=0) back when
    every run only ever fetched the analysis-equivalent field. Now that
    `run_cycle` fans out across every published step itself (see
    ecmwf_ifs.target_steps_for_cycle), this only needs to pick WHICH cycle
    -- the step loop lives entirely inside run_cycle.
    """
    candidate = now - timedelta(hours=publish_lag_hours)
    cycle_hour = max(h for h in SYNOPTIC_HOURS if h <= candidate.hour)
    cycle_dt = candidate.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    return cycle_dt.strftime("%Y-%m-%d"), cycle_hour


# "2026-08-20T18Z_f006.zarr" -> ("2026-08-20", 18) -- same store-directory
# naming as retention.py's _STORE_NAME_RE, but keeping run_hour too (that
# one only needs the date to compare against a max-age cutoff; this one
# needs to know WHICH cycle a store belongs to).
_STORE_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2})Z(?:_f\d{3})?\.zarr$")

# Same store-directory naming, but capturing the step number too -- needed
# by `_steps_present_for_cycle` below to know WHICH steps of a cycle have
# already landed, not just whether the cycle has been touched at all.
_STEP_STORE_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2})Z_f(\d{3})\.zarr$")


def _most_recent_output_cycle(output_root: str, model_name: str) -> tuple[str, int] | None:
    """The most recent (date, run_hour) that `model_name` has ANY published
    store for, found by scanning `output_root/<model_name>/` directly --
    not a separate marker file. A cycle that failed on every step (see the
    2026-08-22 postmortem: fired 30min before the whole cycle had finished
    publishing, 404'd on all of it) never gets a store directory at all
    (`run_cycle` omits a model entirely on zero successful steps), so this
    reflects real success, not just "was attempted". Returns None if the
    model has no output yet at all.
    """
    model_dir = os.path.join(output_root, model_name)
    if not os.path.isdir(model_dir):
        return None
    best: tuple[str, int] | None = None
    for entry in os.listdir(model_dir):
        m = _STORE_NAME_RE.match(entry)
        if not m:
            continue
        found = (m.group(1), int(m.group(2)))
        if best is None or found > best:
            best = found
    return best


def _steps_present_for_cycle(output_root: str, model_name: str, date: str, run_hour: int) -> set[int]:
    """Which step numbers `model_name` already has a written store for, for
    this exact (date, run_hour) cycle -- scans the store directory directly
    (same as `_most_recent_output_cycle`) rather than trusting latest.json,
    since latest.json gets overwritten the moment a NEWER cycle's first
    step lands (see `run_cycle._write_latest_pointer`), while the older
    cycle's already-written stores stay on disk untouched."""
    model_dir = os.path.join(output_root, model_name)
    if not os.path.isdir(model_dir):
        return set()
    present: set[int] = set()
    for entry in os.listdir(model_dir):
        m = _STEP_STORE_NAME_RE.match(entry)
        if m and m.group(1) == date and int(m.group(2)) == run_hour:
            present.add(int(m.group(3)))
    return present


def _cycle_is_complete(output_root: str, model_name: str, date: str, run_hour: int) -> bool:
    """Whether `model_name` has already written every step this cycle's
    run_hour is expected to publish (see `target_steps_for_cycle`). A cycle
    that got through some steps and then had its process killed mid-fetch
    (2026-09-01 postmortem: a step stuck retrying a sustained run of S3
    "503 Slow Down" past the systemd unit's TimeoutStartSec got SIGTERM'd
    mid-retry) is incomplete, not done -- `--poll`'s pending-cycle logic
    below uses this instead of "has ANY output" so a killed cycle gets
    revisited and its missing tail steps re-attempted, rather than being
    considered finished forever the moment a newer cycle supersedes it as
    "most recent"."""
    target = set(target_steps_for_cycle(run_hour))
    return target.issubset(_steps_present_for_cycle(output_root, model_name, date, run_hour))


def _cycles_after(after: tuple[str, int] | None, current: tuple[str, int]) -> list[tuple[str, int]]:
    """Every synoptic cycle strictly after `after` up to and including
    `current`, in ascending order -- the backlog `--poll` needs to check
    for missed runs (e.g. a cycle that 404'd on every step and produced no
    output, or the box being down across a cycle boundary), not just the
    single newest one. If `after` is None (no output on disk at all yet),
    there's no known history to backfill from, so this returns just
    `current` -- matches the old single-cycle behavior for a fresh deploy.
    """
    if after is None:
        return [current]
    cycles: list[tuple[str, int]] = []
    d = datetime.strptime(after[0], "%Y-%m-%d").replace(hour=after[1])
    end = datetime.strptime(current[0], "%Y-%m-%d").replace(hour=current[1])
    while d < end:
        idx = SYNOPTIC_HOURS.index(d.hour)
        d = (
            d.replace(hour=SYNOPTIC_HOURS[idx + 1])
            if idx + 1 < len(SYNOPTIC_HOURS)
            else (d + timedelta(days=1)).replace(hour=SYNOPTIC_HOURS[0])
        )
        cycles.append((d.strftime("%Y-%m-%d"), d.hour))
    return cycles


def build_run_configs(model_dir: str) -> list[ModelRunConfig]:
    return [
        # 2026-10-08: replaced theta-e_uv_q (_best_loss.keras) as the served
        # model. Its manifest is still in config/manifests.py.
        ModelRunConfig(
            manifest=NBS_SQUARE_DILATION_TROF_TT_MANIFEST,
            predictor=KerasPredictor(
                os.path.join(model_dir, NBS_SQUARE_DILATION_TROF_TT_MANIFEST.weights_filename)
            ),
        ),
        # model_1702 disabled -- see module docstring for why. Re-add once
        # the missing 950/900hPa levels have a chosen resolution:
        # ModelRunConfig(
        #     manifest=MODEL_1702_MANIFEST,
        #     predictor=KerasPredictor(os.path.join(model_dir, MODEL_1702_MANIFEST.weights_filename)),
        # ),
        # spix_wce_loss_modified_dilation disabled 2026-10-01 to avoid any GPU
        # use (this deployment runs CPU-only). Its manifest/predictor are still
        # in the codebase; re-add a ModelRunConfig here to bring it back.
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one frontfinder IFS cycle for every active model.")
    parser.add_argument("--model-dir", required=True, help="directory containing the .keras/.h5 weight files")
    parser.add_argument("--output-root", required=True, help="directory to write zarr pyramids + latest.json into")
    parser.add_argument("--cache-dir", default="/tmp/frontfinder_ifs_cache")
    parser.add_argument(
        "--publish-lag-hours",
        type=int,
        default=7,
        help="ignored when --poll is set -- see --poll",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help=(
            "instead of running immediately against a --publish-lag-hours "
            "guess at which cycle should be ready, find the OLDEST synoptic "
            "cycle that doesn't have output on disk yet (scanning "
            "<output-root>/<model>/ directly -- see "
            "_most_recent_output_cycle), and HEAD-check whether ITS data "
            "has landed on the IFS open-data source "
            "(EcmwfOpenDataSource.is_cycle_available) before running it. "
            "This both fires event-driven off real availability instead of "
            "a fixed delay, AND backfills any cycle that was missed "
            "entirely (404'd on every step, box was down, etc) rather than "
            "only ever chasing the newest one. Meant to be invoked "
            "frequently (every 5-10 min) by a tight systemd timer -- see "
            "deploy/systemd/frontfinder-run-cycle.timer. A no-op (exit 0) "
            "when the oldest missing cycle isn't published yet -- one "
            "backlog cycle is run per invocation, so a long backlog "
            "catches up one poll at a time rather than all at once."
        ),
    )
    parser.add_argument(
        "--ifs-source",
        default="aws",
        choices=["aws", "azure", "google", "ecmwf"],
        help="which IFS open-data replica to pull from (default: aws, per registry.opendata.aws/ecmwf-forecasts)",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=10,
        help=(
            "delete published zarr stores and cached GRIB downloads whose "
            "IFS cycle date is older than this many days (default: 10). "
            "Pass 0 to prune everything except the current cycle, or a "
            "very large number to effectively disable pruning."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    # static_data_dir=args.model_dir: geopotential_at_surface (see
    # ecmwf_ifs.STATIC_SINGLE_LEVEL_VARIABLES) is a hand-placed static .npy
    # file, same deployment convention as the .keras weights that already
    # live in --model-dir on the serving VM.
    source = EcmwfOpenDataSource(cache_dir=args.cache_dir, source=args.ifs_source, static_data_dir=args.model_dir)
    run_configs = build_run_configs(args.model_dir)

    if args.poll:
        # 2026-10-02: newest PUBLISHED cycle ONLY -- no backlog backfill. The
        # old oldest-first backfill left the map days behind after any outage
        # (and S3 "503 Slow Down" retries made each backlog cycle slow), so
        # missed older cycles are skipped. Walks back from the nominal
        # current cycle (up to 4 cycles / 24h) to the newest one whose step 0
        # is actually on the source, so the map shows the latest real data
        # while the nominal newest cycle is still being published. A
        # partially-written target cycle is re-run to fill its missing steps
        # (already-written steps are cheap skips in run_cycle).
        d = datetime.strptime("%s %02d" % most_recent_completed_cycle(datetime.now(timezone.utc), publish_lag_hours=0), "%Y-%m-%d %H")
        target = None
        for _ in range(4):
            cand = IFSCycle(date=d.strftime("%Y-%m-%d"), run_hour=d.hour, step=0)
            if source.is_cycle_available(cand):
                target = cand
                break
            logger.info("cycle %s-%02dZ not published yet -- trying the previous cycle", cand.date, cand.run_hour)
            d -= timedelta(hours=6)
        if target is None:
            logger.info("no cycle in the last 24h is published -- will check again next poll")
            return 0
        target_date, target_hour = target.date, target.run_hour
        if all(
            _cycle_is_complete(args.output_root, rc.manifest.name, target_date, target_hour) for rc in run_configs
        ):
            logger.info("cycle %s-%02dZ already has output -- nothing to do", target_date, target_hour)
            return 0
        logger.info("cycle %s-%02dZ available -- triggering run", target_date, target_hour)
        cycle_date, run_hour = target_date, target_hour
    else:
        cycle_date, run_hour = most_recent_completed_cycle(datetime.now(timezone.utc), args.publish_lag_hours)

    results = run_cycle(run_configs, source, cycle_date, run_hour, args.output_root)
    logger.info(
        "cycle %s-%02dZ complete: %s",
        cycle_date, run_hour, {name: len(paths) for name, paths in results.items()},
    )

    # Retention runs regardless of whether this cycle's run_cycle() fully
    # succeeded -- a failed model run shouldn't also block disk cleanup, and
    # pruning failures shouldn't crash a cycle that otherwise succeeded, so
    # this is wrapped in its own try/except rather than left to propagate.
    try:
        now = datetime.now(timezone.utc)
        deleted_stores = prune_old_output_stores(args.output_root, args.retention_days, now)
        deleted_cache = prune_old_cache_files(args.cache_dir, args.retention_days, now)
        logger.info(
            "retention: deleted %d old output store(s), %d old cache file(s) (older than %d days)",
            len(deleted_stores), len(deleted_cache), args.retention_days,
        )
    except Exception:
        logger.exception("retention pruning failed -- leaving old files in place rather than risk a bad delete")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
