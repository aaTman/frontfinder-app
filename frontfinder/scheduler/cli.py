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

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST  # MODEL_1702_MANIFEST -- see module docstring
from frontfinder.inference.engine import KerasPredictor, Predictor
from frontfinder.inference.refiner import (
    REFINER_NORM_FILENAME,
    REFINER_WEIGHTS_FILENAME,
    RefinerPredictor,
    load_refiner_norm,
)
from frontfinder.ingest.base import ForecastCycle
from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource, IFSCycle, target_steps_for_cycle
from frontfinder.ingest.hrrr import HRRRFieldSource, hrrr_target_steps_for_cycle
from frontfinder.scheduler.retention import prune_old_cache_files, prune_old_output_stores
from frontfinder.scheduler.run_cycle import HRRR_3KM_MODEL_KEY, HRRRRunConfig, ModelRunConfig, run_cycle, run_hrrr_cycle

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


def _hrrr_cycle_is_complete(output_root: str, model_key: str, date: str, run_hour: int) -> bool:
    target = set(hrrr_target_steps_for_cycle(run_hour))
    return target.issubset(_steps_present_for_cycle(output_root, model_key, date, run_hour))


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
        # 2026-11-17: replaced nbs_square_dilation_trof_tt_no_inst (which
        # replaced theta-e_uv_q on 2026-10-08) as the served model -- same
        # network retrained without 2 m dewpoint. Earlier manifests are still
        # in config/manifests.py.
        ModelRunConfig(
            manifest=NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST,
            predictor=KerasPredictor(
                os.path.join(model_dir, NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST.weights_filename)
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


def build_hrrr_run_config(model_dir: str, base_predictor: Predictor | None = None) -> HRRRRunConfig | None:
    """Load the C1 3 km HRRR refiner config when weights and norm stats are present."""
    ref_weights = os.path.join(model_dir, REFINER_WEIGHTS_FILENAME)
    ref_norm = os.path.join(model_dir, REFINER_NORM_FILENAME)
    if not (os.path.exists(ref_weights) and os.path.exists(ref_norm)):
        logger.warning("HRRR C1 refiner files not found in %s -- skipping HRRR", model_dir)
        return None
    manifest = NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST
    if base_predictor is None:
        base_predictor = KerasPredictor(os.path.join(model_dir, manifest.weights_filename))
    mean, std = load_refiner_norm(ref_norm)
    return HRRRRunConfig(
        manifest=manifest,
        base_predictor=base_predictor,
        refiner=RefinerPredictor(ref_weights),
        refiner_mean=mean,
        refiner_std=std,
    )


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
    parser.add_argument(
        "--front-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "also derive smooth front lines (splines) from each published step's "
            "probabilities, written as <store>.fronts.geojson next to the zarr store "
            "and listed in latest.json, for the webapp's optional 'Front lines' "
            "overlay (default: on; --no-front-lines to skip). A failure here never "
            "affects inference or publishing."
        ),
    )
    parser.add_argument(
        "--front-line-threshold",
        type=float,
        default=None,
        help=(
            "summed-probability threshold above which a probability band becomes a "
            "front line (default: 0.25). The probabilities "
            "are uncalibrated; tune this against what the overlay looks like."
        ),
    )
    parser.add_argument(
        "--pressure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "also write <store>.pressure.geojson next to the zarr store (MSLP "
            "isobars every 4 hPa, plus H/L centres) from IFS mean sea level "
            "pressure, listed in latest.json as `pressure`, for the webapp's "
            "optional 'Highs & lows' and 'Isobars' overlays (default: on; "
            "--no-pressure to skip). A failure here never affects inference "
            "or publishing."
        ),
    )
    parser.add_argument(
        "--hrrr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also run HRRR 3 km C1 refiner as new HRRR cycles become available (default: on; --no-hrrr to skip)",
    )
    parser.add_argument(
        "--hrrr-cache-dir",
        default=None,
        help="directory for cached HRRR GRIB2 messages (default: <cache-dir>/hrrr)",
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
    base_predictor = next(
        (rc.predictor for rc in run_configs if rc.manifest.name == NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST.name),
        None,
    )

    now = datetime.now(timezone.utc)
    ifs_cycle: tuple[str, int] | None = None
    hrrr_cycle: tuple[str, int] | None = None
    source_3km: HRRRFieldSource | None = None
    source_24km: HRRRFieldSource | None = None

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
        d = datetime.strptime("%s %02d" % most_recent_completed_cycle(now, publish_lag_hours=0), "%Y-%m-%d %H")
        target = None
        for _ in range(4):
            cand = IFSCycle(date=d.strftime("%Y-%m-%d"), run_hour=d.hour, step=0)
            if source.is_cycle_available(cand):
                target = cand
                break
            logger.info("cycle %s-%02dZ not published yet -- trying the previous cycle", cand.date, cand.run_hour)
            d -= timedelta(hours=6)
        if target is None:
            logger.info("no IFS cycle in the last 24h is published")
        elif all(_cycle_is_complete(args.output_root, rc.manifest.name, target.date, target.run_hour) for rc in run_configs):
            logger.info("IFS cycle %s-%02dZ already has output -- nothing to do", target.date, target.run_hour)
        else:
            logger.info("IFS cycle %s-%02dZ available -- triggering run", target.date, target.run_hour)
            ifs_cycle = (target.date, target.run_hour)
    else:
        ifs_cycle = most_recent_completed_cycle(now, args.publish_lag_hours)

    if (
        args.hrrr
        and os.path.exists(os.path.join(args.model_dir, REFINER_WEIGHTS_FILENAME))
        and os.path.exists(os.path.join(args.model_dir, REFINER_NORM_FILENAME))
    ):
        hrrr_cache_dir = args.hrrr_cache_dir or os.path.join(args.cache_dir, "hrrr")
        source_3km = HRRRFieldSource(cache_dir=hrrr_cache_dir, coarsen=1, wind_frame="earth")
        source_24km = HRRRFieldSource(cache_dir=hrrr_cache_dir, coarsen=8, wind_frame="earth")
        if args.poll:
            hd = datetime.strptime("%s %02d" % most_recent_completed_cycle(now, publish_lag_hours=0), "%Y-%m-%d %H")
            htarget = None
            for _ in range(4):
                hcand = ForecastCycle(date=hd.strftime("%Y-%m-%d"), run_hour=hd.hour, step=0)
                if source_24km.is_cycle_available(hcand):
                    htarget = hcand
                    break
                logger.info("HRRR cycle %s-%02dZ not published yet -- trying the previous cycle", hcand.date, hcand.run_hour)
                hd -= timedelta(hours=6)
            if htarget is None:
                logger.info("no HRRR cycle in the last 24h is published")
            elif _hrrr_cycle_is_complete(args.output_root, HRRR_3KM_MODEL_KEY, htarget.date, htarget.run_hour):
                logger.info("HRRR cycle %s-%02dZ already complete -- nothing to do", htarget.date, htarget.run_hour)
            else:
                logger.info("HRRR cycle %s-%02dZ available -- triggering run", htarget.date, htarget.run_hour)
                hrrr_cycle = (htarget.date, htarget.run_hour)
        else:
            hrrr_cycle = most_recent_completed_cycle(now, publish_lag_hours=2)

    if args.poll and ifs_cycle is None and hrrr_cycle is None:
        return 0

    results: dict[str, list[str]] = {}
    if ifs_cycle is not None:
        cycle_date, run_hour = ifs_cycle
        results = run_cycle(
            run_configs,
            source,
            cycle_date,
            run_hour,
            args.output_root,
            front_lines=args.front_lines,
            front_line_threshold=args.front_line_threshold,
            pressure=args.pressure,
        )
        logger.info(
            "cycle %s-%02dZ complete: %s",
            cycle_date, run_hour, {name: len(paths) for name, paths in results.items()},
        )

    hrrr_results: list[str] = []
    if hrrr_cycle is not None and source_3km is not None and source_24km is not None:
        hrrr_date, hrrr_hour = hrrr_cycle
        try:
            hrrr_config = build_hrrr_run_config(args.model_dir, base_predictor=base_predictor)
            if hrrr_config is not None:
                hrrr_results = run_hrrr_cycle(
                    hrrr_config,
                    source_3km,
                    source_24km,
                    hrrr_date,
                    hrrr_hour,
                    args.output_root,
                    front_lines=args.front_lines,
                    front_line_threshold=args.front_line_threshold,
                )
                logger.info("HRRR cycle %s-%02dZ complete: %d step(s)", hrrr_date, hrrr_hour, len(hrrr_results))
        except Exception:
            logger.exception("HRRR cycle %s-%02dZ failed", hrrr_date, hrrr_hour)

    # Retention runs regardless of whether this cycle's run_cycle() fully
    # succeeded -- a failed model run shouldn't also block disk cleanup, and
    # pruning failures shouldn't crash a cycle that otherwise succeeded, so
    # this is wrapped in its own try/except rather than left to propagate.
    try:
        prune_now = datetime.now(timezone.utc)
        deleted_stores = prune_old_output_stores(args.output_root, args.retention_days, prune_now)
        deleted_cache = prune_old_cache_files(args.cache_dir, args.retention_days, prune_now)
        logger.info(
            "retention: deleted %d old output store(s), %d old cache file(s) (older than %d days)",
            len(deleted_stores), len(deleted_cache), args.retention_days,
        )
    except Exception:
        logger.exception("retention pruning failed -- leaving old files in place rather than risk a bad delete")

    return 0 if (results or hrrr_results) else 1


if __name__ == "__main__":
    sys.exit(main())
