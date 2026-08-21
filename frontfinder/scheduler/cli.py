"""CLI entrypoint invoked by the systemd timer on the Proxmox VM. Determines
the most recent completed IFS synoptic cycle from the current UTC time,
loads the active Keras model(s), and runs `run_cycle` for each -- which, as
of 2026-08-21, means every published forecast step for that cycle (every 6h
out to 240h, capped at 90h for 06Z/18Z cycles -- see
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
import sys
from datetime import datetime, timedelta, timezone

from frontfinder.config.manifests import BEST_LOSS_MANIFEST  # , MODEL_1702_MANIFEST -- see module docstring
from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource
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


def build_run_configs(model_dir: str) -> list[ModelRunConfig]:
    import os

    return [
        ModelRunConfig(
            manifest=BEST_LOSS_MANIFEST,
            predictor=KerasPredictor(os.path.join(model_dir, BEST_LOSS_MANIFEST.weights_filename)),
        ),
        # model_1702 disabled -- see module docstring for why. Re-add once
        # the missing 950/900hPa levels have a chosen resolution:
        # ModelRunConfig(
        #     manifest=MODEL_1702_MANIFEST,
        #     predictor=KerasPredictor(os.path.join(model_dir, MODEL_1702_MANIFEST.weights_filename)),
        # ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one frontfinder IFS cycle for both models.")
    parser.add_argument("--model-dir", required=True, help="directory containing the .keras/.h5 weight files")
    parser.add_argument("--output-root", required=True, help="directory to write zarr pyramids + latest.json into")
    parser.add_argument("--cache-dir", default="/tmp/frontfinder_ifs_cache")
    parser.add_argument("--publish-lag-hours", type=int, default=7)
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

    cycle_date, run_hour = most_recent_completed_cycle(datetime.now(timezone.utc), args.publish_lag_hours)
    source = EcmwfOpenDataSource(cache_dir=args.cache_dir, source=args.ifs_source)
    run_configs = build_run_configs(args.model_dir)

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
