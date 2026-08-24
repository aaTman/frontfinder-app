"""Orchestrates one IFS cycle's worth of frontfinder work: for each of the two
models, and for every published forecast lead time ("step") that cycle
carries, fetch/assemble input -> tiled inference -> zarr pyramid write.

Intended to run as a scheduled batch job (systemd timer / cron) on the
Proxmox VM, triggered shortly after each IFS 00/06/12/18Z cycle's 0.25deg
open-data files become available. Not run on a per-request basis -- see the
deployment doc for the timer schedule and offset.

2026-08-21: extended from a single step=0 ("analysis-equivalent") run per
cycle to a full forecast product -- every 6 hours out to 240 hours, per
Taylor's call, prompted by the webapp having no time slider to show since
there was never more than one time per cycle to slide over. See
`ecmwf_ifs.py`'s `target_steps_for_cycle` for which steps a given cycle
actually has available (00Z/12Z go to 240h, 06Z/18Z are capped at 144h --
a real ECMWF publishing asymmetry, not a bug here; see that module's
`available_forecast_steps` for the 2026-08-22 correction from an earlier,
stale 90h assumption). One zarr store is
written per (cycle, step) -- same flat 2D-per-store architecture as before,
just one store per lead time instead of one store per cycle -- and
`latest.json` now lists every step successfully produced for the CURRENT
cycle, sorted ascending, for the webapp's slider to read.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

from frontfinder.config.manifests import MANIFESTS, ModelManifest
from frontfinder.inference.engine import Predictor, run_tiled_inference
from frontfinder.ingest.ecmwf_ifs import (
    IFSCycle,
    IFSFieldSource,
    assemble_model_input,
    target_steps_for_cycle,
)
from frontfinder.zarrio.pyramid import FrontFields, build_front_pyramid, write_front_pyramid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRunConfig:
    manifest: ModelManifest
    predictor: Predictor
    patch_size: int = 256
    overlap: int = 32
    batch_size: int = 8
    n_pyramid_levels: int = 6


def _valid_time(cycle: IFSCycle) -> str:
    """The forecast's actual valid time: cycle init time + step hours.
    Distinct from the cycle's own init time the moment step != 0 -- the two
    used to be conflated (both fields set to the same string) back when
    every run was step=0 and they were numerically identical, which masked
    the bug until steps > 0 existed at all."""
    init = datetime.strptime(f"{cycle.date} {cycle.run_hour:02d}", "%Y-%m-%d %H")
    return (init + timedelta(hours=cycle.step)).strftime("%Y-%m-%dT%H:%M:%S")


def _cycle_time(cycle: IFSCycle) -> str:
    """The IFS cycle's own init time, independent of step."""
    return f"{cycle.date}T{cycle.run_hour:02d}:00:00"


def run_one_model(
    run_config: ModelRunConfig,
    source: IFSFieldSource,
    cycle: IFSCycle,
    output_root: str,
) -> str:
    """Runs one model end-to-end for one (cycle, step). Returns the zarr
    store path written. Does NOT write latest.json -- that's `run_cycle`'s
    job now, once it knows every step that succeeded for this cycle (see
    module docstring); a single step no longer gets to unilaterally decide
    what "latest" means."""
    manifest = run_config.manifest
    logger.info("assembling input for model=%s cycle=%s", manifest.name, cycle)
    input_grid = assemble_model_input(manifest, source, cycle)

    logger.info("running tiled inference for model=%s", manifest.name)
    served_probs = run_tiled_inference(
        run_config.predictor,
        input_grid,
        manifest,
        patch_size=run_config.patch_size,
        overlap=run_config.overlap,
        batch_size=run_config.batch_size,
        lon_deg=source.lon,
    )

    probabilities = {
        cls: served_probs[..., i] for i, cls in enumerate(manifest.served_classes)
    }
    fields = FrontFields(
        probabilities=probabilities,
        lat=source.lat,
        lon=source.lon,
        valid_time=_valid_time(cycle),
        cycle_time=_cycle_time(cycle),
    )

    logger.info("building + writing zarr pyramid for model=%s", manifest.name)
    pyramid = build_front_pyramid(fields, manifest, n_levels=run_config.n_pyramid_levels)

    store_name = _store_name(cycle)
    store_path = os.path.join(output_root, manifest.name, store_name)
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    write_front_pyramid(pyramid, store_path)
    logger.info("wrote %s", store_path)
    return store_path


def _store_name(cycle: IFSCycle) -> str:
    # "2026-08-20T18Z_f000.zarr" -- the `_f<NNN>` step suffix is what lets
    # retention.py's _STORE_NAME_RE and the webapp's latest.json both tell
    # different lead times of the same cycle apart on disk.
    return f"{cycle.date}T{cycle.run_hour:02d}Z_f{cycle.step:03d}.zarr"


def _write_latest_pointer(
    output_root: str, model_name: str, cycle: IFSCycle, step_entries: list[dict]
) -> None:
    """Writes `<output_root>/<model>/latest.json`, read by the webapp to find
    every step successfully published for the most recent cycle, without
    listing the store directory. Only called after at least one step's zarr
    write succeeds (see `run_cycle`), so a wholly-failed cycle never
    clobbers a previous cycle's still-good pointer. `step_entries` is
    already sorted ascending by step_hours by the caller.

    Shape (2026-08-21, replaces the old flat {store, cycle_time}):
        {
          "cycle_time": "2026-08-20T18:00:00",
          "steps": [
            {"step_hours": 0, "valid_time": "2026-08-20T18:00:00", "store": "..."},
            {"step_hours": 6, "valid_time": "2026-08-21T00:00:00", "store": "..."},
            ...
          ]
        }
    """
    pointer_path = os.path.join(output_root, model_name, "latest.json")
    with open(pointer_path, "w") as f:
        json.dump({"cycle_time": _cycle_time(cycle), "steps": step_entries}, f)


def run_cycle(
    run_configs: list[ModelRunConfig],
    source: IFSFieldSource,
    cycle_date: str,
    run_hour: int,
    output_root: str,
    steps: tuple[int, ...] | None = None,
) -> dict[str, list[str]]:
    """Runs every configured model across every step this cycle publishes
    (default: `target_steps_for_cycle(run_hour)`, i.e. every 6h out to 240h
    where the run hour supports it, capped at 144h for 06Z/18Z -- see
    ecmwf_ifs.py). Steps run in ascending order; a failure on one step is
    logged and skipped rather than aborting the remaining steps for that
    model -- ECMWF publishes longer lead times progressively, so a longer
    step 404ing because it isn't published YET is an expected, not
    exceptional, occurrence on some firings. Likewise a failure in one
    model does not stop the others.

    `latest.json` is written PROGRESSIVELY -- after each individual step
    succeeds, not just once at the end of the whole cycle (2026-08-21 fix;
    see the postmortem below). It always lists every step that has
    succeeded so far, so a cycle that only got through step 0-48h before
    something broke still publishes those, rather than nothing.

    2026-08-21 postmortem: a full 00Z/12Z cycle fans out across 41 steps
    (every 6h to 240h), each a full tiled-inference pass over the whole
    grid -- that can take a long time. The original version only called
    `_write_latest_pointer` once, after every step for a model had been
    attempted, so the webapp showed "no published steps" (or a stale
    pre-multi-step latest.json with no "steps" key at all, which the
    frontend treats identically) for the ENTIRE run, even though earlier
    steps had already finished and were sitting on disk. Confirmed live,
    2026-08-21, against a real in-progress cycle. Writing the pointer after
    every successful step means the slider gains steps in near-real-time as
    the cycle progresses, instead of only appearing all at once at the end.

    Returns `{model_name: [store_path, ...]}` for whichever steps
    succeeded, in ascending step order; a model with zero successful steps
    is omitted entirely (mirrors the old single-step "omit on total
    failure" behavior).
    """
    if steps is None:
        steps = target_steps_for_cycle(run_hour)

    results: dict[str, list[str]] = {}
    for run_config in run_configs:
        manifest_name = run_config.manifest.name
        store_paths: list[str] = []
        step_entries: list[dict] = []
        for step in steps:
            cycle = IFSCycle(date=cycle_date, run_hour=run_hour, step=step)
            try:
                store_path = run_one_model(run_config, source, cycle, output_root)
            except Exception:
                logger.exception("model %s failed for cycle %s -- skipping this step", manifest_name, cycle)
                continue
            store_paths.append(store_path)
            step_entries.append(
                {"step_hours": step, "valid_time": _valid_time(cycle), "store": _store_name(cycle)}
            )
            # Publish as soon as this step lands, not after the whole cycle
            # -- see the postmortem above.
            _write_latest_pointer(output_root, manifest_name, cycle, step_entries)

        if store_paths:
            results[manifest_name] = store_paths
        else:
            logger.error("model %s produced zero successful steps for cycle %s-%02dZ", manifest_name, cycle_date, run_hour)

    return results


def known_model_names() -> list[str]:
    return sorted(MANIFESTS)
