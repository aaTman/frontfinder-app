"""Orchestrates one IFS cycle's worth of frontfinder work: for each configured
model, and for every published forecast lead time ("step") that cycle
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
from concurrent.futures import ThreadPoolExecutor
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
    # Mirrors whatever device (e.g. "/GPU:0") `predictor` was itself built
    # with -- run_cycle uses this (not any private attribute on `predictor`)
    # to decide whether models can safely run concurrently for a given step;
    # see `run_cycle`'s docstring. None (every deployment before the
    # two-3090 host) means "let run_cycle process this model exactly like
    # every other None-device model: sequentially, in list order" -- the
    # original, still-default behavior.
    device: str | None = None
    patch_size: int = 256
    # 2026-08-26: raised from 32 -- with a 256px patch and 32px overlap, the
    # row-tile seam between the first two latitude tiles (row_start=0 and
    # row_start=224 on the real 721-row global grid) landed at ~26-34N, and
    # every tile in that band is right at its OWN edge there too (both
    # tiles' 32px blend ramps meet exactly at that seam). Real production
    # output confirmed this as a persistent, weather-independent ridge in
    # served "stationary" front probability pinned at 33.25N across every
    # checked forecast cycle (8 independent real cycles, real weather
    # varying substantially between them, same latitude every time) --
    # reported as a fake stationary front. A real-model comparison run
    # (2026-08-25 18Z cycle) showed the seam's excess probability shrinks
    # sharply as overlap widens (32 -> 96 moved the local excess from
    # roughly 2x the surrounding trend down to a small fraction of it),
    # because a wider overlap band keeps the two tiles' shared crossover
    # point further from each tile's own degraded edge. 96 was chosen as
    # the point past which returns visibly diminish (128 wasn't much
    # better) while only roughly doubling per-step CPU inference time
    # (~11s -> ~24s on this box) -- worth it against the 240h x 41-step
    # forecast product's several-hour cycle window. Tune down if compute
    # budget gets tight, back up if the seam is still visible.
    overlap: int = 96
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
    save_orientation_fields: bool = False,
) -> str:
    """Runs one model end-to-end for one (cycle, step). Returns the zarr
    store path written. Does NOT write latest.json -- that's `run_cycle`'s
    job now, once it knows every step that succeeded for this cycle (see
    module docstring); a single step no longer gets to unilaterally decide
    what "latest" means."""
    manifest = run_config.manifest
    logger.info("assembling input for model=%s cycle=%s", manifest.name, cycle)
    input_grid = assemble_model_input(manifest, source, cycle)
    orientation_fields = None
    if save_orientation_fields:
        # Copied now, before inference gets a chance to reuse the grid. Best
        # effort: the 850 hPa fields only decide which side the viewer's
        # front symbols go on (see postprocess/front_lines.py).
        try:
            from frontfinder.postprocess.front_lines import extract_orientation_fields

            orientation_fields = extract_orientation_fields(manifest, input_grid, source.lat, source.lon)
        except Exception:
            logger.exception("could not extract orientation fields for model=%s -- continuing without", manifest.name)

    logger.info("running tiled inference for model=%s", manifest.name)
    served_probs = run_tiled_inference(
        run_config.predictor,
        input_grid,
        manifest,
        patch_size=run_config.patch_size,
        overlap=run_config.overlap,
        batch_size=run_config.batch_size,
        lon_deg=source.lon,
        lat_deg=source.lat,
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
    if orientation_fields is not None:
        try:
            from frontfinder.postprocess.front_lines import fields_filename, save_orientation_fields as save_fields

            save_fields(orientation_fields, os.path.join(os.path.dirname(store_path), fields_filename(store_name)))
        except Exception:
            logger.exception("could not save orientation fields for %s -- continuing without", store_path)
    return store_path


def _store_name(cycle: IFSCycle) -> str:
    # "2026-08-20T18Z_f000.zarr" -- the `_f<NNN>` step suffix is what lets
    # retention.py's _STORE_NAME_RE and the webapp's latest.json both tell
    # different lead times of the same cycle apart on disk.
    return f"{cycle.date}T{cycle.run_hour:02d}Z_f{cycle.step:03d}.zarr"


def _write_latest_pointer(
    output_root: str,
    model_name: str,
    cycle: IFSCycle,
    step_entries: list[dict],
    expected_steps: tuple[int, ...] | None = None,
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
          ],
          "expected_steps": [0, 6, 12, ...]
        }

    `expected_steps` (lead hours this cycle is targeting, published or not)
    lets the webapp's slider draw a tick for every expected step and mark
    the ones that haven't landed yet. Omitted when the caller doesn't know.
    """
    pointer_path = os.path.join(output_root, model_name, "latest.json")
    with open(pointer_path, "w") as f:
        pointer = {"cycle_time": _cycle_time(cycle), "steps": step_entries}
        if expected_steps is not None:
            pointer["expected_steps"] = list(expected_steps)
        json.dump(pointer, f)


def run_cycle(
    run_configs: list[ModelRunConfig],
    source: IFSFieldSource,
    cycle_date: str,
    run_hour: int,
    output_root: str,
    steps: tuple[int, ...] | None = None,
    front_lines: bool = False,
    front_line_threshold: float | None = None,
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

    2026-09-17: iterates STEP-major (every configured model for step 0, then
    every model for step 6, ...) rather than the original MODEL-major order
    (every step for model A, then every step for model B). Two reasons:

    1. Shared input variables get fetched once per step, not once per
       (step, model). `theta-e_uv_q` and `spix_wce_loss_modified_dilation`
       both need identical pressure-level fields (u/v/specific_humidity/the
       temperature+humidity pair equivalent_potential_temperature derives
       from) at the same 6 levels -- see both manifests in manifests.py.
       `assemble_model_input` always resolves these through the same
       `source` object's own per-(variable, level, cycle) cache
       (`EcmwfOpenDataSource._field_cache`, never evicted for the life of
       one `run_cycle` call), so processing every model for a step back to
       back means the second (and any further) model's call is a hot
       in-memory hit for every field the first model already resolved --
       only fields genuinely unique to that model (spix's 10m wind/2m
       dewpoint/terrain) do any new work. Model-major order still got this
       for free too, since the same `source` persists across the whole
       call -- but only within a single `run_cycle` invocation that
       actually reaches every model for a given step; see point 2 for why
       that reach mattered.
    2. A poll can time out (systemd's TimeoutStartSec) or fail partway
       through -- model-major order meant a cut-off run could finish EVERY
       step for the first model and ZERO steps for the second, so their
       `latest.json`s (and therefore the webapp's shared time slider, once
       more than one model is selected there) drifted out of step by up to
       a whole cycle's worth of runtime. Step-major order means a cut-off
       run leaves every model at roughly the same step, keeping their
       published cycles in lockstep.

    Per-model bookkeeping (already-written skip, ascending order, one
    model's exception never blocking another's, progressive `latest.json`
    writes) is unchanged -- only which step each model tackles next.

    2026-09-17: within one step, the per-model work (`_process_one_model`
    below) runs CONCURRENTLY -- via a thread pool, one worker per model --
    whenever `run_configs` name more than one distinct `device` (e.g. two
    models pinned to two different GPUs: real overlap, since each model's
    TensorFlow graph runs on its own physical device). When every
    `run_config.device` is the same value (including the all-`None`/CPU
    case every deployment used before GPUs existed here), work stays
    sequential in list order -- unchanged from before this date, and
    deliberately so: two models fighting over the SAME CPU cores under
    Python threads would just add contention/GIL overhead for no real
    parallelism, since neither TF op releases the GIL for the other to make
    progress on the same device. `_write_latest_pointer` is only ever
    called from the main thread, after a step's results are all in (each
    model writes its own, differently-named file, so there's no
    cross-model collision to guard against either way).

    `front_lines` (off by default here; the CLI turns it on) additionally
    derives smooth front lines from each published step's probabilities
    (`postprocess.front_lines`) and lists the per-step GeoJSON file as a
    `"fronts"` key on that step's `latest.json` entry, for the webapp's
    optional "Front lines" overlay. It never changes the zarr stores, and
    any failure there is logged and swallowed -- inference and publishing
    carry on exactly as without it. `front_line_threshold` overrides the
    summed-probability threshold for a front's crest (the splines default
    otherwise).
    """
    if steps is None:
        steps = target_steps_for_cycle(run_hour)
    front_line_config = None
    if front_lines:
        try:
            from frontfinder.postprocess.front_lines import default_config

            front_line_config = default_config(front_line_threshold)
        except Exception:
            logger.exception("front lines unavailable (could not import the splines module) -- continuing without them")
            front_lines = False

    store_paths: dict[str, list[str]] = {rc.manifest.name: [] for rc in run_configs}
    step_entries: dict[str, list[dict]] = {rc.manifest.name: [] for rc in run_configs}
    concurrent_dispatch = len({rc.device for rc in run_configs}) > 1

    def process_one_model(run_config: ModelRunConfig) -> tuple[str, str | None]:
        manifest_name = run_config.manifest.name
        existing_store_path = os.path.join(output_root, manifest_name, _store_name(cycle))
        if os.path.exists(existing_store_path):
            # 2026-09-01: cli.py's --poll now re-invokes run_cycle for a
            # cycle that's missing steps rather than abandoning it (see
            # its `_cycle_is_complete`), so this function gets called
            # repeatedly against the SAME cycle as retries land one
            # throttled step at a time. Without this skip, every retry
            # re-ran inference for every step already on disk too
            # (`_fetch_grib`'s cache only ever avoided re-downloading,
            # not re-inferring) -- a cycle stuck on one hot step got
            # slower every single poll as more already-done steps piled
            # up ahead of it. A store directory existing is trusted as
            # a completed write (run_one_model only creates it via
            # write_front_pyramid's mode="w", right before logging
            # "wrote ...").
            logger.info("model %s step %s already written for cycle %s -- skipping re-run", manifest_name, step, cycle)
            return manifest_name, existing_store_path
        try:
            extra = {"save_orientation_fields": True} if front_lines else {}
            return manifest_name, run_one_model(run_config, source, cycle, output_root, **extra)
        except Exception:
            logger.exception("model %s failed for cycle %s -- skipping this step", manifest_name, cycle)
            return manifest_name, None

    for step in steps:
        cycle = IFSCycle(date=cycle_date, run_hour=run_hour, step=step)
        if concurrent_dispatch:
            with ThreadPoolExecutor(max_workers=len(run_configs)) as pool:
                step_results = list(pool.map(process_one_model, run_configs))
        else:
            step_results = [process_one_model(rc) for rc in run_configs]

        for manifest_name, store_path in step_results:
            if store_path is None:
                continue
            store_paths[manifest_name].append(store_path)
            step_entries[manifest_name].append(
                {"step_hours": step, "valid_time": _valid_time(cycle), "store": _store_name(cycle)}
            )
            if front_lines:
                from frontfinder.postprocess.front_lines import attach_front_lines

                # Re-derives every published step so far (tracking needs the
                # whole sequence, in order; ~0.4 s/step) and never raises.
                attach_front_lines(
                    os.path.join(output_root, manifest_name), step_entries[manifest_name], front_line_config
                )
            # Publish as soon as this step lands, not after the whole cycle
            # -- see the postmortem above.
            _write_latest_pointer(
                output_root, manifest_name, cycle, step_entries[manifest_name], expected_steps=tuple(steps)
            )

    results: dict[str, list[str]] = {}
    for run_config in run_configs:
        manifest_name = run_config.manifest.name
        if store_paths[manifest_name]:
            results[manifest_name] = store_paths[manifest_name]
        else:
            logger.error("model %s produced zero successful steps for cycle %s-%02dZ", manifest_name, cycle_date, run_hour)

    return results


def known_model_names() -> list[str]:
    return sorted(MANIFESTS)
