"""Front lines ("splines") for the web viewer's optional overlay.

Turns each published step's probability grids (the level-0 arrays of its zarr
store) into smooth, typed, tracked front lines with `postprocess.splines`, and
writes one GeoJSON file per step next to the zarr stores:

    <output_root>/<model>/<cycle>_f<step>.fronts.geojson

`run_cycle` calls `write_front_lines` after each step lands and lists the file
in `latest.json` (a `"fronts"` key on that step's entry); the webapp fetches it
only when the viewer turns on the "Front lines" option. Nothing here changes
the zarr stores or the probability layers.

Symbol sides (which side of a line the cold-front triangles, warm-front
semicircles, ... go on) are chosen by `splines` from the temperature contrast
and cross-front wind at 850 hPa, plus motion since the previous step. The
850 hPa equivalent potential temperature and wind are saved at inference time
as a small sidecar next to each store (`<store>.fields.npz`, 0.5 degree), so
the lines can be re-derived later without re-fetching IFS data. A step without
a sidecar falls back to motion alone (and the first step then has no clue at
all), which is why `rebuild_orientation_fields` exists for cycles published
before the sidecars did.

Tracking (one event id per front across lead times) depends on the order of
steps, so every call re-derives ALL published steps of the cycle, in order,
rather than keeping state between calls. That is cheap (~0.4 s per step on one
core) and means a resumed or re-run cycle always ends with consistent ids.

Everything in `write_front_lines` is best effort. A failure here is logged and
never propagates to inference or to `latest.json` publishing.

Usage (backfill the cycle `latest.json` already points at; --rebuild-fields
also recreates the orientation sidecars from the GRIB cache):
    python -m frontfinder.postprocess.front_lines --output-root output [--model NAME]
        [--rebuild-fields --cache-dir ifs_cache --model-dir models]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# frontfinder class name -> the type code `postprocess.splines` expects.
CLASS_CODES: dict[str, str] = {
    "cold": "CF",
    "warm": "WF",
    "stationary": "SF",
    "occluded": "OF",
    "dryline": "DL",
    "trough": "TROF",
    "tropical_trough": "TT",
}

# Summed-probability cutoff for a front crest (the splines default is 0.3).
FRONT_LINE_THRESHOLD = 0.25
# Total interest two fronts at consecutive steps need to count as the same event (splines default 0.6).
FRONT_MATCH_THRESHOLD = 0.5
FRONTS_SUFFIX = ".fronts.geojson"
FIELDS_SUFFIX = ".fields.npz"

# Channel names (see ModelManifest.channel_names) the symbol sides are chosen
# from, keyed by the name `splines.extract_fronts` takes them under.
ORIENTATION_CHANNELS: dict[str, str] = {
    "temperature": "equivalent_potential_temperature_850",
    "u": "u_component_of_wind_850",
    "v": "v_component_of_wind_850",
}


def default_config(threshold=None):
    """SplineConfig with frontfinder's thresholds ( overrides the crest cutoff)."""
    from frontfinder.postprocess import splines  # lazy, like the other users

    return splines.SplineConfig(
        threshold=FRONT_LINE_THRESHOLD if threshold is None else threshold,
        match_threshold=FRONT_MATCH_THRESHOLD,
    )


def fronts_filename(store_name: str) -> str:
    """`2026-10-31T06Z_f006.zarr` -> `2026-10-31T06Z_f006.fronts.geojson`."""
    if not store_name.endswith(".zarr"):
        raise ValueError(f"not a zarr store name: {store_name!r}")
    return store_name[: -len(".zarr")] + FRONTS_SUFFIX


def fields_filename(store_name: str) -> str:
    """`2026-10-31T06Z_f006.zarr` -> `2026-10-31T06Z_f006.fields.npz`."""
    if not store_name.endswith(".zarr"):
        raise ValueError(f"not a zarr store name: {store_name!r}")
    return store_name[: -len(".zarr")] + FIELDS_SUFFIX


def extract_orientation_fields(
    manifest, input_grid: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> dict[str, np.ndarray] | None:
    """Copies the 850 hPa theta-e and wind channels out of an assembled model
    input grid (H, W, n_channels), thinned to every other point on a global
    0.25 degree grid and rolled to the signed -180..180 longitude convention
    the zarr stores use. Returns None if the model doesn't carry those
    channels. The copy matters: inference may reuse `input_grid`."""
    names = manifest.channel_names()
    if any(name not in names for name in ORIENTATION_CHANNELS.values()):
        return None
    from frontfinder.zarrio.pyramid import _roll_lon_0_360_to_signed

    stride = 2 if min(input_grid.shape[:2]) >= 200 else 1
    arrays = {
        key: np.array(input_grid[::stride, ::stride, names.index(name)], dtype=np.float32)
        for key, name in ORIENTATION_CHANNELS.items()
    }
    lat, lon = np.asarray(lat)[::stride], np.asarray(lon)[::stride]
    lon, arrays = _roll_lon_0_360_to_signed(lon, arrays)
    return {**arrays, "lat": np.asarray(lat, dtype=np.float32), "lon": np.asarray(lon, dtype=np.float32)}


def save_orientation_fields(fields: dict[str, np.ndarray], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **fields)
    os.replace(tmp, path)


def load_orientation_fields(path: str) -> dict[str, xr.DataArray]:
    """The sidecar as `{temperature, u, v}` DataArrays (dims latitude,
    longitude), or `{}` if there is none or it can't be read."""
    if not os.path.exists(path):
        return {}
    try:
        with np.load(path) as z:
            coords = {"latitude": z["lat"].astype(np.float64), "longitude": z["lon"].astype(np.float64)}
            return {
                key: xr.DataArray(z[key], dims=("latitude", "longitude"), coords=coords)
                for key in ORIENTATION_CHANNELS
            }
    except Exception:
        logger.exception("unreadable orientation fields %s -- orienting without them", path)
        return {}


def load_probabilities(store_path: str) -> xr.Dataset:
    """Reads a published store's full-resolution (level 0) class probabilities
    in the layout `postprocess.splines` expects: one variable per front type
    code (`CF`, `WF`, ...), dims (latitude, longitude)."""
    ds = xr.open_zarr(store_path, group="0", consolidated=False, zarr_format=3)
    present = {cls: code for cls, code in CLASS_CODES.items() if cls in ds}
    if not present:
        raise ValueError(f"{store_path} has none of the front classes {sorted(CLASS_CODES)}")
    ds = ds[list(present)].rename(present).rename(lat="latitude", lon="longitude")
    return ds.drop_vars("spatial_ref", errors="ignore").load()


def _viewer_features(collection: dict) -> list[dict]:
    # `overall_geometry` repeats every line of a front once more; the viewer
    # draws the per-type sections, so leaving it out roughly halves the file.
    return [f for f in collection["features"] if f["properties"]["name"] != "overall_geometry"]


def _write_json_atomically(path: str, payload: dict, compact: bool = True) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":") if compact else None)
    os.replace(tmp, path)


def write_front_lines(
    model_dir: str, step_entries: list[dict], config: object | None = None
) -> dict[str, str]:
    """Writes one GeoJSON file per entry of `step_entries` (the `latest.json`
    `steps` list: dicts with `valid_time` and `store`, ascending) into
    `model_dir`.

    Returns `{store_name: fronts_filename}` for the steps that got a file. A
    step whose store can't be read or processed is logged and skipped (the
    next step is then tracked against the last good one); the rest of the
    cycle is unaffected.

    `config` is a `postprocess.splines.SplineConfig` (default settings if
    None).
    """
    from frontfinder.postprocess import splines  # lazy: scikit-image is only needed when this actually runs

    config = config or default_config()
    new_ids = (f"front_{n:04d}" for n in itertools.count())
    written: dict[str, str] = {}
    previous: list = []
    for entry in step_entries:
        store_name = entry["store"]
        try:
            probabilities = load_probabilities(os.path.join(model_dir, store_name))
            fields = load_orientation_fields(os.path.join(model_dir, fields_filename(store_name)))
            boundaries = splines.extract_fronts(
                probabilities,
                config,
                temperature=fields.get("temperature"),
                u=fields.get("u"),
                v=fields.get("v"),
                previous=previous,
            )
            splines.track_fronts(previous, boundaries, config, new_ids)
            collection = splines.to_geojson(boundaries, entry["valid_time"])
            payload = {
                "type": "FeatureCollection",
                "valid_time": entry["valid_time"],
                # True when the symbol sides were chosen with the 850 hPa fields; the viewer only
                # draws symbols then (motion alone is a weak clue, and missing at the first step).
                "oriented": all(key in fields for key in ORIENTATION_CHANNELS),
                "features": _viewer_features(collection),
            }
            name = fronts_filename(store_name)
            _write_json_atomically(os.path.join(model_dir, name), payload)
        except Exception:
            logger.exception("front lines failed for %s -- skipping this step", store_name)
            continue
        written[store_name] = name
        previous = boundaries
    return written


def attach_front_lines(model_dir: str, step_entries: list[dict], config: object | None = None) -> None:
    """Runs `write_front_lines` and sets `entry["fronts"]` on every entry that
    got a file (and removes a stale `fronts` key from the ones that didn't).
    Never raises: front lines are an optional extra and must not be able to
    break inference or `latest.json` publishing."""
    try:
        written = write_front_lines(model_dir, step_entries, config)
    except Exception:
        logger.exception("front lines unavailable for %s -- continuing without them", model_dir)
        written = {}
    for entry in step_entries:
        name = written.get(entry["store"])
        if name is not None:
            entry["fronts"] = name
        else:
            entry.pop("fronts", None)


def backfill(output_root: str, model: str | None = None, config: object | None = None) -> dict[str, int]:
    """Generates front lines for the cycle each model's `latest.json` points
    at and adds the `fronts` keys to that file. Returns `{model: n_steps}`.

    Not safe to run while `run_cycle` is publishing to the same model (both
    rewrite `latest.json`); the scheduler's next step would otherwise
    overwrite the keys added here, though it adds them back itself.
    """
    models = [model] if model else sorted(
        d for d in os.listdir(output_root) if os.path.isfile(os.path.join(output_root, d, "latest.json"))
    )
    counts: dict[str, int] = {}
    for name in models:
        model_dir = os.path.join(output_root, name)
        pointer_path = os.path.join(model_dir, "latest.json")
        with open(pointer_path) as f:
            pointer = json.load(f)
        before = json.dumps(pointer, sort_keys=True)
        attach_front_lines(model_dir, pointer["steps"], config)
        if json.dumps(pointer, sort_keys=True) != before:  # leave retired models' pointers alone
            _write_json_atomically(pointer_path, pointer, compact=False)  # same formatting run_cycle uses
        counts[name] = sum(1 for s in pointer["steps"] if "fronts" in s)
    return counts


def rebuild_orientation_fields(output_root: str, model_name: str, source) -> int:
    """Rebuilds the `.fields.npz` sidecars for the cycle `latest.json` points
    at, for steps that have a store but no sidecar (cycles published before
    sidecars existed). Re-reads the IFS fields through `source` -- from its
    GRIB cache when the files are still there -- but only theta-e and wind at
    850 hPa, not the whole model input. Returns the number written; a step
    that fails is logged and skipped. Models that are not in `MANIFESTS`
    (retired ones) are skipped."""
    from frontfinder.config.manifests import MANIFESTS, ModelManifest, VariableSpec
    from frontfinder.ingest.ecmwf_ifs import IFSCycle, assemble_model_input

    full = MANIFESTS.get(model_name)
    model_dir = os.path.join(output_root, model_name)
    if full is None or not os.path.isfile(os.path.join(model_dir, "latest.json")):
        return 0
    # Just the three channels the sides are chosen from, not all 28.
    reduced = ModelManifest(
        name=full.name,
        weights_filename=full.weights_filename,
        variables=tuple(
            VariableSpec(name.rsplit("_", 1)[0], levels=(int(name.rsplit("_", 1)[1]),))
            for name in ORIENTATION_CHANNELS.values()
        ),
    )
    with open(os.path.join(model_dir, "latest.json")) as f:
        pointer = json.load(f)
    cycle_time = pointer["cycle_time"]
    written = 0
    for entry in pointer["steps"]:
        fields_path = os.path.join(model_dir, fields_filename(entry["store"]))
        if os.path.exists(fields_path) or not os.path.isdir(os.path.join(model_dir, entry["store"])):
            continue
        cycle = IFSCycle(date=cycle_time[:10], run_hour=int(cycle_time[11:13]), step=entry["step_hours"])
        try:
            grid = assemble_model_input(reduced, source, cycle)
            fields = extract_orientation_fields(reduced, grid, source.lat, source.lon)
            if fields is None:
                raise ValueError("no orientation channels")
            save_orientation_fields(fields, fields_path)
        except Exception:
            logger.exception("could not rebuild orientation fields for %s -- skipping", entry["store"])
            continue
        written += 1
        logger.info("rebuilt %s", os.path.basename(fields_path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate viewer front lines for the published cycle.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", help="one model directory name (default: every model with a latest.json)")
    parser.add_argument("--threshold", type=float, help="summed-probability threshold for a front's crest")
    parser.add_argument(
        "--rebuild-fields",
        action="store_true",
        help=(
            "first rebuild the 850 hPa orientation fields (symbol sides) for steps that lack them, "
            "from the IFS GRIB cache (needs --cache-dir and --model-dir; downloads anything missing)"
        ),
    )
    parser.add_argument(
        "--fields-only",
        action="store_true",
        help=(
            "with --rebuild-fields: only write the missing orientation sidecars; do not regenerate lines "
            "or touch latest.json. Safe while a cycle is being published (it only adds files); the next "
            "scheduler process to publish a step re-derives the lines from them."
        ),
    )
    parser.add_argument("--cache-dir", help="the scheduler's GRIB cache (with --rebuild-fields)")
    parser.add_argument("--model-dir", help="directory holding the static geopotential file (with --rebuild-fields)")
    parser.add_argument("--ifs-source", default="aws", choices=["aws", "azure", "google", "ecmwf"])
    args = parser.parse_args()
    if args.fields_only and not args.rebuild_fields:
        parser.error("--fields-only needs --rebuild-fields")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = None
    if args.threshold is not None:
        from frontfinder.postprocess import splines

        config = default_config(args.threshold)
    models = [args.model] if args.model else sorted(
        d for d in os.listdir(args.output_root) if os.path.isfile(os.path.join(args.output_root, d, "latest.json"))
    )
    if args.rebuild_fields:
        if not (args.cache_dir and args.model_dir):
            parser.error("--rebuild-fields needs --cache-dir and --model-dir")
        from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource

        source = EcmwfOpenDataSource(cache_dir=args.cache_dir, source=args.ifs_source, static_data_dir=args.model_dir)
        for name in models:
            print(f"{name}: rebuilt orientation fields for {rebuild_orientation_fields(args.output_root, name, source)} step(s)")
    if args.fields_only:
        return
    for name, n in backfill(args.output_root, args.model, config).items():
        print(f"{name}: front lines for {n} step(s)")


if __name__ == "__main__":
    main()
