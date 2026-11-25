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

HRRR 3 km (`hrrr-3km`) stores stay on HRRR's native Lambert conformal grid:
level 0 is 1059x1799 with 1-D `x`/`y` coordinates in metres and no lat/lon.
Front lines for it are extracted ON the native 3 km grid (reprojecting the
probabilities to lat/lon first would throw away exactly the detail the model
exists to add); the `probs.npz` sidecar then carries `x`/`y`/`crs` keys
instead of `lat`/`lon`, the `fields.npz` sidecar likewise (thinned to every
other native cell), and `postprocess.grid_geom.NativeGridCoords` adapts
between the pixel and lon/lat worlds (see `splines.extract_fronts`'s `geom`).
The emitted GeoJSON is WGS84 lon/lat like the IFS one, since the viewer draws
plain GeoJSON. A published native-grid store with no sidecar has no CRS to
recover, so its geometry comes from `ingest.hrrr_grid.native_grid` matched to
the store's own grid name/shape -- and if the sidecar is corrupt or missing
for a store the loader can't place at all, that step is logged and skipped.

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
import dataclasses
import itertools
import json
import logging
import os

import numpy as np
import xarray as xr

from frontfinder.postprocess.grid_geom import NativeGridCoords

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
# HRRR's native 3 km probabilities are block-averaged and blurred to roughly the IFS resolution before
# crest extraction (see `splines.coarsen_probabilities`); blurring lowers the peaks, so the native crest
# cutoff sits below FRONT_LINE_THRESHOLD. 0.20 keeps total front length within ~7% of the un-coarsened
# lines on the 2026-11-16/18/20 12Z f000 fields (`scripts/hrrr_spline_parity.py`).
NATIVE_FRONT_LINE_THRESHOLD = 0.20
NATIVE_BLOCK = 3
NATIVE_BLUR_CELLS = 3.5
FRONTS_SUFFIX = ".fronts.geojson"
FIELDS_SUFFIX = ".fields.npz"
PROBS_SUFFIX = ".probs.npz"

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


def native_config(threshold=None):
    """SplineConfig for a model kept on its native projected grid (HRRR): the default settings with the
    probabilities coarsened before crest extraction, at the native crest cutoff."""
    return dataclasses.replace(
        default_config(NATIVE_FRONT_LINE_THRESHOLD if threshold is None else threshold),
        native_block=NATIVE_BLOCK,
        native_blur_cells=NATIVE_BLUR_CELLS,
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


def probabilities_filename(store_name: str) -> str:
    """`2026-10-31T06Z_f006.zarr` -> `2026-10-31T06Z_f006.probs.npz`."""
    if not store_name.endswith(".zarr"):
        raise ValueError(f"not a zarr store name: {store_name!r}")
    return store_name[: -len(".zarr")] + PROBS_SUFFIX


def save_probabilities_sidecar(level0: xr.Dataset, path: str, coords: NativeGridCoords | None = None) -> None:
    """Saves the exact float32 level-0 class grids of a store about to be
    uint8-packed, so the front lines can still be derived from them.

    A packed store only holds the probabilities to within 1/254, which is
    enough to draw but moves the spline crests a little and can add or drop a
    short front near the threshold. `load_probabilities` prefers this sidecar,
    so with packing on the lines come out exactly as they did from float32
    stores. Call it BEFORE the store is written (packing mutates the tree).

    `coords` selects the coordinate convention the sidecar is written with:
    None (IFS) stores 1-D `lat`/`lon`; a `NativeGridCoords` (HRRR's projected
    grid) stores the 1-D `x`/`y` cell coordinates and the CRS instead, since a
    projected grid has no 1-D lat/lon.
    """
    arrays = {
        code: np.asarray(level0[cls].values, dtype=np.float32) for cls, code in CLASS_CODES.items() if cls in level0
    }
    if coords is None:
        grids = {"lat": np.asarray(level0["lat"].values), "lon": np.asarray(level0["lon"].values), **arrays}
    else:
        bad = {code: grid.shape for code, grid in arrays.items() if grid.shape != coords.shape}
        if bad:
            raise ValueError(f"probability grids {bad} do not match the grid shape {coords.shape}")
        grids = {"x": np.asarray(coords.x, dtype=np.float64), "y": np.asarray(coords.y, dtype=np.float64), "crs": np.array(coords.crs), **arrays}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **grids)
    os.replace(tmp, path)


def _load_probabilities_sidecar(path: str) -> tuple[xr.Dataset, NativeGridCoords | None]:
    """`(dataset, geom)` from a sidecar written by `save_probabilities_sidecar`
    -- the regular (latitude, longitude) layout with geom None, or the native
    (y, x) layout with the `NativeGridCoords` its `x`/`y`/`crs` describe."""
    with np.load(path) as z:
        if "lat" not in z.files:
            x = np.asarray(z["x"], dtype=np.float64)
            y = np.asarray(z["y"], dtype=np.float64)
            crs = str(z["crs"].item())
            dataset = xr.Dataset(
                {code: (("y", "x"), z[code]) for code in CLASS_CODES.values() if code in z.files},
                coords={"y": y, "x": x},
            )
            return dataset, NativeGridCoords.from_crs(crs, x, y)
        coords = {"latitude": z["lat"], "longitude": z["lon"]}
        return (
            xr.Dataset(
                {code: (("latitude", "longitude"), z[code]) for code in CLASS_CODES.values() if code in z.files},
                coords=coords,
            ),
            None,
        )


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


def extract_native_orientation_fields(
    manifest, input_grid: np.ndarray, grid
) -> dict[str, np.ndarray] | None:
    """`extract_orientation_fields` for a projected grid: copies the 850 hPa
    theta-e and wind channels out of an assembled model input grid (H, W,
    n_channels) laid out on a native `GridSpec`, thinned to every other cell
    (the same thinning; symbol sides are sampled 150 km off the line, far
    coarser than 6 km). Coordinates go along as the thinned 1-D `x`/`y` arrays
    plus the CRS string; nothing is rolled, since projected lon/lat is
    already signed. The copy matters: inference may reuse `input_grid`.
    Returns None if the model doesn't carry those channels."""
    names = manifest.channel_names()
    if any(name not in names for name in ORIENTATION_CHANNELS.values()):
        return None
    stride = 2 if min(input_grid.shape[:2]) >= 200 else 1
    arrays = {
        key: np.array(input_grid[::stride, ::stride, names.index(name)], dtype=np.float32)
        for key, name in ORIENTATION_CHANNELS.items()
    }
    return {
        **arrays,
        "x": np.asarray(grid.x[::stride], dtype=np.float64),
        "y": np.asarray(grid.y[::stride], dtype=np.float64),
        "crs": np.array(grid.crs),
    }


def save_orientation_fields(fields: dict[str, np.ndarray], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **fields)
    os.replace(tmp, path)


def _load_orientation_fields(path: str) -> tuple[dict[str, xr.DataArray], NativeGridCoords | None]:
    """`load_orientation_fields` with the grid adapter: the sidecar's regular
    (latitude, longitude) DataArrays with a None geom, or its native (y, x)
    ones with the `NativeGridCoords` (no lat/lon table -- sampling only needs
    the inverse) for the thinned grid its `x`/`y`/`crs` describe."""
    if not os.path.exists(path):
        return {}, None
    try:
        with np.load(path) as z:
            if "lat" not in z.files:
                x = np.asarray(z["x"], dtype=np.float64)
                y = np.asarray(z["y"], dtype=np.float64)
                crs = str(z["crs"].item())
                coords = {"y": y, "x": x}
                return (
                    {key: xr.DataArray(z[key], dims=("y", "x"), coords=coords) for key in ORIENTATION_CHANNELS},
                    NativeGridCoords(crs=crs, x=x, y=y),
                )
            coords = {"latitude": z["lat"].astype(np.float64), "longitude": z["lon"].astype(np.float64)}
            return (
                {
                    key: xr.DataArray(z[key], dims=("latitude", "longitude"), coords=coords)
                    for key in ORIENTATION_CHANNELS
                },
                None,
            )
    except Exception:
        logger.exception("unreadable orientation fields %s -- orienting without them", path)
        return {}, None


def load_orientation_fields(path: str) -> dict[str, xr.DataArray]:
    """The sidecar as `{temperature, u, v}` DataArrays (dims latitude,
    longitude), or `{}` if there is none or it can't be read."""
    return _load_orientation_fields(path)[0]


def load_probabilities(store_path: str) -> xr.Dataset:
    """Reads a published store's full-resolution (level 0) class probabilities
    in the layout `postprocess.splines` expects: one variable per front type
    code (`CF`, `WF`, ...), dims (latitude, longitude) -- or dims (y, x) for a
    store on a projected grid, which `_load_probabilities` pairs with the
    grid adapter the sampling needs.

    Uses the `<store>.probs.npz` float32 sidecar when there is one (stores
    written with uint8 packing on -- see `save_probabilities_sidecar`),
    otherwise the store itself. A published native-grid store carries no CRS
    of its own, so without a sidecar its geometry is taken from the ingest
    grid definition (`ingest.hrrr_grid.native_grid`) matched by the store's
    own grid name/shape; a store the loader cannot place raises, which its
    callers log and skip.
    """
    return _load_probabilities(store_path)[0]


def _load_probabilities(store_path: str) -> tuple[xr.Dataset, NativeGridCoords | None]:
    sidecar = os.path.join(os.path.dirname(store_path), probabilities_filename(os.path.basename(store_path)))
    if os.path.exists(sidecar):
        try:
            return _load_probabilities_sidecar(sidecar)
        except Exception:
            logger.exception("unreadable probabilities sidecar %s -- using the store's own values", sidecar)
    ds = xr.open_zarr(store_path, group="0", consolidated=False, zarr_format=3)
    if any("scale_factor" in ds[cls].encoding for cls in CLASS_CODES if cls in ds):
        logger.warning("%s is uint8-packed and has no probabilities sidecar -- front lines use the packed values", store_path)
    present = {cls: code for cls, code in CLASS_CODES.items() if cls in ds}
    if not present:
        raise ValueError(f"{store_path} has none of the front classes {sorted(CLASS_CODES)}")
    if "lat" in ds.coords and "lon" in ds.coords:
        ds = ds[list(present)].rename(present).rename(lat="latitude", lon="longitude")
        return ds.drop_vars("spatial_ref", errors="ignore").load(), None
    if not {"x", "y"}.issubset(ds.coords):
        raise ValueError(f"{store_path} has neither lat/lon nor x/y coordinates -- cannot place it on the earth")
    return (
        ds[list(present)].rename(present).drop_vars("spatial_ref", errors="ignore").load(),
        _native_coords_for_store(ds),
    )


def _native_coords_for_store(ds: xr.Dataset) -> NativeGridCoords:
    """The grid adapter for a published store kept on a projected grid.

    The store itself carries only its 1-D x/y cell coordinates (the viewer
    reads the CRS from `latest.json`), so the 2-D lat/lon geometry comes from
    the ingest grid definition that produced it, matched by the store's own
    `grid` attribute, falling back to its shape."""
    from frontfinder.ingest.hrrr_grid import SUPPORTED_COARSEN, native_grid

    shape = (int(ds.sizes["y"]), int(ds.sizes["x"]))
    name = ds.attrs.get("grid")
    for coarsen in SUPPORTED_COARSEN:
        grid = native_grid(coarsen)
        if grid.shape == shape and (name is None or name == grid.name):
            return NativeGridCoords.from_grid_spec(grid)
    raise ValueError(f"no ingest grid matches the published store's shape {shape} (grid name {name!r})")


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
            probabilities, probs_geom = _load_probabilities(os.path.join(model_dir, store_name))
            fields, field_geom = _load_orientation_fields(os.path.join(model_dir, fields_filename(store_name)))
            boundaries = splines.extract_fronts(
                probabilities,
                config,
                temperature=fields.get("temperature"),
                u=fields.get("u"),
                v=fields.get("v"),
                previous=previous,
                geom=probs_geom,
                field_geom=field_geom,
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


def _config_for_pointer(pointer: dict):
    """The spline config a published pointer's model wants: HRRR grids (a
    `grid` key naming an `hrrr-*` grid) get the coarsened `native_config()`,
    everything else the IFS `default_config()`."""
    grid = pointer.get("grid") or {}
    return native_config() if str(grid.get("name", "")).startswith("hrrr") else default_config()


def backfill(output_root: str, model: str | None = None, config: object | None = None) -> dict[str, int]:
    """Generates front lines for the cycle each model's `latest.json` points
    at and adds the `fronts` keys to that file. Returns `{model: n_steps}`.

    With `config` None each model gets its own: `native_config()` when its
    `latest.json` names an HRRR grid, `default_config()` otherwise.

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
        attach_front_lines(model_dir, pointer["steps"], config or _config_for_pointer(pointer))
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
