"""Publish one finished HRRR pilot run in the layout the webapp reads:
<out>/<model>/<case>.zarr, domain.geojson, and latest.json. Local only.

    uv run --frozen python scripts/publish_hrrr_pilot.py \
        --root ./hrrr-pilot --out ./hrrr-pilot/viewer-data
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any

import numpy as np

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST
from frontfinder.ingest.base import GridSpec
from frontfinder.ingest.hrrr_grid import (
    LOW_CONFIDENCE_MARGIN_CELLS,
    RING_STRIDE,
    domain_geojson,
    edge_bounds,
    native_grid,
)
from frontfinder.zarrio.pyramid import RegionalFrontFields, build_regional_pyramid, write_front_pyramid

DEFAULT_CASE = "2026-11-20T12Z_f000"
DEFAULT_CONFIG = "hrrr_3km_c1"
MODEL_KEY = "hrrr-3km"
COARSEN_BY_KM = {3: 1, 6: 2, 12: 4, 24: 8}


def config_grid(config: str) -> GridSpec:
    """The native grid of an `hrrr_<km>km_<tag>` config (e.g. `hrrr_3km_c1`, `hrrr_24km_earth`)."""
    parts = config.split("_")
    if len(parts) != 3 or parts[0] != "hrrr" or not parts[1].endswith("km") or parts[1][:-2] not in map(str, COARSEN_BY_KM):
        raise ValueError(f"{config!r} is not a native-grid HRRR config")
    return native_grid(COARSEN_BY_KM[int(parts[1][:-2])])


def config_model_key(config: str) -> str:
    """The viewer model directory key for a native-grid HRRR config (e.g. `hrrr-3km`)."""
    config_grid(config)
    return f"hrrr-{config.split('_')[1]}"


def _write_json(path: str, payload: dict[str, Any]) -> None:
    temp = f"{path}.tmp"
    with open(temp, "w") as handle:
        json.dump(payload, handle)
    os.replace(temp, path)


def publish(
    root: str,
    out: str,
    case: str = DEFAULT_CASE,
    config: str = DEFAULT_CONFIG,
    model_key: str | None = None,
) -> str:
    """Write the store and pointer for one run; returns the store path."""
    run_dir = os.path.join(root, "runs", case, config)
    with open(os.path.join(run_dir, "meta.json")) as handle:
        meta = json.load(handle)
    if meta.get("status") != "ok":
        raise ValueError(f"{run_dir} did not finish cleanly")
    grid = config_grid(config)
    if meta["grid"]["name"] != grid.name:
        raise ValueError(f"run grid {meta['grid']['name']!r} != {grid.name!r}")

    manifest = NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST
    cycle = meta["cycle"]
    cycle_time = f"{cycle['date']}T{cycle['run_hour']:02d}:00:00"
    valid_time = str(meta["valid_time"]).replace("Z", "")[:19]
    with np.load(os.path.join(run_dir, "probs.npz")) as saved:
        probabilities = {cls: saved[cls] for cls in manifest.served_classes}

    model_dir = os.path.join(out, model_key or config_model_key(config))
    os.makedirs(model_dir, exist_ok=True)
    store_name = f"{case}.zarr"
    store_path = os.path.join(model_dir, store_name)
    if os.path.exists(store_path):
        raise FileExistsError(f"{store_path} exists; stores are immutable, publish to a new --out")
    fields = RegionalFrontFields(probabilities, grid, valid_time, cycle_time, f"NOAA HRRR analysis, {config}")
    temp_store = f"{store_path}.tmp"
    shutil.rmtree(temp_store, ignore_errors=True)
    write_front_pyramid(build_regional_pyramid(fields, manifest), temp_store, pack_uint8=True)
    os.rename(temp_store, store_path)

    _write_json(os.path.join(model_dir, "domain.geojson"), domain_geojson(grid))
    _write_json(
        os.path.join(model_dir, "latest.json"),
        {
            "cycle_time": cycle_time,
            "steps": [{"step_hours": int(cycle["step"]), "valid_time": valid_time, "store": store_name}],
            "expected_steps": [int(cycle["step"])],
            "source_label": "HRRR",
            "domain": "domain.geojson",
            "grid": {
                "name": grid.name,
                "proj4": grid.crs,
                "bounds": edge_bounds(grid),
                "lat_ascending": False,
                "dimensions": {"lat": "y", "lon": "x"},
            },
        },
    )
    return store_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish one HRRR pilot run for the local viewer.")
    parser.add_argument("--root", required=True, help="pilot data root containing runs/")
    parser.add_argument("--out", required=True, help="viewer data directory (DATA_DIR for scripts/dev_server.mjs)")
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(publish(os.path.abspath(args.root), os.path.abspath(args.out), args.case, args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
