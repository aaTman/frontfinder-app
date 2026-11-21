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
from frontfinder.ingest.hrrr_grid import native_grid
from frontfinder.zarrio.pyramid import RegionalFrontFields, build_regional_pyramid, write_front_pyramid

DEFAULT_CASE = "2026-11-20T12Z_f000"
DEFAULT_CONFIG = "hrrr_24km_earth"
MODEL_KEY = "hrrr-24km"
COARSEN_BY_KM = {3: 1, 6: 2, 12: 4, 24: 8}
# About 770 km at 24 km, the margin the results audit found still loses context.
LOW_CONFIDENCE_MARGIN_CELLS = 32
RING_STRIDE = 4


def config_grid(config: str) -> GridSpec:
    """The native grid of an `hrrr_<km>km_<winds>` config."""
    parts = config.split("_")
    if len(parts) != 3 or parts[0] != "hrrr" or not parts[1].endswith("km") or parts[1][:-2] not in map(str, COARSEN_BY_KM):
        raise ValueError(f"{config!r} is not a native-grid HRRR config")
    return native_grid(COARSEN_BY_KM[int(parts[1][:-2])])


def edge_bounds(grid: GridSpec) -> list[float]:
    """[xMin, yMin, xMax, yMax] of the cell edges, in CRS units."""
    dx = abs(float(grid.x[1] - grid.x[0]))
    dy = abs(float(grid.y[1] - grid.y[0]))
    return [
        float(grid.x.min()) - dx / 2,
        float(grid.y.min()) - dy / 2,
        float(grid.x.max()) + dx / 2,
        float(grid.y.max()) + dy / 2,
    ]


def _ring(grid: GridSpec, margin: int) -> list[list[float]]:
    """Closed lon/lat ring around the cells `margin` in from each edge."""
    rows, cols = grid.shape
    top, bottom, left, right = margin, rows - 1 - margin, margin, cols - 1 - margin
    cells = (
        [(top, c) for c in range(left, right, RING_STRIDE)]
        + [(r, right) for r in range(top, bottom, RING_STRIDE)]
        + [(bottom, c) for c in range(right, left, -RING_STRIDE)]
        + [(r, left) for r in range(bottom, top, -RING_STRIDE)]
    )
    ring = [
        [round((float(grid.lon[r, c]) + 180.0) % 360.0 - 180.0, 4), round(float(grid.lat[r, c]), 4)] for r, c in cells
    ]
    return [*ring, ring[0]]


def domain_geojson(grid: GridSpec, margin: int = LOW_CONFIDENCE_MARGIN_CELLS) -> dict[str, Any]:
    """The lower-confidence edge band: the domain with its interior cut out."""
    if 2 * margin >= min(grid.shape):
        raise ValueError(f"margin {margin} leaves no interior on a {grid.shape} grid")
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"kind": "low_confidence_edge", "margin_cells": margin},
                "geometry": {"type": "Polygon", "coordinates": [_ring(grid, 0), _ring(grid, margin)[::-1]]},
            }
        ],
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    temp = f"{path}.tmp"
    with open(temp, "w") as handle:
        json.dump(payload, handle)
    os.replace(temp, path)


def publish(root: str, out: str, case: str = DEFAULT_CASE, config: str = DEFAULT_CONFIG) -> str:
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

    model_dir = os.path.join(out, MODEL_KEY)
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
