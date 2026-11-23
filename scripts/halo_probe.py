"""Measure how far real outside-domain context reaches into v2.0.1's output.

Production scores HRRR's CONUS domain with v2.0.1 at 24 km and reflect-pads
every edge (`inference/regional.py`), so cells near an edge are scored with
fabricated context. The published low-confidence band is 256 cells (768 km)
wide at 3 km because that width came from the network's ~180-pixel reach
before C1 existed -- it was never measured. This script measures it.

For each of the domain's four edges it re-runs the 24 km baseline with that
edge's padded context replaced by REAL fields: global IFS 0.25 deg (~28 km, the
same order as the 24 km input grid) regridded onto the Lambert conformal grid
and blended across the seam, so the model sees continuous, physically real
context outside HRRR's domain instead of a mirror image of the boundary strip.
Then it reports |delta probability| per cell against the production run, binned
by distance from that edge, next to a control that repeats the same run with the
halo filled by HRRR's own reflect instead of IFS -- the control is the
machinery's own noise floor, since it reproduces what production already does.

Read the result as sensitivity, not skill: a cell whose probability moves when
its outside-domain context becomes real was being scored on fabricated context
at that distance. Where the delta sits at the control's noise floor, the padding
was never being used and the band is over-wide there.

Usage (on the pilot box, everything already cached -- no network needed for a
case that has been run before):

  uv run --frozen python scripts/halo_probe.py \
      --model-dir /home/orca/hrrr-pilot/weights \
      --hrrr-cache /home/orca/hrrr-pilot/cache/hrrr \
      --ifs-cache /home/orca/hrrr-pilot/cache/ifs \
      --date 2026-11-16 --run-hour 12 --step 0 \
      --out /home/orca/hrrr-pilot/halo-probe/2026-11-16T12Z_f000.json

Statics (surface geopotential) always come from HRRR's own field, reflect-
extended: terrain is cycle-invariant, and the difference between two models'
orography clippings would only add seam noise to a measurement about dynamic
context.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
from pyproj import Transformer
from scipy.ndimage import map_coordinates

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST as MANIFEST
from frontfinder.ingest.assembly import assemble_model_input
from frontfinder.ingest.base import FieldSource, ForecastCycle, GridSpec
from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource
from frontfinder.ingest.hrrr import HRRRFieldSource, HRRR_CRS
from frontfinder.inference.engine import KerasPredictor
from frontfinder.inference.regional import run_regional_inference

EDGES = ("south", "east", "north", "west")


def _extended_grid(grid: GridSpec, edge: str, halo: int) -> GridSpec:
    """`grid` grown by `halo` cells beyond one edge, with 2-D lat/lon filled in."""
    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])  # negative: rows run north to south
    if edge == "south":
        y = np.concatenate([y, y[-1] + dy * np.arange(1, halo + 1)])
    elif edge == "north":
        y = np.concatenate([y[0] - dy * np.arange(halo, 0, -1), y])
    elif edge == "east":
        x = np.concatenate([x, x[-1] + dx * np.arange(1, halo + 1)])
    elif edge == "west":
        x = np.concatenate([x[0] - dx * np.arange(halo, 0, -1), x])
    else:
        raise ValueError(f"unknown edge {edge!r}")
    to_wgs84 = Transformer.from_crs(HRRR_CRS, "EPSG:4326", always_xy=True)
    xx, yy = np.meshgrid(x, y)
    lon, lat = to_wgs84.transform(xx, yy)
    return GridSpec(
        name=f"{grid.name}-{edge}-halo",
        crs=HRRR_CRS,
        x=x,
        y=y,
        lat=np.asarray(lat, dtype=np.float64),
        lon=np.asarray(lon, dtype=np.float64),
    )


def _inner_slice(shape: tuple[int, int], edge: str, halo: int) -> tuple[slice, slice]:
    rows, cols = shape
    if edge == "north":
        return slice(halo, halo + rows), slice(0, cols)
    if edge == "west":
        return slice(0, rows), slice(halo, halo + cols)
    return slice(0, rows), slice(0, cols)


def _seam_weights(shape: tuple[int, int], edge: str, blend: int) -> np.ndarray:
    """Weight of the inner (HRRR) field across the blend band, 1 -> 0 toward the seam."""
    rows, cols = shape
    ramp = (blend - np.arange(blend) - 0.5) / blend  # 1 deep inside the domain -> 0 at the seam
    w = np.ones(shape, dtype=np.float64)
    if edge == "south":
        w[rows - blend : rows] = ramp[:, None]
    elif edge == "north":
        w[0:blend] = ramp[::-1][:, None]
    elif edge == "east":
        w[:, cols - blend : cols] = ramp[None, :]
    elif edge == "west":
        w[:, 0:blend] = ramp[::-1][None, :]
    return w


def _regrid_from_latlon(arr: np.ndarray, src: GridSpec, tgt: GridSpec) -> np.ndarray:
    """Bilinear resample of a rectilinear, north-up source field onto `tgt`."""
    lat = np.asarray(src.lat, dtype=np.float64)
    lon = np.asarray(src.lon, dtype=np.float64)
    dlon = float(lon[1] - lon[0])
    dlat = float(lat[1] - lat[0])
    col = (tgt.lon - lon[0]) / dlon
    row = (tgt.lat - lat[0]) / dlat
    if src.is_global:
        col = col % len(lon)
    coords = np.stack([np.asarray(row).ravel(), np.asarray(col).ravel()])
    return map_coordinates(np.asarray(arr, dtype=np.float64), coords, order=1, mode="nearest").reshape(tgt.shape)


def _reflect_across(edge: str, arr: np.ndarray, halo: int) -> np.ndarray:
    """`arr` grown beyond one edge by mirroring its own boundary strip."""
    pad = {
        "south": ((0, halo), (0, 0)),
        "north": ((halo, 0), (0, 0)),
        "east": ((0, 0), (0, halo)),
        "west": ((0, 0), (halo, 0)),
    }[edge]
    return np.pad(arr, pad, mode="reflect")


class EdgeHaloSource(FieldSource):
    """Inner fields where they exist, donor fields beyond one edge of the domain.

    `donor="ifs"` supplies real outside-domain context regridded from the outer
    source; `donor="reflect"` reproduces production's boundary behaviour and so
    measures the probe's own noise floor.
    """

    source_id = "edge-halo"

    def __init__(
        self,
        inner: FieldSource,
        outer: FieldSource,
        edge: str,
        halo: int,
        blend: int,
        donor: str = "ifs",
    ):
        if donor not in ("ifs", "reflect"):
            raise ValueError(f"donor must be 'ifs' or 'reflect', got {donor!r}")
        self._inner = inner
        self._outer = outer
        self._edge = edge
        self._donor = donor
        self._halo = halo
        self._grid_spec = _extended_grid(inner.grid, edge, halo)
        self._sl = _inner_slice(inner.grid.shape, edge, halo)
        self._w = _seam_weights(inner.grid.shape, edge, blend)

    @property
    def grid(self) -> GridSpec:
        return self._grid_spec

    def _combine(self, inner_arr: np.ndarray, outer_arr: np.ndarray | None) -> np.ndarray:
        if self._donor == "reflect":
            return _reflect_across(self._edge, inner_arr, self._halo).astype(np.float32)
        donor = _regrid_from_latlon(outer_arr, self._outer.grid, self.grid)
        full = np.asarray(donor, dtype=np.float32)
        full[self._sl] = (
            np.asarray(donor[self._sl], dtype=np.float32) * (1.0 - self._w)
            + np.asarray(inner_arr, dtype=np.float32) * self._w
        )
        return full

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: ForecastCycle) -> np.ndarray:
        inner_arr = self._inner.fetch_pressure_level(variable, level_hpa, cycle)
        if self._donor == "reflect":
            return self._combine(inner_arr, None)
        return self._combine(inner_arr, self._outer.fetch_pressure_level(variable, level_hpa, cycle))

    def fetch_single_level(self, variable: str, cycle: ForecastCycle) -> np.ndarray:
        inner_arr = self._inner.fetch_single_level(variable, cycle)
        if self._donor == "reflect":
            return self._combine(inner_arr, None)
        return self._combine(inner_arr, self._outer.fetch_single_level(variable, cycle))

    def fetch_static(self, variable: str) -> np.ndarray:
        # Terrain is cycle-invariant; mirroring HRRR's own keeps the seam clean.
        return _reflect_across(self._edge, self._inner.fetch_static(variable), self._halo)


def _distances_km(shape: tuple[int, int], edge: str, cell_km: float) -> np.ndarray:
    rows, cols = shape
    r = np.arange(rows, dtype=np.float64)
    c = np.arange(cols, dtype=np.float64)
    if edge == "south":
        d = (rows - 1 - r + 0.5)[:, None] * np.ones((1, cols))
    elif edge == "north":
        d = (r + 0.5)[:, None] * np.ones((1, cols))
    elif edge == "east":
        d = np.ones((rows, 1)) * (cols - 1 - c + 0.5)[None, :]
    else:
        d = np.ones((rows, 1)) * (c + 0.5)[None, :]
    return d * cell_km


def _profile(delta_max: np.ndarray, per_class: dict[str, np.ndarray], dist: np.ndarray, cell_km: float) -> dict[str, Any]:
    """Mean/95th-percentile delta binned by distance from the probed edge."""
    nbins = int(dist.max() // cell_km) + 1
    idx = np.clip((dist / cell_km).astype(int), 0, nbins - 1)
    bins = []
    for b in range(nbins):
        m = idx == b
        if not m.any():
            continue
        bins.append(
            {
                "cell": b,
                "km": round((b + 0.5) * cell_km, 1),
                "dp_max_mean": float(delta_max[m].mean()),
                "dp_max_p95": float(np.percentile(delta_max[m], 95)),
                **{k: float(v[m].mean()) for k, v in per_class.items()},
            }
        )
    return {"cell_km": cell_km, "bins": bins}


def _reach_km(profile: dict[str, Any], control: dict[str, Any], floor_x: float = 3.0, run_len: int = 3) -> float:
    """Furthest bin still clearly above the control, requiring `run_len` in a row."""
    bins = profile["bins"]
    ctrl = {b["cell"]: b["dp_max_mean"] for b in control["bins"]}
    far = [b["dp_max_mean"] for b in bins if b["cell"] >= len(bins) // 2] or [0.0]
    threshold = max(floor_x * max(ctrl.values() or [0.0]), 3.0 * float(np.mean(far)), 1e-4)
    above = [b["dp_max_mean"] > threshold for b in bins]
    last = 0.0
    for i in range(len(bins) - run_len + 1):
        if all(above[i : i + run_len]):
            last = bins[i + run_len - 1]["km"]
    return last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--hrrr-cache", required=True)
    ap.add_argument("--ifs-cache", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--run-hour", type=int, default=12)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--ifs-source", default="google", help="ecmwf-open-data mirror (the AWS mirror 503s persistently)")
    ap.add_argument("--halo", type=int, default=192, help="donor cells beyond the probed edge (>180 px network reach)")
    ap.add_argument("--blend", type=int, default=12, help="cells over which HRRR fades to the donor across the seam")
    ap.add_argument("--edges", default=",".join(EDGES))
    ap.add_argument("--band-km", type=float, default=768.0, help="the published low-confidence band width to compare against")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cycle = ForecastCycle(date=args.date, run_hour=args.run_hour, step=args.step)
    inner = HRRRFieldSource(cache_dir=args.hrrr_cache, coarsen=8, wind_frame="earth")
    outer = EcmwfOpenDataSource(cache_dir=args.ifs_cache, source=args.ifs_source, static_data_dir=args.model_dir)
    predictor = KerasPredictor(os.path.join(args.model_dir, MANIFEST.weights_filename))

    shape = inner.grid.shape
    cell_m = float(abs(inner.grid.x[1] - inner.grid.x[0]))
    cell_km = cell_m / 1000.0
    sl0 = (slice(0, shape[0]), slice(0, shape[1]))

    base = run_regional_inference(predictor, assemble_model_input(MANIFEST, inner, cycle), MANIFEST)
    classes = list(MANIFEST.served_classes)
    print(f"baseline {shape[0]}x{shape[1]} @ {cell_km:.1f} km, classes={classes}", flush=True)

    report: dict[str, Any] = {
        "cycle": f"{args.date}T{args.run_hour:02d}Z_f{args.step:03d}",
        "grid": {"shape": list(shape), "cell_km": cell_km, "halo_cells": args.halo, "blend_cells": args.blend},
        "band_km": args.band_km,
        "classes": classes,
        "edges": {},
    }
    for edge in [e.strip() for e in args.edges.split(",") if e.strip()]:
        src = EdgeHaloSource(inner, outer, edge, args.halo, args.blend, donor="ifs")
        ctrl = EdgeHaloSource(inner, outer, edge, args.halo, args.blend, donor="reflect")
        p_halo = run_regional_inference(predictor, assemble_model_input(MANIFEST, src, cycle), MANIFEST)[sl0]
        p_ctrl = run_regional_inference(predictor, assemble_model_input(MANIFEST, ctrl, cycle), MANIFEST)[sl0]
        d = np.abs(p_halo - base)
        c = np.abs(p_ctrl - base)
        dist = _distances_km(shape, edge, cell_km)
        prof = _profile(d.max(axis=-1), {f"dp_{k}": d[..., i] for i, k in enumerate(classes)}, dist, cell_km)
        ctrl_prof = _profile(c.max(axis=-1), {}, dist, cell_km)
        near = dist <= args.band_km
        report["edges"][edge] = {
            "reach_km": _reach_km(prof, ctrl_prof),
            "dp_max_mean_in_band": float(d.max(axis=-1)[near].mean()),
            "dp_max_mean_out_of_band": float(d.max(axis=-1)[~near].mean()),
            "control_dp_max_mean_in_band": float(c.max(axis=-1)[near].mean()),
            "control_dp_max_mean_out_of_band": float(c.max(axis=-1)[~near].mean()),
            "profile": prof,
            "control_profile": ctrl_prof,
        }
        print(
            f"{edge:6s} reach={report['edges'][edge]['reach_km']:7.0f} km  "
            f"in-band dp={report['edges'][edge]['dp_max_mean_in_band']:.4f} "
            f"(control {report['edges'][edge]['control_dp_max_mean_in_band']:.4f})  "
            f"out-of-band dp={report['edges'][edge]['dp_max_mean_out_of_band']:.4f} "
            f"(control {report['edges'][edge]['control_dp_max_mean_out_of_band']:.4f})",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
