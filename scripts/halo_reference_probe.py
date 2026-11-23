"""Which HRRR boundary treatment agrees best with the same model fed IFS?

`scripts/halo_probe.py` found that replacing one edge's reflect-padded context
with real IFS fields moves v2.0.1's probabilities by ~0.03 (max over classes)
inband -- and that deepening the MIRROR by the same amount moves them by
practically as much, on a profile that reaches the far side of the domain
because the network's ~180-pixel reach (~4,300 km) exceeds the domain's
3,200 km short axis. So "real vs mirrored context" cannot be separated from
"thin pad vs deep halo" by comparing both against production.

This script adds the third leg: run the same v2.0.1 on IFS inputs over a box
with a generous margin (lat 10..65, lon -155..-35 -- everything we score sits
well inside it, so its own edges barely register), sample it onto HRRR's 24 km
Lambert conformal cells, and score each HRRR treatment against that reference:

  base   production: 2-row reflect pad at the domain edge
  mirror deep 192-cell halo made of HRRR's own mirrored strip
  ifs    deep 192-cell halo of real IFS fields blended across the seam

Agreement is consistency between two runs of one network, not skill: it says
which treatment behaves like the version of the model that had real context all
the way around. Where `base` agrees as well as `ifs`, the published
low-confidence band is over-wide; if `ifs` agrees clearly better out to depth D,
the band is honest at D and an IFS halo is what buys trust, not a smaller flag.

Usage:
  uv run --frozen python scripts/halo_reference_probe.py \
      --model-dir /home/orca/hrrr-pilot/weights \
      --hrrr-cache /home/orca/hrrr-pilot/cache/hrrr \
      --ifs-cache /home/orca/hrrr-pilot/cache/ifs \
      --date 2026-11-16 --run-hour 12 --step 0 \
      --out /home/orca/hrrr-pilot/halo-probe/reference-2026-11-16T12Z_f000.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from halo_probe import EDGES, EdgeHaloSource, _distances_km  # noqa: E402

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST as MANIFEST  # noqa: E402
from frontfinder.ingest.assembly import assemble_model_input  # noqa: E402
from frontfinder.ingest.base import FieldSource, ForecastCycle, GridSpec  # noqa: E402
from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource  # noqa: E402
from frontfinder.ingest.hrrr import HRRRFieldSource  # noqa: E402
from frontfinder.inference.engine import KerasPredictor  # noqa: E402
from frontfinder.inference.regional import run_regional_inference  # noqa: E402


class CroppedLatLonSource(FieldSource):
    """A latitude/longitude window onto a global rectilinear source."""

    source_id = "latlon-crop"

    def __init__(self, inner: FieldSource, lat_range: tuple[float, float], lon_range: tuple[float, float]):
        lat = np.asarray(inner.grid.lat, dtype=np.float64)
        lon = np.asarray(inner.grid.lon, dtype=np.float64)
        rows = np.flatnonzero((lat <= lat_range[1]) & (lat >= lat_range[0]))
        # IFS open data indexes longitude 0..360; the window is named in -180..180.
        span = (((np.array(lon_range, dtype=np.float64) - float(lon[0])) % 360.0) + float(lon[0]))
        if span[1] < span[0]:
            span = span[::-1]
        cols = np.flatnonzero((lon >= span[0]) & (lon <= span[1]))
        if len(rows) < 8 or len(cols) < 8:
            raise ValueError(f"crop {lat_range}, {lon_range} is too small on the {inner.grid.name} grid")
        self._rows = slice(int(rows[0]), int(rows[-1]) + 1)
        self._cols = slice(int(cols[0]), int(cols[-1]) + 1)
        self._inner = inner
        # Re-label the window in -180..180 so sampling against pyproj output lines up.
        lon_w = ((lon[self._cols] + 180.0) % 360.0) - 180.0
        self._grid = GridSpec.regular_latlon(f"{inner.grid.name}-crop", lat[self._rows], lon_w)

    @property
    def grid(self) -> GridSpec:
        return self._grid

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: ForecastCycle) -> np.ndarray:
        return self._inner.fetch_pressure_level(variable, level_hpa, cycle)[self._rows, self._cols]

    def fetch_single_level(self, variable: str, cycle: ForecastCycle) -> np.ndarray:
        return self._inner.fetch_single_level(variable, cycle)[self._rows, self._cols]

    def fetch_static(self, variable: str) -> np.ndarray:
        return self._inner.fetch_static(variable)[self._rows, self._cols]


def _frontness(probs: np.ndarray) -> np.ndarray:
    """Probability that some feature class fires -- what the map actually shows."""
    return probs.max(axis=-1)


def _sample_to_hrrr(probs: np.ndarray, src: GridSpec, tgt: GridSpec) -> np.ndarray:
    """Sample a rectilinear field's class grids at every target cell centre."""
    lat = np.asarray(src.lat, dtype=np.float64)
    lon = np.asarray(src.lon, dtype=np.float64)
    row = (np.asarray(tgt.lat, dtype=np.float64) - lat[0]) / float(lat[1] - lat[0])
    col = (np.asarray(tgt.lon, dtype=np.float64) - lon[0]) / float(lon[1] - lon[0])
    if src.is_global:
        col = col % len(lon)
    out = np.empty(tgt.shape + (probs.shape[-1],), dtype=np.float32)
    for k in range(probs.shape[-1]):
        out[..., k] = map_coordinates(probs[..., k], [row.ravel(), col.ravel()], order=1, mode="nearest").reshape(
            tgt.shape
        )
    return out


def _profile(score: np.ndarray, dist: np.ndarray, cell_km: float) -> list[dict[str, float]]:
    nbins = int(dist.max() // cell_km) + 1
    idx = np.clip((dist / cell_km).astype(int), 0, nbins - 1)
    return [
        {
            "cell": b,
            "km": round((b + 0.5) * cell_km, 1),
            "mad": float(score[idx == b].mean()),
            "p95": float(np.percentile(score[idx == b], 95)),
        }
        for b in range(nbins)
        if (idx == b).any()
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--hrrr-cache", required=True)
    ap.add_argument("--ifs-cache", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--run-hour", type=int, default=12)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--ifs-source", default="google")
    ap.add_argument("--static-dir", default="/home/orca/hrrr-pilot/static", help="dir holding the cycle-invariant IFS fields")
    ap.add_argument("--halo", type=int, default=192)
    ap.add_argument("--blend", type=int, default=12)
    ap.add_argument("--edges", default=",".join(EDGES))
    ap.add_argument("--band-km", type=float, default=768.0)
    ap.add_argument(
        "--reference-box",
        default="10.0,65.0,-155.0,-35.0",
        help="lat_min,lat_max,lon_min,lon_max for the IFS-context reference run",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cycle = ForecastCycle(date=args.date, run_hour=args.run_hour, step=args.step)
    inner = HRRRFieldSource(cache_dir=args.hrrr_cache, coarsen=8, wind_frame="earth")
    ifs = EcmwfOpenDataSource(
        cache_dir=args.ifs_cache, source=args.ifs_source, static_data_dir=args.static_dir
    )
    predictor = KerasPredictor(os.path.join(args.model_dir, MANIFEST.weights_filename))
    cell_km = float(abs(inner.grid.x[1] - inner.grid.x[0])) / 1000.0
    box = [float(x) for x in args.reference_box.split(",")]
    if len(box) != 4:
        raise ValueError("--reference-box wants lat_min,lat_max,lon_min,lon_max")

    print("reference run: v2.0.1 on IFS, generous margin", flush=True)
    crop = CroppedLatLonSource(ifs, (box[0], box[1]), (box[2], box[3]))
    p_ref_crop = run_regional_inference(predictor, assemble_model_input(MANIFEST, crop, cycle), MANIFEST)
    p_ref = _sample_to_hrrr(p_ref_crop, crop.grid, inner.grid)
    ref = _frontness(p_ref)
    print(f"  crop {crop.grid.shape}, reference frontness mean {ref.mean():.3f}", flush=True)

    print("production run: 2-row reflect pad", flush=True)
    base = _frontness(run_regional_inference(predictor, assemble_model_input(MANIFEST, inner, cycle), MANIFEST))

    report: dict[str, object] = {
        "cycle": f"{args.date}T{args.run_hour:02d}Z_f{args.step:03d}",
        "cell_km": cell_km,
        "band_km": args.band_km,
        "halo_cells": args.halo,
        "reference_box": box,
        "reference_shape": list(crop.grid.shape),
        "edges": {},
    }
    # Whole-domain summary first: how far apart are these treatments at all?
    print(f"  |base - reference| mean {np.abs(base - ref).mean():.4f}", flush=True)

    for edge in [e.strip() for e in args.edges.split(",") if e.strip()]:
        dist = _distances_km(inner.grid.shape, edge, cell_km)
        rows: dict[str, list[dict[str, float]]] = {"base": _profile(np.abs(base - ref), dist, cell_km)}
        means: dict[str, float] = {"base_in": float(np.abs(base - ref)[dist <= args.band_km].mean()),
                                   "base_out": float(np.abs(base - ref)[dist > args.band_km].mean())}
        for donor, label in (("ifs", "ifs"), ("reflect", "mirror")):
            src = EdgeHaloSource(inner, ifs, edge, args.halo, args.blend, donor=donor)
            p = _frontness(run_regional_inference(predictor, assemble_model_input(MANIFEST, src, cycle), MANIFEST)[
                slice(0, inner.grid.shape[0]), slice(0, inner.grid.shape[1])
            ])
            score = np.abs(p - ref)
            rows[label] = _profile(score, dist, cell_km)
            means[f"{label}_in"] = float(score[dist <= args.band_km].mean())
            means[f"{label}_out"] = float(score[dist > args.band_km].mean())
            means[f"{label}_vs_base"] = float(np.abs(p - base).mean())
        report["edges"][edge] = {"means": means, "profiles": rows}
        print(
            f"{edge:6s} in-band MAD  base {means['base_in']:.4f}  mirror {means['mirror_in']:.4f}  "
            f"ifs {means['ifs_in']:.4f}   out-of-band  base {means['base_out']:.4f}  "
            f"mirror {means['mirror_out']:.4f}  ifs {means['ifs_out']:.4f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
