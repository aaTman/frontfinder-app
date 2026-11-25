"""Measure whether HRRR front lines look like the IFS ones, and render them.

HRRR's probabilities live on a 3 km grid while `postprocess.splines` was tuned
on the IFS 0.25 deg (~28 km) grid, so HRRR's crest skeleton is fragmented: the
per-step turning angle of its output lines is several times the IFS one and
short arms come out as tight hooks. The candidate fix is to coarsen and blur
the HRRR probabilities to an IFS-like resolution before crest extraction, but
blurring also lowers the peaks, so the crest threshold has to move with it.

This script measures both sides of that trade-off on real probability
sidecars. For every sidecar it runs `splines.extract_fronts` once with the
production (IFS) settings -- the baseline -- and once per candidate threshold
with the coarsened settings, then prints the geometry of the resulting lines:

    n_lines        one line = all sections of one `FrontBoundary.lines[i]`
                   concatenated, with the junction vertex a type change
                   repeats dropped (see `line_points`); lines with fewer than
                   four output vertices are ignored
    total_km       summed haversine segment length of those lines
    len_med_km     median line length
    frac_lt_300km  fraction of lines shorter than 300 km
    turn_mean      mean |change of heading between consecutive output
                   segments|, degrees, pooled over all lines. Heading is
                   arctan2(dlat, dlon * cos(lat)) of the segment, unwrapped
                   along the line so a wrap through +/-180 deg is not a turn
    turn_p90       90th percentile of the same pooled turns
    hook150_med    per line, the largest accumulated turn across six
                   consecutive heading changes (150 km at the 25 km output
                   spacing); median over lines. Shorter lines contribute their
                   whole accumulated turn. This is the "tight hook" measure: a
                   line that doubles back on itself scores ~180 here
    hook150_p90    90th percentile of the same per-line hooks

plus `km_ratio = total_km / baseline total_km` per row. The last block picks
the threshold by the plan's rule: the value whose worst-case `|km_ratio - 1|`
across all dates is smallest, i.e. the candidate that keeps total front length
closest to the baseline's without being a mere echo of it.

`--render` draws the baseline and the selected threshold per date as PNGs
(white equirectangular canvas over CONUS, the lines in their front-type
colours, a dot at every output vertex, and the summed class probability as a
light grey underlay) for the human visual sign-off the metrics cannot replace.

Usage (the defaults are the three real HRRR 12Z f000 sidecars the plan was
calibrated on, plus a scratch out-dir; nothing here needs the network):

    uv run python scripts/hrrr_spline_parity.py
    uv run python scripts/hrrr_spline_parity.py --probs A.probs.npz B.probs.npz \
        --thresholds 0.16:0.24:0.01 --render --out-dir /tmp/parity
    uv run python scripts/hrrr_spline_parity.py --ifs-probs step.probs.npz

Longitudes are wrapped to [-180, 180) before measuring, which is what the
published GeoJSON uses; a global line that crosses the antimeridian therefore
has one heading computed across a 360 deg jump (the HRRR domain never does).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from typing import Any

import numpy as np
import xarray as xr

from frontfinder.postprocess import front_lines, splines

DEFAULT_PROBS_DIR = "/home/orca/scratch/hrrr-fronts/measure_out/hrrr-3km"
DEFAULT_PROBS = [
    os.path.join(DEFAULT_PROBS_DIR, f"{date}T12Z_f000.probs.npz") for date in ("2026-11-16", "2026-11-18", "2026-11-20")
]
DEFAULT_OUT_DIR = "/home/orca/scratch/hrrr-fronts/parity_out"
DEFAULT_THRESHOLDS = "0.16:0.24:0.01"

# Coarsening settings for the candidate rows, used only when this checkout
# predates `front_lines.native_config` (see `candidate_config`).
NATIVE_BLOCK = 3
NATIVE_BLUR_CELLS = 3.5

# Six consecutive output segments at the 25 km spacing the splines emit: 150 km.
HOOK_WINDOW = 6
SHORT_LINE_KM = 300.0

# Column order for the printed tables ("km_ratio" is added to the metrics).
METRICS: tuple[str, ...] = (
    "n_lines",
    "total_km",
    "len_med_km",
    "frac_lt_300km",
    "turn_mean",
    "turn_p90",
    "hook150_med",
    "hook150_p90",
)
DECIMALS = {"n_lines": 0, "total_km": 0, "len_med_km": 0, "frac_lt_300km": 2, "km_ratio": 3}

# Rendering: an equirectangular canvas over the HRRR domain.
CANVAS_PX = (1400, 800)
CANVAS_LON = (-125.0, -66.0)
CANVAS_LAT = (24.0, 50.0)
LINE_WIDTH_PX = 3
DOT_RADIUS_PX = 2.0
TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "CF": (21, 68, 214),  # cold: blue
    "WF": (214, 39, 40),  # warm: red
    "SF": (148, 64, 181),  # stationary: purple
    "OF": (76, 26, 122),  # occluded: dark purple
    "DL": (230, 126, 34),  # dryline: orange
    "TROF": (110, 110, 110),  # trough: grey
    "TT": (60, 140, 140),  # tropical trough: teal
    "INST": (180, 180, 60),
}
UNKNOWN_COLOR = (130, 130, 130)
UNDERLAY_FLOOR = 0.02
UNDERLAY_GREY = 145


def log(message: str) -> None:
    """Progress and per-run timings go to stderr, so stdout stays the report."""
    print(message, file=sys.stderr, flush=True)


def parse_thresholds(spec: str) -> list[float]:
    """`"0.16:0.24:0.01"` -> `[0.16, ..., 0.24]`, both ends included."""
    start, stop, step = (float(part) for part in spec.split(":"))
    if step <= 0 or stop < start:
        raise ValueError(f"thresholds {spec!r} must be start:stop:step with step > 0 and stop >= start")
    return [round(float(value), 4) for value in np.arange(start, stop + step / 2.0, step)]


def line_points(boundaries: list[splines.FrontBoundary]) -> list[tuple[np.ndarray, np.ndarray]]:
    """One `(lon, lat)` pair per front line, for measuring.

    A line is one entry of a `FrontBoundary.lines`, i.e. its typed sections
    concatenated in order. Longitudes are wrapped to [-180, 180), the published
    GeoJSON's convention, and lines with fewer than four vertices are left out:
    a turning statistic on a two-segment stub measures nothing.

    Neighbouring sections share their junction vertex. The repeat is dropped:
    left in, it is a zero-length segment per type change whose heading reads
    as arctan2(0, 0) = 0 (due east), which scores two phantom 90 deg turns for
    every type change and rewards a config merely for having fewer sections
    (on 2026-11-16 that artifact was ~74% of the baseline's mean turn).
    """
    points = []
    for boundary in boundaries:
        for line in boundary.lines:
            lon = np.concatenate([section.lon for section in line])
            lat = np.concatenate([section.lat for section in line])
            keep = np.concatenate([[True], (np.diff(lon) != 0) | (np.diff(lat) != 0)])
            lon, lat = lon[keep], lat[keep]
            if len(lon) < 4:
                continue
            points.append(((lon + 180.0) % 360.0 - 180.0, lat))
    return points


def turn_degrees(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Signed heading change (degrees) at each interior vertex of a line.

    The heading of a segment is `arctan2(dlat, dlon * cos(lat))` at its middle
    latitude -- degrees of latitude against longitude shrunk by the parallel it
    is measured along, which is the flat-earths-but-right-at-25-km reading of a
    direction. Unwrapped along the line first, so a heading passing through
    +/-180 deg is a straight run rather than a turn.
    """
    dlon = np.diff(lon) * np.cos(np.radians(0.5 * (lat[:-1] + lat[1:])))
    heading = np.unwrap(np.arctan2(np.diff(lat), dlon))
    return np.degrees(np.diff(heading))


def hook_degrees(turns: np.ndarray, window: int = HOOK_WINDOW) -> float:
    """Largest accumulated turn across `window` consecutive heading changes.

    Six at the 25 km output spacing is a 150 km run of line. A line that keeps
    turning one way for that stretch scores the sum of those six turns; one
    that wiggles scores little, because the signed turns cancel. Lines with
    fewer heading changes than the window score their whole accumulated turn.
    """
    if turns.size == 0:
        return 0.0
    span = min(window, turns.size)
    cumulative = np.concatenate([[0.0], np.cumsum(turns)])
    return float(np.max(np.abs(cumulative[span:] - cumulative[:-span])))


def line_metrics(boundaries: list[splines.FrontBoundary]) -> dict[str, float]:
    """The geometry measures of one `extract_fronts` result (see the module docstring).

    Every measured line has at least four vertices, so it has at least two
    heading changes: with lines to report, none of these statistics is ever
    taken over an empty set.
    """
    lines = line_points(boundaries)
    if not lines:
        return {name: float("nan") for name in METRICS} | {"n_lines": 0.0, "total_km": 0.0}
    lengths = np.array(
        [float(np.sum(splines.haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:]))) for lon, lat in lines]
    )
    turns = [turn_degrees(lon, lat) for lon, lat in lines]
    pooled = np.abs(np.concatenate(turns))
    hooks = np.array([hook_degrees(turn) for turn in turns])
    return {
        "n_lines": float(len(lines)),
        "total_km": float(lengths.sum()),
        "len_med_km": float(np.median(lengths)),
        "frac_lt_300km": float(np.mean(lengths < SHORT_LINE_KM)),
        "turn_mean": float(pooled.mean()),
        "turn_p90": float(np.percentile(pooled, 90)),
        "hook150_med": float(np.median(hooks)),
        "hook150_p90": float(np.percentile(hooks, 90)),
    }


def candidate_config(threshold: float) -> object | None:
    """The coarsened `SplineConfig` for one candidate threshold, or None if unbuildable.

    `front_lines.native_config` is the production builder and the one the
    scheduler uses. It is being added in parallel with this script, so fall
    back to assembling the same settings off `default_config()` when it is not
    there yet -- and when neither it nor the coarsening fields exist (a
    checkout predating the coarsening work), say so by returning None: the
    baseline rows are still a real measurement, the candidate rows are not.
    """
    native_config = getattr(front_lines, "native_config", None)
    if native_config is not None:
        return native_config(threshold)
    config = front_lines.default_config(threshold)
    if not (hasattr(config, "native_block") or hasattr(config, "native_blur_cells")):
        return None
    return dataclasses.replace(config, native_block=NATIVE_BLOCK, native_blur_cells=NATIVE_BLUR_CELLS)


def run_sidecar(path: str, thresholds: list[float]) -> dict[str, Any]:
    """Metrics for one sidecar: a baseline row plus one row per usable candidate threshold.

    Symbol sides are not computed (no 850 hPa fields, no previous step): they
    do not move any of these measures and cost most of the runtime.
    """
    probabilities, geom = front_lines._load_probabilities_sidecar(path)
    rows: dict[str, dict[str, Any]] = {}
    boundaries: dict[str, list[splines.FrontBoundary]] = {}
    configs: dict[str, tuple[object | None, float | None]] = {"baseline": (front_lines.default_config(), None)}
    for threshold in thresholds:
        config = candidate_config(threshold)
        if config is not None:
            configs[f"native {threshold:.2f}"] = (config, threshold)
    for label, (config, threshold) in configs.items():
        started = time.perf_counter()
        found = splines.extract_fronts(probabilities, config, geom=geom)
        seconds = time.perf_counter() - started
        rows[label] = {"label": label, "threshold": threshold, "seconds": seconds, **line_metrics(found)}
        boundaries[label] = found
        log(f"  {os.path.basename(path):32s} {label:12s} {seconds:5.2f}s")
    return {
        "label": os.path.basename(path)[: -len(front_lines.PROBS_SUFFIX)] or path,
        "path": path,
        "rows": rows,
        "boundaries": boundaries,
    }


def add_ratios(result: dict[str, Any]) -> None:
    """Add `km_ratio` to every row of one date, against that date's baseline length."""
    baseline = result["rows"]["baseline"]["total_km"]
    for row in result["rows"].values():
        row["km_ratio"] = row["total_km"] / baseline if baseline else float("nan")


def worst_case_deviation(ratios_by_threshold: dict[float, list[float]]) -> dict[float, float]:
    """Worst-case `|km_ratio - 1|` of each candidate, over all the dates it was measured on.

    Candidates with no usable ratio (a baseline of zero length, say) are left
    out rather than scored as a perfect match.
    """
    worst: dict[float, float] = {}
    for threshold, ratios in ratios_by_threshold.items():
        deviations = [abs(ratio - 1.0) for ratio in ratios if np.isfinite(ratio)]
        if deviations:
            worst[threshold] = max(deviations)
    return worst


def select_threshold(worst: dict[float, float]) -> float | None:
    """The candidate whose worst case is smallest, ties going to the lowest threshold."""
    if not worst:
        return None
    return min(worst.items(), key=lambda item: (item[1], item[0]))[0]


def threshold_selection(results: list[dict[str, Any]]) -> tuple[float | None, dict[float, float]]:
    """The plan's threshold rule over measured dates: minimise the worst-case `|km_ratio - 1|`.

    Returns the selected threshold (None when no candidate row was computed) and
    the worst-case deviation per candidate.
    """
    ratios: dict[float, list[float]] = {}
    for result in results:
        for row in result["rows"].values():
            if row["threshold"] is not None:
                ratios.setdefault(row["threshold"], []).append(row["km_ratio"])
    worst = worst_case_deviation(ratios)
    return select_threshold(worst), worst


def markdown_row(row: dict[str, Any]) -> str:
    cells = [row["label"]]
    for name in METRICS:
        value = row[name]
        digits = DECIMALS.get(name, 2)
        cells.append("-" if not np.isfinite(value) else f"{value:.{digits}f}")
    cells.append("-" if not np.isfinite(row["km_ratio"]) else f"{row['km_ratio']:.3f}")
    return "| " + " | ".join(cells) + " |"


def print_table(result: dict[str, Any], suffix: str = "") -> None:
    header = "| run | " + " | ".join(METRICS) + " | km_ratio |"
    rule = "|" + "---|" + "---:|" * (len(METRICS) + 1)
    print(f"\n### {result['label']}{suffix}\n")
    print(header)
    print(rule)
    for row in result["rows"].values():
        print(markdown_row(row))


def print_selection(results: list[dict[str, Any]], selected: float | None, worst: dict[float, float]) -> None:
    print("\n## threshold selection\n")
    if not worst:
        print("no candidate rows were computed, so no threshold can be selected.\n")
        return
    print("worst-case absolute deviation of km_ratio from 1, over all dates:\n")
    print("| threshold | worst km_ratio deviation |")
    print("|---:|---:|")
    for threshold, deviation in sorted(worst.items()):
        mark = " **(selected)**" if threshold == selected else ""
        print(f"| {threshold:.2f} | {deviation:.3f}{mark} |")
    print(f"\nselected threshold: {selected:.2f} (worst-case deviation {worst[selected]:.3f})")
    print(f"rule: arg-min over thresholds of the worst case over {len(results)} dates.\n")


def project(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canvas pixel coordinates of lon/lat points (x east, y south, 0 at the top left)."""
    x = (np.asarray(lon) - CANVAS_LON[0]) / (CANVAS_LON[1] - CANVAS_LON[0]) * (CANVAS_PX[0] - 1)
    y = (CANVAS_LAT[1] - np.asarray(lat)) / (CANVAS_LAT[1] - CANVAS_LAT[0]) * (CANVAS_PX[1] - 1)
    return x, y


def clip_segment(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float] | None:
    """The part of one segment inside the canvas (Liang-Barsky), or None if it misses."""
    dx, dy = x1 - x0, y1 - y0
    low, high = 0.0, 1.0
    for p, q in (
        (-dx, x0 - 0.0),
        (dx, (CANVAS_PX[0] - 1) - x0),
        (-dy, y0 - 0.0),
        (dy, (CANVAS_PX[1] - 1) - y0),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high:
            return None
    return x0 + low * dx, y0 + low * dy, x0 + high * dx, y0 + high * dy


def probability_underlay(probabilities: xr.Dataset, geom) -> np.ndarray:
    """The summed class probability resampled onto the canvas, for the grey underlay.

    Every pixel is placed back on the probabilities' own grid through the same
    sampler the crest extraction uses (`splines.sample`, which goes lon/lat ->
    fractional pixel through the grid adapter on a projected grid), so the
    shading cannot drift away from the grid the lines came from. Off-grid
    pixels read as no probability.
    """
    columns = [code for code in front_lines.CLASS_CODES.values() if code in probabilities]
    if not columns:
        raise ValueError("no front class variables to draw")
    grid = probabilities[columns].to_array(dim="class").sum("class")
    cols, rows = np.meshgrid(
        np.linspace(CANVAS_LON[0], CANVAS_LON[1], CANVAS_PX[0]),
        np.linspace(CANVAS_LAT[1], CANVAS_LAT[0], CANVAS_PX[1]),
    )
    total = splines.sample(grid, cols.ravel(), rows.ravel(), geom=geom).reshape(CANVAS_PX[::-1])
    total = np.nan_to_num(total, nan=0.0)
    grey = np.full(total.shape + (3,), 255, dtype=np.uint8)
    shade = total >= UNDERLAY_FLOOR
    value = np.round(255.0 - (255.0 - UNDERLAY_GREY) * np.clip(total, 0.0, 1.0)).astype(np.uint8)
    grey[shade] = np.repeat(value[shade][:, None], 3, axis=1)
    return grey


def render_png(path: str, boundaries: list[splines.FrontBoundary], probabilities: xr.Dataset, geom) -> None:
    """Write one canvas: the probability underlay, then the lines and their vertices.

    The underlay is the native probability field for every row, so a baseline
    and a coarsened candidate render over the same reference shading.
    """
    from PIL import Image, ImageDraw  # pillow rides in with scikit-image's dependencies

    canvas = Image.fromarray(probability_underlay(probabilities, geom), "RGB")
    draw = ImageDraw.Draw(canvas)
    for boundary in boundaries:
        for line in boundary.lines:
            for section in line:
                x, y = project(section.lon, section.lat)
                color = TYPE_COLORS.get(section.front_type, UNKNOWN_COLOR)
                for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:]):
                    clipped = clip_segment(float(x0), float(y0), float(x1), float(y1))
                    if clipped is not None:
                        draw.line(clipped, fill=color, width=LINE_WIDTH_PX)
            for section in line:  # vertices last, so they sit on top of the lines
                x, y = project(section.lon, section.lat)
                color = TYPE_COLORS.get(section.front_type, UNKNOWN_COLOR)
                inside = (x >= 0) & (x <= CANVAS_PX[0] - 1) & (y >= 0) & (y <= CANVAS_PX[1] - 1)
                for xi, yi in zip(x[inside], y[inside]):
                    draw.ellipse(
                        [xi - DOT_RADIUS_PX, yi - DOT_RADIUS_PX, xi + DOT_RADIUS_PX, yi + DOT_RADIUS_PX], fill=color
                    )
    canvas.save(path)
    log(f"  wrote {path} ({os.path.getsize(path)} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--probs",
        nargs="+",
        default=DEFAULT_PROBS,
        help="HRRR .probs.npz sidecars (default: the three 12Z f000 steps the plan was calibrated on)",
    )
    parser.add_argument("--ifs-probs", help="one IFS .probs.npz sidecar, measured as a reference row")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="candidate crest thresholds as start:stop:step")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="where --render writes its PNGs")
    parser.add_argument("--render", action="store_true", help="write the baseline and selected-threshold PNGs")
    args = parser.parse_args(argv)

    missing = [
        path for path in [*(args.probs or []), *([args.ifs_probs] if args.ifs_probs else [])] if not os.path.exists(path)
    ]
    if missing:
        for path in missing:
            print(f"no such sidecar: {path}", file=sys.stderr)
        return 1

    thresholds = parse_thresholds(args.thresholds)
    results = []
    for path in args.probs:
        log(f"extracting {path}")
        result = run_sidecar(path, thresholds)
        add_ratios(result)
        results.append(result)

    reference = None
    if args.ifs_probs:
        log(f"extracting the IFS reference {args.ifs_probs}")
        reference = run_sidecar(args.ifs_probs, [])  # the IFS path has nothing to coarsen, so baseline only
        add_ratios(reference)

    selected, worst = threshold_selection(results)
    if not worst:
        print(
            "\nnative settings not available in this checkout: `front_lines.native_config` and the "
            "`SplineConfig.native_block`/`native_blur_cells` fields are both absent, so no coarsened "
            "candidate could be built. The tables below are baseline-only.\n"
        )

    for result in results:
        print_table(result)
    if reference is not None:
        print_table(reference, " (IFS reference)")
    print_selection(results, selected, worst)

    if args.render:
        os.makedirs(args.out_dir, exist_ok=True)
        log("rendering")
        for result in [*results, *([reference] if reference is not None else [])]:
            probabilities, geom = front_lines._load_probabilities_sidecar(result["path"])
            labels = ["baseline"] + ([f"native {selected:.2f}"] if selected is not None and result is not reference else [])
            for label in labels:
                name = f"{result['label']}_{label.replace(' ', '_')}.png"
                render_png(os.path.join(args.out_dir, name), result["boundaries"][label], probabilities, geom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
