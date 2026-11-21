"""Compare finished HRRR pilot runs against the IFS run on a common grid.

Reads only `<root>/runs/<case>/<config>/{meta.json,probs.npz}`. Each run's
placement comes from its stored `GridSpec.to_metadata()`, so nothing here
depends on which model produced it. A directory without `meta.json` is an
unfinished run and is ignored.

Usage:
    uv run --frozen --with matplotlib --with cartopy python scripts/report_hrrr_pilot.py \
        --root ./hrrr-pilot --out ./hrrr-pilot/report
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import re
import socket
import sys
from typing import Any

import numpy as np
import pyproj

from frontfinder.ingest.base import GridSpec
from frontfinder.ingest.hrrr_grid import eval_grid_0p25
from frontfinder.postprocess.front_lines import FRONT_LINE_THRESHOLD

logger = logging.getLogger("report_hrrr_pilot")

SCHEMA_VERSION = 1
REFERENCE_CONFIG = "ifs_0p25"
# IFS run on the same grid and regional path, so lost boundary context affects
# both sides of the comparison.
SAME_DOMAIN_REFERENCES = {
    "hrrr_latlon_0p25_earth": "ifs_regional_0p25",
    "hrrr_24km_earth": "ifs_regional_24km",
    "hrrr_24km_grid": "ifs_regional_24km",
}
KM_PER_DEGREE = 111.32
EVAL_SPACING_DEG = 0.25
FSS_SCALES_KM = (100, 250)
MASK_EROSION_CELLS = 9
TARGET_SPACING_M = 24000.0
PLOT_GROUP = "frontal"

# Production's SplineConfig.groups for the served classes, by served class name.
SCORED_GROUPS: dict[str, tuple[str, ...]] = {
    "frontal": ("cold", "warm", "stationary", "occluded"),
    "dryline": ("dryline",),
    "trough": ("trough",),
    "tropical_trough": ("tropical_trough",),
}

PANEL_ROWS = (
    (REFERENCE_CONFIG, "hrrr_latlon_0p25_earth"),
    ("ifs_regional_0p25", "ifs_regional_24km"),
    ("hrrr_3km_earth", "hrrr_3km_grid"),
    ("hrrr_6km_earth", "hrrr_6km_grid"),
    ("hrrr_12km_earth", "hrrr_12km_grid"),
    ("hrrr_24km_earth", "hrrr_24km_grid"),
)
SEQUENTIAL_RAMP = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")

METRIC_COLUMNS = (
    "case",
    "config",
    "source_id",
    "grid",
    "group",
    "threshold",
    "fss_100km",
    "fss_250km",
    "front_area_km2",
    "ifs_area_km2",
    "area_ratio_to_ifs",
    "same_domain_reference",
    "fss_same_domain_100km",
    "fss_same_domain_250km",
    "area_ratio_same_domain",
    "worst_input_channel",
    "worst_frac_outside_norm",
    "runtime_fetch_s",
    "runtime_inference_s",
    "runtime_total_s",
    "peak_rss_gb",
)


def fss_radius(km: float) -> int:
    return round(km / (EVAL_SPACING_DEG * KM_PER_DEGREE))


def eroded_mask(shape: tuple[int, int], cells: int = MASK_EROSION_CELLS) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[cells : shape[0] - cells, cells : shape[1] - cells] = True
    return mask


def neighborhood_fractions(binary: np.ndarray, radius: int) -> np.ndarray:
    """Event fraction in a `(2r+1)`-square window around every cell of the
    full grid, by integral image so the counts are exact."""
    height, width = binary.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    integral[1:, 1:] = binary.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    top = np.clip(np.arange(height) - radius, 0, height)[:, np.newaxis]
    bottom = np.clip(np.arange(height) + radius + 1, 0, height)[:, np.newaxis]
    left = np.clip(np.arange(width) - radius, 0, width)
    right = np.clip(np.arange(width) + radius + 1, 0, width)
    counts = integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]
    return counts / ((bottom - top) * (right - left))


def fractions_score(forecast: np.ndarray, reference: np.ndarray, radius: int, mask: np.ndarray) -> float:
    """FSS with fractions from the full grid, scored inside `mask`. NaN when
    neither field has an event in reach of the mask."""
    forecast_fraction = neighborhood_fractions(forecast, radius)[mask]
    reference_fraction = neighborhood_fractions(reference, radius)[mask]
    worst_case = float(np.sum(forecast_fraction**2) + np.sum(reference_fraction**2))
    if worst_case == 0.0:
        return math.nan
    return 1.0 - float(np.sum((forecast_fraction - reference_fraction) ** 2)) / worst_case


def cell_areas_km2(lat_deg: np.ndarray, n_cols: int) -> np.ndarray:
    side = EVAL_SPACING_DEG * KM_PER_DEGREE
    return np.repeat((side**2 * np.cos(np.radians(lat_deg)))[:, np.newaxis], n_cols, axis=1)


def group_probabilities(probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        group: np.sum([probs[name] for name in members], axis=0, dtype=np.float64)
        for group, members in SCORED_GROUPS.items()
    }


def _axis(metadata: dict[str, Any]) -> np.ndarray:
    return metadata["start"] + metadata["step"] * np.arange(metadata["size"], dtype=np.float64)


def _is_on_grid(grid_meta: dict[str, Any], target: GridSpec) -> bool:
    if tuple(grid_meta["shape"]) != target.shape or grid_meta["crs"] != target.crs:
        return False
    return bool(
        np.allclose(_axis(grid_meta["x"]), target.x, atol=1e-6)
        and np.allclose(_axis(grid_meta["y"]), target.y, atol=1e-6)
    )


def block_average_to_target(
    field: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a north-up field to `TARGET_SPACING_M`, blocks anchored SW."""
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    if dx <= 0 or dy >= 0 or not np.isclose(dx, -dy):
        raise ValueError(f"expected equal spacing, x increasing and y decreasing; got dx={dx}, dy={dy}")
    block = round(TARGET_SPACING_M / dx)
    if block < 1 or not np.isclose(block * dx, TARGET_SPACING_M):
        raise ValueError(f"spacing {dx} m does not divide {TARGET_SPACING_M} m")
    if block == 1:
        return np.asarray(field, dtype=np.float64), x, y
    rows, cols = field.shape[0] // block, field.shape[1] // block
    first_row = field.shape[0] - rows * block
    kept = np.asarray(field[first_row:, : cols * block], dtype=np.float64)
    averaged = kept.reshape(rows, block, cols, block).mean(axis=(1, 3))
    return (
        averaged,
        x[: cols * block].reshape(cols, block).mean(axis=1),
        y[first_row:].reshape(rows, block).mean(axis=1),
    )


def bilinear(field: np.ndarray, x: np.ndarray, y: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Bilinear samples at projected points. Raises rather than extrapolate."""
    col = (px - x[0]) / (x[1] - x[0])
    row = (py - y[0]) / (y[1] - y[0])
    # Only absorbs floating-point noise for a node sitting exactly on the hull.
    eps = 1e-9
    if (col < -eps).any() or (col > len(x) - 1 + eps).any() or (row < -eps).any() or (row > len(y) - 1 + eps).any():
        raise ValueError("evaluation node outside the source grid's cell-center hull")
    col = np.clip(col, 0, len(x) - 1)
    row = np.clip(row, 0, len(y) - 1)
    col0 = np.minimum(np.floor(col).astype(int), len(x) - 2)
    row0 = np.minimum(np.floor(row).astype(int), len(y) - 2)
    col_weight = col - col0
    row_weight = row - row0
    return (
        field[row0, col0] * (1 - row_weight) * (1 - col_weight)
        + field[row0, col0 + 1] * (1 - row_weight) * col_weight
        + field[row0 + 1, col0] * row_weight * (1 - col_weight)
        + field[row0 + 1, col0 + 1] * row_weight * col_weight
    )


def regrid_to_eval(field: np.ndarray, grid_meta: dict[str, Any], target: GridSpec) -> np.ndarray:
    """Bring a stored field onto the rectilinear lat/lon `target`."""
    if tuple(field.shape) != tuple(grid_meta["shape"]):
        raise ValueError(f"field shape {field.shape} does not match grid {grid_meta['shape']}")
    if _is_on_grid(grid_meta, target):
        return np.asarray(field, dtype=np.float64)
    if not pyproj.CRS.from_user_input(grid_meta["crs"]).is_projected:
        raise ValueError(f"no regridding path from unprojected grid {grid_meta['name']!r}")
    coarse, x, y = block_average_to_target(field, _axis(grid_meta["x"]), _axis(grid_meta["y"]))
    lat, lon = np.meshgrid(target.lat, target.lon, indexing="ij")
    to_grid = pyproj.Transformer.from_crs("EPSG:4326", grid_meta["crs"], always_xy=True)
    px, py = to_grid.transform((lon + 180.0) % 360.0 - 180.0, lat)
    return bilinear(coarse, x, y, np.asarray(px), np.asarray(py))


def discover_runs(root: str) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    """Finished runs as {case: {config: meta}}, plus recorded failures."""
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    runs_dir = os.path.join(root, "runs")
    if not os.path.isdir(runs_dir):
        return runs, failures
    for case in sorted(os.listdir(runs_dir)):
        case_dir = os.path.join(runs_dir, case)
        if not os.path.isdir(case_dir):
            continue
        for config in sorted(os.listdir(case_dir)):
            meta_path = os.path.join(case_dir, config, "meta.json")
            failure_path = os.path.join(case_dir, config, "failure.json")
            if os.path.exists(meta_path):
                with open(meta_path) as handle:
                    meta = json.load(handle)
                if meta.get("schema_version") != SCHEMA_VERSION or meta.get("status") != "ok":
                    raise ValueError(f"{meta_path}: unsupported schema or status")
                runs.setdefault(case, {})[config] = meta
            elif os.path.exists(failure_path):
                with open(failure_path) as handle:
                    failure = json.load(handle)
                failures.append({"case": case, "config": config, "error_type": failure.get("error_type", "?")})
    return runs, failures


def load_probs(root: str, case: str, config: str, meta: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(os.path.join(root, "runs", case, config, "probs.npz")) as stored:
        return {name: stored[name] for name in meta["served_classes"]}


def worst_input(meta: dict[str, Any]) -> tuple[str | None, float | None]:
    scored = [
        (name, stats["frac_outside_norm_range"])
        for name, stats in meta["input_stats"].items()
        if stats["frac_outside_norm_range"] is not None
    ]
    return max(scored, key=lambda item: item[1]) if scored else (None, None)


def score_case(root: str, case: str, metas: dict[str, dict[str, Any]], target: GridSpec) -> list[dict[str, Any]]:
    """One metrics row per (config, group) for a case."""
    fields = {
        config: {
            group: regrid_to_eval(field, meta["grid"], target)
            for group, field in group_probabilities(load_probs(root, case, config, meta)).items()
        }
        for config, meta in metas.items()
    }
    mask = eroded_mask(target.shape)
    areas = cell_areas_km2(target.lat, target.shape[1])
    reference = fields.get(REFERENCE_CONFIG)

    rows: list[dict[str, Any]] = []
    for config, meta in metas.items():
        channel, frac = worst_input(meta)
        for group in SCORED_GROUPS:
            events = fields[config][group] >= FRONT_LINE_THRESHOLD
            area = float(areas[events & mask].sum())
            row: dict[str, Any] = {
                "case": case,
                "config": config,
                "source_id": meta["source_id"],
                "grid": meta["grid"]["name"],
                "group": group,
                "threshold": FRONT_LINE_THRESHOLD,
                "front_area_km2": area,
                "worst_input_channel": channel,
                "worst_frac_outside_norm": frac,
                "runtime_fetch_s": meta["runtime_s"]["fetch"],
                "runtime_inference_s": meta["runtime_s"]["inference"],
                "runtime_total_s": meta["runtime_s"]["total"],
                "peak_rss_gb": meta["peak_rss_gb"],
            }
            if reference is not None:
                reference_events = reference[group] >= FRONT_LINE_THRESHOLD
                ifs_area = float(areas[reference_events & mask].sum())
                row["ifs_area_km2"] = ifs_area
                row["area_ratio_to_ifs"] = area / ifs_area if ifs_area > 0 else None
                if config != REFERENCE_CONFIG:
                    for km in FSS_SCALES_KM:
                        row[f"fss_{km}km"] = fractions_score(events, reference_events, fss_radius(km), mask)
            same_domain = SAME_DOMAIN_REFERENCES.get(config)
            if same_domain in fields:
                same_domain_events = fields[same_domain][group] >= FRONT_LINE_THRESHOLD
                same_domain_area = float(areas[same_domain_events & mask].sum())
                row["same_domain_reference"] = same_domain
                row["area_ratio_same_domain"] = area / same_domain_area if same_domain_area > 0 else None
                for km in FSS_SCALES_KM:
                    row[f"fss_same_domain_{km}km"] = fractions_score(events, same_domain_events, fss_radius(km), mask)
            rows.append(row)
    return rows


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


def metrics_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(METRIC_COLUMNS)
    for row in rows:
        writer.writerow([_cell(row.get(column)) for column in METRIC_COLUMNS])
    return buffer.getvalue()


def _defined(rows: list[dict[str, Any]], column: str) -> list[float]:
    values = [row.get(column) for row in rows]
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_range(values: list[float]) -> str:
    return f"{min(values):.2f} to {max(values):.2f}" if values else "n/a"


def _fmt_mean_n(values: list[float]) -> str:
    """Mean with its case count, since FSS is undefined when neither field has a front."""
    return f"{_fmt(_mean(values))} (n={len(values)})" if values else "n/a"


def _config_order(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [config for pair in PANEL_ROWS for config in pair]
    present = {row["config"] for row in rows}
    return [c for c in preferred if c in present] + sorted(present - set(preferred))


def render_summary(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> str:
    configs = _config_order(rows)
    cases = sorted({row["case"] for row in rows})
    lines = ["# HRRR pilot summary", ""]
    if any(str(row["source_id"]).startswith("synthetic") for row in rows):
        lines += ["**SYNTHETIC DRY-RUN DATA. These numbers say nothing about any model.**", ""]
    lines += [
        "**Agreement with IFS is consistency, not skill.** IFS-driven output is a reference here, not "
        "observed truth, and no config is verified against observations.",
        "",
        f"Cases: {', '.join(cases)}. Threshold: {FRONT_LINE_THRESHOLD} on each group's summed probability. "
        f"FSS is a spatial agreement score against `{REFERENCE_CONFIG}` at 100 and 250 km, over the eval grid "
        f"eroded by {MASK_EROSION_CELLS} cells. Area ratio is total front area over total IFS front area, "
        "with the per-case range beside it because offsetting cases can cancel in the total.",
        "",
        "Same-domain columns score a run against IFS on the same grid and regional inference path "
        f"({', '.join(f'`{name}`' for name in sorted(set(SAME_DOMAIN_REFERENCES.values())))}), so lost boundary "
        f"context affects both sides. Scores against `{REFERENCE_CONFIG}` also include that context penalty.",
        "",
    ]
    for group, members in SCORED_GROUPS.items():
        lines += [
            f"## {group} ({' + '.join(members)})",
            "",
            "| Config | FSS 100 km | FSS 250 km | Area ratio | Per-case ratio | Same-domain FSS 100 km "
            "| Same-domain FSS 250 km | Same-domain ratio | Cases |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for config in configs:
            selected = [r for r in rows if r["config"] == config and r["group"] == group]
            paired = [r for r in selected if r.get("ifs_area_km2") is not None]
            ifs_total = sum(_defined(paired, "ifs_area_km2"))
            ratio = sum(_defined(paired, "front_area_km2")) / ifs_total if ifs_total > 0 else None
            fss = [_fmt(_mean(_defined(selected, f"fss_{km}km"))) for km in FSS_SCALES_KM]
            if config == REFERENCE_CONFIG:
                fss = ["reference", "reference"]
            same_domain_fss = [_fmt_mean_n(_defined(selected, f"fss_same_domain_{km}km")) for km in FSS_SCALES_KM]
            lines.append(
                f"| `{config}` | {fss[0]} | {fss[1]} | {_fmt(ratio)} "
                f"| {_fmt_range(_defined(selected, 'area_ratio_to_ifs'))} "
                f"| {same_domain_fss[0]} | {same_domain_fss[1]} "
                f"| {_fmt_range(_defined(selected, 'area_ratio_same_domain'))} | {len(selected)} |"
            )
        lines.append("")

    lines += [
        "## Input range, runtime, memory",
        "",
        "Input check is the worst channel's fraction of cells outside the model's normalization range, "
        "over all cases. Runtime is the per-case mean. Memory is the largest per-run process peak.",
        "",
        "| Config | Worst outside-range | Channel | Fetch s | Inference s | Total s | Peak RSS GB |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for config in configs:
        selected = [r for r in rows if r["config"] == config and r["group"] == PLOT_GROUP]
        with_frac = [r for r in selected if r.get("worst_frac_outside_norm") is not None]
        worst = max(with_frac, key=lambda r: r["worst_frac_outside_norm"]) if with_frac else None
        peaks = _defined(selected, "peak_rss_gb")
        lines.append(
            f"| `{config}` | {_fmt(worst['worst_frac_outside_norm'], 5) if worst else 'n/a'} "
            f"| {worst['worst_input_channel'] if worst else 'n/a'} "
            f"| {_fmt(_mean(_defined(selected, 'runtime_fetch_s')), 1)} "
            f"| {_fmt(_mean(_defined(selected, 'runtime_inference_s')), 1)} "
            f"| {_fmt(_mean(_defined(selected, 'runtime_total_s')), 1)} "
            f"| {_fmt(max(peaks) if peaks else None, 2)} |"
        )
    lines.append("")

    missing = [case for case in cases if not any(r["case"] == case and r["config"] == REFERENCE_CONFIG for r in rows)]
    if missing:
        lines += [f"**No `{REFERENCE_CONFIG}` run for:** {', '.join(missing)}. FSS and ratios are blank there.", ""]
    if failures:
        lines += ["## Failed runs", ""]
        lines += [f"- {f['case']} `{f['config']}`: {f['error_type']}" for f in failures]
        lines.append("")
    return "\n".join(lines)


def load_coastlines() -> list[Any] | None:
    """Natural Earth coastlines, or None if they can't be read or downloaded."""
    import cartopy.io.shapereader as shapereader

    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        path = shapereader.natural_earth(resolution="50m", category="physical", name="coastline")
        return list(shapereader.Reader(path).geometries())
    except Exception as exc:
        logger.warning("coastlines unavailable, plotting without them: %s", exc)
        return None
    finally:
        socket.setdefaulttimeout(previous_timeout)


def grid_lonlat(grid_meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """2-D signed lon and lat of a stored grid's cell centers."""
    x, y = np.meshgrid(_axis(grid_meta["x"]), _axis(grid_meta["y"]))
    crs = pyproj.CRS.from_user_input(grid_meta["crs"])
    if crs.is_projected:
        to_geo = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = to_geo.transform(x, y)
        return np.asarray(lon), np.asarray(lat)
    return (x + 180.0) % 360.0 - 180.0, y


def _panel_label(config: str) -> str:
    if config == REFERENCE_CONFIG:
        return "IFS 0.25° (reference)"
    if config == "hrrr_latlon_0p25_earth":
        return "HRRR 0.25°, earth winds"
    if config == "ifs_regional_0p25":
        return "IFS 0.25°, HRRR-sized box"
    if config == "ifs_regional_24km":
        return "IFS on HRRR 24 km grid"
    match = re.fullmatch(r"hrrr_(\d+)km_(earth|grid)", config)
    return f"HRRR {match[1]} km, {match[2]} winds" if match else config


def plot_case(
    root: str,
    case: str,
    metas: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    target: GridSpec,
    out: str,
    coastlines: list[Any] | None,
) -> str:
    """Stored-resolution `frontal` panels, one per config, on shared axes."""
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    ink, muted = "#1f1f1e", "#62615c"
    cmap = LinearSegmentedColormap.from_list("frontal", SEQUENTIAL_RAMP)
    cmap.set_bad((1, 1, 1, 0))
    norm = Normalize(0.0, 1.0)
    plate = ccrs.PlateCarree()
    lons = (target.lon + 180.0) % 360.0 - 180.0
    extent = [float(lons.min()), float(lons.max()), float(target.lat.min()), float(target.lat.max())]
    scores = {r["config"]: r for r in rows if r["case"] == case and r["group"] == PLOT_GROUP}

    fig, axes = plt.subplots(
        len(PANEL_ROWS), 2, figsize=(11.5, 18.5), subplot_kw={"projection": plate}, layout="constrained"
    )
    for row_axes, pair in zip(axes, PANEL_ROWS):
        for ax, config in zip(row_axes, pair):
            ax.set_extent(extent, crs=plate)
            ax.spines["geo"].set_edgecolor("#c9c8c2")
            ax.spines["geo"].set_linewidth(0.6)
            label = _panel_label(config)
            if config in metas:
                meta = metas[config]
                field = group_probabilities(load_probs(root, case, config, meta))[PLOT_GROUP]
                lon, lat = grid_lonlat(meta["grid"])
                ax.pcolormesh(
                    lon,
                    lat,
                    np.ma.masked_less(field, 0.05),
                    cmap=cmap,
                    norm=norm,
                    shading="nearest",
                    transform=plate,
                    rasterized=True,
                )
                if (field >= FRONT_LINE_THRESHOLD).any():
                    ax.contour(
                        lon, lat, field, levels=[FRONT_LINE_THRESHOLD], colors=ink, linewidths=0.6, transform=plate
                    )
                row = scores.get(config, {})
                if config != REFERENCE_CONFIG and row.get("fss_100km") is not None:
                    label += f"  ·  FSS {_fmt(row['fss_100km'], 2)} / {_fmt(row.get('fss_250km'), 2)}"
            else:
                ax.text(0.5, 0.5, "no finished run", ha="center", va="center", color=muted, transform=ax.transAxes)
            if coastlines is not None:
                ax.add_geometries(coastlines, plate, facecolor="none", edgecolor=muted, linewidth=0.45)
            ax.set_title(label, loc="left", fontsize=10.5, color=ink)

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal", fraction=0.022, pad=0.012, aspect=60
    )
    colorbar.set_label(
        f"Summed frontal probability (cold + warm + stationary + occluded). Line: {FRONT_LINE_THRESHOLD}",
        color=ink,
        fontsize=10,
    )
    colorbar.ax.axvline(FRONT_LINE_THRESHOLD, color=ink, linewidth=1.0)
    colorbar.ax.tick_params(labelsize=9, colors=muted)
    title = f"{case}: frontal group by config (FSS vs IFS at 100 / 250 km; consistency, not verification)"
    if any(str(meta["source_id"]).startswith("synthetic") for meta in metas.values()):
        title = f"SYNTHETIC DRY RUN. {title}"
    fig.suptitle(title, fontsize=12, color=ink)
    if coastlines is None:
        fig.supxlabel("Coastlines unavailable (Natural Earth download failed)", fontsize=9, color=muted)
    path = os.path.join(out, f"{case}_frontal.png")
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path


def build_report(root: str, out: str, plots: bool = True) -> list[dict[str, Any]]:
    """Write metrics and summary, then the figures, so a plotting problem
    never costs the numbers."""
    runs, failures = discover_runs(root)
    if not runs:
        raise FileNotFoundError(f"no finished runs under {os.path.join(root, 'runs')}")
    os.makedirs(out, exist_ok=True)
    target = eval_grid_0p25()
    rows: list[dict[str, Any]] = []
    for case, metas in runs.items():
        rows.extend(score_case(root, case, metas, target))
    _write_text(os.path.join(out, "metrics.csv"), metrics_csv(rows))
    _write_text(os.path.join(out, "summary.md"), render_summary(rows, failures))
    if plots:
        coastlines = load_coastlines()
        for case, metas in runs.items():
            logger.info("wrote %s", plot_case(root, case, metas, rows, target, out, coastlines))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score HRRR pilot runs against the IFS run.")
    parser.add_argument("--root", required=True, help="pilot data root containing runs/")
    parser.add_argument("--out", required=True, help="directory for metrics.csv, summary.md, and PNGs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        rows = build_report(os.path.abspath(args.root), os.path.abspath(args.out))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        if exc.name is None or exc.name.split(".")[0] not in ("matplotlib", "cartopy"):
            raise
        print(
            "error: metrics.csv and summary.md were written, but plotting needs "
            "`uv run --frozen --with matplotlib --with cartopy`",
            file=sys.stderr,
        )
        return 1
    print(f"wrote {len(rows)} metric rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
