"""Run the HRRR pilot matrix: every case through every source/resolution config.

Each run writes `<root>/runs/<case>/<config>/probs.npz` and then `meta.json`.
`meta.json` lands atomically and last, so its presence is what marks a run
finished; anything else in the directory is treated as an interrupted attempt
and redone. A config that fails records `failure.json` and the matrix moves on.

Usage:
    uv run --frozen python scripts/run_hrrr_pilot.py --root ./hrrr-pilot
    uv run --frozen python scripts/run_hrrr_pilot.py --root /tmp/pilot --dry-run

`--dry-run` swaps in a synthetic source and a fake predictor: no network, no
TensorFlow, and nothing written outside `--root`.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import glob
import json
import logging
import os
import re
import sys
import time
import traceback
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import IO, Any, Literal

import numpy as np
from scipy.ndimage import map_coordinates

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST, ModelManifest
from frontfinder.ingest.assembly import assemble_model_input
from frontfinder.ingest.base import FieldSource, ForecastCycle, GridSpec
from frontfinder.ingest.hrrr_grid import eval_grid_0p25, native_grid

logger = logging.getLogger("run_hrrr_pilot")

SCHEMA_VERSION = 1
DEFAULT_CASES = ("2026-11-16T12Z_f000", "2026-11-18T12Z_f000", "2026-11-20T12Z_f000")
NORMALIZATION_LAYER = "input_normalization"
PROBABILITY_TOLERANCE = 1e-5


@dataclass(frozen=True)
class PilotConfig:
    name: str
    source: Literal["hrrr", "ifs", "ifs_regional"]
    coarsen: int = 1
    wind_frame: Literal["earth", "grid"] = "earth"
    target: Literal["native", "latlon_0p25"] = "native"


def _build_configs() -> dict[str, PilotConfig]:
    configs = [
        PilotConfig(f"hrrr_{km}km_{frame}", "hrrr", coarsen=coarsen, wind_frame=frame)
        for km, coarsen in ((3, 1), (6, 2), (12, 4), (24, 8))
        for frame in ("earth", "grid")
    ]
    configs.append(PilotConfig("hrrr_latlon_0p25_earth", "hrrr", coarsen=8, target="latlon_0p25"))
    configs.append(PilotConfig("ifs_0p25", "ifs"))
    # IFS on the same grids and regional path as HRRR, so a comparison against
    # these isn't dominated by the boundary context HRRR's domain lacks.
    configs.append(PilotConfig("ifs_regional_0p25", "ifs_regional", coarsen=8, target="latlon_0p25"))
    configs.append(PilotConfig("ifs_regional_24km", "ifs_regional", coarsen=8))
    return {config.name: config for config in configs}


CONFIGS = _build_configs()


@dataclass
class Backend:
    """Everything a run needs from the outside world, so a dry run and tests
    can replace it without touching the network or TensorFlow."""

    make_source: Callable[[PilotConfig], FieldSource]
    load_predictor: Callable[[], Any]
    load_norm_ranges: Callable[[], tuple[np.ndarray, np.ndarray]]
    regional: Callable[..., np.ndarray]
    tiled: Callable[..., np.ndarray]


def case_id(cycle: ForecastCycle) -> str:
    return f"{cycle.date}T{cycle.run_hour:02d}Z_f{cycle.step:03d}"


def parse_case(text: str) -> ForecastCycle:
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})T(\d{2})Z_f(\d{3})", text)
    if match is None:
        raise argparse.ArgumentTypeError(f"case must look like 2026-11-20T12Z_f000, got {text!r}")
    return ForecastCycle(match[1], int(match[2]), int(match[3]))


def valid_time(cycle: ForecastCycle) -> str:
    start = datetime.strptime(cycle.date, "%Y-%m-%d").replace(hour=cycle.run_hour, tzinfo=timezone.utc)
    return (start + timedelta(hours=cycle.step)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _once(factory: Callable[[], Any]) -> Callable[[], Any]:
    """Keep the first successful result; a failed load is retried next call."""
    results: list[Any] = []

    def get() -> Any:
        if not results:
            results.append(factory())
        return results[0]

    return get


def _atomic_write(path: str, write: Callable[[IO[bytes]], None]) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    dir_fd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, allow_nan=False).encode()
    _atomic_write(path, lambda handle: handle.write(body))


def write_probs_atomic(path: str, probs: dict[str, np.ndarray]) -> None:
    _atomic_write(path, lambda handle: np.savez_compressed(handle, **probs))


def _clear_partial(run_dir: str) -> None:
    """Drop an interrupted attempt. `failure.json` stays until a run succeeds."""
    for path in [os.path.join(run_dir, "probs.npz"), *glob.glob(os.path.join(run_dir, "*.tmp-*"))]:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


def reset_peak_rss() -> bool:
    """Reset the kernel's RSS high-water mark. Without this, every run after
    the largest would report that run's peak."""
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
    except OSError:
        return False
    return True


def read_peak_rss_gb() -> float | None:
    try:
        with open("/proc/self/status") as handle:
            match = re.search(r"VmHWM:\s+(\d+)\s+kB", handle.read())
    except OSError:
        return None
    return int(match[1]) / 1024**2 if match else None


def load_norm_ranges(weights_path: str, n_channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (low, high) from the saved Rescaling layer, read from the
    `.keras` archive so it doesn't need TensorFlow."""
    with zipfile.ZipFile(weights_path) as archive:
        config = json.loads(archive.read("config.json"))
    layer = _find_layer(config, NORMALIZATION_LAYER)
    if layer is None or layer.get("class_name") != "Rescaling":
        raise ValueError(f"{weights_path}: no Rescaling layer named {NORMALIZATION_LAYER!r}")
    scale = _config_array(layer["config"]["scale"])
    offset = _config_array(layer["config"]["offset"])
    if scale.shape != (n_channels,) or offset.shape != (n_channels,):
        raise ValueError(
            f"{weights_path}: normalization shapes {scale.shape}/{offset.shape}, expected ({n_channels},)"
        )
    if not (np.isfinite(scale).all() and np.isfinite(offset).all() and (scale > 0).all()):
        raise ValueError(f"{weights_path}: normalization scale must be finite and positive")
    return -offset / scale, (1.0 - offset) / scale


def _find_layer(node: Any, name: str) -> dict[str, Any] | None:
    if isinstance(node, dict):
        config = node.get("config")
        if isinstance(config, dict) and config.get("name") == name and "class_name" in node:
            return node
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _find_layer(child, name)
        if found is not None:
            return found
    return None


def _config_array(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value["config"]["value"]
    return np.asarray(value, dtype=np.float64)


def channel_stats(
    input_grid: np.ndarray, names: list[str], low: np.ndarray, high: np.ndarray
) -> dict[str, dict[str, float | int | None]]:
    """Per-channel range check. `nan_count` counts every non-finite cell, and
    the outside-range fraction is taken over the finite ones."""
    stats: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(names):
        channel = input_grid[..., index]
        finite = channel[np.isfinite(channel)]
        outside = np.count_nonzero((finite < low[index]) | (finite > high[index]))
        stats[name] = {
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "nan_count": int(channel.size - finite.size),
            "frac_outside_norm_range": outside / finite.size if finite.size else None,
        }
    return stats


def _exact_lookup(axis: np.ndarray, wanted: np.ndarray, period: float | None = None) -> np.ndarray:
    delta = axis[:, np.newaxis] - wanted[np.newaxis, :]
    if period is not None:
        delta = (delta + period / 2.0) % period - period / 2.0
    indices = np.abs(delta).argmin(axis=0)
    if not (np.abs(delta[indices, np.arange(len(wanted))]) <= 1e-6).all():
        raise ValueError("target grid has nodes that are not on the source grid")
    return indices


def crop_indices(source: GridSpec, target: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of `target`'s nodes in a rectilinear `source`."""
    if not (source.is_rectilinear and target.is_rectilinear):
        raise ValueError("an exact node crop needs two rectilinear lat/lon grids")
    return _exact_lookup(source.lat, target.lat), _exact_lookup(source.lon, target.lon, period=360.0)


def resample_to_grid(input_grid: np.ndarray, source: GridSpec, target: GridSpec) -> np.ndarray:
    """(H, W, C) from a regular lat/lon `source` onto `target`: exact nodes for
    a rectilinear target, bilinear for a projected one. Raises rather than
    extrapolate."""
    if target.is_rectilinear:
        return input_grid[np.ix_(*crop_indices(source, target))]
    rows = (target.lat - source.lat[0]) / (source.lat[1] - source.lat[0])
    cols = ((target.lon - source.lon[0]) % 360.0) / (source.lon[1] - source.lon[0])
    if rows.min() < 0 or rows.max() > len(source.lat) - 1 or cols.max() > len(source.lon) - 1:
        raise ValueError("target grid has nodes outside the source grid")
    channels = [
        map_coordinates(input_grid[..., index], [rows, cols], order=1) for index in range(input_grid.shape[-1])
    ]
    return np.stack(channels, axis=-1).astype(input_grid.dtype)


def _validate_output(output: np.ndarray, shape: tuple[int, int], manifest: ModelManifest) -> None:
    expected = (*shape, len(manifest.served_classes))
    if output.shape != expected:
        raise ValueError(f"inference output shape {output.shape}, expected {expected}")
    if not np.isfinite(output).all():
        raise ValueError("inference output contains non-finite values")
    if output.min() < -PROBABILITY_TOLERANCE or output.max() > 1.0 + PROBABILITY_TOLERANCE:
        raise ValueError(f"inference output outside [0, 1]: {output.min()}..{output.max()}")


def _execute(
    run_dir: str, config: PilotConfig, cycle: ForecastCycle, backend: Backend, manifest: ModelManifest
) -> dict[str, Any]:
    # Model load stays outside the timers so only the first run doesn't pay for it.
    predictor = backend.load_predictor()
    low, high = backend.load_norm_ranges()
    peak_is_per_run = reset_peak_rss()

    started = time.perf_counter()
    source = backend.make_source(config)
    input_grid = assemble_model_input(manifest, source, cycle)
    stored_grid = source.grid
    if config.source == "ifs_regional":
        stored_grid = eval_grid_0p25() if config.target == "latlon_0p25" else native_grid(config.coarsen)
        input_grid = resample_to_grid(input_grid, source.grid, stored_grid)
    fetched = time.perf_counter()

    if config.source == "ifs":
        output = backend.tiled(
            predictor,
            input_grid,
            manifest,
            patch_size=256,
            overlap=96,
            batch_size=8,
            lon_deg=source.lon,
            lat_deg=source.lat,
        )
    else:
        output = backend.regional(predictor, input_grid, manifest)
    inferred = time.perf_counter()

    if config.source == "ifs":
        stored_grid = eval_grid_0p25()
        cells = np.ix_(*crop_indices(source.grid, stored_grid))
        output = output[cells]
        # Stats cover the stored domain so IFS and HRRR are checked over
        # comparable areas rather than IFS's whole globe.
        input_grid = input_grid[cells]

    _validate_output(output, stored_grid.shape, manifest)
    input_stats = channel_stats(input_grid, manifest.channel_names(), low, high)
    probs = {name: output[..., i].astype(np.float32) for i, name in enumerate(manifest.served_classes)}
    write_probs_atomic(os.path.join(run_dir, "probs.npz"), probs)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_id": source.source_id,
        "config": config.name,
        "model": manifest.name,
        "cycle": {"date": cycle.date, "run_hour": cycle.run_hour, "step": cycle.step},
        "valid_time": valid_time(cycle),
        "grid": stored_grid.to_metadata(),
        "served_classes": list(manifest.served_classes),
        "input_stats": input_stats,
        "runtime_s": {
            "fetch": fetched - started,
            "inference": inferred - fetched,
            "total": time.perf_counter() - started,
        },
        "peak_rss_gb": read_peak_rss_gb() if peak_is_per_run else None,
        "status": "ok",
    }


def run_config(
    root: str, cycle: ForecastCycle, config: PilotConfig, backend: Backend, manifest: ModelManifest
) -> Literal["ok", "skipped", "failed"]:
    run_dir = os.path.join(root, "runs", case_id(cycle), config.name)
    meta_path = os.path.join(run_dir, "meta.json")
    failure_path = os.path.join(run_dir, "failure.json")
    if os.path.exists(meta_path):
        return "skipped"

    os.makedirs(run_dir, exist_ok=True)
    _clear_partial(run_dir)
    try:
        meta = _execute(run_dir, config, cycle, backend, manifest)
        # Cleared before meta.json so that file stays the last thing written.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(failure_path)
        write_json_atomic(meta_path, meta)
    except Exception as exc:
        logger.exception("%s %s failed", case_id(cycle), config.name)
        _clear_partial(run_dir)
        try:
            write_json_atomic(
                failure_path,
                {
                    "config": config.name,
                    "case": case_id(cycle),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        except OSError:
            logger.exception("could not record failure for %s", config.name)
        return "failed"
    finally:
        gc.collect()
    return "ok"


def run_matrix(
    root: str,
    cycles: list[ForecastCycle],
    configs: list[PilotConfig],
    backend: Backend,
    manifest: ModelManifest = NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST,
) -> Counter[str]:
    outcomes: Counter[str] = Counter()
    for cycle in cycles:
        for config in configs:
            outcome = run_config(root, cycle, config, backend, manifest)
            outcomes[outcome] += 1
            logger.info("%s %s: %s", case_id(cycle), config.name, outcome)
    return outcomes


def real_backend(root: str, manifest: ModelManifest, ifs_source: str = "aws") -> Backend:
    weights_path = os.path.join(root, "weights", manifest.weights_filename)

    def make_source(config: PilotConfig) -> FieldSource:
        if config.source != "hrrr":
            from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource

            return EcmwfOpenDataSource(
                cache_dir=os.path.join(root, "cache", "ifs"),
                source=ifs_source,
                static_data_dir=os.path.join(root, "static"),
            )
        from frontfinder.ingest.hrrr import HRRRFieldSource

        return HRRRFieldSource(
            cache_dir=os.path.join(root, "cache", "hrrr"),
            coarsen=config.coarsen,
            wind_frame=config.wind_frame,
            target=config.target,
        )

    def load_predictor() -> Any:
        from frontfinder.inference.engine import KerasPredictor

        return KerasPredictor(weights_path)

    def regional(*args: Any, **kwargs: Any) -> np.ndarray:
        from frontfinder.inference.regional import run_regional_inference

        return run_regional_inference(*args, **kwargs)

    def tiled(*args: Any, **kwargs: Any) -> np.ndarray:
        from frontfinder.inference.engine import run_tiled_inference

        return run_tiled_inference(*args, **kwargs)

    return Backend(
        make_source=make_source,
        load_predictor=_once(load_predictor),
        load_norm_ranges=_once(lambda: load_norm_ranges(weights_path, manifest.n_channels)),
        regional=regional,
        tiled=tiled,
    )


class SyntheticSource(FieldSource):
    """Two smooth analytic bands placed by lat/lon, so every grid sees the
    same pattern. `shift_deg` moves them to give configs a known disagreement."""

    def __init__(self, source_id: str, grid: GridSpec, shift_deg: float = 0.0):
        self._source_id = source_id
        self._grid = grid
        self._shift_deg = shift_deg
        self._bands: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def grid(self) -> GridSpec:
        return self._grid

    def _front_bands(self) -> tuple[np.ndarray, np.ndarray]:
        if self._bands is None:
            lat, lon = self._grid.lat, self._grid.lon
            if self._grid.is_rectilinear:
                lat, lon = np.meshgrid(lat, lon, indexing="ij")
            lon = (lon + 180.0) % 360.0 - 180.0
            envelope = np.exp(-(((lon + 97.0) / 20.0) ** 4))
            first = 37.0 + self._shift_deg + 6.0 * np.sin(np.radians((lon + 100.0) * 6.0))
            second = 45.0 + self._shift_deg - 3.0 * np.cos(np.radians((lon + 90.0) * 8.0))
            self._bands = (
                (10.0 * envelope * np.exp(-(((lat - first) / 1.4) ** 2))).astype(np.float32),
                (10.0 * envelope * np.exp(-(((lat - second) / 1.0) ** 2))).astype(np.float32),
            )
        return self._bands

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: ForecastCycle) -> np.ndarray:
        first, second = self._front_bands()
        if variable == "u_component_of_wind":
            return first
        if variable == "v_component_of_wind":
            return second
        if variable == "temperature":
            return np.full(self._grid.shape, 290.0 * (level_hpa / 1000.0) ** 0.286, dtype=np.float32)
        if variable == "specific_humidity":
            return np.full(self._grid.shape, 0.004 * level_hpa / 1000.0, dtype=np.float32)
        raise ValueError(f"synthetic source has no pressure-level {variable!r}")

    def fetch_single_level(self, variable: str, cycle: ForecastCycle) -> np.ndarray:
        first, second = self._front_bands()
        if variable == "10m_u_component_of_wind":
            return first
        if variable == "10m_v_component_of_wind":
            return second
        raise ValueError(f"synthetic source has no single-level {variable!r}")

    def fetch_static(self, variable: str) -> np.ndarray:
        return np.zeros(self._grid.shape, dtype=np.float32)


class FakePredictor:
    """Turns the synthetic wind bands into softmax-like class probabilities."""

    def __init__(self, manifest: ModelManifest):
        names = manifest.channel_names()
        self._u = names.index("u_component_of_wind_850")
        self._v = names.index("v_component_of_wind_850")
        self._classes = list(manifest.all_classes)

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        output = np.full((*patches.shape[:3], len(self._classes)), 0.005, dtype=np.float32)
        output[..., self._classes.index("cold")] = 0.85 * np.clip(patches[..., self._u] / 10.0, 0.0, 1.0)
        output[..., self._classes.index("warm")] = 0.7 * np.clip(patches[..., self._v] / 10.0, 0.0, 1.0)
        background = self._classes.index("background")
        output[..., background] = 0.0
        output[..., background] = np.clip(1.0 - output.sum(axis=-1), 0.05, None)
        return output / output.sum(axis=-1, keepdims=True)


def _synthetic_shift(config: PilotConfig) -> float:
    if config.source != "hrrr":
        return 0.0
    if config.target == "latlon_0p25":
        return 0.2
    by_coarsen = {1: 0.0, 2: 0.15, 4: 0.35, 8: 0.6}
    return by_coarsen[config.coarsen] + (0.5 if config.wind_frame == "grid" else 0.0)


def _stub_regional(predictor: Any, input_grid: np.ndarray, manifest: ModelManifest) -> np.ndarray:
    output = predictor.predict_batch(input_grid[np.newaxis, ...])[0]
    return output[..., manifest.served_class_indices()].astype(np.float32)


def dry_run_backend(manifest: ModelManifest) -> Backend:
    def make_source(config: PilotConfig) -> FieldSource:
        shift = _synthetic_shift(config)
        if config.source != "hrrr":
            grid = GridSpec.regular_latlon(
                "synthetic-ifs-0p25", np.linspace(90.0, -90.0, 721), np.linspace(0.0, 359.75, 1440)
            )
            return SyntheticSource("synthetic_ifs", grid, shift)
        grid = native_grid(config.coarsen) if config.target == "native" else eval_grid_0p25()
        return SyntheticSource("synthetic_hrrr", grid, shift)

    try:
        from frontfinder.inference.regional import run_regional_inference as regional
    except ImportError:
        regional = _stub_regional
    from frontfinder.inference.engine import run_tiled_inference

    n_channels = manifest.n_channels
    return Backend(
        make_source=make_source,
        load_predictor=_once(lambda: FakePredictor(manifest)),
        load_norm_ranges=lambda: (np.full(n_channels, -100.0), np.full(n_channels, 500.0)),
        regional=regional,
        tiled=run_tiled_inference,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the HRRR pilot case/config matrix.")
    parser.add_argument("--root", required=True, help="pilot data root; runs land in <root>/runs")
    parser.add_argument("--cases", nargs="+", type=parse_case, default=[parse_case(c) for c in DEFAULT_CASES])
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--ifs-source", choices=["aws", "google", "ecmwf"], default="aws", help="IFS open-data mirror")
    parser.add_argument("--dry-run", action="store_true", help="synthetic source and fake predictor")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = os.path.abspath(args.root)
    os.makedirs(root, exist_ok=True)
    manifest = NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST
    backend = dry_run_backend(manifest) if args.dry_run else real_backend(root, manifest, args.ifs_source)
    outcomes = run_matrix(root, args.cases, [CONFIGS[name] for name in args.configs], backend, manifest)
    print(f"ok={outcomes['ok']} skipped={outcomes['skipped']} failed={outcomes['failed']}")
    return 1 if outcomes["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
