"""HRRR pilot runner and report: no network, TensorFlow, or matplotlib."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pyproj
import pytest

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST as MANIFEST
from frontfinder.ingest.base import ForecastCycle, GridSpec
from frontfinder.ingest.hrrr_grid import HRRR_CRS, eval_grid_0p25, native_grid
from frontfinder.postprocess.front_lines import CLASS_CODES, default_config

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
CYCLE = ForecastCycle("2026-11-20", 12, 0)
CASE = "2026-11-20T12Z_f000"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"pilot_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("run_hrrr_pilot")
report = _load("report_hrrr_pilot")


def _counting_backend() -> tuple[object, list[str]]:
    backend = runner.dry_run_backend(MANIFEST)
    calls: list[str] = []
    make_source = backend.make_source

    def counted(config):
        calls.append(config.name)
        return make_source(config)

    backend.make_source = counted
    return backend, calls


def _run_dir(root: Path, config: str) -> Path:
    return root / "runs" / CASE / config


def test_fss_identical_fields_score_one() -> None:
    field = np.zeros((9, 9), dtype=bool)
    field[4, 2:7] = True
    assert report.fractions_score(field, field, 1, report.eroded_mask((9, 9), 1)) == pytest.approx(1.0)


def test_fss_disjoint_fields_score_zero() -> None:
    forecast = np.zeros((20, 20), dtype=bool)
    reference = np.zeros((20, 20), dtype=bool)
    forecast[4, 4] = True
    reference[15, 15] = True
    assert report.fractions_score(forecast, reference, 1, report.eroded_mask((20, 20), 1)) == pytest.approx(0.0)


def test_fss_partial_overlap_matches_hand_calculation() -> None:
    forecast = np.zeros((5, 5), dtype=bool)
    reference = np.zeros((5, 5), dtype=bool)
    forecast[2, 2] = True
    reference[2, 3] = True
    # In the 3x3 mask: forecast is 1/9 in all 9 cells, reference 1/9 in 6.
    # MSE sum = 3/81 and the reference sum = (9 + 6)/81, so FSS = 1 - 3/15.
    assert report.fractions_score(forecast, reference, 1, report.eroded_mask((5, 5), 1)) == pytest.approx(0.8)


def test_fss_uses_events_outside_the_mask_for_fractions() -> None:
    forecast = np.zeros((7, 7), dtype=bool)
    reference = np.zeros((7, 7), dtype=bool)
    # Both events sit outside the mask but inside windows centered in it.
    forecast[0, 3] = True
    reference[0, 3] = True
    mask = report.eroded_mask((7, 7), 1)
    assert report.neighborhood_fractions(forecast, 1)[1, 3] == pytest.approx(1 / 9)
    assert report.fractions_score(forecast, reference, 1, mask) == pytest.approx(1.0)


def test_fss_is_undefined_when_both_fields_are_empty() -> None:
    empty = np.zeros((9, 9), dtype=bool)
    assert np.isnan(report.fractions_score(empty, empty, 1, report.eroded_mask((9, 9), 1)))


def test_fss_radii_and_mask_keep_every_window_inside_the_grid() -> None:
    assert (report.fss_radius(100), report.fss_radius(250)) == (4, 9)
    shape = eval_grid_0p25().shape
    mask = report.eroded_mask(shape)
    rows, cols = np.where(mask)
    radius = report.fss_radius(250)
    assert rows.min() - radius == 0 and rows.max() + radius == shape[0] - 1
    assert cols.min() - radius == 0 and cols.max() + radius == shape[1] - 1
    assert mask.sum() == (shape[0] - 18) * (shape[1] - 18)


def test_cell_area_weights_by_cos_latitude() -> None:
    areas = report.cell_areas_km2(np.array([60.0, 0.0]), 3)
    assert areas.shape == (2, 3)
    assert areas[1, 0] == pytest.approx((0.25 * 111.32) ** 2)
    assert areas[0, 2] == pytest.approx(areas[1, 0] / 2)


def test_scored_groups_match_production_spline_groups() -> None:
    served = set(MANIFEST.served_classes)
    code_to_class = {code: name for name, code in CLASS_CODES.items()}
    production = [
        tuple(code_to_class[code] for code in group if code_to_class.get(code) in served)
        for group in default_config().groups
    ]
    assert [group for group in production if group] == list(report.SCORED_GROUPS.values())


def test_group_probabilities_sum_frontal_classes() -> None:
    probs = {name: np.full((2, 2), 0.1, dtype=np.float32) for name in MANIFEST.served_classes}
    groups = report.group_probabilities(probs)
    assert groups["frontal"] == pytest.approx(np.full((2, 2), 0.4))
    assert groups["dryline"] == pytest.approx(np.full((2, 2), 0.1))


@pytest.mark.parametrize("coarsen", [1, 2, 4, 8])
def test_regrid_reproduces_a_linear_field_in_projected_space(coarsen: int) -> None:
    grid = native_grid(coarsen)
    xx, yy = np.meshgrid(grid.x, grid.y)
    field = 2e-6 * xx - 3e-6 * yy + 0.5
    target = eval_grid_0p25()
    lat, lon = np.meshgrid(target.lat, target.lon, indexing="ij")
    px, py = pyproj.Transformer.from_crs("EPSG:4326", HRRR_CRS, always_xy=True).transform(lon - 360.0, lat)
    result = report.regrid_to_eval(field, grid.to_metadata(), target)
    assert result == pytest.approx(2e-6 * px - 3e-6 * py + 0.5, abs=1e-7)


def test_block_average_is_anchored_southwest() -> None:
    x = 3000.0 * np.arange(10, dtype=float)
    y = 3000.0 * np.arange(9, dtype=float)[::-1]
    field = np.arange(90, dtype=float).reshape(9, 10)
    coarse, coarse_x, coarse_y = report.block_average_to_target(field, x, y)
    assert coarse.shape == (1, 1)
    # 8x8 block: drop the north row and the two east columns.
    assert coarse[0, 0] == pytest.approx(field[1:, :8].mean())
    assert coarse_x[0] == pytest.approx(x[:8].mean())
    assert coarse_y[0] == pytest.approx(y[1:].mean())


def test_eval_grid_output_is_used_directly() -> None:
    target = eval_grid_0p25()
    field = np.random.default_rng(0).random(target.shape)
    assert np.array_equal(report.regrid_to_eval(field, target.to_metadata(), target), field)


def test_regrid_raises_instead_of_extrapolating() -> None:
    x = np.array([0.0, 24000.0, 48000.0])
    y = np.array([48000.0, 24000.0, 0.0])
    with pytest.raises(ValueError, match="hull"):
        report.bilinear(np.zeros((3, 3)), x, y, np.array([-1.0]), np.array([24000.0]))
    assert report.bilinear(np.ones((3, 3)), x, y, np.array([48000.0]), np.array([0.0]))[0] == pytest.approx(1.0)


def test_every_eval_node_is_inside_the_24km_hull() -> None:
    grid = native_grid(8)
    target = eval_grid_0p25()
    result = report.regrid_to_eval(np.ones(grid.shape), grid.to_metadata(), target)
    assert result.shape == target.shape


def test_ifs_crop_finds_exact_nodes_and_rejects_off_grid_targets() -> None:
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.linspace(0.0, 359.75, 1440)
    source = GridSpec.regular_latlon("ifs", lat, lon)
    target = eval_grid_0p25()
    rows, cols = runner.crop_indices(source, target)
    assert np.array_equal(lat[rows], target.lat)
    assert np.array_equal(lon[cols], target.lon)
    shifted = GridSpec.regular_latlon("off", target.lat + 0.1, target.lon)
    with pytest.raises(ValueError, match="not on the source grid"):
        runner.crop_indices(source, shifted)
    with pytest.raises(ValueError, match="rectilinear"):
        runner.crop_indices(native_grid(8), target)


def test_channel_stats_counts_outside_range_and_nans() -> None:
    grid = np.array([[[0.0], [2.0]], [[np.nan], [0.5]]], dtype=np.float32)
    stats = runner.channel_stats(grid, ["a"], np.array([0.0]), np.array([1.0]))["a"]
    assert stats["min"] == 0.0 and stats["max"] == 2.0
    assert stats["nan_count"] == 1
    assert stats["frac_outside_norm_range"] == pytest.approx(1 / 3)


def test_norm_ranges_come_from_the_rescaling_layer(tmp_path: Path) -> None:
    layer = {
        "class_name": "Rescaling",
        "config": {
            "name": "input_normalization",
            "scale": {"class_name": "__numpy__", "config": {"value": [0.5, 2.0]}},
            "offset": {"class_name": "__numpy__", "config": {"value": [-1.0, 1.0]}},
        },
    }
    path = tmp_path / "model.keras"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps({"config": {"layers": [layer]}}))
    low, high = runner.load_norm_ranges(str(path), 2)
    assert low == pytest.approx([2.0, -0.5])
    assert high == pytest.approx([4.0, 0.0])
    with pytest.raises(ValueError, match="normalization shapes"):
        runner.load_norm_ranges(str(path), 3)


def test_finished_run_is_skipped_on_resume(tmp_path: Path) -> None:
    backend, calls = _counting_backend()
    config = runner.CONFIGS["hrrr_latlon_0p25_earth"]
    assert runner.run_config(str(tmp_path), CYCLE, config, backend, MANIFEST) == "ok"
    assert runner.run_config(str(tmp_path), CYCLE, config, backend, MANIFEST) == "skipped"
    assert calls == ["hrrr_latlon_0p25_earth"]


def test_interrupted_run_without_meta_is_redone(tmp_path: Path) -> None:
    backend, calls = _counting_backend()
    config = runner.CONFIGS["hrrr_latlon_0p25_earth"]
    run_dir = _run_dir(tmp_path, config.name)
    run_dir.mkdir(parents=True)
    (run_dir / "probs.npz").write_bytes(b"partial")
    (run_dir / "probs.npz.tmp-1").write_bytes(b"partial")
    assert runner.run_config(str(tmp_path), CYCLE, config, backend, MANIFEST) == "ok"
    assert calls == [config.name]
    assert not list(run_dir.glob("*.tmp-*"))
    with np.load(run_dir / "probs.npz") as stored:
        assert set(stored.files) == set(MANIFEST.served_classes)


def test_failed_config_is_recorded_and_does_not_stop_the_matrix(tmp_path: Path) -> None:
    backend, _ = _counting_backend()
    make_source = backend.make_source

    def flaky(config):
        if config.name == "hrrr_24km_earth":
            raise RuntimeError("fetch exploded")
        return make_source(config)

    backend.make_source = flaky
    configs = [runner.CONFIGS["hrrr_24km_earth"], runner.CONFIGS["hrrr_latlon_0p25_earth"]]
    outcomes = runner.run_matrix(str(tmp_path), [CYCLE], configs, backend, MANIFEST)
    assert (outcomes["failed"], outcomes["ok"]) == (1, 1)

    failed_dir = _run_dir(tmp_path, "hrrr_24km_earth")
    assert not (failed_dir / "meta.json").exists()
    failure = json.loads((failed_dir / "failure.json").read_text())
    assert failure["config"] == "hrrr_24km_earth"
    assert failure["error_type"] == "RuntimeError"
    assert failure["message"] == "fetch exploded"
    assert "fetch exploded" in failure["traceback"]
    assert (_run_dir(tmp_path, "hrrr_latlon_0p25_earth") / "meta.json").exists()

    backend.make_source = make_source
    outcomes = runner.run_matrix(str(tmp_path), [CYCLE], configs, backend, MANIFEST)
    assert (outcomes["ok"], outcomes["skipped"], outcomes["failed"]) == (1, 1, 0)
    assert (failed_dir / "meta.json").exists()
    assert not (failed_dir / "failure.json").exists()


def test_meta_is_written_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend, _ = _counting_backend()
    config = runner.CONFIGS["hrrr_latlon_0p25_earth"]
    run_dir = _run_dir(tmp_path, config.name)
    run_dir.mkdir(parents=True)
    (run_dir / "failure.json").write_text("{}")
    seen: list[tuple[str, bool]] = []
    atomic_write = runner._atomic_write

    def recording(path, write):
        seen.append((Path(path).name, (run_dir / "failure.json").exists()))
        atomic_write(path, write)

    monkeypatch.setattr(runner, "_atomic_write", recording)
    assert runner.run_config(str(tmp_path), CYCLE, config, backend, MANIFEST) == "ok"
    assert [name for name, _ in seen] == ["probs.npz", "meta.json"]
    assert seen[-1] == ("meta.json", False)


def test_killed_run_leaves_no_meta(tmp_path: Path) -> None:
    backend, _ = _counting_backend()

    def killed():
        raise KeyboardInterrupt

    backend.load_predictor = killed
    config = runner.CONFIGS["hrrr_latlon_0p25_earth"]
    with pytest.raises(KeyboardInterrupt):
        runner.run_config(str(tmp_path), CYCLE, config, backend, MANIFEST)
    assert not (_run_dir(tmp_path, config.name) / "meta.json").exists()


def test_invalid_output_is_a_failure_not_a_finished_run(tmp_path: Path) -> None:
    backend, _ = _counting_backend()
    backend.regional = lambda predictor, grid, manifest: np.full(
        (*grid.shape[:2], len(manifest.served_classes)), np.nan, dtype=np.float32
    )
    config = runner.CONFIGS["hrrr_latlon_0p25_earth"]
    assert runner.run_config(str(tmp_path), CYCLE, config, backend, MANIFEST) == "failed"
    run_dir = _run_dir(tmp_path, config.name)
    assert not (run_dir / "meta.json").exists()
    assert not (run_dir / "probs.npz").exists()


def test_dry_run_end_to_end(tmp_path: Path) -> None:
    configs = ["hrrr_24km_earth", "hrrr_12km_grid", "hrrr_latlon_0p25_earth", "ifs_0p25"]
    assert runner.main(["--root", str(tmp_path), "--dry-run", "--cases", CASE, "--configs", *configs]) == 0

    for name in configs:
        run_dir = _run_dir(tmp_path, name)
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["schema_version"] == 1 and meta["status"] == "ok" and meta["config"] == name
        assert meta["valid_time"] == "2026-11-20T12:00:00Z"
        assert meta["served_classes"] == list(MANIFEST.served_classes)
        assert list(meta["input_stats"]) == MANIFEST.channel_names()
        assert set(meta["runtime_s"]) == {"fetch", "inference", "total"}
        with np.load(run_dir / "probs.npz") as stored:
            assert set(stored.files) == set(MANIFEST.served_classes)
            for key in stored.files:
                assert stored[key].dtype == np.float32
                assert stored[key].shape == tuple(meta["grid"]["shape"])

    expected_grid = eval_grid_0p25().to_metadata()
    assert json.loads((_run_dir(tmp_path, "ifs_0p25") / "meta.json").read_text())["grid"] == expected_grid
    native_meta = json.loads((_run_dir(tmp_path, "hrrr_24km_earth") / "meta.json").read_text())
    assert native_meta["grid"] == native_grid(8).to_metadata()

    out = tmp_path / "report"
    rows = report.build_report(str(tmp_path), str(out), plots=False)
    assert len(rows) == len(configs) * len(report.SCORED_GROUPS)
    frontal = {r["config"]: r for r in rows if r["group"] == "frontal"}
    assert "fss_100km" not in frontal["ifs_0p25"]
    assert frontal["ifs_0p25"]["area_ratio_to_ifs"] == pytest.approx(1.0)
    for name in configs[:3]:
        assert 0.0 <= frontal[name]["fss_100km"] <= 1.0
        assert frontal[name]["fss_100km"] <= frontal[name]["fss_250km"] + 1e-9

    header = (out / "metrics.csv").read_text().splitlines()[0].split(",")
    assert header == list(report.METRIC_COLUMNS)
    summary = (out / "summary.md").read_text()
    assert "Agreement with IFS is consistency, not skill." in summary
    assert summary.lower().count("skill") == 1
    assert "accuracy" not in summary.lower()
    assert "SYNTHETIC" in summary


def test_resample_reproduces_a_linear_field_on_the_projected_grid() -> None:
    source = GridSpec.regular_latlon("global", np.linspace(90.0, -90.0, 721), np.linspace(0.0, 359.75, 1440))
    lat, lon = np.meshgrid(source.lat, source.lon, indexing="ij")
    field = np.stack([2.0 * lat, 3.0 * lon], axis=-1)
    target = native_grid(8)

    result = runner.resample_to_grid(field, source, target)

    assert result.shape == (*target.shape, 2)
    np.testing.assert_allclose(result[..., 0], 2.0 * target.lat, atol=1e-8)
    np.testing.assert_allclose(result[..., 1], 3.0 * (target.lon % 360.0), atol=1e-8)


def test_resample_crops_exact_nodes_for_a_lat_lon_target() -> None:
    source = GridSpec.regular_latlon("global", np.linspace(90.0, -90.0, 721), np.linspace(0.0, 359.75, 1440))
    lat, lon = np.meshgrid(source.lat, source.lon, indexing="ij")
    target = eval_grid_0p25()

    result = runner.resample_to_grid(np.stack([lat, lon], axis=-1), source, target)

    np.testing.assert_array_equal(result[:, 0, 0], target.lat)
    np.testing.assert_array_equal(result[0, :, 1], target.lon)


def test_resample_refuses_to_extrapolate_past_the_source_edge() -> None:
    source = GridSpec.regular_latlon("global", np.linspace(90.0, -90.0, 721), np.linspace(0.0, 359.75, 1440))
    lat = np.full((2, 2), 40.0)
    lon = np.full((2, 2), 359.9)
    target = GridSpec("seam", "EPSG:3857", np.arange(2.0), np.arange(2.0), lat, lon)

    with pytest.raises(ValueError, match="outside the source grid"):
        runner.resample_to_grid(np.zeros((721, 1440, 1)), source, target)


def test_same_domain_references_are_scored_and_summarized(tmp_path: Path) -> None:
    configs = ["hrrr_24km_earth", "hrrr_latlon_0p25_earth", "ifs_0p25", "ifs_regional_24km", "ifs_regional_0p25"]
    assert runner.main(["--root", str(tmp_path), "--dry-run", "--cases", CASE, "--configs", *configs]) == 0
    expected = {
        "ifs_regional_24km": native_grid(8).to_metadata(),
        "ifs_regional_0p25": eval_grid_0p25().to_metadata(),
    }
    for name, grid in expected.items():
        assert json.loads((_run_dir(tmp_path, name) / "meta.json").read_text())["grid"] == grid

    out = tmp_path / "report"
    rows = report.build_report(str(tmp_path), str(out), plots=False)

    frontal = {r["config"]: r for r in rows if r["group"] == "frontal"}
    for name, reference in report.SAME_DOMAIN_REFERENCES.items():
        if name not in frontal:
            continue
        assert frontal[name]["same_domain_reference"] == reference
        assert 0.0 <= frontal[name]["fss_same_domain_100km"] <= 1.0
        assert frontal[name]["area_ratio_same_domain"] > 0.0
    assert "same_domain_reference" not in frontal["ifs_0p25"]
    summary = (out / "summary.md").read_text()
    assert "Per-case ratio" in summary and "Same-domain FSS 100 km" in summary


def test_dry_run_needs_no_network_tensorflow_or_matplotlib(tmp_path: Path) -> None:
    program = f"""
import importlib.util, socket, sys
def blocked(*args, **kwargs):
    raise RuntimeError("network used in dry run")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.getaddrinfo = blocked
spec = importlib.util.spec_from_file_location("pilot", {str(SCRIPTS / "run_hrrr_pilot.py")!r})
module = importlib.util.module_from_spec(spec)
sys.modules["pilot"] = module
spec.loader.exec_module(module)
code = module.main(["--root", {str(tmp_path)!r}, "--dry-run", "--cases", {CASE!r},
                    "--configs", "hrrr_latlon_0p25_earth"])
assert code == 0
for name in ("tensorflow", "keras", "matplotlib", "cartopy"):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["runs"]


def test_c1_refiner_config_runs_in_dry_run_and_reuses_cached_24km(tmp_path: Path) -> None:
    standalone_root = tmp_path / "standalone"
    assert (
        runner.main(["--root", str(standalone_root), "--dry-run", "--cases", CASE, "--configs", "hrrr_3km_c1"]) == 0
    )
    meta = json.loads((_run_dir(standalone_root, "hrrr_3km_c1") / "meta.json").read_text())
    assert meta["status"] == "ok" and meta["grid"] == native_grid(1).to_metadata()
    with np.load(_run_dir(standalone_root, "hrrr_3km_c1") / "probs.npz") as stored:
        assert set(stored.files) == set(MANIFEST.served_classes)
        for cls in stored.files:
            assert stored[cls].shape == native_grid(1).shape
            assert stored[cls].dtype == np.float32

    cached_root = tmp_path / "cached"
    assert (
        runner.main(["--root", str(cached_root), "--dry-run", "--cases", CASE, "--configs", "hrrr_24km_earth"]) == 0
    )
    backend, calls = _counting_backend()
    assert runner.run_config(str(cached_root), CYCLE, runner.CONFIGS["hrrr_3km_c1"], backend, MANIFEST) == "ok"
    assert calls == ["hrrr_3km_c1"]
    with (
        np.load(_run_dir(standalone_root, "hrrr_3km_c1") / "probs.npz") as standalone_probs,
        np.load(_run_dir(cached_root, "hrrr_3km_c1") / "probs.npz") as cached_probs,
    ):
        for cls in MANIFEST.served_classes:
            np.testing.assert_allclose(cached_probs[cls], standalone_probs[cls], atol=1e-6)
