"""Regional pyramid and HRRR viewer publishing. No network or TensorFlow."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST as MANIFEST
from frontfinder.ingest.hrrr_grid import HRRR_CRS, native_grid
from frontfinder.zarrio.pyramid import RegionalFrontFields, build_regional_pyramid, write_front_pyramid

CASE = "2026-11-20T12Z_f000"
CONFIG = "hrrr_24km_earth"


@pytest.fixture(scope="module")
def publisher():
    path = Path(__file__).resolve().parents[2] / "scripts" / "publish_hrrr_pilot.py"
    spec = importlib.util.spec_from_file_location("publish_hrrr_pilot", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_probabilities(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    ramp = (rows + cols) / (shape[0] + shape[1])
    return {cls: (ramp / (i + 1)).astype(np.float32) for i, cls in enumerate(MANIFEST.served_classes)}


def write_run(
    root: Path,
    grid_name: str = "hrrr-conus-24km",
    status: str = "ok",
    config: str = CONFIG,
    coarsen: int = 8,
) -> Path:
    run_dir = root / "runs" / CASE / config
    run_dir.mkdir(parents=True)
    np.savez(run_dir / "probs.npz", **fake_probabilities(native_grid(coarsen).shape))
    meta = {
        "status": status,
        "grid": {"name": grid_name},
        "cycle": {"date": "2026-11-20", "run_hour": 12, "step": 0},
        "valid_time": "2026-11-20T12:00:00Z",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta))
    return run_dir


class TestRegionalPyramid:
    def test_round_trip_keeps_projected_axes_and_values(self, tmp_path):
        grid = native_grid(8)
        probabilities = fake_probabilities(grid.shape)
        fields = RegionalFrontFields(probabilities, grid, "2026-11-20T12:00:00", "2026-11-20T12:00:00", "test")
        store = tmp_path / "store.zarr"
        write_front_pyramid(build_regional_pyramid(fields, MANIFEST), str(store), pack_uint8=True)
        level = xr.open_zarr(store, group="0", consolidated=False)
        assert level.sizes == {"y": grid.shape[0], "x": grid.shape[1]}
        assert np.allclose(level["y"].values, grid.y)
        assert np.allclose(level["x"].values, grid.x)
        for cls in MANIFEST.served_classes:
            assert np.abs(level[cls].values - probabilities[cls]).max() <= 0.5 / 254 + 1e-6
        root = json.loads((store / "zarr.json").read_text())
        assert "+proj=lcc" in root["attributes"]["proj:code"]

    def test_rejects_wrong_shape_and_missing_class(self):
        grid = native_grid(8)
        good = fake_probabilities(grid.shape)
        wrong_shape = {**good, "cold": np.zeros((3, 3), np.float32)}
        with pytest.raises(ValueError, match="do not match grid shape"):
            build_regional_pyramid(RegionalFrontFields(wrong_shape, grid, "t", "t", "test"), MANIFEST)
        missing = {k: v for k, v in good.items() if k != "cold"}
        with pytest.raises(ValueError, match="served classes"):
            build_regional_pyramid(RegionalFrontFields(missing, grid, "t", "t", "test"), MANIFEST)


class TestPublish:
    def test_config_grid_parses_native_configs_only(self, publisher):
        assert publisher.config_grid("hrrr_24km_grid").shape == native_grid(8).shape
        assert publisher.config_grid("hrrr_3km_earth").shape == native_grid(1).shape
        assert publisher.config_grid("hrrr_3km_c1").shape == native_grid(1).shape
        assert publisher.config_model_key("hrrr_24km_earth") == "hrrr-24km"
        assert publisher.config_model_key("hrrr_3km_c1") == "hrrr-3km"
        assert publisher.DEFAULT_CONFIG == "hrrr_3km_c1"
        assert publisher.MODEL_KEY == "hrrr-3km"
        for bad in ("hrrr_latlon_0p25_earth", "ifs_0p25", "hrrr_5km_earth"):
            with pytest.raises(ValueError, match="native-grid"):
                publisher.config_grid(bad)

    def test_edge_bounds_are_half_a_cell_outside_centers(self, publisher):
        grid = native_grid(8)
        x_min, y_min, x_max, y_max = publisher.edge_bounds(grid)
        assert x_min == pytest.approx(grid.x.min() - 12_000)
        assert x_max == pytest.approx(grid.x.max() + 12_000)
        assert y_min == pytest.approx(grid.y.min() - 12_000)
        assert y_max == pytest.approx(grid.y.max() + 12_000)

    def test_domain_is_a_closed_band_around_the_interior(self, publisher):
        for coarsen, expected_margin in ((8, 32), (1, 256)):
            grid = native_grid(coarsen)
            feature = publisher.domain_geojson(grid)["features"][0]
            assert feature["properties"]["margin_cells"] == expected_margin
            outer, hole = (np.array(ring) for ring in feature["geometry"]["coordinates"])
            assert (outer[0] == outer[-1]).all() and (hole[0] == hole[-1]).all()
            assert outer[:, 0].min() < hole[:, 0].min() and hole[:, 0].max() < outer[:, 0].max()
            assert outer[:, 1].min() < hole[:, 1].min() and hole[:, 1].max() < outer[:, 1].max()

    def test_domain_refuses_a_margin_with_no_interior(self, publisher):
        with pytest.raises(ValueError, match="no interior"):
            publisher.domain_geojson(native_grid(8), margin=66)

    def test_publish_writes_store_domain_and_pointer(self, publisher, tmp_path):
        root, out = tmp_path / "root", tmp_path / "out"
        write_run(root)
        store = Path(publisher.publish(str(root), str(out), CASE, CONFIG))
        assert store.name == f"{CASE}.zarr" and (store / "zarr.json").exists()
        model_dir = out / publisher.config_model_key(CONFIG)
        assert not list(model_dir.glob("*.tmp"))
        latest = json.loads((model_dir / "latest.json").read_text())
        assert latest["cycle_time"] == "2026-11-20T12:00:00"
        assert latest["steps"] == [{"step_hours": 0, "valid_time": "2026-11-20T12:00:00", "store": store.name}]
        assert latest["source_label"] == "HRRR"
        assert latest["grid"]["proj4"] == HRRR_CRS
        assert latest["grid"]["dimensions"] == {"lat": "y", "lon": "x"}
        assert (model_dir / latest["domain"]).exists()

    def test_cli_publishes_default_case_and_config(self, publisher, tmp_path):
        root, out = tmp_path / "root", tmp_path / "out"
        write_run(root, grid_name="hrrr-conus-3km", config=publisher.DEFAULT_CONFIG, coarsen=1)
        assert publisher.main(["--root", str(root), "--out", str(out)]) == 0
        model_dir = out / publisher.MODEL_KEY
        assert (model_dir / f"{publisher.DEFAULT_CASE}.zarr" / "zarr.json").exists()
        domain = json.loads((model_dir / "domain.geojson").read_text())
        assert domain["features"][0]["properties"]["margin_cells"] == 256

    def test_publish_never_overwrites_a_store(self, publisher, tmp_path):
        root, out = tmp_path / "root", tmp_path / "out"
        write_run(root)
        publisher.publish(str(root), str(out), CASE, CONFIG)
        with pytest.raises(FileExistsError):
            publisher.publish(str(root), str(out), CASE, CONFIG)

    @pytest.mark.parametrize(("grid_name", "status"), [("hrrr-conus-3km", "ok"), ("hrrr-conus-24km", "failed")])
    def test_publish_refuses_a_bad_run(self, publisher, tmp_path, grid_name, status):
        root = tmp_path / "root"
        write_run(root, grid_name, status)
        with pytest.raises(ValueError):
            publisher.publish(str(root), str(tmp_path / "out"), CASE, CONFIG)
