import json

import numpy as np
import pytest
import xarray as xr

from frontfinder.config.manifests import ALL_CLASSES, BEST_LOSS_MANIFEST, MODEL_1702_MANIFEST
from frontfinder.ingest.ecmwf_ifs import FakeIFSFieldSource, IFSCycle
from frontfinder.scheduler.run_cycle import ModelRunConfig, run_cycle, run_one_model


class FakePredictor:
    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        n, h, w, _ = patches.shape
        out = np.random.default_rng(1).random((n, h, w, len(ALL_CLASSES))).astype(np.float32)
        return out / out.sum(axis=-1, keepdims=True)  # looks like a softmax output


@pytest.fixture
def tiny_source():
    lat = np.linspace(25.0, 56.75, 64)
    lon = np.linspace(228.0, 299.75, 64)
    return FakeIFSFieldSource(lat, lon, seed=7)


def test_run_one_model_writes_a_readable_zarr_pyramid(tiny_source, tmp_path):
    cycle = IFSCycle(date="2026-08-19", run_hour=0)
    run_config = ModelRunConfig(
        manifest=MODEL_1702_MANIFEST,
        predictor=FakePredictor(),
        patch_size=64,
        overlap=16,
        n_pyramid_levels=2,
    )
    store_path = run_one_model(run_config, tiny_source, cycle, str(tmp_path))

    assert store_path.endswith("2026-08-19T00Z_f000.zarr")
    reopened = xr.open_datatree(store_path, engine="zarr", consolidated=True)
    lvl0 = reopened["0"].to_dataset()
    assert set(lvl0.data_vars) == set(MODEL_1702_MANIFEST.served_classes)
    assert lvl0.attrs["model"] == "model_1702"
    assert lvl0.attrs["cycle_time"] == "2026-08-19T00:00:00"
    assert lvl0.attrs["valid_time"] == "2026-08-19T00:00:00"  # step=0 -> valid_time == cycle_time


def test_run_one_model_step_offset_shows_up_in_valid_time_not_cycle_time(tiny_source, tmp_path):
    # 2026-08-21 regression coverage: valid_time and cycle_time used to be
    # set to the identical string regardless of step, which only happened
    # to be correct at step=0. A nonzero step must offset valid_time by
    # that many hours while cycle_time stays anchored to the cycle's own
    # init time.
    cycle = IFSCycle(date="2026-08-19", run_hour=18, step=6)
    run_config = ModelRunConfig(
        manifest=MODEL_1702_MANIFEST,
        predictor=FakePredictor(),
        patch_size=64,
        overlap=16,
        n_pyramid_levels=2,
    )
    store_path = run_one_model(run_config, tiny_source, cycle, str(tmp_path))

    assert store_path.endswith("2026-08-19T18Z_f006.zarr")
    reopened = xr.open_datatree(store_path, engine="zarr", consolidated=True)
    lvl0 = reopened["0"].to_dataset()
    assert lvl0.attrs["cycle_time"] == "2026-08-19T18:00:00"
    assert lvl0.attrs["valid_time"] == "2026-08-20T00:00:00"  # +6h, and rolls to the next day


def test_run_one_model_does_not_write_latest_pointer(tiny_source, tmp_path):
    # 2026-08-21: latest.json writing moved to run_cycle (it needs to see
    # every step across the whole cycle before it can write one consolidated
    # pointer) -- run_one_model doing it unilaterally would race/clobber
    # across steps of the same cycle.
    cycle = IFSCycle(date="2026-08-19", run_hour=0)
    run_config = ModelRunConfig(
        manifest=MODEL_1702_MANIFEST,
        predictor=FakePredictor(),
        patch_size=64,
        overlap=16,
        n_pyramid_levels=2,
    )
    run_one_model(run_config, tiny_source, cycle, str(tmp_path))

    assert not (tmp_path / "model_1702" / "latest.json").exists()


def test_run_cycle_runs_both_models_and_returns_both_paths(tiny_source, tmp_path):
    configs = [
        ModelRunConfig(manifest=BEST_LOSS_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    results = run_cycle(configs, tiny_source, "2026-08-19", 6, str(tmp_path), steps=(0,))
    assert set(results.keys()) == {"best_loss", "model_1702"}
    for paths in results.values():
        assert len(paths) == 1
        assert paths[0].endswith(".zarr")


class AlwaysFailsPredictor:
    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        raise RuntimeError("boom")


def test_run_cycle_continues_after_one_model_fails(tiny_source, tmp_path):
    configs = [
        ModelRunConfig(manifest=BEST_LOSS_MANIFEST, predictor=AlwaysFailsPredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    results = run_cycle(configs, tiny_source, "2026-08-19", 6, str(tmp_path), steps=(0,))
    assert "best_loss" not in results
    assert "model_1702" in results


def test_run_cycle_runs_every_requested_step_and_writes_them_all(tiny_source, tmp_path):
    configs = [
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    steps = (0, 6, 12)
    results = run_cycle(configs, tiny_source, "2026-08-19", 0, str(tmp_path), steps=steps)
    assert len(results["model_1702"]) == 3
    for step, path in zip(steps, results["model_1702"]):
        assert path.endswith(f"2026-08-19T00Z_f{step:03d}.zarr")


def test_run_cycle_writes_one_consolidated_latest_pointer_listing_every_successful_step(tiny_source, tmp_path):
    configs = [
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    run_cycle(configs, tiny_source, "2026-08-19", 12, str(tmp_path), steps=(0, 6, 12))

    pointer_path = tmp_path / "model_1702" / "latest.json"
    assert pointer_path.exists()
    pointer = json.loads(pointer_path.read_text())
    assert pointer["cycle_time"] == "2026-08-19T12:00:00"
    assert [s["step_hours"] for s in pointer["steps"]] == [0, 6, 12]
    assert pointer["steps"][0]["valid_time"] == "2026-08-19T12:00:00"
    assert pointer["steps"][1]["valid_time"] == "2026-08-19T18:00:00"
    assert pointer["steps"][2]["valid_time"] == "2026-08-20T00:00:00"
    assert pointer["steps"][1]["store"] == "2026-08-19T12Z_f006.zarr"


def test_run_cycle_skips_a_failing_step_but_publishes_the_rest(tiny_source, tmp_path):
    # Fail deterministically for exactly one step (12h, an unpublished
    # long-range step in this scenario) by raising out of the field source
    # rather than counting predictor calls -- tile count per step depends on
    # patch_size/overlap internals we shouldn't need to know here.
    class FlakySource:
        def __init__(self, inner):
            self._inner = inner

        @property
        def lat(self):
            return self._inner.lat

        @property
        def lon(self):
            return self._inner.lon

        def fetch_pressure_level(self, variable, level_hpa, cycle):
            if cycle.step == 6:
                raise RuntimeError("simulated 404 for an unpublished long-range step")
            return self._inner.fetch_pressure_level(variable, level_hpa, cycle)

        def fetch_single_level(self, variable, cycle):
            return self._inner.fetch_single_level(variable, cycle)

    configs = [
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    results = run_cycle(configs, FlakySource(tiny_source), "2026-08-19", 0, str(tmp_path), steps=(0, 6, 12))
    assert "model_1702" in results
    assert [p.split("_f")[-1] for p in results["model_1702"]] == ["000.zarr", "012.zarr"]


def test_run_cycle_defaults_to_target_steps_for_cycle_when_steps_omitted(tiny_source, tmp_path):
    # run_hour=18 -> capped at 90h/16 steps per ecmwf_ifs.target_steps_for_cycle;
    # exercising the real default here (not just the fast-test steps=(...)
    # override used elsewhere) would fetch 16 full inference passes, so this
    # just checks the plumbing picks up SOME multi-step default rather than
    # silently reverting to a single step=0 run.
    configs = [
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    from frontfinder.ingest.ecmwf_ifs import target_steps_for_cycle

    expected_steps = target_steps_for_cycle(18)
    assert len(expected_steps) == 16  # sanity: this is really exercising the multi-step default

    results = run_cycle(configs, tiny_source, "2026-08-19", 18, str(tmp_path))
    assert len(results["model_1702"]) == 16
