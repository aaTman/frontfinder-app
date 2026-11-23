import json
import os

import numpy as np
import pytest
import xarray as xr

from frontfinder.config.manifests import ALL_CLASSES, THETA_E_UV_Q_MANIFEST, MODEL_1702_MANIFEST
from frontfinder.ingest.base import FieldSource
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
        ModelRunConfig(manifest=THETA_E_UV_Q_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    results = run_cycle(configs, tiny_source, "2026-08-19", 6, str(tmp_path), steps=(0,))
    assert set(results.keys()) == {"theta-e_uv_q", "model_1702"}
    for paths in results.values():
        assert len(paths) == 1
        assert paths[0].endswith(".zarr")


class AlwaysFailsPredictor:
    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        raise RuntimeError("boom")


def test_run_cycle_continues_after_one_model_fails(tiny_source, tmp_path):
    configs = [
        ModelRunConfig(manifest=THETA_E_UV_Q_MANIFEST, predictor=AlwaysFailsPredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    results = run_cycle(configs, tiny_source, "2026-08-19", 6, str(tmp_path), steps=(0,))
    assert "theta-e_uv_q" not in results
    assert "model_1702" in results


def test_run_cycle_processes_step_major_across_models_not_model_major(tiny_source, tmp_path, monkeypatch):
    # 2026-09-17: run_cycle used to be MODEL-major (every step for model A,
    # then every step for model B) -- restructured to STEP-major (every
    # model for step 0, then every model for step 6, ...) so a poll that
    # gets cut off partway leaves every model's published cycle at roughly
    # the same step, instead of one model finishing its whole cycle while
    # another gets zero steps -- see run_cycle's docstring. Pins the actual
    # call order rather than just the final result shape (which is
    # order-independent and wouldn't catch a regression back to model-major).
    import frontfinder.scheduler.run_cycle as run_cycle_module

    calls: list[tuple[int, str]] = []
    original = run_cycle_module.run_one_model

    def tracking(run_config, source, cycle, output_root):
        calls.append((cycle.step, run_config.manifest.name))
        return original(run_config, source, cycle, output_root)

    monkeypatch.setattr(run_cycle_module, "run_one_model", tracking)

    configs = [
        ModelRunConfig(manifest=THETA_E_UV_Q_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    run_cycle(configs, tiny_source, "2026-08-19", 6, str(tmp_path), steps=(0, 6))

    assert [step for step, _ in calls] == [0, 0, 6, 6]
    assert {name for _, name in calls[:2]} == {"theta-e_uv_q", "model_1702"}


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
    assert pointer["expected_steps"] == [0, 6, 12]
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


def test_run_cycle_resume_does_not_refetch_or_rerun_already_written_steps(tiny_source, tmp_path):
    # Mirrors cli.py's --poll resume path (2026-09-01 postmortem): a cycle
    # that got through some steps before a transient failure (S3 "503 Slow
    # Down" in production) gets run_cycle() called again for the SAME
    # cycle/output_root once the failure clears. That must not re-fetch or
    # re-run inference for steps already on disk -- only the previously
    # missing one.
    fetched_steps: list[int] = []

    class FlakyOnceSource:
        def __init__(self, inner):
            self._inner = inner
            self._failed_once = False

        @property
        def lat(self):
            return self._inner.lat

        @property
        def lon(self):
            return self._inner.lon

        def fetch_pressure_level(self, variable, level_hpa, cycle):
            if cycle.step == 6 and not self._failed_once:
                self._failed_once = True
                raise RuntimeError("simulated transient throttling (e.g. S3 503 Slow Down)")
            fetched_steps.append(cycle.step)
            return self._inner.fetch_pressure_level(variable, level_hpa, cycle)

        def fetch_single_level(self, variable, cycle):
            return self._inner.fetch_single_level(variable, cycle)

    configs = [
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    source = FlakyOnceSource(tiny_source)

    first = run_cycle(configs, source, "2026-08-19", 0, str(tmp_path), steps=(0, 6, 12))
    assert [p.split("_f")[-1] for p in first["model_1702"]] == ["000.zarr", "012.zarr"]

    fetched_steps.clear()
    second = run_cycle(configs, source, "2026-08-19", 0, str(tmp_path), steps=(0, 6, 12))
    assert [p.split("_f")[-1] for p in second["model_1702"]] == ["000.zarr", "006.zarr", "012.zarr"]
    # Steps 0 and 12 were already written -- only step 6 should have
    # actually fetched anything the second time around.
    assert set(fetched_steps) == {6}


class SleepingPredictor:
    """Sleeps `delay_s` in predict_batch -- lets a test assert on wall-clock
    overlap (or the lack of it) between two models' inference, which is the
    actual thing `run_cycle`'s concurrent_dispatch is for; asserting on
    final results alone can't tell concurrent from sequential."""

    def __init__(self, delay_s: float):
        self._delay_s = delay_s

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        import time

        time.sleep(self._delay_s)
        n, h, w, _ = patches.shape
        out = np.random.default_rng(1).random((n, h, w, len(ALL_CLASSES))).astype(np.float32)
        return out / out.sum(axis=-1, keepdims=True)


def test_run_cycle_dispatches_models_concurrently_when_devices_differ(tiny_source, tmp_path):
    # 2026-09-17 (two-3090 host): models pinned to different devices should
    # run their inference in parallel, not one after another -- that's the
    # entire point of pinning them to separate GPUs. Each model calls
    # predict_batch twice per step (engine.run_tiled_inference's two-pass
    # hemisphere correction runs regardless of whether the grid actually
    # has southern rows -- confirmed against this exact fixture/patch size),
    # so one model's own inference work is ~2*DELAY_S; concurrent means the
    # two models' ~2*DELAY_S each overlap (~2*DELAY_S total), sequential
    # means they stack (~4*DELAY_S total). DELAY_S is large enough that
    # fixed overhead (fetch/tile/write) can't blur that distinction.
    DELAY_S = 0.5
    per_model_s = 2 * DELAY_S
    configs = [
        ModelRunConfig(
            manifest=THETA_E_UV_Q_MANIFEST, predictor=SleepingPredictor(DELAY_S), device="/GPU:0",
            patch_size=64, overlap=16, n_pyramid_levels=2,
        ),
        ModelRunConfig(
            manifest=MODEL_1702_MANIFEST, predictor=SleepingPredictor(DELAY_S), device="/GPU:1",
            patch_size=64, overlap=16, n_pyramid_levels=2,
        ),
    ]
    import time

    start = time.monotonic()
    results = run_cycle(configs, tiny_source, "2026-08-19", 6, str(tmp_path), steps=(0,))
    elapsed = time.monotonic() - start

    assert set(results.keys()) == {"theta-e_uv_q", "model_1702"}
    assert elapsed < 1.5 * per_model_s, (
        f"took {elapsed:.2f}s -- expected concurrent (~{per_model_s}s), not sequential (~{2 * per_model_s}s)"
    )


def test_run_cycle_stays_sequential_when_devices_match(tiny_source, tmp_path):
    # The CPU-only case (both None, every deployment before GPUs existed
    # here) -- and any case where models share one device -- must stay
    # sequential: two models on the same device gain nothing from Python
    # threads (neither's TF op frees that device for the other) and would
    # just add scheduling overhead.
    DELAY_S = 0.3
    per_model_s = 2 * DELAY_S  # see the concurrent test above for why x2
    configs = [
        ModelRunConfig(
            manifest=THETA_E_UV_Q_MANIFEST, predictor=SleepingPredictor(DELAY_S), device=None,
            patch_size=64, overlap=16, n_pyramid_levels=2,
        ),
        ModelRunConfig(
            manifest=MODEL_1702_MANIFEST, predictor=SleepingPredictor(DELAY_S), device=None,
            patch_size=64, overlap=16, n_pyramid_levels=2,
        ),
    ]
    import time

    start = time.monotonic()
    run_cycle(configs, tiny_source, "2026-08-19", 6, str(tmp_path), steps=(0,))
    elapsed = time.monotonic() - start

    assert elapsed >= 1.8 * per_model_s, (
        f"took {elapsed:.2f}s -- expected sequential (~{2 * per_model_s}s), looked concurrent instead"
    )


def test_run_cycle_defaults_to_target_steps_for_cycle_when_steps_omitted(tiny_source, tmp_path):
    # run_hour=18 -> capped at 144h/25 steps per ecmwf_ifs.target_steps_for_cycle
    # (2026-08-22: corrected from a stale 90h/16-step assumption -- see that
    # module's available_forecast_steps docstring); exercising the real
    # default here (not just the fast-test steps=(...) override used
    # elsewhere) would fetch 25 full inference passes, so this just checks
    # the plumbing picks up SOME multi-step default rather than silently
    # reverting to a single step=0 run.
    configs = [
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]
    from frontfinder.ingest.ecmwf_ifs import target_steps_for_cycle

    expected_steps = target_steps_for_cycle(18)
    assert len(expected_steps) == 25  # sanity: this is really exercising the multi-step default

    results = run_cycle(configs, tiny_source, "2026-08-19", 18, str(tmp_path))
    assert len(results["model_1702"]) == 25


def _model_1702_config():
    return [
        ModelRunConfig(manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2),
    ]


def test_run_cycle_leaves_front_lines_out_unless_asked(tiny_source, tmp_path):
    run_cycle(_model_1702_config(), tiny_source, "2026-08-19", 12, str(tmp_path), steps=(0, 6))
    pointer = json.loads((tmp_path / "model_1702" / "latest.json").read_text())
    assert all("fronts" not in s for s in pointer["steps"])
    assert not list((tmp_path / "model_1702").glob("*.fronts.geojson"))


def test_run_cycle_with_front_lines_lists_a_geojson_per_step_in_latest_json(tiny_source, tmp_path):
    run_cycle(_model_1702_config(), tiny_source, "2026-08-19", 12, str(tmp_path), steps=(0, 6, 12), front_lines=True)
    model_dir = tmp_path / "model_1702"
    pointer = json.loads((model_dir / "latest.json").read_text())
    assert [s["fronts"] for s in pointer["steps"]] == [
        "2026-08-19T12Z_f000.fronts.geojson",
        "2026-08-19T12Z_f006.fronts.geojson",
        "2026-08-19T12Z_f012.fronts.geojson",
    ]
    collection = json.loads((model_dir / pointer["steps"][1]["fronts"]).read_text())
    assert collection["type"] == "FeatureCollection"
    assert collection["valid_time"] == "2026-08-19T18:00:00"


def test_a_crash_in_front_lines_never_stops_publishing(tiny_source, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("splines exploded")

    monkeypatch.setattr("frontfinder.postprocess.front_lines.write_front_lines", boom)
    results = run_cycle(_model_1702_config(), tiny_source, "2026-08-19", 12, str(tmp_path), steps=(0, 6), front_lines=True)
    assert len(results["model_1702"]) == 2
    pointer = json.loads((tmp_path / "model_1702" / "latest.json").read_text())
    assert [s["step_hours"] for s in pointer["steps"]] == [0, 6]
    assert all("fronts" not in s for s in pointer["steps"])


def test_run_cycle_with_front_lines_saves_orientation_fields_next_to_each_store(tiny_source, tmp_path):
    from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_MANIFEST as NBS

    class NbsPredictor:
        def predict_batch(self, patches):
            n, h, w, _ = patches.shape
            out = np.random.default_rng(1).random((n, h, w, len(NBS.all_classes))).astype(np.float32)
            return out / out.sum(axis=-1, keepdims=True)

    configs = [ModelRunConfig(manifest=NBS, predictor=NbsPredictor(), patch_size=64, overlap=16, n_pyramid_levels=2)]
    run_cycle(configs, tiny_source, "2026-08-19", 12, str(tmp_path), steps=(0, 6), front_lines=True)
    model_dir = tmp_path / NBS.name
    assert sorted(p.name for p in model_dir.glob("*.fields.npz")) == [
        "2026-08-19T12Z_f000.fields.npz",
        "2026-08-19T12Z_f006.fields.npz",
    ]
    with np.load(model_dir / "2026-08-19T12Z_f000.fields.npz") as z:
        assert set(z.files) == {"temperature", "u", "v", "lat", "lon"}
        assert z["temperature"].shape == (len(z["lat"]), len(z["lon"]))


def test_run_cycle_without_front_lines_writes_no_orientation_fields(tiny_source, tmp_path):
    run_cycle(_model_1702_config(), tiny_source, "2026-08-19", 12, str(tmp_path), steps=(0,))
    assert not list((tmp_path / "model_1702").glob("*.fields.npz"))


def test_rebuild_orientation_fields_fills_in_a_cycle_published_without_them(tiny_source, tmp_path):
    from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_MANIFEST as NBS
    from frontfinder.postprocess import front_lines

    class NbsPredictor:
        def predict_batch(self, patches):
            n, h, w, _ = patches.shape
            out = np.random.default_rng(1).random((n, h, w, len(NBS.all_classes))).astype(np.float32)
            return out / out.sum(axis=-1, keepdims=True)

    configs = [ModelRunConfig(manifest=NBS, predictor=NbsPredictor(), patch_size=64, overlap=16, n_pyramid_levels=2)]
    run_cycle(configs, tiny_source, "2026-08-19", 12, str(tmp_path), steps=(0, 6))  # front_lines off: no sidecars
    model_dir = tmp_path / NBS.name
    assert not list(model_dir.glob("*.fields.npz"))

    assert front_lines.rebuild_orientation_fields(str(tmp_path), NBS.name, tiny_source) == 2
    assert sorted(p.name for p in model_dir.glob("*.fields.npz")) == [
        "2026-08-19T12Z_f000.fields.npz",
        "2026-08-19T12Z_f006.fields.npz",
    ]
    assert front_lines.rebuild_orientation_fields(str(tmp_path), NBS.name, tiny_source) == 0  # nothing left to do
    assert front_lines.rebuild_orientation_fields(str(tmp_path), "not_a_model", tiny_source) == 0


def _nbs_front_lines_config():
    from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_MANIFEST as NBS

    class NbsPredictor:
        def predict_batch(self, patches):
            n, h, w, _ = patches.shape
            out = np.random.default_rng(1).random((n, h, w, len(NBS.all_classes))).astype(np.float32)
            return out / out.sum(axis=-1, keepdims=True)

    return NBS, [ModelRunConfig(manifest=NBS, predictor=NbsPredictor(), patch_size=64, overlap=16, n_pyramid_levels=2)]


def test_run_cycle_packing_with_front_lines_saves_exact_float32_probabilities(tiny_source, tmp_path, monkeypatch):
    # Packing quantizes the store, and the front lines are derived from the store
    # on disk -- so the exact float32 grids must be saved beside it first.
    monkeypatch.setenv("FRONTFINDER_PACK_UINT8", "1")
    NBS, configs = _nbs_front_lines_config()
    run_cycle(configs, tiny_source, "2026-08-19", 12, str(tmp_path), steps=(0, 6), front_lines=True)
    model_dir = tmp_path / NBS.name
    assert sorted(p.name for p in model_dir.glob("*.probs.npz")) == [
        "2026-08-19T12Z_f000.probs.npz",
        "2026-08-19T12Z_f006.probs.npz",
    ]
    ds = xr.open_zarr(str(model_dir / "2026-08-19T12Z_f000.zarr"), group="0", consolidated=False, mask_and_scale=False)
    assert ds["cold"].dtype == np.uint8  # the store really is packed
    with np.load(model_dir / "2026-08-19T12Z_f000.probs.npz") as z:
        assert z["CF"].dtype == np.float32
        # sidecar holds exact values; the packed store is within the packing error of them
        decoded = ds["cold"].values.astype(np.float64) / 254.0
        assert np.abs(decoded - z["CF"]).max() <= 0.5 / 254 + 1e-6
        assert not np.array_equal(decoded, z["CF"])


def test_run_cycle_without_packing_writes_no_probabilities_sidecar(tiny_source, tmp_path, monkeypatch):
    monkeypatch.delenv("FRONTFINDER_PACK_UINT8", raising=False)
    NBS, configs = _nbs_front_lines_config()
    run_cycle(configs, tiny_source, "2026-08-19", 12, str(tmp_path), steps=(0,), front_lines=True)
    assert not list((tmp_path / NBS.name).glob("*.probs.npz"))


def test_run_cycle_packing_without_front_lines_writes_no_probabilities_sidecar(tiny_source, tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTFINDER_PACK_UINT8", "1")
    run_cycle(_model_1702_config(), tiny_source, "2026-08-19", 12, str(tmp_path), steps=(0,))
    assert not list((tmp_path / "model_1702").glob("*.probs.npz"))


class PressureSource:
    """Wraps the fake source and serves the two single-level variables the
    pressure layer fetches: a smooth synthetic MSLP field with one low (like
    IFS `msl`, but deterministic and offline) and whatever the fake source
    gives for `surface_pressure` (a plausible flat-ish sp field)."""

    def __init__(self, inner):
        self._inner = inner

    @property
    def lat(self):
        return self._inner.lat

    @property
    def lon(self):
        return self._inner.lon

    def fetch_pressure_level(self, variable, level_hpa, cycle):
        return self._inner.fetch_pressure_level(variable, level_hpa, cycle)

    def fetch_single_level(self, variable, cycle):
        if variable == "mean_sea_level_pressure":
            lat, lon = np.meshgrid(self._inner.lat, self._inner.lon, indexing="ij")
            low_pa = 1500.0 * np.exp(-(((lat - 40.0) / 6.0) ** 2 + ((lon - 265.0) / 8.0) ** 2))
            return 101325.0 - low_pa  # a ~15 hPa low near (40N, 265E)
        return self._inner.fetch_single_level(variable, cycle)


class RaisingSource:
    """Wraps a source and raises for whichever single-level variables are
    named in `raise_for` -- the failure mode _write_pressure_layer must
    survive (a feed that doesn't carry the variable, a bad GRIB, etc)."""

    def __init__(self, inner, raise_for=()):
        self._inner = inner
        self._raise_for = set(raise_for)

    @property
    def lat(self):
        return self._inner.lat

    @property
    def lon(self):
        return self._inner.lon

    def fetch_pressure_level(self, variable, level_hpa, cycle):
        return self._inner.fetch_pressure_level(variable, level_hpa, cycle)

    def fetch_single_level(self, variable, cycle):
        if variable in self._raise_for:
            raise RuntimeError(f"simulated missing variable: {variable}")
        return self._inner.fetch_single_level(variable, cycle)


def _model_1702_run_config():
    return ModelRunConfig(
        manifest=MODEL_1702_MANIFEST, predictor=FakePredictor(), patch_size=64, overlap=16, n_pyramid_levels=2
    )


def test_run_one_model_with_save_pressure_writes_a_pressure_geojson_next_to_the_store(tiny_source, tmp_path):
    cycle = IFSCycle(date="2026-08-19", run_hour=0)
    store_path = run_one_model(_model_1702_run_config(), PressureSource(tiny_source), cycle, str(tmp_path), save_pressure=True)

    path = tmp_path / "model_1702" / "2026-08-19T00Z_f000.pressure.geojson"
    assert os.path.dirname(store_path) == str(tmp_path / "model_1702")
    collection = json.loads(path.read_text())
    assert collection["type"] == "FeatureCollection"
    assert collection["valid_time"] == "2026-08-19T00:00:00"
    assert len(collection["features"]) >= 1
    assert any(f["properties"]["kind"] == "isobar" for f in collection["features"])


def test_run_one_model_without_save_pressure_writes_no_pressure_file(tiny_source, tmp_path):
    cycle = IFSCycle(date="2026-08-19", run_hour=0)
    run_one_model(_model_1702_run_config(), PressureSource(tiny_source), cycle, str(tmp_path), save_pressure=False)
    assert not list((tmp_path / "model_1702").glob("*.pressure.geojson"))


def test_write_pressure_layer_survives_a_source_that_raises_for_mslp(tiny_source, tmp_path):
    # The whole point of the helper being best effort: a feed that can't
    # serve MSLP must leave no file and no exception, not break the step.
    from frontfinder.scheduler.run_cycle import _write_pressure_layer

    model_dir = tmp_path / "model_1702"
    model_dir.mkdir()
    source = RaisingSource(PressureSource(tiny_source), raise_for={"mean_sea_level_pressure"})
    _write_pressure_layer(source, IFSCycle(date="2026-08-19", run_hour=0), str(model_dir / "2026-08-19T00Z_f000.zarr"))
    assert not list(model_dir.glob("*.pressure.geojson"))


def test_write_pressure_layer_still_writes_when_only_surface_pressure_raises(tiny_source, tmp_path):
    # surface_pressure only refines the H/L placement -- losing it must not
    # cost us the isobars.
    from frontfinder.scheduler.run_cycle import _write_pressure_layer

    model_dir = tmp_path / "model_1702"
    model_dir.mkdir()
    source = RaisingSource(PressureSource(tiny_source), raise_for={"surface_pressure"})
    _write_pressure_layer(source, IFSCycle(date="2026-08-19", run_hour=0), str(model_dir / "2026-08-19T00Z_f000.zarr"))
    collection = json.loads((model_dir / "2026-08-19T00Z_f000.pressure.geojson").read_text())
    assert collection["features"]


def test_write_pressure_layer_does_not_overwrite_an_existing_file(tiny_source, tmp_path):
    # Resumed cycles call this for steps that already have their layer;
    # rewriting it would be wasted work at best.
    from frontfinder.scheduler.run_cycle import _write_pressure_layer

    model_dir = tmp_path / "model_1702"
    model_dir.mkdir()
    existing = model_dir / "2026-08-19T00Z_f000.pressure.geojson"
    existing.write_text("sentinel")
    source = PressureSource(tiny_source)
    _write_pressure_layer(source, IFSCycle(date="2026-08-19", run_hour=0), str(model_dir / "2026-08-19T00Z_f000.zarr"))
    assert existing.read_text() == "sentinel"


def test_run_cycle_leaves_pressure_out_unless_asked(tiny_source, tmp_path):
    run_cycle(_model_1702_config(), PressureSource(tiny_source), "2026-08-19", 12, str(tmp_path), steps=(0, 6))
    pointer = json.loads((tmp_path / "model_1702" / "latest.json").read_text())
    assert all("pressure" not in s for s in pointer["steps"])
    assert not list((tmp_path / "model_1702").glob("*.pressure.geojson"))


def test_run_cycle_with_pressure_lists_a_geojson_per_step_in_latest_json(tiny_source, tmp_path):
    run_cycle(
        _model_1702_config(), PressureSource(tiny_source), "2026-08-19", 12, str(tmp_path), steps=(0, 6, 12), pressure=True
    )
    model_dir = tmp_path / "model_1702"
    pointer = json.loads((model_dir / "latest.json").read_text())
    assert [s["pressure"] for s in pointer["steps"]] == [
        "2026-08-19T12Z_f000.pressure.geojson",
        "2026-08-19T12Z_f006.pressure.geojson",
        "2026-08-19T12Z_f012.pressure.geojson",
    ]
    collection = json.loads((model_dir / pointer["steps"][1]["pressure"]).read_text())
    assert collection["type"] == "FeatureCollection"
    assert collection["valid_time"] == "2026-08-19T18:00:00"


def test_run_cycle_resume_adds_pressure_to_steps_it_skips(tiny_source, tmp_path):
    # cli.py's --poll resume re-runs a cycle with already-written stores;
    # with pressure on, the skipped steps must still get their layer.
    configs = _model_1702_config()
    source = PressureSource(tiny_source)
    run_cycle(configs, source, "2026-08-19", 12, str(tmp_path), steps=(0, 6))  # no pressure the first time
    assert not list((tmp_path / "model_1702").glob("*.pressure.geojson"))

    run_cycle(configs, source, "2026-08-19", 12, str(tmp_path), steps=(0, 6), pressure=True)
    pointer = json.loads((tmp_path / "model_1702" / "latest.json").read_text())
    assert [s["pressure"] for s in pointer["steps"]] == [
        "2026-08-19T12Z_f000.pressure.geojson",
        "2026-08-19T12Z_f006.pressure.geojson",
    ]


class _FakeHRRRSource(FieldSource):
    """Minimal synthetic `FieldSource` on an HRRR grid, offline with no GRIB decoding."""

    def __init__(self, grid):
        self._grid = grid
        self.fetched_steps: list[int] = []

    @property
    def source_id(self) -> str:
        return "fake_hrrr"

    @property
    def grid(self):
        return self._grid

    def _band(self) -> np.ndarray:
        lat, lon = self._grid.lat, (self._grid.lon + 180.0) % 360.0 - 180.0
        return (10.0 * np.exp(-(((lat - 38.0) / 2.0) ** 2 + ((lon + 97.0) / 8.0) ** 2))).astype(np.float32)

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: IFSCycle | object) -> np.ndarray:
        self.fetched_steps.append(cycle.step)
        if variable in ("u_component_of_wind", "v_component_of_wind"):
            return self._band()
        if variable == "temperature":
            return np.full(self._grid.shape, 285.0 * (level_hpa / 1000.0) ** 0.286, dtype=np.float32)
        if variable == "specific_humidity":
            return np.full(self._grid.shape, 0.005 * level_hpa / 1000.0, dtype=np.float32)
        raise ValueError(f"unsupported variable {variable!r}")

    def fetch_single_level(self, variable: str, cycle: IFSCycle | object) -> np.ndarray:
        if variable in ("10m_u_component_of_wind", "10m_v_component_of_wind"):
            return self._band()
        raise ValueError(f"unsupported single level {variable!r}")

    def fetch_static(self, variable: str) -> np.ndarray:
        return np.zeros(self._grid.shape, dtype=np.float32)


def test_run_hrrr_cycle_writes_3km_pyramid_domain_and_latest_pointer(tmp_path):
    from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST as NBS
    from frontfinder.inference.refiner import FakeRefinerPredictor
    from frontfinder.ingest.hrrr_grid import native_grid
    from frontfinder.scheduler.run_cycle import HRRR_3KM_MODEL_KEY, HRRRRunConfig, run_hrrr_cycle

    class NbsPredictor:
        def predict_batch(self, patches: np.ndarray) -> np.ndarray:
            out = np.full((*patches.shape[:3], len(NBS.all_classes)), 0.01, dtype=np.float32)
            out[..., 0] = 0.0
            out[..., 0] = np.clip(1.0 - out.sum(axis=-1), 0.1, None)
            return out / out.sum(axis=-1, keepdims=True)

    run_config = HRRRRunConfig(
        manifest=NBS,
        base_predictor=NbsPredictor(),
        refiner=FakeRefinerPredictor(),
        refiner_mean=np.zeros(NBS.n_channels, dtype=np.float32),
        refiner_std=np.ones(NBS.n_channels, dtype=np.float32),
    )
    source_3km = _FakeHRRRSource(native_grid(1))
    source_24km = _FakeHRRRSource(native_grid(8))

    paths = run_hrrr_cycle(run_config, source_3km, source_24km, "2026-08-19", 12, str(tmp_path), steps=(0, 6))

    assert [os.path.basename(p) for p in paths] == ["2026-08-19T12Z_f000.zarr", "2026-08-19T12Z_f006.zarr"]
    model_dir = tmp_path / HRRR_3KM_MODEL_KEY
    pointer = json.loads((model_dir / "latest.json").read_text())
    assert pointer["cycle_time"] == "2026-08-19T12:00:00"
    assert [s["step_hours"] for s in pointer["steps"]] == [0, 6]
    assert pointer["expected_steps"] == [0, 6]
    assert pointer["source_label"] == "HRRR"
    assert pointer["grid"]["name"] == "hrrr-conus-3km"
    domain = json.loads((model_dir / pointer["domain"]).read_text())
    assert domain["features"][0]["properties"]["margin_cells"] == 256


def test_run_hrrr_cycle_skips_already_written_steps_on_resume(tmp_path):
    from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST as NBS
    from frontfinder.inference.refiner import FakeRefinerPredictor
    from frontfinder.ingest.hrrr_grid import native_grid
    from frontfinder.scheduler.run_cycle import HRRRRunConfig, run_hrrr_cycle

    class NbsPredictor:
        def predict_batch(self, patches: np.ndarray) -> np.ndarray:
            out = np.full((*patches.shape[:3], len(NBS.all_classes)), 0.01, dtype=np.float32)
            return out / out.sum(axis=-1, keepdims=True)

    run_config = HRRRRunConfig(
        manifest=NBS,
        base_predictor=NbsPredictor(),
        refiner=FakeRefinerPredictor(),
        refiner_mean=np.zeros(NBS.n_channels, dtype=np.float32),
        refiner_std=np.ones(NBS.n_channels, dtype=np.float32),
    )
    source_3km = _FakeHRRRSource(native_grid(1))
    source_24km = _FakeHRRRSource(native_grid(8))

    first = run_hrrr_cycle(run_config, source_3km, source_24km, "2026-08-19", 12, str(tmp_path), steps=(0,))
    assert len(first) == 1

    source_3km.fetched_steps.clear()
    source_24km.fetched_steps.clear()
    second = run_hrrr_cycle(run_config, source_3km, source_24km, "2026-08-19", 12, str(tmp_path), steps=(0, 6))
    assert len(second) == 2
    assert set(source_3km.fetched_steps) == {6}
    assert set(source_24km.fetched_steps) == {6}
