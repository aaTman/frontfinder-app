import numpy as np
import pytest
import xarray as xr

from frontfinder.config.manifests import BEST_LOSS_MANIFEST, MODEL_1702_MANIFEST
from frontfinder.ingest.ecmwf_ifs import (
    EcmwfOpenDataSource,
    FakeIFSFieldSource,
    IFSCycle,
    assemble_model_input,
)


@pytest.fixture
def small_source():
    lat = np.linspace(20.0, 50.0, 17)
    lon = np.linspace(200.0, 300.0, 21)
    return FakeIFSFieldSource(lat, lon, seed=42)


def test_ifs_cycle_rejects_bad_run_hour():
    with pytest.raises(ValueError):
        IFSCycle(date="2026-08-19", run_hour=9)


def test_ifs_cycle_rejects_bad_date():
    with pytest.raises(ValueError):
        IFSCycle(date="not-a-date", run_hour=0)


def test_ifs_cycle_accepts_valid_synoptic_hours():
    for h in (0, 6, 12, 18):
        IFSCycle(date="2026-08-19", run_hour=h)


def test_assemble_best_loss_input_has_24_channels(small_source):
    # 30 originally (5 vars x 6 levels); potential_vorticity dropped
    # 2026-08-20 -- see BEST_LOSS_MANIFEST's docstring -- leaving 4 vars x 6
    # levels = 24 (theta-e stays; Taylor confirmed the retrained model still
    # uses it, just not PV).
    cycle = IFSCycle(date="2026-08-19", run_hour=12)
    arr = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle)
    assert arr.shape == (17, 21, 24)
    assert np.all(np.isfinite(arr))


def test_assemble_model_1702_input_has_25_channels(small_source):
    cycle = IFSCycle(date="2026-08-19", run_hour=12)
    arr = assemble_model_input(MODEL_1702_MANIFEST, small_source, cycle)
    assert arr.shape == (17, 21, 25)
    assert np.all(np.isfinite(arr))


def test_assemble_is_deterministic_for_same_cycle(small_source):
    cycle = IFSCycle(date="2026-08-19", run_hour=12)
    arr1 = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle)
    arr2 = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle)
    np.testing.assert_array_equal(arr1, arr2)


def test_assemble_differs_between_cycles(small_source):
    cycle_a = IFSCycle(date="2026-08-19", run_hour=0)
    cycle_b = IFSCycle(date="2026-08-19", run_hour=12)
    arr_a = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle_a)
    arr_b = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle_b)
    assert not np.array_equal(arr_a, arr_b)


def test_assemble_channel_order_is_level_major_not_variable_major(small_source):
    # 2026-08-20 regression test: assemble_model_input used to stack
    # channels variable-major (all 6 levels of theta_e, then all 6 of u,
    # ...), but fronts/data/inputs.py's inputs_ds_to_dataarray() -- what
    # actually built the training tensor -- stacks level-major (theta_e/u/v/q
    # at 1000hPa, then all four again at 925hPa, ...). Getting this
    # backwards silently fed the model's baked-in normalization stats to the
    # wrong channels (see ecmwf_ifs.py's assemble_model_input docstring).
    # Only channel 0 (theta_e@1000) should be in a plausible Kelvin range;
    # channel 1 (u@1000, level-major) should NOT be -- it's a wind speed.
    cycle = IFSCycle(date="2026-08-19", run_hour=12)
    arr = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle)
    assert np.all(arr[..., 0] > 200) and np.all(arr[..., 0] < 500)  # theta_e@1000
    assert not np.all((arr[..., 1] > 200) & (arr[..., 1] < 500))  # u@1000, not theta_e@925


def test_assemble_channel_order_matches_manifest_channel_names_exactly(small_source):
    # Stronger version of the above: fetch every DIRECT (variable, level)
    # pair straight from the same fake source and confirm
    # assemble_model_input's output is a bit-exact match, position by
    # position, against BEST_LOSS_MANIFEST.channel_names()'s documented
    # order for every non-derived channel -- not just a plausibility check
    # on a couple of channels.
    cycle = IFSCycle(date="2026-08-19", run_hour=12)
    arr = assemble_model_input(BEST_LOSS_MANIFEST, small_source, cycle)
    names = BEST_LOSS_MANIFEST.channel_names()
    assert len(names) == arr.shape[-1]

    direct_lookup = {
        "u_component_of_wind": small_source.fetch_pressure_level,
        "v_component_of_wind": small_source.fetch_pressure_level,
        "specific_humidity": small_source.fetch_pressure_level,
    }
    for i, name in enumerate(names):
        var_name, _, level_str = name.rpartition("_")
        level = int(level_str)
        if var_name in direct_lookup:
            expected = small_source.fetch_pressure_level(var_name, level, cycle)
            np.testing.assert_array_equal(arr[..., i], expected)
        # equivalent_potential_temperature is derived, not directly
        # comparable to a raw fetched field -- the plausible-range check
        # above is what catches a real ordering bug for it.


# --- EcmwfOpenDataSource.fetch_pressure_level's isobaricInhPa handling ---
#
# This is the one piece of EcmwfOpenDataSource that's pure decode logic
# rather than a network call, so it's tested directly against synthetic
# xarray Datasets shaped the way cfgrib really returns them (confirmed via
# scripts/smoke_test_ecmwf.py against a live request, 2026-08-20) --
# `_fetch_grib` is monkeypatched to avoid any network/cfgrib dependency.


def _fake_grib_scalar_level(level_hpa: float, value: float) -> xr.Dataset:
    """Mimics what cfgrib returns when a single `levelist` value is
    requested: isobaricInhPa is a 0-d scalar coordinate, not a dimension."""
    data = np.full((3, 4), value, dtype=np.float32)
    da = xr.DataArray(data, dims=("latitude", "longitude"))
    da = da.assign_coords(isobaricInhPa=level_hpa)
    return xr.Dataset({"t": da})


def _fake_grib_dimensioned_levels(values_by_level: dict) -> xr.Dataset:
    """Mimics what cfgrib returns when multiple levels are present:
    isobaricInhPa is a real, indexable dimension."""
    levels = sorted(values_by_level)
    stacked = np.stack(
        [np.full((3, 4), values_by_level[lvl], dtype=np.float32) for lvl in levels], axis=0
    )
    da = xr.DataArray(
        stacked, dims=("isobaricInhPa", "latitude", "longitude"), coords={"isobaricInhPa": levels}
    )
    return xr.Dataset({"t": da})


def test_fetch_pressure_level_handles_scalar_isobaric_coordinate(tmp_path, monkeypatch):
    source = EcmwfOpenDataSource(cache_dir=str(tmp_path))
    monkeypatch.setattr(source, "_fetch_grib", lambda param, cycle, levelist: _fake_grib_scalar_level(850, 42.0))
    cycle = IFSCycle(date="2026-08-19", run_hour=12)

    arr = source.fetch_pressure_level("temperature", 850, cycle)

    assert arr.shape == (3, 4)
    np.testing.assert_allclose(arr, 42.0)


def test_fetch_pressure_level_handles_dimensioned_isobaric_coordinate(tmp_path, monkeypatch):
    source = EcmwfOpenDataSource(cache_dir=str(tmp_path))
    monkeypatch.setattr(
        source,
        "_fetch_grib",
        lambda param, cycle, levelist: _fake_grib_dimensioned_levels({700: 10.0, 850: 20.0}),
    )
    cycle = IFSCycle(date="2026-08-19", run_hour=12)

    arr = source.fetch_pressure_level("temperature", 850, cycle)

    assert arr.shape == (3, 4)
    np.testing.assert_allclose(arr, 20.0)


def test_fetch_pressure_level_raises_if_scalar_level_mismatches_request(tmp_path, monkeypatch):
    # if cfgrib ever silently ignores/reinterprets the levelist request, fail
    # loudly rather than quietly serving the wrong pressure level's data.
    source = EcmwfOpenDataSource(cache_dir=str(tmp_path))
    monkeypatch.setattr(source, "_fetch_grib", lambda param, cycle, levelist: _fake_grib_scalar_level(700, 1.0))
    cycle = IFSCycle(date="2026-08-19", run_hour=12)

    with pytest.raises(ValueError):
        source.fetch_pressure_level("temperature", 850, cycle)


def test_fetch_pressure_level_caches_by_variable_level_and_cycle(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_grib(param, cycle, levelist):
        calls.append((param, tuple(levelist)))
        return _fake_grib_scalar_level(levelist[0], float(levelist[0]))

    source = EcmwfOpenDataSource(cache_dir=str(tmp_path))
    monkeypatch.setattr(source, "_fetch_grib", fake_fetch_grib)
    cycle = IFSCycle(date="2026-08-19", run_hour=12)

    source.fetch_pressure_level("temperature", 850, cycle)
    source.fetch_pressure_level("temperature", 850, cycle)  # should hit the in-memory cache

    assert calls == [("t", (850,))]

