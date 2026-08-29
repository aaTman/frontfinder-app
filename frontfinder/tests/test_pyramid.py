import json

import numpy as np
import pytest

from frontfinder.config.manifests import MODEL_1702_MANIFEST
from frontfinder.zarrio.pyramid import (
    FrontFields,
    build_front_pyramid,
    build_level0_dataset,
    write_front_pyramid,
)


@pytest.fixture
def small_fields():
    lat = np.linspace(25.0, 56.75, 64)
    lon = np.linspace(228.0, 299.75, 64)
    rng = np.random.default_rng(0)
    probs = {
        cls: rng.random((64, 64)).astype(np.float32) for cls in MODEL_1702_MANIFEST.served_classes
    }
    return FrontFields(
        probabilities=probs,
        lat=lat,
        lon=lon,
        valid_time="2026-08-19T12:00:00",
        cycle_time="2026-08-19T12:00:00",
    )


@pytest.fixture
def global_fields():
    lat = np.linspace(89.875, -89.875, 8)
    lon = np.linspace(0.0, 360.0, 16, endpoint=False)  # uniform-step global 0..360 axis, coarse for test speed
    rng = np.random.default_rng(1)
    probs = {cls: rng.random((8, 16)).astype(np.float32) for cls in MODEL_1702_MANIFEST.served_classes}
    return FrontFields(
        probabilities=probs,
        lat=lat,
        lon=lon,
        valid_time="2026-08-19T12:00:00",
        cycle_time="2026-08-19T12:00:00",
    )


def test_front_fields_rejects_shape_mismatch():
    lat = np.linspace(0, 1, 10)
    lon = np.linspace(0, 1, 10)
    probs = {cls: np.zeros((5, 5)) for cls in MODEL_1702_MANIFEST.served_classes}
    with pytest.raises(ValueError):
        FrontFields(probabilities=probs, lat=lat, lon=lon, valid_time="t", cycle_time="t")


def test_build_level0_dataset_has_only_served_classes(small_fields):
    ds = build_level0_dataset(small_fields, MODEL_1702_MANIFEST)
    assert set(ds.data_vars) == set(MODEL_1702_MANIFEST.served_classes)
    assert "dryline" not in ds.data_vars
    assert "background" not in ds.data_vars


def test_build_level0_dataset_rejects_missing_class(small_fields):
    small_fields.probabilities.pop("cold")
    with pytest.raises(ValueError):
        build_level0_dataset(small_fields, MODEL_1702_MANIFEST)


def test_build_level0_dataset_carries_provenance_attrs(small_fields):
    ds = build_level0_dataset(small_fields, MODEL_1702_MANIFEST)
    assert ds.attrs["model"] == "model_1702"
    assert ds.attrs["cycle_time"] == "2026-08-19T12:00:00"


def test_build_front_pyramid_has_requested_number_of_levels(small_fields):
    pyramid = build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=3)
    level_groups = [g for g in pyramid.datatree.groups if g not in ("", "/")]
    assert len(level_groups) == 3


def test_build_front_pyramid_coarsens_each_level_by_half(small_fields):
    pyramid = build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=3)
    lvl0 = pyramid.datatree["0"].to_dataset()
    lvl1 = pyramid.datatree["1"].to_dataset()
    assert lvl0.sizes["lat"] == 64
    assert lvl1.sizes["lat"] == 32


def test_build_level0_dataset_embeds_colormap_and_clim_style_attrs(small_fields):
    ds = build_level0_dataset(small_fields, MODEL_1702_MANIFEST)
    for cls in MODEL_1702_MANIFEST.served_classes:
        assert ds[cls].attrs["colormap"]
        assert ds[cls].attrs["clim"] == [0.0, 1.0]


def test_build_front_pyramid_rejects_zero_levels(small_fields):
    with pytest.raises(ValueError):
        build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=0)


def test_build_level0_dataset_rolls_global_lon_to_signed_convention(global_fields):
    # 2026-08-29 fix: @carbonplan/zarr-layer's region-visibility math
    # assumes -180..180, so a global grid's 0..360 IFS-native axis gets
    # rolled at serve time -- see _roll_lon_0_360_to_signed's docstring.
    ds = build_level0_dataset(global_fields, MODEL_1702_MANIFEST)
    lon = ds["lon"].values
    assert np.all(np.diff(lon) > 0)
    assert lon[0] == pytest.approx(-180.0)
    assert lon[-1] == pytest.approx(157.5)
    assert lon.min() >= -180.0 and lon.max() < 180.0


def test_build_level0_dataset_preserves_data_at_each_lon_after_roll(global_fields):
    # The value at a given real-world longitude must survive the reindex,
    # not just the coordinate labels -- roll the data columns in lockstep
    # with the lon axis, or this fix silently scrambles the map.
    original_lon = global_fields.lon
    original_cold = global_fields.probabilities["cold"]

    ds = build_level0_dataset(global_fields, MODEL_1702_MANIFEST)
    rolled_lon = ds["lon"].values
    rolled_cold = ds["cold"].values

    for i, lon_val in enumerate(original_lon):
        signed = lon_val - 360 if lon_val >= 180 else lon_val
        j = np.argmin(np.abs(rolled_lon - signed))
        np.testing.assert_allclose(rolled_cold[:, j], original_cold[:, i])


def test_build_level0_dataset_leaves_regional_lon_untouched(small_fields):
    # small_fields' lon (228..299.75) is a regional window, not a full
    # 360deg-spanning global grid -- the roll must no-op for it, exactly
    # like inference.engine.run_tiled_inference's is_global_lon check.
    ds = build_level0_dataset(small_fields, MODEL_1702_MANIFEST)
    np.testing.assert_allclose(ds["lon"].values, small_fields.lon)
    np.testing.assert_allclose(ds["cold"].values, small_fields.probabilities["cold"])


def test_write_front_pyramid_roundtrips_through_zarr(small_fields, tmp_path):
    import xarray as xr

    pyramid = build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=2)
    store_path = str(tmp_path / "test_pyramid.zarr")
    write_front_pyramid(pyramid, store_path)

    reopened = xr.open_datatree(store_path, engine="zarr")
    lvl0 = reopened["0"].to_dataset()
    np.testing.assert_allclose(
        lvl0["cold"].values, small_fields.probabilities["cold"], atol=1e-5
    )
    assert lvl0["cold"].attrs["colormap"] == "blues"


def test_write_front_pyramid_populates_consolidated_metadata(small_fields, tmp_path):
    # 2026-08-22: `datatree.to_zarr` alone writes a `consolidated_metadata`
    # block on every group node but leaves its `metadata` dict EMPTY (
    # confirmed live against a real store) -- the frontend's zarr client
    # needs it populated to skip a per-array metadata fetch per pyramid
    # level per class on every page load. write_front_pyramid now follows
    # the write with an explicit zarr.consolidate_metadata() call; this
    # checks the root zarr.json actually ends up with real per-array
    # entries (shape/chunk info), not just empty per-level placeholders.
    pyramid = build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=2)
    store_path = str(tmp_path / "test_pyramid.zarr")
    write_front_pyramid(pyramid, store_path)

    with open(f"{store_path}/zarr.json") as f:
        root = json.load(f)
    consolidated = root["consolidated_metadata"]["metadata"]
    assert "0/cold" in consolidated
    assert consolidated["0/cold"]["shape"] == [64, 64]
