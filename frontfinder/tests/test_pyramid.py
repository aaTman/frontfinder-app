import json
import os

import numpy as np
import pytest

from frontfinder.config.manifests import MODEL_1702_MANIFEST
from frontfinder.zarrio.pyramid import (
    PACKED_ADD_OFFSET,
    PACKED_FILL_VALUE,
    PACKED_SCALE_FACTOR,
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
def fields_with_extremes():
    # Deliberately puts 0, 1, 0.5 and NaN at known level-0 cells: the packed
    # uint8 grid (scale_factor 1/254 + 255 NaN sentinel) must round-trip the
    # endpoints exactly, preserve NaNs, and stay within the 0.004 budget
    # everywhere else.
    lat = np.linspace(25.0, 56.75, 64)
    lon = np.linspace(228.0, 299.75, 64)
    rng = np.random.default_rng(7)
    probs = {
        cls: rng.random((64, 64)).astype(np.float32) for cls in MODEL_1702_MANIFEST.served_classes
    }
    probs["cold"][0, 0] = 0.0
    probs["cold"][0, 1] = 1.0
    probs["cold"][0, 2] = 0.5
    probs["cold"][0, 3] = np.nan
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
    assert "dryline" in ds.data_vars
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
    write_front_pyramid(pyramid, store_path, pack_uint8=True)

    reopened = xr.open_datatree(store_path, engine="zarr")
    lvl0 = reopened["0"].to_dataset()
    # 2026-11-16: written as packed uint8 (scale_factor 1/254), so compare
    # against the 0.004 quantization budget rather than 1e-5 float equality.
    np.testing.assert_allclose(
        lvl0["cold"].values, small_fields.probabilities["cold"], atol=0.004
    )
    assert lvl0["cold"].attrs["colormap"] == "blues"


def test_write_front_pyramid_preserves_nan_and_endpoint_values(fields_with_extremes, tmp_path):
    import xarray as xr

    pyramid = build_front_pyramid(fields_with_extremes, MODEL_1702_MANIFEST, n_levels=2)
    store_path = str(tmp_path / "test_pyramid.zarr")
    write_front_pyramid(pyramid, store_path, pack_uint8=True)

    # open_datatree defaults to mask_and_scale=True in the zarr backend, so
    # this is exactly what the client-side decoder sees: float values after
    # applying scale_factor/add_offset and masking _FillValue.
    reopened = xr.open_datatree(store_path, engine="zarr")
    for cls in MODEL_1702_MANIFEST.served_classes:
        decoded = reopened["0"][cls].values
        expected = fields_with_extremes.probabilities[cls]

        # NaNs survive at exactly the same cells, rather than turning into a
        # valid probability (the 255 sentinel's whole point).
        np.testing.assert_array_equal(np.isnan(decoded), np.isnan(expected))

        finite = np.isfinite(expected)
        assert np.max(np.abs(decoded[finite] - expected[finite])) <= 0.004

    cold = reopened["0"]["cold"].values
    assert cold[0, 0] == 0.0
    assert cold[0, 1] == 1.0  # 1.0 still has its own code point (254/254)
    assert cold[0, 2] == 0.5  # 0.5 lands exactly on 127/254
    assert np.isnan(cold[0, 3])


def test_write_front_pyramid_packs_every_level_and_class_as_uint8(small_fields, tmp_path):
    pyramid = build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=3)
    store_path = str(tmp_path / "test_pyramid.zarr")
    write_front_pyramid(pyramid, store_path, pack_uint8=True)

    for level in ("0", "1", "2"):
        for cls in MODEL_1702_MANIFEST.served_classes:
            with open(os.path.join(store_path, level, cls, "zarr.json")) as f:
                metadata = json.load(f)
            assert metadata["data_type"] == "uint8", f"{level}/{cls}"
            assert metadata["attributes"]["scale_factor"] == pytest.approx(PACKED_SCALE_FACTOR)
            assert metadata["attributes"]["add_offset"] == PACKED_ADD_OFFSET
            assert metadata["attributes"]["_FillValue"] == PACKED_FILL_VALUE
        # Non-probability vars keep their native dtypes.
        with open(os.path.join(store_path, level, "lat", "zarr.json")) as f:
            assert json.load(f)["data_type"] == "float64"
        with open(os.path.join(store_path, level, "lon", "zarr.json")) as f:
            assert json.load(f)["data_type"] == "float64"
        with open(os.path.join(store_path, level, "spatial_ref", "zarr.json")) as f:
            assert json.load(f)["data_type"] == "int64"


def test_write_front_pyramid_consolidated_metadata_carries_packing_attrs(small_fields, tmp_path):
    # The client reads scale_factor/add_offset out of the CONSOLIDATED
    # metadata block, not the per-array zarr.json -- both must agree.
    pyramid = build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=2)
    store_path = str(tmp_path / "test_pyramid.zarr")
    write_front_pyramid(pyramid, store_path, pack_uint8=True)

    with open(os.path.join(store_path, "zarr.json")) as f:
        consolidated = json.load(f)["consolidated_metadata"]["metadata"]
    for level in ("0", "1"):
        entry = consolidated[f"{level}/cold"]
        assert entry["data_type"] == "uint8"
        assert entry["attributes"]["scale_factor"] == pytest.approx(PACKED_SCALE_FACTOR)
        assert entry["attributes"]["add_offset"] == PACKED_ADD_OFFSET
        assert entry["attributes"]["_FillValue"] == PACKED_FILL_VALUE


def test_quantized_level0_shard_is_much_smaller_than_float32(tmp_path):
    # A realistic sparse probability field: a coherent frontal ridge over a
    # field of tiny non-zero floats (softmax outputs are never exactly 0, and
    # those noisy mantissas are exactly what float32 zstd handles badly).
    n = 512
    lat = np.linspace(0.0, 63.0, n)
    lon = np.linspace(200.0, 263.0, n)
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:n, 0:n]
    ridge = np.exp(-((xx - 256) ** 2) / (2 * 30.0**2))
    base = np.clip(ridge * 0.8 + rng.random((n, n)) * 0.004, 0.0, 1.0)
    probs = {
        cls: (base * (0.5 + 0.5 * rng.random((n, n)))).astype(np.float32)
        for cls in MODEL_1702_MANIFEST.served_classes
    }
    fields = FrontFields(
        probabilities=probs, lat=lat, lon=lon, valid_time="t", cycle_time="t"
    )

    packed_store = str(tmp_path / "packed.zarr")
    write_front_pyramid(build_front_pyramid(fields, MODEL_1702_MANIFEST, n_levels=2), packed_store, pack_uint8=True)

    float_store = str(tmp_path / "float.zarr")
    float_pyramid = build_front_pyramid(fields, MODEL_1702_MANIFEST, n_levels=2)
    float_pyramid.datatree.to_zarr(float_store, encoding=float_pyramid.encoding, mode="w")

    def largest_level0_shard(store_path: str) -> int:
        shard_dir = os.path.join(store_path, "0", "cold", "c")
        sizes = [
            os.path.getsize(os.path.join(root, name))
            for root, _, names in os.walk(shard_dir)
            for name in names
        ]
        return max(sizes)

    assert largest_level0_shard(packed_store) < largest_level0_shard(float_store) * 0.5


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
    write_front_pyramid(pyramid, store_path, pack_uint8=True)

    with open(f"{store_path}/zarr.json") as f:
        root = json.load(f)
    consolidated = root["consolidated_metadata"]["metadata"]
    assert "0/cold" in consolidated
    assert consolidated["0/cold"]["shape"] == [64, 64]


def test_write_front_pyramid_is_float32_by_default(small_fields, tmp_path, monkeypatch):
    # Packing is opt-in (see PACK_UINT8_ENV): until the dev viewer is promoted,
    # the live write path must keep producing exactly what production serves.
    import xarray as xr

    monkeypatch.delenv("FRONTFINDER_PACK_UINT8", raising=False)
    store_path = str(tmp_path / "default.zarr")
    write_front_pyramid(build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=2), store_path)
    ds = xr.open_zarr(store_path, group="0", consolidated=False, mask_and_scale=False)
    assert ds["cold"].dtype == np.float32
    assert "scale_factor" not in ds["cold"].attrs


def test_write_front_pyramid_env_var_opts_into_uint8(small_fields, tmp_path, monkeypatch):
    import xarray as xr

    monkeypatch.setenv("FRONTFINDER_PACK_UINT8", "1")
    store_path = str(tmp_path / "env.zarr")
    write_front_pyramid(build_front_pyramid(small_fields, MODEL_1702_MANIFEST, n_levels=2), store_path)
    ds = xr.open_zarr(store_path, group="0", consolidated=False, mask_and_scale=False)
    assert ds["cold"].dtype == np.uint8
