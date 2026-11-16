import json

import numpy as np
import pytest

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_MANIFEST as MANIFEST
from frontfinder.postprocess import front_lines
from frontfinder.zarrio.pyramid import FrontFields, build_front_pyramid, write_front_pyramid

LAT = np.linspace(55.0, 25.0, 61)  # descending, 0.5 deg, like the IFS grid
LON = np.linspace(228.0, 298.0, 141)  # 0.5 deg, regional


def write_store(model_dir, store_name, valid_time, cold_row: int | None):
    """A tiny published store whose only front is a cold ridge along row `cold_row`."""
    probs = {cls: np.zeros((len(LAT), len(LON)), dtype=np.float32) for cls in MANIFEST.served_classes}
    if cold_row is not None:
        probs["cold"][cold_row, 20:120] = 0.9
    fields = FrontFields(probabilities=probs, lat=LAT, lon=LON, valid_time=valid_time, cycle_time="2026-10-31T06:00:00")
    write_front_pyramid(build_front_pyramid(fields, MANIFEST, n_levels=2), str(model_dir / store_name))


def entry(store_name, valid_time):
    return {"step_hours": 0, "valid_time": valid_time, "store": store_name}


def test_fronts_filename_swaps_the_zarr_suffix():
    assert front_lines.fronts_filename("2026-10-31T06Z_f006.zarr") == "2026-10-31T06Z_f006.fronts.geojson"
    with pytest.raises(ValueError):
        front_lines.fronts_filename("latest.json")


def test_load_probabilities_uses_splines_layout(tmp_path):
    write_store(tmp_path, "2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00", cold_row=20)
    ds = front_lines.load_probabilities(str(tmp_path / "2026-10-31T06Z_f000.zarr"))
    assert set(ds.data_vars) == {"CF", "WF", "SF", "OF", "DL", "TROF", "TT"}
    assert set(ds.dims) == {"latitude", "longitude"}
    assert float(ds["CF"].max()) == pytest.approx(0.9)


def test_write_front_lines_writes_a_file_per_step_and_tracks_ids(tmp_path):
    write_store(tmp_path, "2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00", cold_row=20)
    write_store(tmp_path, "2026-10-31T06Z_f006.zarr", "2026-10-31T12:00:00", cold_row=21)  # drifted 0.5 deg
    entries = [
        entry("2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00"),
        entry("2026-10-31T06Z_f006.zarr", "2026-10-31T12:00:00"),
    ]
    written = front_lines.write_front_lines(str(tmp_path), entries)
    assert written == {
        "2026-10-31T06Z_f000.zarr": "2026-10-31T06Z_f000.fronts.geojson",
        "2026-10-31T06Z_f006.zarr": "2026-10-31T06Z_f006.fronts.geojson",
    }
    first, second = (json.loads((tmp_path / n).read_text()) for n in written.values())
    for collection, valid_time in ((first, "2026-10-31T06:00:00"), (second, "2026-10-31T12:00:00")):
        assert collection["type"] == "FeatureCollection"
        assert collection["valid_time"] == valid_time
        names = {f["properties"]["name"] for f in collection["features"]}
        assert names == {"cold_front"}  # no overall_geometry duplicate
        assert all(f["geometry"]["type"] == "MultiLineString" for f in collection["features"])
    # the same front keeps its event id from one step to the next
    assert {f["properties"]["event_id"] for f in first["features"]} == {
        f["properties"]["event_id"] for f in second["features"]
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_a_quiet_step_gets_an_empty_collection(tmp_path):
    write_store(tmp_path, "2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00", cold_row=None)
    written = front_lines.write_front_lines(str(tmp_path), [entry("2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00")])
    assert json.loads((tmp_path / written["2026-10-31T06Z_f000.zarr"]).read_text())["features"] == []


def test_an_unreadable_step_is_skipped_without_losing_the_others(tmp_path):
    write_store(tmp_path, "2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00", cold_row=20)
    write_store(tmp_path, "2026-10-31T06Z_f012.zarr", "2026-10-31T18:00:00", cold_row=20)
    entries = [
        entry("2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00"),
        entry("2026-10-31T06Z_f006.zarr", "2026-10-31T12:00:00"),  # never written
        entry("2026-10-31T06Z_f012.zarr", "2026-10-31T18:00:00"),
    ]
    written = front_lines.write_front_lines(str(tmp_path), entries)
    assert set(written) == {"2026-10-31T06Z_f000.zarr", "2026-10-31T06Z_f012.zarr"}


def test_attach_sets_and_clears_the_fronts_key_and_never_raises(tmp_path, monkeypatch):
    write_store(tmp_path, "2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00", cold_row=20)
    entries = [
        entry("2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00"),
        {**entry("2026-10-31T06Z_f006.zarr", "2026-10-31T12:00:00"), "fronts": "stale.fronts.geojson"},
    ]
    front_lines.attach_front_lines(str(tmp_path), entries)
    assert entries[0]["fronts"] == "2026-10-31T06Z_f000.fronts.geojson"
    assert "fronts" not in entries[1]  # store missing -> stale pointer removed

    def boom(*args, **kwargs):
        raise RuntimeError("scikit-image exploded")

    monkeypatch.setattr(front_lines, "write_front_lines", boom)
    front_lines.attach_front_lines(str(tmp_path), entries)  # must not raise
    assert all("fronts" not in e for e in entries)


def test_backfill_adds_fronts_to_latest_json(tmp_path):
    model_dir = tmp_path / "some_model"
    model_dir.mkdir()
    write_store(model_dir, "2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00", cold_row=20)
    pointer = {"cycle_time": "2026-10-31T06:00:00", "steps": [entry("2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00")]}
    (model_dir / "latest.json").write_text(json.dumps(pointer))
    assert front_lines.backfill(str(tmp_path)) == {"some_model": 1}
    updated = json.loads((model_dir / "latest.json").read_text())
    assert updated["steps"][0]["fronts"] == "2026-10-31T06Z_f000.fronts.geojson"
    assert updated["cycle_time"] == "2026-10-31T06:00:00"


def test_backfill_leaves_a_pointer_alone_when_its_stores_are_gone(tmp_path):
    model_dir = tmp_path / "retired_model"
    model_dir.mkdir()
    pointer_text = json.dumps(
        {"cycle_time": "2026-10-08T06:00:00", "steps": [entry("2026-10-08T06Z_f000.zarr", "2026-10-08T06:00:00")]}
    )
    (model_dir / "latest.json").write_text(pointer_text)
    assert front_lines.backfill(str(tmp_path)) == {"retired_model": 0}
    assert (model_dir / "latest.json").read_text() == pointer_text  # byte-for-byte untouched
    assert not list(model_dir.glob("*.geojson"))


# --- symbol orientation fields (the .fields.npz sidecar) ---


def test_fields_filename_swaps_the_zarr_suffix():
    assert front_lines.fields_filename("2026-10-31T06Z_f006.zarr") == "2026-10-31T06Z_f006.fields.npz"
    with pytest.raises(ValueError):
        front_lines.fields_filename("latest.json")


def test_extract_orientation_fields_picks_the_850_channels_rolls_longitude_and_copies():
    names = MANIFEST.channel_names()
    lat = np.linspace(89.875, -89.875, 8)
    lon = np.linspace(0.0, 360.0, 16, endpoint=False)  # global, IFS-native 0..360
    grid = np.zeros((8, 16, len(names)), dtype=np.float32)
    for k, name in enumerate(front_lines.ORIENTATION_CHANNELS.values()):
        grid[..., names.index(name)] = (k + 1) * 100 + np.arange(16)[None, :]
    out = front_lines.extract_orientation_fields(MANIFEST, grid, lat, lon)
    assert set(out) == {"temperature", "u", "v", "lat", "lon"}
    assert out["lon"][0] == -180 and np.all(np.diff(out["lon"]) > 0) and out["lon"][-1] < 180
    assert out["temperature"][0, 0] == 100 + 8  # column 8 (180E) is first after the roll
    assert out["u"][0, 0] == 200 + 8 and out["v"][0, 0] == 300 + 8
    grid[:] = -1  # the sidecar must not alias the (reusable) input grid
    assert out["temperature"][0, 0] == 108


def test_extract_orientation_fields_is_none_for_a_model_without_those_channels():
    from frontfinder.config.manifests import ModelManifest, VariableSpec

    only_u = ModelManifest(
        name="x", weights_filename="x.keras", variables=(VariableSpec("u_component_of_wind", levels=(850,)),)
    )
    assert front_lines.extract_orientation_fields(only_u, np.zeros((8, 16, 1), np.float32), np.arange(8), np.arange(16)) is None


def test_orientation_fields_roundtrip_and_a_missing_or_corrupt_file_is_empty(tmp_path):
    fields = {
        "temperature": np.full((3, 4), 300.0, np.float32), "u": np.ones((3, 4), np.float32),
        "v": np.zeros((3, 4), np.float32), "lat": np.array([10.0, 5.0, 0.0], np.float32),
        "lon": np.array([0.0, 1.0, 2.0, 3.0], np.float32),
    }
    path = str(tmp_path / "a.fields.npz")
    front_lines.save_orientation_fields(fields, path)
    loaded = front_lines.load_orientation_fields(path)
    assert set(loaded) == {"temperature", "u", "v"}
    assert loaded["temperature"].dims == ("latitude", "longitude")
    assert list(loaded["temperature"]["latitude"].values) == [10.0, 5.0, 0.0]
    assert front_lines.load_orientation_fields(str(tmp_path / "missing.npz")) == {}
    (tmp_path / "bad.fields.npz").write_bytes(b"not an npz")
    assert front_lines.load_orientation_fields(str(tmp_path / "bad.fields.npz")) == {}


@pytest.mark.parametrize("warm_south", [True, False])
def test_a_cold_front_gets_its_symbols_on_the_warm_air_side(tmp_path, warm_south):
    store = "2026-10-31T06Z_f000.zarr"
    write_store(tmp_path, store, "2026-10-31T06:00:00", cold_row=20)  # a west-east ridge at 45N
    sign = 1.0 if warm_south else -1.0
    theta_e = 300.0 + sign * np.where(LAT[:, None] < 45.0, 8.0, -8.0) * np.ones((1, len(LON)))
    zeros = np.zeros_like(theta_e, dtype=np.float32)
    front_lines.save_orientation_fields(
        {"temperature": theta_e.astype(np.float32), "u": zeros, "v": zeros,
         "lat": LAT.astype(np.float32), "lon": LON.astype(np.float32)},
        str(tmp_path / front_lines.fields_filename(store)),
    )
    written = front_lines.write_front_lines(str(tmp_path), [entry(store, "2026-10-31T06:00:00")])
    collection = json.loads((tmp_path / written[store]).read_text())
    (feature,) = [f for f in collection["features"] if f["properties"]["name"] == "cold_front"]
    assert collection["oriented"] is True  # the sidecar was used
    sides = feature["properties"]["summary"]["symbol_sides"]
    coords = feature["geometry"]["coordinates"]
    assert len(sides) == len(coords) == 1
    eastward = coords[0][-1][0] > coords[0][0][0]  # walking east: left is north, right is south
    toward_warm = ("right" if eastward else "left") if warm_south else ("left" if eastward else "right")
    assert sides == [toward_warm]


def test_without_a_sidecar_the_step_still_gets_lines(tmp_path):
    store = "2026-10-31T06Z_f000.zarr"
    write_store(tmp_path, store, "2026-10-31T06:00:00", cold_row=20)
    written = front_lines.write_front_lines(str(tmp_path), [entry(store, "2026-10-31T06:00:00")])
    assert store in written
    assert json.loads((tmp_path / written[store]).read_text())["oriented"] is False  # so the viewer draws no symbols
