import json
import os
import time

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from frontfinder.postprocess import pressure

LAT = np.linspace(80.0, -80.0, 321)  # descending, 0.5 deg, like the IFS grid
LON = np.linspace(0.0, 359.5, 720)
SIGNED_LON = np.linspace(-180.0, 179.5, 720)


def synthetic(lat, lon, centres):
    """hPa field: a 1013 background plus Gaussian bumps `(lon_deg, lat_deg, amplitude, sigma_deg)`."""
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    field = np.full(lat_grid.shape, 1013.0)
    for lon0, lat0, amplitude, sigma in centres:
        dlon = (lon_grid - lon0 + 180.0) % 360.0 - 180.0  # the grid wraps, the bump does not split
        dlat = lat_grid - lat0
        field += amplitude * np.exp(-((dlon * np.cos(np.radians(lat0))) ** 2 + dlat**2) / (2 * sigma**2))
    return field


LOW = (200.0, 50.0, -30.0, 8.0)
HIGH = (330.0, 30.0, 20.0, 8.0)


def centres_of(collection, kind):
    return [f for f in collection["features"] if f["properties"]["kind"] == kind]


def isobars_of(collection, hpa):
    return [f for f in collection["features"] if f["properties"]["kind"] == "isobar" and f["properties"]["hpa"] == hpa]


def two_lows(saddle_hpa: float) -> np.ndarray:
    """hPa field with a deep low at lat 60, a shallow one at lat 38 (2400 km apart, both
    inside each other's window) and a saddle of `saddle_hpa` between them.

    The profile runs along the 200 degree meridian and fades out with longitude, so the
    two lows are separate features rather than one band around the globe. A monotone
    cubic keeps the profile from wobbling between the keypoints.
    """
    profile = PchipInterpolator(
        np.array([-80.0, -22.0, 22.0, 30.0, 38.0, 48.0, 60.0, 68.0, 80.0]),
        np.array([1013.0, 1013.0, 1013.0, 1004.0, 994.0, saddle_hpa, 989.0, 1000.0, 1013.0]),
    )
    lon_grid, lat_grid = np.meshgrid(LON, LAT)
    dx = (((lon_grid - 200.0 + 180.0) % 360.0) - 180.0) * np.cos(np.radians(lat_grid))
    around = np.exp(-dx**2 / (2 * 20.0**2))
    return 1013.0 + (profile(LAT) - 1013.0)[:, None] * around


def shape_key(feature) -> tuple:
    """A line's coordinates with longitudes folded to [-180, 180), so the two copies of one
    contour on either side of the seam come out identical."""
    coordinates = feature["geometry"]["coordinates"]
    return tuple(sorted((round((lon + 180.0) % 360.0 - 180.0, 2), round(lat, 2)) for lon, lat in coordinates))


def test_pressure_filename_swaps_the_zarr_suffix():
    assert pressure.pressure_filename("2026-10-31T06Z_f006.zarr") == "2026-10-31T06Z_f006.pressure.geojson"
    with pytest.raises(ValueError):
        pressure.pressure_filename("latest.json")


def test_build_pressure_geojson_finds_the_low_and_the_high():
    field = synthetic(LAT, LON, [LOW, HIGH])
    collection = pressure.build_pressure_geojson(field * 100.0, LAT, LON, "2026-10-31T06:00:00")
    assert collection["type"] == "FeatureCollection"
    assert collection["valid_time"] == "2026-10-31T06:00:00"

    lows, highs = centres_of(collection, "L"), centres_of(collection, "H")
    assert len(lows) == 1 and len(highs) == 1
    # lon 200 on the 0..360 grid is -160 in the file's convention
    assert lows[0]["geometry"]["type"] == "Point"
    assert abs(lows[0]["geometry"]["coordinates"][0] - -160.0) <= 2.0
    assert abs(lows[0]["geometry"]["coordinates"][1] - 50.0) <= 2.0
    assert lows[0]["properties"]["hpa"] == pytest.approx(983, abs=2)
    assert abs(highs[0]["geometry"]["coordinates"][0] - -30.0) <= 2.0
    assert abs(highs[0]["geometry"]["coordinates"][1] - 30.0) <= 2.0
    assert highs[0]["properties"]["hpa"] == pytest.approx(1033, abs=2)

    lines = [f for f in collection["features"] if f["properties"]["kind"] == "isobar"]
    assert lines and all(len(f["geometry"]["coordinates"]) >= 3 for f in lines)
    assert all(f["geometry"]["type"] == "LineString" for f in lines)
    assert all(f["properties"]["hpa"] % 4 == 0 for f in lines)
    assert all(f["properties"]["hpa"] == int(f["properties"]["hpa"]) for f in lines)
    for hpa, major in ((1000, True), (1020, True), (1004, False)):
        present = isobars_of(collection, hpa)
        assert present, f"no isobar at {hpa}"
        assert {f["properties"]["major"] for f in present} == {major}
    # isobars first, then the centres, L before H
    kinds = [f["properties"]["kind"] for f in collection["features"]]
    assert kinds[: len(lines)] == ["isobar"] * len(lines)
    assert kinds[len(lines) :] == ["L", "H"]


def test_seam_spanning_low_is_found_once_with_single_piece_isobars():
    field = synthetic(LAT, LON, [(359.5, 45.0, -30.0, 8.0)])
    collection = pressure.build_pressure_geojson(field * 100.0, LAT, LON, None)
    assert len(centres_of(collection, "L")) == 1  # not one per side of the grid

    levels = {f["properties"]["hpa"] for f in collection["features"] if f["properties"]["kind"] == "isobar"}
    assert levels
    for hpa in levels:
        lines = isobars_of(collection, hpa)
        keys = [shape_key(f) for f in lines]
        assert len(keys) == len(set(keys)) == 1, f"isobar at {hpa} is not one piece"
        assert -180.0 <= lines[0]["geometry"]["coordinates"][0][0] < 180.0

    loop = max(isobars_of(collection, 1000), key=lambda f: len(f["geometry"]["coordinates"]))["geometry"]["coordinates"]
    assert loop[0] == loop[-1]  # closed isobars repeat their first coordinate
    assert len([c for c in loop if c == loop[0]]) == 2
    assert max(c[0] for c in loop) - min(c[0] for c in loop) < 180.0  # carries on, not cut in half


def test_antimeridian_lines_are_never_cut():
    field = synthetic(LAT, LON, [(180.0, 45.0, -30.0, 8.0)])  # straddles the date line
    collection = pressure.build_pressure_geojson(field * 100.0, LAT, LON, None)
    lines = [f for f in collection["features"] if f["properties"]["kind"] == "isobar"]
    assert lines
    for line in lines:
        assert -180.0 <= line["geometry"]["coordinates"][0][0] < 180.0
    spanning = [f for f in lines if any(lon > 180.0 or lon < -180.0 for lon, _ in f["geometry"]["coordinates"])]
    assert spanning, "no isobar carries on across 180"


def test_globe_spanning_isobar_stays_one_line():
    # A zonal gradient (isobars are circles of latitude) crosses every longitude on
    # its way round, so it has no padded copy to keep and must not be dropped.
    lat = np.linspace(80.0, -80.0, 321)
    _, lat_grid = np.meshgrid(LON, lat)
    lines = pressure.isobars(1013.0 + 30.0 * np.sin(np.radians(lat_grid)), lat, LON)
    assert lines
    for line in lines:
        coordinates = line["geometry"]["coordinates"]
        assert -180.0 <= coordinates[0][0] < 180.0
        lons = [c[0] for c in coordinates]
        assert max(lons) - min(lons) > 300.0  # carries round the world, never cut


def test_flat_field_has_no_centres_and_no_isobars():
    field = np.full((LAT.size, LON.size), 1013.0)
    collection = pressure.build_pressure_geojson(field * 100.0, LAT, LON, None)
    assert collection["features"] == []


def test_surface_pressure_hides_centres_but_not_isobars():
    field = synthetic(LAT, LON, [LOW, HIGH])
    plateau = np.full(field.shape, 60000.0)  # 600 hPa, above any low-level analysis
    collection = pressure.build_pressure_geojson(field * 100.0, LAT, LON, None, surface_pressure_pa=plateau)
    assert not centres_of(collection, "L")
    assert not centres_of(collection, "H")
    assert [f for f in collection["features"] if f["properties"]["kind"] == "isobar"]


def test_flat_bottomed_low_gives_one_centre_not_one_per_plateau_pixel():
    # Clipping the field flattens the bottom of the low: every pixel of that plateau is the same
    # local minimum, so the thinning (not the depth test) is what leaves a single centre.
    hpa = np.maximum(synthetic(LAT, LON, [LOW, HIGH]), 983.5)  # a 3-degree-wide flat bottom
    lows = pressure.pressure_centers(hpa, LAT, LON)
    assert [f["properties"]["kind"] for f in lows] == ["L", "H"]
    assert abs(lows[0]["geometry"]["coordinates"][0] - -160.0) <= 2.0
    assert lows[0]["properties"]["hpa"] == 984  # the plateau's own value


def test_a_bump_shallower_than_one_isobar_is_not_a_centre():
    # Prominence, not depth under a ring: a bump the field never rises a full isobar
    # interval around has no closed isobar of its own, so no centre.
    shallow = synthetic(LAT, LON, [(200.0, 50.0, -2.0, 8.0)])
    assert pressure.pressure_centers(shallow, LAT, LON) == []
    deep = pressure.pressure_centers(synthetic(LAT, LON, [(200.0, 50.0, -6.0, 8.0)]), LAT, LON)
    assert [f["properties"]["kind"] for f in deep] == ["L"]
    assert deep[0]["geometry"]["coordinates"] == [-160.0, 50.0]
    assert deep[0]["properties"]["hpa"] == 1007


def test_a_broad_high_is_a_centre_whatever_its_ring_mean_depth_is():
    # The subtropical highs, which is what the ring-mean depth test deleted: +8 hPa spread
    # over 20 degrees is barely below the mean of a 6 degree ring (its whole depth is a few
    # hPa measured that way), but its 1017 hPa isobar closes around itself and around
    # nothing else, which is exactly what makes it a high.
    highs = pressure.pressure_centers(synthetic(LAT, LON, [(120.0, -20.0, 8.0, 20.0)]), LAT, LON)
    assert [f["properties"]["kind"] for f in highs] == ["H"]
    assert abs(highs[0]["geometry"]["coordinates"][0] - 120.0) <= 2.0
    assert abs(highs[0]["geometry"]["coordinates"][1] + 20.0) <= 2.0
    assert highs[0]["properties"]["hpa"] == 1021


def test_a_low_that_drains_over_a_shallow_saddle_is_not_a_centre():
    # Only the lows: the field's 1013 background is the highest ground anywhere, so what
    # the highs of this field say is a question about the background, not about the saddle.
    def lows(hpa):
        centres = pressure.pressure_centers(hpa, LAT, LON)
        return [f["properties"]["hpa"] for f in centres if f["properties"]["kind"] == "L"]

    # 2 hPa of saddle above the shallow low: the water poured in around it slips over that
    # saddle into the deeper low, so it has no closed isobar and no centre. The deeper low
    # is boxed in by ground 7 hPa above it and keeps its own.
    assert lows(two_lows(996.0)) == [989]
    # 6 hPa of saddle and the shallow low encloses an isobar of its own, so both are centres
    assert lows(two_lows(1000.0)) == [989, 994]


def test_ascending_latitude_and_signed_longitude_give_the_same_centres():
    descending = synthetic(LAT, LON, [LOW, HIGH])
    ascending = synthetic(LAT[::-1], LON, [LOW, HIGH])
    signed = synthetic(LAT, SIGNED_LON, [LOW, HIGH])
    expected = pressure.pressure_centers(descending, LAT, LON)
    assert pressure.pressure_centers(ascending, LAT[::-1], LON) == expected
    assert pressure.pressure_centers(signed, LAT, SIGNED_LON) == expected


def test_write_pressure_layer_and_attach_pressure(tmp_path):
    field = synthetic(LAT, LON, [LOW, HIGH])
    store = "2026-10-31T06Z_f006.zarr"
    name = pressure.write_pressure_layer(str(tmp_path), store, field * 100.0, LAT, LON, "2026-10-31T06:00:00")
    assert name == "2026-10-31T06Z_f006.pressure.geojson"
    path = tmp_path / name
    assert json.loads(path.read_text())["valid_time"] == "2026-10-31T06:00:00"
    assert path.read_text().startswith('{"type":"FeatureCollection","valid_time"')  # compact, no spaces
    assert not (tmp_path / (name + ".tmp")).exists()

    # a missing directory or a field that does not fit the grid is logged, not raised
    assert pressure.write_pressure_layer(str(tmp_path / "nope"), store, field * 100.0, LAT, LON, None) is None
    assert pressure.write_pressure_layer(str(tmp_path), store, field[:4, :4] * 100.0, LAT, LON, None) is None
    assert not (tmp_path / (name + ".tmp")).exists()

    entries = [{"store": store}, {"store": "2026-10-31T06Z_f012.zarr"}, {"store": "latest.json"}, {}]
    pressure.attach_pressure(str(tmp_path), entries)
    assert entries[0]["pressure"] == name
    assert "pressure" not in entries[1]
    assert "pressure" not in entries[2] and "pressure" not in entries[3]
    os.remove(path)
    pressure.attach_pressure(str(tmp_path), entries)
    assert "pressure" not in entries[0]


def test_global_field_stays_small_and_fast():
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.linspace(0.0, 359.75, 1440)
    field = synthetic(lat, lon, [LOW, HIGH])
    field += np.random.default_rng(0).normal(0.0, 0.3, field.shape)
    started = time.perf_counter()
    collection = pressure.build_pressure_geojson(field * 100.0, lat, lon, "2026-10-31T06:00:00")
    elapsed = time.perf_counter() - started
    size = len(json.dumps(collection, separators=(",", ":")).encode())
    print(f"\nglobal field: {size / 1024:.1f} kB, {elapsed:.2f} s, {len(collection['features'])} features")
    assert size < 400_000
    assert elapsed < 3.0
