import itertools
import json
import pathlib

import numpy as np
import pytest
import xarray as xr

from frontfinder.postprocess import splines

# Descending latitude, like ERA5 and IFS grids.
LAT = np.arange(60.0, 19.99, -0.25)
LON = np.arange(230.0, 290.0, 0.25)
CONFIG = splines.SplineConfig()


def ridge(
    points: list[tuple[float, float]], width: float = 0.75, peak: float = 0.9, lon: np.ndarray = LON
) -> np.ndarray:
    """Gaussian probability band around a polyline of (lon, lat) points."""
    lat2d, lon2d = np.meshgrid(LAT, lon, indexing="ij")
    dist = np.full(lat2d.shape, np.inf)
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((lon2d - x0) * dx + (lat2d - y0) * dy) / (dx**2 + dy**2), 0, 1)
        dist = np.minimum(dist, np.hypot(lon2d - (x0 + t * dx), lat2d - (y0 + t * dy)))
    return peak * np.exp(-(dist**2) / (2 * width**2))


def grid_field(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values, dims=["latitude", "longitude"], coords={"latitude": LAT, "longitude": LON})


def probabilities(**fields: np.ndarray) -> xr.Dataset:
    return xr.Dataset({name: grid_field(values) for name, values in fields.items()})


def warm_south() -> xr.DataArray:
    """Temperature falling 0.5 K per degree of latitude toward the pole."""
    return grid_field(np.broadcast_to(300.0 - 0.5 * (LAT[:, None] - 20.0), (len(LAT), len(LON))).copy())


def constant(value: float) -> xr.DataArray:
    return grid_field(np.full((len(LAT), len(LON)), value))


def symbols_point_north(section: splines.FrontSection) -> bool:
    _, north = splines.left_normals(section.lon, section.lat)
    sign = 1.0 if section.symbol_side == "left" else -1.0
    return float(np.mean(north)) * sign > 0


def mask_from(pixels: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((len(LAT), len(LON)), dtype=bool)
    for row, col in pixels:
        mask[row, col] = True
    return mask


# --- crest_mask ---


def test_crest_mask_follows_band_center() -> None:
    mask = splines.crest_mask(ridge([(240.0, 40.0), (270.0, 40.0)]), 0.3)
    rows, cols = np.nonzero(mask)
    inner = (LON[cols] > 242) & (LON[cols] < 268)
    assert inner.sum() > 80
    assert np.all(np.abs(LAT[rows[inner]] - 40.0) <= 0.25)


def test_crest_mask_ignores_nan() -> None:
    probs = ridge([(240.0, 40.0), (270.0, 40.0)])
    probs[:5] = np.nan
    assert splines.crest_mask(probs, 0.3).any()


# --- trace_paths ---


def test_trace_paths_orders_an_l_shape() -> None:
    pixels = [(10, c) for c in range(5, 26)] + [(r, 25) for r in range(11, 31)]
    crests = splines.trace_paths(mask_from(pixels), LAT, LON, min_length_km=100.0)
    assert len(crests) == 1
    assert len(crests[0]) == 1
    path = crests[0][0]
    assert len(path) == len(pixels)
    assert {tuple(path[0]), tuple(path[-1])} == {(10, 5), (30, 25)}
    assert np.all(np.abs(np.diff(path, axis=0)).max(axis=1) == 1)


def test_trace_paths_drops_short_spurs() -> None:
    pixels = [(10, c) for c in range(0, 61)] + [(r, 30) for r in range(11, 14)]
    crests = splines.trace_paths(mask_from(pixels), LAT, LON, min_length_km=250.0)
    assert [len(path) for path in crests[0]] == [61]


def test_trace_paths_keeps_long_branches_joined_to_the_main_path() -> None:
    # At 35N each arm of the row (~680 km) is longer than the branch (~556 km), so the row is the main path.
    pixels = [(100, c) for c in range(0, 61)] + [(r, 30) for r in range(101, 121)]
    crests = splines.trace_paths(mask_from(pixels), LAT, LON, min_length_km=250.0)
    main, branch = crests[0]
    assert len(main) == 61
    assert tuple(branch[0]) == (100, 30)
    assert tuple(branch[-1]) == (120, 30)


def test_trace_paths_opens_loops() -> None:
    pixels = sorted({(r, c) for r in range(10, 31) for c in range(10, 31) if r in (10, 30) or c in (10, 30)})
    crests = splines.trace_paths(mask_from(pixels), LAT, LON, min_length_km=100.0)
    assert len(crests[0][0]) == len(pixels)


def test_trace_paths_drops_short_crests() -> None:
    crests = splines.trace_paths(mask_from([(10, c) for c in range(5)]), LAT, LON, min_length_km=250.0)
    assert crests == []


# --- geometry helpers ---


def test_haversine_one_degree_of_latitude() -> None:
    assert splines.haversine_km(0.0, 40.0, 0.0, 41.0) == pytest.approx(111.19, abs=0.1)


def test_line_length_km() -> None:
    lat = np.arange(30.0, 40.01, 0.5)
    assert splines.line_length_km(np.full_like(lat, 250.0), lat) == pytest.approx(1111.9, rel=1e-3)


def test_left_normals_point_north_when_heading_east() -> None:
    lon = np.arange(240.0, 250.0, 1.0)
    east, north = splines.left_normals(lon, np.full_like(lon, 40.0))
    np.testing.assert_allclose(east, 0.0, atol=1e-12)
    np.testing.assert_allclose(north, 1.0)


def test_smooth_line_removes_zigzag_and_spaces_points_evenly() -> None:
    lon = np.arange(240.0, 260.01, 0.25)
    lat = 40.0 + np.where(np.arange(len(lon)) % 2 == 0, 0.25, -0.25)
    slon, slat = splines.smooth_line(lon, lat, smoothing_deg=0.25, spacing_km=25.0)
    steps = splines.haversine_km(slon[:-1], slat[:-1], slon[1:], slat[1:])
    assert np.max(np.abs(slat - 40.0)) < 0.15
    assert np.all(np.abs(steps - 25.0) < 2.5)
    assert abs(slon[0] - 240.0) < 0.3
    assert abs(slon[-1] - 260.0) < 0.3


def test_smooth_line_handles_two_points() -> None:
    slon, slat = splines.smooth_line(np.array([250.0, 251.0]), np.array([40.0, 40.0]), 0.25, 25.0)
    assert len(slon) >= 2
    np.testing.assert_allclose(slat, 40.0)


def test_split_sections_folds_short_blips_into_neighbours() -> None:
    lon = np.arange(240.0, 260.01, 0.25)
    lat = np.full_like(lon, 40.0)
    cold = np.where(lon < 252.0, 0.8, 0.1)
    cold[(lon >= 245.0) & (lon < 245.5)] = 0.0  # ~43 km of "warm" inside the cold stretch
    probs = np.stack([cold, 1.0 - cold])
    sections = splines.split_sections(probs, lon, lat, min_section_km=100.0)
    assert [k for k, _, _ in sections] == [0, 1]
    (_, start0, stop0), (_, start1, stop1) = sections
    assert (start0, stop1) == (0, len(lon) - 1)
    assert stop0 == start1
    assert lon[start1] == pytest.approx(252.0)


def test_extract_fronts_single_cold_front() -> None:
    (boundary,) = splines.extract_fronts(probabilities(CF=ridge([(240.0, 40.0), (270.0, 40.0)])), CONFIG)
    ((section,),) = boundary.lines
    assert section.front_type == "CF"
    assert np.all(np.abs(section.lat - 40.0) < 0.2)
    assert splines.line_length_km(section.lon, section.lat) > 2000
    assert np.all((section.probability >= 0) & (section.probability <= 1))


def test_extract_fronts_splits_one_boundary_by_type() -> None:
    band = ridge([(240.0, 40.0), (270.0, 40.0)])
    west = LON[None, :] < 255.0
    probs = probabilities(CF=np.where(west, band, 0.1 * band), WF=np.where(west, 0.1 * band, band))
    (boundary,) = splines.extract_fronts(probs, CONFIG)
    (line,) = boundary.lines
    by_lon = sorted(line, key=lambda s: float(np.mean(s.lon)))
    assert [s.front_type for s in by_lon] == ["CF", "WF"]
    assert by_lon[0].lon[-1] == pytest.approx(by_lon[1].lon[0]) or by_lon[0].lon[0] == pytest.approx(by_lon[1].lon[-1])


def test_extract_fronts_keeps_separate_bands_apart() -> None:
    probs = probabilities(CF=ridge([(240.0, 30.0), (270.0, 30.0)]) + ridge([(240.0, 50.0), (270.0, 50.0)]))
    assert len(splines.extract_fronts(probs, CONFIG)) == 2


def test_extract_fronts_drops_short_bands() -> None:
    assert splines.extract_fronts(probabilities(CF=ridge([(250.0, 40.0), (251.5, 40.0)])), CONFIG) == []


def test_extract_fronts_keeps_groups_apart() -> None:
    probs = probabilities(CF=ridge([(240.0, 30.0), (270.0, 30.0)]), DL=ridge([(255.0, 35.0), (255.0, 55.0)]))
    types = sorted(s.front_type for b in splines.extract_fronts(probs, CONFIG) for line in b.lines for s in line)
    assert types == ["CF", "DL"]


@pytest.mark.parametrize(("front_type", "north"), [("CF", False), ("WF", True), ("SF", False)])
def test_symbols_follow_temperature_contrast(front_type: str, north: bool) -> None:
    probs = probabilities(**{front_type: ridge([(240.0, 40.0), (270.0, 40.0)])})
    (boundary,) = splines.extract_fronts(probs, CONFIG, temperature=warm_south())
    ((section,),) = boundary.lines
    assert symbols_point_north(section) is north


@pytest.mark.parametrize(("v", "north"), [(5.0, True), (-5.0, False)])
def test_occluded_symbols_follow_cross_front_wind(v: float, north: bool) -> None:
    probs = probabilities(OF=ridge([(240.0, 40.0), (270.0, 40.0)]))
    (boundary,) = splines.extract_fronts(probs, CONFIG, temperature=warm_south(), u=constant(0.0), v=constant(v))
    ((section,),) = boundary.lines
    assert symbols_point_north(section) is north


@pytest.mark.parametrize(("previous_lat", "north"), [(41.0, False), (39.0, True)])
def test_symbols_follow_motion_since_previous_time(previous_lat: float, north: bool) -> None:
    previous = splines.extract_fronts(probabilities(CF=ridge([(240.0, previous_lat), (270.0, previous_lat)])), CONFIG)
    probs = probabilities(CF=ridge([(240.0, 40.0), (270.0, 40.0)]))
    (boundary,) = splines.extract_fronts(probs, CONFIG, previous=previous)
    ((section,),) = boundary.lines
    assert symbols_point_north(section) is north


# --- to_geojson ---


def test_to_geojson_layout() -> None:
    band = ridge([(240.0, 40.0), (270.0, 40.0)])
    west = LON[None, :] < 255.0
    probs = probabilities(CF=np.where(west, band, 0.1 * band), WF=np.where(west, 0.1 * band, band))
    collection = splines.to_geojson(splines.extract_fronts(probs, CONFIG), valid_time="2024-01-01T00:00:00")
    json.dumps(collection)
    assert collection["type"] == "FeatureCollection"
    names = [f["properties"]["name"] for f in collection["features"]]
    assert names == ["overall_geometry", "cold_front", "warm_front"]
    for feature in collection["features"]:
        assert feature["geometry"]["type"] == "MultiLineString"
        assert feature["properties"]["event_id"] == "front_0000"
        assert feature["properties"]["valid_time"] == "2024-01-01T00:00:00"
        for part in feature["geometry"]["coordinates"]:
            lon, lat = np.array(part).T
            assert np.all((lon >= -180) & (lon <= 180))
            assert np.all(np.abs(lat - 40.0) < 0.2)
    overall, cold, _ = collection["features"]
    assert overall["properties"]["summary"]["classifications"] == ["cold", "warm"]
    summary = cold["properties"]["summary"]
    assert summary["classification"] == "cold"
    assert summary["symbol_sides"] in (["left"], ["right"])
    assert summary["length_km"] > 0
    assert 0 <= summary["probability_diagnostics"]["mean_probability"] <= 1
    assert 0 <= summary["probability_diagnostics"]["max_probability"] <= 1


def series(*frames: dict[str, np.ndarray], lon: np.ndarray = LON) -> xr.Dataset:
    """Probabilities with one time per frame, 6 hours apart. Types missing from a frame are zero."""
    times = np.datetime64("2024-01-01T00", "ns") + np.arange(len(frames)) * np.timedelta64(6, "h")
    zeros = np.zeros((len(LAT), len(lon)))
    names = sorted({name for frame in frames for name in frame})
    return xr.Dataset(
        {name: (["time", "latitude", "longitude"], np.stack([f.get(name, zeros) for f in frames])) for name in names},
        coords={"time": times, "latitude": LAT, "longitude": lon},
    )


def forecast(lats: list[float]) -> xr.Dataset:
    """Cold-front probabilities with one time per entry, the front sitting at each latitude in turn."""
    return series(*({"CF": ridge([(240.0, lat), (270.0, lat)])} for lat in lats))


def test_extract_fronts_over_time_uses_the_previous_time_for_motion() -> None:
    results = splines.extract_fronts_over_time(forecast([41.0, 40.0]), CONFIG)
    assert [time for time, _ in results] == ["2024-01-01T00:00:00", "2024-01-01T06:00:00"]
    ((section,),) = results[1][1][0].lines
    assert symbols_point_north(section) is False


def test_extract_fronts_over_time_reads_fields_at_one_level() -> None:
    warm = warm_south().values
    # Warm air is south at 850 hPa but north at 700 hPa, so reading the wrong level flips the symbols.
    fields = xr.Dataset(
        {"theta_e": (["level", "latitude", "longitude"], np.stack([600.0 - warm, warm]))},
        coords={"level": [700, 850], "latitude": LAT, "longitude": LON},
    )
    ((_, (boundary,)),) = splines.extract_fronts_over_time(
        forecast([40.0]), CONFIG, fields=fields, temperature_var="theta_e", level=850
    )
    ((section,),) = boundary.lines
    assert symbols_point_north(section) is False


def test_extract_fronts_on_an_empty_map() -> None:
    assert splines.extract_fronts(probabilities(CF=np.zeros((len(LAT), len(LON)))), CONFIG) == []


def test_sample_interpolates_on_descending_latitude_and_gives_nan_off_grid() -> None:
    values = splines.sample(warm_south(), np.array([250.1, 250.1, 300.0]), np.array([40.0, 40.1, 40.0]))
    np.testing.assert_allclose(values[:2], [290.0, 289.95])
    assert np.isnan(values[2])


# --- the grid seam ---

GLOBAL_LON = np.arange(0.0, 360.0, 0.25)


def seam_ridge(lat: float) -> np.ndarray:
    """A band from 350E to 10E, crossing the 0/360 edge of a global grid."""
    points = [(350.0, lat), (370.0, lat)]
    return np.maximum(ridge(points, lon=GLOBAL_LON), ridge(points, lon=GLOBAL_LON + 360.0))


def test_is_periodic() -> None:
    assert splines.is_periodic(GLOBAL_LON)
    assert not splines.is_periodic(LON)


def test_sample_wraps_across_the_seam() -> None:
    field = xr.DataArray(
        np.broadcast_to(np.cos(np.radians(GLOBAL_LON)), (len(LAT), len(GLOBAL_LON))).copy(),
        dims=["latitude", "longitude"],
        coords={"latitude": LAT, "longitude": GLOBAL_LON},
    )
    values = splines.sample(field, np.array([359.9, -0.1, 0.1, 360.1]), np.full(4, 40.0))
    np.testing.assert_allclose(values, np.cos(np.radians(0.1)), atol=1e-4)


def test_extract_fronts_keeps_a_front_whole_across_the_seam() -> None:
    probs = xr.Dataset(
        {"CF": (["latitude", "longitude"], seam_ridge(40.0))}, coords={"latitude": LAT, "longitude": GLOBAL_LON}
    )
    (boundary,) = splines.extract_fronts(probs, CONFIG)
    ((section,),) = boundary.lines
    assert np.all(np.abs(np.diff(section.lon)) < 1.0)
    assert np.ptp(section.lon) == pytest.approx(20.0, abs=1.5)
    assert np.all(np.abs(section.lat - 40.0) < 0.2)


def test_to_geojson_keeps_lines_crossing_180_in_one_piece() -> None:
    section = splines.FrontSection("WF", np.array([170.0, 175.0, 185.0, 190.0]), np.full(4, 40.0), np.full(4, 0.5))
    collection = splines.to_geojson([splines.FrontBoundary([[section]])])
    for feature in collection["features"]:
        (part,) = feature["geometry"]["coordinates"]
        lon = np.array(part)[:, 0]
        np.testing.assert_allclose(lon, [170.0, 175.0, 185.0, 190.0])
    assert collection["features"][1]["properties"]["summary"]["symbol_sides"] == ["left"]


def test_to_geojson_starts_each_line_inside_minus_180_to_180() -> None:
    section = splines.FrontSection("CF", np.array([350.0, 355.0, 365.0]), np.full(3, 40.0), np.full(3, 0.5))
    overall = splines.to_geojson([splines.FrontBoundary([[section]])])["features"][0]
    (part,) = overall["geometry"]["coordinates"]
    np.testing.assert_allclose(np.array(part)[:, 0], [-10.0, -5.0, 5.0])


# --- tracking ---


def event_ids(results: list[tuple[str | None, list[splines.FrontBoundary]]]) -> list[list[str]]:
    return [sorted(b.event_id for b in boundaries) for _, boundaries in results]


def test_a_moving_front_keeps_its_event_id() -> None:
    results = splines.extract_fronts_over_time(forecast([41.0, 40.0]), CONFIG)
    assert event_ids(results) == [["front_0000"], ["front_0000"]]


def test_a_new_front_gets_a_new_event_id() -> None:
    first = {"CF": ridge([(240.0, 30.0), (270.0, 30.0)])}
    second = {"CF": ridge([(240.0, 30.5), (270.0, 30.5)]) + ridge([(240.0, 52.0), (270.0, 52.0)])}
    results = splines.extract_fronts_over_time(series(first, second), CONFIG)
    assert event_ids(results) == [["front_0000"], ["front_0000", "front_0001"]]


def test_fronts_in_different_groups_do_not_match() -> None:
    results = splines.extract_fronts_over_time(
        series({"CF": ridge([(240.0, 40.0), (270.0, 40.0)])}, {"DL": ridge([(240.0, 40.0), (270.0, 40.0)])}), CONFIG
    )
    assert event_ids(results) == [["front_0000"], ["front_0001"]]


def test_a_front_that_splits_records_its_parent() -> None:
    whole = {"CF": ridge([(235.0, 40.0), (285.0, 40.0)])}
    halves = {"CF": ridge([(235.0, 40.0), (258.0, 40.0)]) + ridge([(262.0, 40.0), (285.0, 40.0)])}
    results = splines.extract_fronts_over_time(series(whole, halves), CONFIG)
    (parent,) = results[0][1]
    children = results[1][1]
    assert parent.event_id in [b.event_id for b in children]
    (new,) = [b for b in children if b.event_id != parent.event_id]
    assert new.parents == [parent.event_id]
    events = splines.event_index(results)
    assert events[parent.event_id]["lineage"]["children"] == [new.event_id]
    assert events[new.event_id]["lineage"]["parents"] == [parent.event_id]


def test_fronts_that_merge_record_both_parents() -> None:
    halves = {"CF": ridge([(235.0, 40.0), (258.0, 40.0)]) + ridge([(262.0, 40.0), (285.0, 40.0)])}
    whole = {"CF": ridge([(235.0, 40.0), (285.0, 40.0)])}
    results = splines.extract_fronts_over_time(series(halves, whole), CONFIG)
    first_ids = {b.event_id for b in results[0][1]}
    (merged,) = results[1][1]
    assert merged.event_id in first_ids
    assert merged.parents == sorted(first_ids - {merged.event_id})


def test_a_front_crossing_the_seam_is_tracked_as_one() -> None:
    results = splines.extract_fronts_over_time(
        series({"CF": seam_ridge(41.0)}, {"CF": seam_ridge(40.0)}, lon=GLOBAL_LON), CONFIG
    )
    assert event_ids(results) == [["front_0000"], ["front_0000"]]


def test_match_interest_falls_with_distance() -> None:
    def boundary(lat: float) -> splines.FrontBoundary:
        lon = np.linspace(240.0, 270.0, 100)
        return splines.FrontBoundary(
            [[splines.FrontSection("CF", lon, np.full_like(lon, lat), np.full_like(lon, 0.8))]]
        )

    near, far, gone = (splines.match_interest(boundary(40.0), boundary(lat), CONFIG) for lat in (40.5, 42.5, 50.0))
    assert 1.0 >= near > far
    assert gone == 0.0
