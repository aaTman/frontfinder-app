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


# The point a wave cyclone's cold, warm and occluded arms meet at, with one arm each way from it: the cold
# arm toward the southwest, the warm arm toward the east, the occluded arm toward the west-northwest.
MEET = (255.0, 40.0)
Y_ARMS: dict[str, list[tuple[float, float]]] = {
    "CF": [MEET, (245.0, 28.0)],
    "WF": [MEET, (280.0, 38.0)],
    "OF": [MEET, (240.0, 43.0)],
}


def y_probabilities(of_arm: list[tuple[float, float]] | None = None) -> xr.Dataset:
    """Probabilities for three fronts meeting at one point, the Y junction every junction test uses."""
    arms = dict(Y_ARMS, OF=of_arm) if of_arm is not None else Y_ARMS
    return probabilities(**{name: ridge(points) for name, points in arms.items()})


def y_crest() -> list[np.ndarray]:
    """The traced crest of the Y fixture, the lines every junction test is drawn along."""
    probs = y_probabilities()
    total = np.sum([np.nan_to_num(probs[name].values) for name in Y_ARMS], axis=0)
    mask = splines.crest_mask(total, CONFIG.threshold, periodic=splines.is_periodic(LON))
    return splines.trace_paths(mask, LAT, LON, CONFIG.min_length_km, CONFIG.min_branch_km)[0]


def line_polyline(line: list[splines.FrontSection]) -> tuple[np.ndarray, np.ndarray]:
    """Every point of a line, its sections joined up end to end."""
    return np.concatenate([s.lon for s in line]), np.concatenate([s.lat for s in line])


def line_endpoints(boundary: splines.FrontBoundary) -> list[tuple[float, float]]:
    """The first and last point of each of a boundary's lines."""
    ends = []
    for line in boundary.lines:
        line_lon, line_lat = line_polyline(line)
        ends.append((float(line_lon[0]), float(line_lat[0])))
        ends.append((float(line_lon[-1]), float(line_lat[-1])))
    return ends


def distance_to_polyline(lon: float, lat: float, poly_lon: np.ndarray, poly_lat: np.ndarray) -> float:
    """Return how far a point is from the closest part of a polyline, in km."""
    east = np.radians(poly_lon - lon) * splines.EARTH_RADIUS_KM * np.cos(np.radians(lat))
    north = np.radians(poly_lat - lat) * splines.EARTH_RADIUS_KM
    de, dn = np.diff(east), np.diff(north)
    span = np.where(de**2 + dn**2 > 0.0, de**2 + dn**2, 1.0)
    step = np.clip(-(east[:-1] * de + north[:-1] * dn) / span, 0.0, 1.0)
    return float(np.min(np.hypot(east[:-1] + step * de, north[:-1] + step * dn)))


def point_km_along(lon: np.ndarray, lat: np.ndarray, from_start: bool, km: float) -> tuple[float, float]:
    """Return the point about ``km`` along a line from one of its ends."""
    dist = np.concatenate([[0.0], np.cumsum(splines.haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:]))])
    if not from_start:
        lon, lat, dist = lon[::-1], lat[::-1], dist[-1] - dist[::-1]
    i = int(np.argmin(np.abs(dist - km)))
    return float(lon[i]), float(lat[i])


def turn_at_deg(before: tuple[float, float], at: tuple[float, float], after: tuple[float, float]) -> float:
    """Return how far (degrees) a line turns at a point, read from the legs either side of it."""
    def leg(point: tuple[float, float]) -> tuple[float, float]:
        east = (point[0] - at[0]) * np.cos(np.radians(at[1])) * splines.KM_PER_DEG
        return east, (point[1] - at[1]) * splines.KM_PER_DEG

    coming = tuple(-value for value in leg(before))  # the direction of travel arriving at the point
    going = leg(after)  # and leaving it
    cos = (coming[0] * going[0] + coming[1] * going[1]) / (np.hypot(*coming) * np.hypot(*going))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


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


def test_trace_paths_min_branch_km_keeps_short_attached_branch() -> None:
    # A ~195 km branch off a ~1000 km line, and a detached crest of the same length well away from it.
    pixels = [(100, c) for c in range(0, 61)] + [(r, 30) for r in range(101, 108)] + [(r, 55) for r in range(20, 28)]
    crests = splines.trace_paths(mask_from(pixels), LAT, LON, min_length_km=250.0, min_branch_km=150.0)
    assert [len(path) for path in crests[0]] == [61, 8]
    assert tuple(crests[0][1][0]) == (100, 30)
    # Without the extra setting a branch is held to the same length as a whole crest, and the detached
    # crest is too short for either setting, so neither shows up twice.
    for kwargs in ({}, {"min_branch_km": 250.0}):
        strict = splines.trace_paths(mask_from(pixels), LAT, LON, min_length_km=250.0, **kwargs)
        assert [len(path) for path in strict[0]] == [61]
        assert len(strict) == 1


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


def seam_y_probabilities() -> xr.Dataset:
    """The junction fixture wrapped around the 0/360 edge of a global grid, meeting at 358E."""
    arms = {
        "CF": [(358.0, 40.0), (348.0, 28.0)],
        "WF": [(358.0, 40.0), (383.0, 38.0)],  # the warm arm carries on past the edge, to 23E
        "OF": [(358.0, 40.0), (343.0, 43.0)],
    }
    return xr.Dataset(
        {
            name: (
                ["latitude", "longitude"],
                np.maximum(ridge(points, lon=GLOBAL_LON), ridge(points, lon=GLOBAL_LON + 360.0)),
            )
            for name, points in arms.items()
        },
        coords={"latitude": LAT, "longitude": GLOBAL_LON},
    )


def test_y_junction_across_the_seam_keeps_one_vertex() -> None:
    (boundary,) = splines.extract_fronts(seam_y_probabilities(), CONFIG)
    assert len(boundary.lines) == 3
    for line in boundary.lines:
        line_lon, line_lat = line_polyline(line)
        assert np.all(np.abs(np.diff(line_lon)) < 1.0)  # no line jumps the width of the grid
    # Lines are unwrapped one at a time, so the vertex the arms share is the same place on Earth even when
    # the lines name it 360 degrees apart.
    ends = [p for p in line_endpoints(boundary) if splines.haversine_km(p[0], p[1], 356.8, 40.0) < 60.0]
    assert len(ends) == 3
    for lon, lat in ends[1:]:
        assert float(splines.haversine_km(lon, lat, ends[0][0], ends[0][1])) < 1e-6


def line_end_at(line: list[splines.FrontSection], at: tuple[float, float]) -> bool:
    """Return whether a line's end nearest a point is its first one."""
    line_lon, line_lat = line_polyline(line)
    return bool(
        splines.haversine_km(line_lon[0], line_lat[0], at[0], at[1])
        < splines.haversine_km(line_lon[-1], line_lat[-1], at[0], at[1])
    )


def outer_section(
    line: list[splines.FrontSection], at: tuple[float, float]
) -> tuple[tuple[float, float], splines.FrontSection]:
    """Return a line's end farthest from a point, and the section that end belongs to."""
    line_lon, line_lat = line_polyline(line)
    if line_end_at(line, at):
        return (float(line_lon[-1]), float(line_lat[-1])), line[-1]
    return (float(line_lon[0]), float(line_lat[0])), line[0]


def junction_vertex(boundary: splines.FrontBoundary) -> tuple[float, float]:
    """Return the endpoint of a boundary closest to where its arms meet."""
    return min(line_endpoints(boundary), key=lambda p: float(splines.haversine_km(p[0], p[1], *MEET)))


def line_for(
    boundary: splines.FrontBoundary, at: tuple[float, float], tip: tuple[float, float]
) -> list[splines.FrontSection]:
    """Return the line of a boundary whose far end is closest to a tip."""
    return min(
        boundary.lines, key=lambda line: float(splines.haversine_km(*outer_section(line, at)[0], tip[0], tip[1]))
    )


# --- junctions ---


def test_find_junctions_returns_parent_position_and_branch() -> None:
    main = np.array([[10, 5], [10, 6], [10, 7], [10, 8]])
    branch = np.array([[10, 7], [11, 7], [12, 7]])
    assert splines.find_junctions([main, branch]) == [(0, 2, 1)]
    # A branch off a branch reports the branch it left, not the main path.
    offshoot = np.array([[11, 7], [11, 8], [11, 9]])
    assert splines.find_junctions([main, branch, offshoot]) == [(0, 2, 1), (1, 1, 2)]
    # A pixel that two earlier paths both hold is read against the first of them.
    assert splines.find_junctions([main, branch, np.array([[10, 7], [12, 9]])]) == [(0, 2, 1), (0, 2, 2)]


def test_pin_end_hits_target_exactly_and_blends() -> None:
    lon = np.arange(240.0, 250.01, 0.25)  # ~850 km of east-heading line at 40N, a point every 21 km
    lat = np.full_like(lon, 40.0)
    for at_start in (True, False):
        end, far = (0, -1) if at_start else (-1, 0)
        new_lon, new_lat = splines.pin_end(lon, lat, at_start, 244.0, 41.5, 75.0)
        # The pinned end lands on the target exactly, so the lines that meet there share a vertex.
        assert new_lon[end] == 244.0
        assert new_lat[end] == 41.5
        moved = splines.haversine_km(new_lon, new_lat, lon, lat)
        order = slice(None) if at_start else slice(None, None, -1)
        assert np.all(np.diff(moved[order]) <= 1e-9)  # the move fades out along the line
        assert moved[1 if at_start else -2] < moved[end]  # its neighbour moves only part of the way
        assert moved[far] == 0.0  # and past the taper, 75 km of the 850 km line, nothing moves
    # A target a whole turn of 360 degrees away is the same place, read in the line's own longitude frame.
    wrapped = splines.pin_end(lon, lat, True, 244.0 - 360.0, 41.5, 75.0)
    np.testing.assert_allclose(wrapped[0], splines.pin_end(lon, lat, True, 244.0, 41.5, 75.0)[0])


def test_y_junction_arms_share_one_vertex() -> None:
    (boundary,) = splines.extract_fronts(y_probabilities(), CONFIG)
    assert len(boundary.lines) == 3
    # The crest's own junction sits about 100 km from where the three bands' centre lines cross, since
    # thinning reads where the three bands are widest together, so look around the meeting point.
    ends = [p for p in line_endpoints(boundary) if splines.haversine_km(p[0], p[1], *MEET) < 150.0]
    assert len(ends) >= 3
    for lon, lat in ends:
        assert lon == pytest.approx(ends[0][0], abs=1e-9)
        assert lat == pytest.approx(ends[0][1], abs=1e-9)


def test_y_junction_keeps_the_sharp_corner() -> None:
    crest = y_crest()
    parent, k, _branch = splines.find_junctions(crest)[0]
    crest_lon, crest_lat = splines.path_points(crest[parent], LAT, LON)
    assert splines.turn_angle_deg(crest_lon, crest_lat, k, CONFIG.corner_window_km) >= CONFIG.corner_deg
    pixels = np.vstack(crest)  # every pixel of the crest, the line every arm is drawn along
    pixel_lon, pixel_lat = LON[pixels[:, 1]], LAT[pixels[:, 0]]

    (boundary,) = splines.extract_fronts(y_probabilities(), CONFIG)
    vertex = junction_vertex(boundary)
    tips = [(244.5, 27.75), (280.25, 38.25), (239.75, 43.0)]  # the cold, warm and occluded arm tips
    lines = [line_for(boundary, vertex, tip) for tip in tips]
    # Each arm runs up to the junction and stops on a vertex there, staying within a pixel or two of the
    # crest it was drawn along, instead of rounding the corner off.
    for line in lines:
        arm_lon, arm_lat = line_polyline(line)
        near = splines.haversine_km(arm_lon, arm_lat, *vertex) < 100.0
        assert near.sum() >= 2
        for lon, lat in zip(arm_lon[near], arm_lat[near]):
            assert float(splines.haversine_km(lon, lat, pixel_lon, pixel_lat).min()) < 25.0
    # The cold and warm arms keep the crest's corner between them, read 100 km out along each arm.
    legs = [point_km_along(*line_polyline(line), line_end_at(line, vertex), 100.0) for line in lines[:2]]
    assert turn_at_deg(legs[0], vertex, legs[1]) >= CONFIG.corner_deg
    # Fitting the whole crest as one spline, which is what the main path used to get, irons that corner
    # out: it comes nowhere near where the arms meet, and turns far less than a corner as it passes by.
    whole_lon, whole_lat = splines.smooth_line(crest_lon, crest_lat, CONFIG.smoothing_deg, CONFIG.spacing_km)
    i = int(np.argmin(splines.haversine_km(whole_lon, whole_lat, *vertex)))
    assert float(splines.haversine_km(whole_lon[i], whole_lat[i], *vertex)) > 25.0
    dist = np.concatenate(
        [[0.0], np.cumsum(splines.haversine_km(whole_lon[:-1], whole_lat[:-1], whole_lon[1:], whole_lat[1:]))]
    )
    back = int(np.argmin(np.abs(dist - dist[i] + 100.0)))
    ahead = int(np.argmin(np.abs(dist - dist[i] - 100.0)))
    assert (
        turn_at_deg(
            (float(whole_lon[back]), float(whole_lat[back])),
            (float(whole_lon[i]), float(whole_lat[i])),
            (float(whole_lon[ahead]), float(whole_lat[ahead])),
        )
        < CONFIG.corner_deg
    )


def test_y_junction_arms_keep_their_types() -> None:
    (boundary,) = splines.extract_fronts(y_probabilities(), CONFIG)
    vertex = junction_vertex(boundary)
    tips = {"CF": (244.5, 27.75), "WF": (280.25, 38.25), "OF": (239.75, 43.0)}
    for front_type, tip in tips.items():
        point, section = outer_section(line_for(boundary, vertex, tip), vertex)
        assert splines.haversine_km(point[0], point[1], tip[0], tip[1]) < 60.0
        assert section.front_type == front_type


def test_straight_line_with_a_branch_is_not_cut() -> None:
    # A branch leaving a front at right angles: the front carries straight on through the junction, so
    # there is no corner there, and only the branch's end moves onto the front it came off.
    probs = probabilities(CF=ridge([(240.0, 40.0), (270.0, 40.0)]) + ridge([(255.0, 40.0), (255.0, 46.0)]))
    (boundary,) = splines.extract_fronts(probs, CONFIG)
    main, branch = sorted(boundary.lines, key=lambda line: -splines.line_length_km(*line_polyline(line)))
    main_lon, main_lat = line_polyline(main)
    assert splines.line_length_km(main_lon, main_lat) > 2000.0
    branch_lon, branch_lat = line_polyline(branch)
    assert splines.line_length_km(branch_lon, branch_lat) > 400.0
    assert distance_to_polyline(float(branch_lon[0]), float(branch_lat[0]), main_lon, main_lat) < 1.0


def test_short_attached_arm_survives() -> None:
    # An occluded arm barely 230 km long is too short to stand on its own, but it is what makes the
    # junction, so it is kept against the shorter length branches are held to.
    probs = y_probabilities(of_arm=[MEET, (253.0, 41.4)])
    (boundary,) = splines.extract_fronts(probs, CONFIG)
    assert any(s.front_type == "OF" for line in boundary.lines for s in line)
    assert len(boundary.lines) == 3
    # Held to the length of a free-standing crest it goes, and with it the corner it made.
    (strict,) = splines.extract_fronts(probs, splines.SplineConfig(min_branch_km=CONFIG.min_length_km))
    assert not any(s.front_type == "OF" for line in strict.lines for s in line)
    assert len(strict.lines) == 1


def test_symbol_side_ignores_samples_near_other_arms() -> None:
    lon = np.linspace(240.0, 270.0, 121)
    section = splines.FrontSection("CF", lon, np.full_like(lon, 40.0), np.full_like(lon, 0.8))
    # Warm air 150 km north of the middle of the line, which is exactly where its left-hand samples land.
    # It belongs to another arm, not across this front, so the samples that read it have to go.
    warm_blob = (LON[None, :] >= 249.0) & (LON[None, :] <= 261.0) & (LAT[:, None] > 40.5) & (LAT[:, None] <= 42.5)
    temperature = grid_field(warm_south().values + 50.0 * warm_blob)
    assert splines.symbol_side(section, CONFIG, temperature=temperature) == "left"
    # The other arm of the crest, running along where those left-hand samples sit.
    others = splines.PointIndex(np.linspace(249.0, 261.0, 49), np.full(49, 41.35))
    side = splines.symbol_side(section, CONFIG, temperature=temperature, others=others)
    assert side == "right"  # the warm air is south of the line, so the symbols point south


def test_symbol_side_falls_back_when_everything_is_masked() -> None:
    lon = np.linspace(240.0, 270.0, 121)
    section = splines.FrontSection("CF", lon, np.full_like(lon, 40.0), np.full_like(lon, 0.8))
    warm_blob = (LON[None, :] >= 249.0) & (LON[None, :] <= 261.0) & (LAT[:, None] > 40.5) & (LAT[:, None] <= 42.5)
    temperature = grid_field(warm_south().values + 50.0 * warm_blob)
    # An arm along the whole line leaves nothing to read, so the clue falls back to all its samples.
    others = splines.PointIndex(np.linspace(240.0, 270.0, 121), np.full(121, 41.35))
    plain = splines.symbol_side(section, CONFIG, temperature=temperature)
    assert plain == "left"
    assert splines.symbol_side(section, CONFIG, temperature=temperature, others=others) == plain


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
