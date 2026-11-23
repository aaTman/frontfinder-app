"""HRRR source tests. No network: idx text, HTTP, and GRIB messages are synthetic."""

from __future__ import annotations

import io
import os
import urllib.error

import numpy as np
import pyproj
import pytest

from frontfinder.ingest import hrrr
from frontfinder.ingest.base import ForecastCycle
from frontfinder.ingest.hrrr import (
    HRRR_BASE_URL,
    HRRR_STATIC_CYCLE,
    STANDARD_GRAVITY,
    HRRRFieldSource,
    _block_mean,
    _byte_range,
    _decode_message,
    _eval_points_xy_south_up,
    _GribField,
    _http_get,
    _interpolate_to_eval,
    _is_complete_grib2,
    _parse_idx,
    hrrr_target_steps_for_cycle,
    rotate_grid_to_earth,
)
from frontfinder.ingest.hrrr_grid import (
    HRRR_CRS,
    HRRR_FIRST_LAT_DEG,
    HRRR_FIRST_LON_DEG,
    HRRR_NX,
    HRRR_NY,
    eval_grid_0p25,
    native_grid,
)

CYCLE = ForecastCycle("2026-11-20", 12, 0)
TMP_850 = _GribField("prs", "TMP", "850 mb")
U_10M = _GribField("sfc", "UGRD", "10 m above ground")

IDX = """\
1:0:d=2026112012:TMP:850 mb:anl:
2:100:d=2026112012:UGRD:850 mb:anl:
3:250:d=2026112012:VGRD:850 mb:anl:
4:400:d=2026112012:TMP:2 m above ground:anl:
"""


def fake_grib(size: int = 32, fill: bytes = b"\0") -> bytes:
    return b"GRIB\0\0\0\x02" + size.to_bytes(8, "big") + fill * (size - 20) + b"7777"


def build_grib(values: np.ndarray | None = None, field: _GribField = TMP_850, **overrides) -> bytes:
    """A real HRRR-shaped GRIB2 message written with eccodes."""
    import eccodes

    parameter = hrrr._GRIB_PARAMETERS[field.var]
    surface_type, surface_value = hrrr._expected_surface(field)
    keys: dict = {
        "gridType": "lambert",
        "Nx": HRRR_NX,
        "Ny": HRRR_NY,
        "latitudeOfFirstGridPointInDegrees": HRRR_FIRST_LAT_DEG,
        "longitudeOfFirstGridPointInDegrees": HRRR_FIRST_LON_DEG,
        "LoVInDegrees": 262.5,
        "LaDInDegrees": 38.5,
        "Latin1InDegrees": 38.5,
        "Latin2InDegrees": 38.5,
        "DxInMetres": 3000.0,
        "DyInMetres": 3000.0,
        "iScansNegatively": 0,
        "jScansPositively": 1,
        "jPointsAreConsecutive": 0,
        "shapeOfTheEarth": 6,
        "uvRelativeToGrid": 1,
        "discipline": parameter[0],
        "parameterCategory": parameter[1],
        "parameterNumber": parameter[2],
        "typeOfFirstFixedSurface": surface_type,
        "dataDate": 20261120,
        "dataTime": 1200,
        "stepRange": "0",
        "packingType": "grid_simple",
        "bitsPerValue": 24,
    }
    if surface_value is not None:
        keys["scaleFactorOfFirstFixedSurface"] = 0
        keys["scaledValueOfFirstFixedSurface"] = int(surface_value)
    keys.update(overrides)
    gid = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        for key, value in keys.items():
            eccodes.codes_set(gid, key, value)
        if values is None:
            values = np.full(HRRR_NY * HRRR_NX, 280.0)
        if np.any(values == 9999.0):
            eccodes.codes_set(gid, "bitmapPresent", 1)
            eccodes.codes_set(gid, "missingValue", 9999.0)
        eccodes.codes_set_values(gid, np.asarray(values, dtype=np.float64).ravel())
        return eccodes.codes_get_message(gid)
    finally:
        eccodes.codes_release(gid)


def inject_fields(monkeypatch, arrays: dict[tuple[str, str], np.ndarray]) -> list[_GribField]:
    """Bypass network and decode; returns a log of decoded fields."""
    calls: list[_GribField] = []

    def decode(_payload, field, _cycle):
        calls.append(field)
        return arrays[(field.var, field.level)].astype(np.float64)

    monkeypatch.setattr(hrrr, "_decode_message", decode)
    monkeypatch.setattr(HRRRFieldSource, "_message_bytes", lambda self, field, cycle: b"")
    return calls


def south_up_index_field() -> np.ndarray:
    rows, cols = np.mgrid[0:HRRR_NY, 0:HRRR_NX]
    return rows * 10_000.0 + cols


class TestIdx:
    def test_parse(self):
        entries = _parse_idx(IDX)
        assert [entry.offset for entry in entries] == [0, 100, 250, 400]
        assert (entries[1].var, entries[1].level, entries[1].forecast) == ("UGRD", "850 mb", "anl")

    def test_byte_ranges_including_last_message(self):
        entries = _parse_idx(IDX)
        assert _byte_range(entries, TMP_850, CYCLE) == (0, 99)
        assert _byte_range(entries, _GribField("prs", "VGRD", "850 mb"), CYCLE) == (250, 399)
        assert _byte_range(entries, _GribField("sfc", "TMP", "2 m above ground"), CYCLE) == (400, None)

    def test_missing_match_raises(self):
        with pytest.raises(ValueError, match="found 0"):
            _byte_range(_parse_idx(IDX), _GribField("prs", "HGT", "850 mb"), CYCLE)

    def test_ambiguous_match_raises(self):
        text = IDX + "5:500:d=2026112012:TMP:850 mb:anl:\n"
        with pytest.raises(ValueError, match="found 2"):
            _byte_range(_parse_idx(text), TMP_850, CYCLE)

    def test_forecast_descriptor_is_exact(self):
        text = (
            "1:0:d=2026112012:UGRD:10 m above ground:6 hour fcst:\n"
            "2:90:d=2026112012:UGRD:10 m above ground:5-6 hour max fcst:\n"
            "3:190:d=2026112012:UGRD:10 m above ground:anl:\n"
        )
        entries = _parse_idx(text)
        assert _byte_range(entries, U_10M, ForecastCycle("2026-11-20", 12, 6)) == (0, 89)
        assert _byte_range(entries, U_10M, CYCLE) == (190, None)
        with pytest.raises(ValueError, match="found 0"):
            _byte_range(entries, U_10M, ForecastCycle("2026-11-20", 12, 5))

    def test_wrong_cycle_raises(self):
        with pytest.raises(ValueError, match="expected 2026112112"):
            _byte_range(_parse_idx(IDX), TMP_850, ForecastCycle("2026-11-21", 12, 0))

    def test_submessage_offset_raises(self):
        text = "1.1:0:d=2026112012:TMP:850 mb:anl:\n1.2:0:d=2026112012:UGRD:850 mb:anl:\n"
        with pytest.raises(ValueError, match="shares its byte offset"):
            _byte_range(_parse_idx(text), TMP_850, CYCLE)

    @pytest.mark.parametrize("line", ["garbage", "1:0:2026112012:TMP:850 mb:anl:"])
    def test_malformed_line_raises(self, line):
        with pytest.raises(ValueError, match="malformed"):
            _parse_idx(line)

    def test_backwards_offset_raises(self):
        text = "1:100:d=2026112012:TMP:850 mb:anl:\n2:50:d=2026112012:UGRD:850 mb:anl:\n"
        with pytest.raises(ValueError, match="backwards"):
            _parse_idx(text)


class _Response(io.BytesIO):
    def __init__(self, body: bytes, status: int, content_range: str | None):
        super().__init__(body)
        self.status = status
        self.headers = {} if content_range is None else {"Content-Range": content_range}


class TestHttp:
    def test_sends_range_and_accepts_206(self, monkeypatch):
        seen = []

        def urlopen(request, timeout):
            seen.append(request.get_header("Range"))
            return _Response(b"abc", 206, "bytes 5-7/100")

        monkeypatch.setattr(hrrr.urllib.request, "urlopen", urlopen)
        assert _http_get("http://x/f", 5, 7) == b"abc"
        assert seen == ["bytes=5-7"]

    def test_open_ended_range_for_last_message(self, monkeypatch):
        seen = []

        def urlopen(request, timeout):
            seen.append(request.get_header("Range"))
            return _Response(b"abc", 206, "bytes 97-99/100")

        monkeypatch.setattr(hrrr.urllib.request, "urlopen", urlopen)
        assert _http_get("http://x/f", 97, None) == b"abc"
        assert seen == ["bytes=97-"]

    @pytest.mark.parametrize(
        ("status", "content_range"), [(200, None), (206, "bytes 6-8/100"), (206, "bytes 5-9/100")]
    )
    def test_rejects_unhonored_range(self, monkeypatch, status, content_range):
        monkeypatch.setattr(
            hrrr.urllib.request, "urlopen", lambda request, timeout: _Response(b"abc", status, content_range)
        )
        with pytest.raises(OSError, match="range request not honored"):
            _http_get("http://x/f", 5, 7)

    def test_retries_transient_status(self, monkeypatch):
        attempts = []

        def urlopen(request, timeout):
            attempts.append(1)
            if len(attempts) < 3:
                raise urllib.error.HTTPError(request.full_url, 503, "Slow Down", {}, None)
            return _Response(b"ok", 200, None)

        monkeypatch.setattr(hrrr.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(hrrr.time, "sleep", lambda _s: None)
        assert _http_get("http://x/f") == b"ok"
        assert len(attempts) == 3

    def test_does_not_retry_404(self, monkeypatch):
        attempts = []

        def urlopen(request, timeout):
            attempts.append(1)
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

        monkeypatch.setattr(hrrr.urllib.request, "urlopen", urlopen)
        with pytest.raises(urllib.error.HTTPError):
            _http_get("http://x/f")
        assert len(attempts) == 1


def finite_difference_earth_wind(u_grid, v_grid, lon, lat, delta=1e-4):
    """Earth-relative wind from the projection's own east and north directions."""
    to_lcc = pyproj.Transformer.from_crs("EPSG:4326", HRRR_CRS, always_xy=True)

    def unit(dlon, dlat):
        x0, y0 = to_lcc.transform(lon - dlon, lat - dlat)
        x1, y1 = to_lcc.transform(lon + dlon, lat + dlat)
        length = np.hypot(x1 - x0, y1 - y0)
        return (x1 - x0) / length, (y1 - y0) / length

    east_x, east_y = unit(delta, 0.0)
    north_x, north_y = unit(0.0, delta)
    return u_grid * east_x + v_grid * east_y, u_grid * north_x + v_grid * north_y


class TestWindRotation:
    @pytest.mark.parametrize(
        ("lon", "lat"),
        [
            (-122.0, 47.5),  # far west
            (-118.0, 25.0),
            (-105.0, 40.0),  # west of the central meridian
            (-97.5, 38.5),  # on it
            (-90.0, 30.0),  # east of it
            (-75.0, 42.0),
            (-65.0, 47.0),  # far east
        ],
    )
    @pytest.mark.parametrize(("u_grid", "v_grid"), [(10.0, 0.0), (0.0, 10.0), (-7.0, 13.0)])
    def test_matches_projection_finite_differences(self, lon, lat, u_grid, v_grid):
        u_earth, v_earth = rotate_grid_to_earth(np.array(u_grid), np.array(v_grid), np.array(lon))
        expected_u, expected_v = finite_difference_earth_wind(u_grid, v_grid, lon, lat)
        assert u_earth == pytest.approx(expected_u, abs=1e-6)
        assert v_earth == pytest.approx(expected_v, abs=1e-6)

    @pytest.mark.parametrize(("lon", "sign"), [(-75.0, -1.0), (-120.0, 1.0)])
    def test_sign_either_side_of_central_meridian(self, lon, sign):
        # Grid +x tilts south of true east on the east side, north of it on the west.
        _, v_earth = rotate_grid_to_earth(np.array(10.0), np.array(0.0), np.array(lon))
        assert np.sign(v_earth) == sign
        assert abs(v_earth) > 1.0

    def test_no_rotation_on_central_meridian(self):
        u_earth, v_earth = rotate_grid_to_earth(np.array(3.0), np.array(4.0), np.array(-97.5))
        assert (float(u_earth), float(v_earth)) == pytest.approx((3.0, 4.0))

    def test_accepts_0_360_longitude(self):
        signed = rotate_grid_to_earth(np.array(3.0), np.array(4.0), np.array(-75.0))
        unsigned = rotate_grid_to_earth(np.array(3.0), np.array(4.0), np.array(285.0))
        assert np.allclose(signed, unsigned)

    def test_source_rotates_every_native_point_with_its_own_longitude(self, monkeypatch, tmp_path):
        grid = native_grid(1)
        u_grid = np.full((HRRR_NY, HRRR_NX), 10.0)
        v_grid = np.full((HRRR_NY, HRRR_NX), -4.0)
        inject_fields(monkeypatch, {("UGRD", "850 mb"): u_grid, ("VGRD", "850 mb"): v_grid})
        source = HRRRFieldSource(str(tmp_path))
        u_earth = source.fetch_pressure_level("u_component_of_wind", 850, CYCLE)
        v_earth = source.fetch_pressure_level("v_component_of_wind", 850, CYCLE)
        for row, col in ((5, 5), (5, 1790), (1050, 5), (1050, 1790), (530, 900), (200, 300), (900, 1500)):
            expected_u, expected_v = finite_difference_earth_wind(
                10.0, -4.0, float(grid.lon[row, col]), float(grid.lat[row, col])
            )
            assert u_earth[row, col] == pytest.approx(expected_u, abs=1e-5)
            assert v_earth[row, col] == pytest.approx(expected_v, abs=1e-5)

    def test_grid_frame_is_untouched(self, monkeypatch, tmp_path):
        u_grid = south_up_index_field() / 1e4
        v_grid = -u_grid
        inject_fields(monkeypatch, {("UGRD", "850 mb"): u_grid, ("VGRD", "850 mb"): v_grid})
        source = HRRRFieldSource(str(tmp_path), wind_frame="grid")
        result = source.fetch_pressure_level("u_component_of_wind", 850, CYCLE)
        assert np.array_equal(result, np.flipud(u_grid).astype(np.float32))

    def test_fetching_one_component_loads_the_pair_once(self, monkeypatch, tmp_path):
        zeros = np.zeros((HRRR_NY, HRRR_NX))
        calls = inject_fields(monkeypatch, {("UGRD", "850 mb"): zeros, ("VGRD", "850 mb"): zeros})
        source = HRRRFieldSource(str(tmp_path), coarsen=8)
        source.fetch_pressure_level("v_component_of_wind", 850, CYCLE)
        source.fetch_pressure_level("u_component_of_wind", 850, CYCLE)
        assert sorted(field.var for field in calls) == ["UGRD", "VGRD"]


class TestProcessingOrder:
    def test_rotation_happens_at_3km_before_coarsening(self, monkeypatch, tmp_path):
        # Columns alternate, so a block's mean depends on the angle at each 3 km cell.
        alternating = np.where(np.arange(HRRR_NX) % 2 == 0, 1000.0, -1000.0)
        u_grid = np.tile(alternating, (HRRR_NY, 1))
        v_grid = np.full((HRRR_NY, HRRR_NX), 5.0)
        inject_fields(monkeypatch, {("UGRD", "850 mb"): u_grid, ("VGRD", "850 mb"): v_grid})
        source = HRRRFieldSource(str(tmp_path), coarsen=8)
        actual = source.fetch_pressure_level("v_component_of_wind", 850, CYCLE)

        lon_south_up = native_grid(1).lon[::-1]
        _, v_fine = rotate_grid_to_earth(u_grid, v_grid, lon_south_up)
        rotate_first = np.flipud(_block_mean(v_fine, 8))
        assert np.allclose(actual, rotate_first, atol=1e-3)

        coarse = native_grid(8)
        _, coarsen_first = rotate_grid_to_earth(
            np.flipud(_block_mean(u_grid, 8)), np.flipud(_block_mean(v_grid, 8)), coarse.lon
        )
        assert np.abs(actual - coarsen_first).min() > 0.1

    def test_flip_is_after_block_mean(self, monkeypatch, tmp_path):
        # 1059 rows drop 3 at the north edge. Flipping first would drop the south rows.
        field = south_up_index_field()
        inject_fields(monkeypatch, {("TMP", "850 mb"): field})
        source = HRRRFieldSource(str(tmp_path), coarsen=8)
        result = source.fetch_pressure_level("temperature", 850, CYCLE)
        assert result[-1, 0] == pytest.approx(field[:8, :8].mean())
        assert result[0, 0] == pytest.approx(field[1048:1056, :8].mean())


class TestRowOrder:
    def test_latitude_field_is_north_up(self, monkeypatch, tmp_path):
        grid = native_grid(1)
        inject_fields(monkeypatch, {("TMP", "850 mb"): grid.lat[::-1].copy()})
        source = HRRRFieldSource(str(tmp_path))
        result = source.fetch_pressure_level("temperature", 850, CYCLE)
        assert np.allclose(result, source.lat, atol=1e-4)
        center = result[:, HRRR_NX // 2]
        assert np.all(np.diff(center) < 0)
        assert center[0] > 52.0 and center[-1] < 25.0

    @pytest.mark.parametrize("coarsen", [2, 4, 8])
    def test_coarsened_latitude_stays_on_its_grid(self, monkeypatch, tmp_path, coarsen):
        inject_fields(monkeypatch, {("TMP", "850 mb"): native_grid(1).lat[::-1].copy()})
        source = HRRRFieldSource(str(tmp_path), coarsen=coarsen)
        result = source.fetch_pressure_level("temperature", 850, CYCLE)
        # A block's mean latitude is not exactly its center's on a curved projection.
        assert np.abs(result - source.lat).max() < 0.01

    def test_real_grib_scan_order(self):
        import eccodes

        grid = native_grid(1)
        south_up_lat = grid.lat[::-1]
        payload = build_grib(south_up_lat)
        decoded = _decode_message(payload, TMP_850, CYCLE)
        assert np.allclose(decoded, south_up_lat, atol=1e-4)

        gid = eccodes.codes_new_from_message(payload)
        try:
            message_lat = eccodes.codes_get_array(gid, "latitudes").reshape(HRRR_NY, HRRR_NX)
            message_lon = eccodes.codes_get_array(gid, "longitudes").reshape(HRRR_NY, HRRR_NX)
        finally:
            eccodes.codes_release(gid)
        assert np.allclose(message_lat[::-1], grid.lat, atol=1e-5)
        assert np.allclose((message_lon[::-1] + 180) % 360 - 180, grid.lon, atol=1e-5)


class TestUnits:
    def test_geopotential_is_scaled(self, monkeypatch, tmp_path):
        inject_fields(monkeypatch, {("HGT", "850 mb"): np.full((HRRR_NY, HRRR_NX), 1500.0)})
        result = HRRRFieldSource(str(tmp_path)).fetch_pressure_level("geopotential", 850, CYCLE)
        assert np.allclose(result, 1500.0 * STANDARD_GRAVITY)

    def test_terrain_is_scaled_and_uses_the_static_cycle(self, monkeypatch, tmp_path):
        seen = []
        inject_fields(monkeypatch, {("HGT", "surface"): np.full((HRRR_NY, HRRR_NX), 2000.0)})
        real_decode = hrrr._decode_message

        def spy(payload, field, cycle):
            seen.append(cycle)
            return real_decode(payload, field, cycle)

        monkeypatch.setattr(hrrr, "_decode_message", spy)
        result = HRRRFieldSource(str(tmp_path)).fetch_static("geopotential_at_surface")
        assert np.allclose(result, 2000.0 * STANDARD_GRAVITY)
        assert seen == [HRRR_STATIC_CYCLE]

    @pytest.mark.parametrize(
        ("variable", "key"),
        [
            ("2m_temperature", ("TMP", "2 m above ground")),
            ("2m_dewpoint_temperature", ("DPT", "2 m above ground")),
            ("surface_pressure", ("PRES", "surface")),
            ("mean_sea_level_pressure", ("MSLMA", "mean sea level")),
        ],
    )
    def test_single_level_needs_no_conversion(self, monkeypatch, tmp_path, variable, key):
        inject_fields(monkeypatch, {key: np.full((HRRR_NY, HRRR_NX), 1234.5)})
        result = HRRRFieldSource(str(tmp_path)).fetch_single_level(variable, CYCLE)
        assert np.allclose(result, 1234.5)
        assert result.dtype == np.float32

    def test_unknown_variables_raise(self, tmp_path):
        source = HRRRFieldSource(str(tmp_path))
        with pytest.raises(ValueError):
            source.fetch_pressure_level("2m_temperature", 850, CYCLE)
        with pytest.raises(ValueError):
            source.fetch_single_level("temperature", CYCLE)
        with pytest.raises(ValueError):
            source.fetch_static("land_sea_mask")


class TestBelowGroundHumidity:
    @staticmethod
    def arrays(surface_pressure: np.ndarray, level: str = "850 mb") -> dict:
        return {
            ("SPFH", level): np.full((HRRR_NY, HRRR_NX), 0.008),
            ("SPFH", "2 m above ground"): np.full((HRRR_NY, HRRR_NX), 0.003),
            ("PRES", "surface"): surface_pressure,
        }

    def test_only_below_ground_cells_take_the_2m_value(self, monkeypatch, tmp_path):
        surface_pressure = np.full((HRRR_NY, HRRR_NX), 100_000.0)
        surface_pressure[0, 0] = 80_000.0  # level is below ground
        surface_pressure[0, 1] = 85_000.0  # surface exactly on the level
        inject_fields(monkeypatch, self.arrays(surface_pressure))
        result = HRRRFieldSource(str(tmp_path)).fetch_pressure_level("specific_humidity", 850, CYCLE)
        # Row 0 is south in the decoded field and last after the north-up flip.
        assert result[-1, 0] == pytest.approx(0.003)
        assert result[-1, 1] == pytest.approx(0.008)
        assert result[0, 0] == pytest.approx(0.008)

    def test_mask_uses_each_levels_own_pressure(self, monkeypatch, tmp_path):
        surface_pressure = np.full((HRRR_NY, HRRR_NX), 80_000.0)
        arrays = {**self.arrays(surface_pressure), **self.arrays(surface_pressure, "700 mb")}
        inject_fields(monkeypatch, arrays)
        source = HRRRFieldSource(str(tmp_path))
        assert np.allclose(source.fetch_pressure_level("specific_humidity", 850, CYCLE), 0.003)
        assert np.allclose(source.fetch_pressure_level("specific_humidity", 700, CYCLE), 0.008)

    def test_fill_happens_before_coarsening(self, monkeypatch, tmp_path):
        surface_pressure = np.full((HRRR_NY, HRRR_NX), 100_000.0)
        surface_pressure[0, 0] = 80_000.0
        inject_fields(monkeypatch, self.arrays(surface_pressure))
        result = HRRRFieldSource(str(tmp_path), coarsen=2).fetch_pressure_level("specific_humidity", 850, CYCLE)
        # One of four 3 km cells is below ground in the SW block.
        assert result[-1, 0] == pytest.approx((3 * 0.008 + 0.003) / 4)


class TestBlockMean:
    def test_values_anchor_southwest_and_drop_trailing_cells(self):
        field = np.arange(5 * 7, dtype=np.float64).reshape(5, 7)
        result = _block_mean(field, 2)
        assert result.shape == (2, 3)
        assert result[0, 0] == pytest.approx(field[:2, :2].mean())
        assert result[1, 2] == pytest.approx(field[2:4, 4:6].mean())

    def test_coarsen_one_is_identity(self):
        field = np.arange(6.0).reshape(2, 3)
        assert _block_mean(field, 1) is field

    @pytest.mark.parametrize("coarsen", [1, 2, 4, 8])
    def test_shape_and_projected_coordinates_match_grid(self, monkeypatch, tmp_path, coarsen):
        fine = native_grid(1)
        x_field = np.tile(fine.x, (HRRR_NY, 1))
        y_field = np.tile(fine.y[::-1][:, None], (1, HRRR_NX))
        inject_fields(monkeypatch, {("TMP", "850 mb"): x_field, ("HGT", "850 mb"): y_field})
        source = HRRRFieldSource(str(tmp_path), coarsen=coarsen)
        x_result = source.fetch_pressure_level("temperature", 850, CYCLE)
        y_result = source.fetch_pressure_level("geopotential", 850, CYCLE) / STANDARD_GRAVITY
        assert x_result.shape == source.grid.shape
        assert x_result.dtype == np.float32
        # float32 at ~2.7e6 m resolves about 0.25 m.
        assert np.abs(x_result - source.grid.x[None, :]).max() < 1.0
        assert np.abs(y_result - source.grid.y[:, None]).max() < 1.0


class TestLatlonTarget:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"coarsen": 1},
            {"coarsen": 2},
            {"coarsen": 4},
            {"coarsen": 8, "wind_frame": "grid"},
        ],
    )
    def test_requires_coarsen_8_and_earth_winds(self, tmp_path, kwargs):
        with pytest.raises(ValueError, match="requires coarsen=8"):
            HRRRFieldSource(str(tmp_path), target="latlon_0p25", **kwargs)

    def test_grid_is_the_eval_grid(self, tmp_path):
        source = HRRRFieldSource(str(tmp_path), coarsen=8, target="latlon_0p25")
        assert source.grid is eval_grid_0p25()
        assert source.grid.shape == (103, 207)

    def test_bilinear_is_exact_for_a_plane_in_projected_coordinates(self):
        grid = native_grid(1)
        xx, yy = np.meshgrid(grid.x, grid.y[::-1])
        result = _interpolate_to_eval(2.0 * xx - 3.0 * yy + 7.0, 1)
        eval_x, eval_y = _eval_points_xy_south_up()
        assert result.shape == eval_grid_0p25().shape
        assert np.allclose(result, 2.0 * eval_x - 3.0 * eval_y + 7.0, rtol=0.0, atol=1e-5)

    def test_eval_points_run_south_to_north(self):
        grid = eval_grid_0p25()
        to_geo = pyproj.Transformer.from_crs(HRRR_CRS, "EPSG:4326", always_xy=True)
        x, y = _eval_points_xy_south_up()
        _, lat = to_geo.transform(x[:, 0], y[:, 0])
        assert np.allclose(lat, grid.lat[::-1], atol=1e-8)

    @pytest.mark.parametrize("coarsen", [1, 2, 4, 8])
    def test_real_eval_grid_is_inside_every_cell_center_hull(self, coarsen):
        grid = native_grid(coarsen)
        x, y = _eval_points_xy_south_up()
        assert grid.x.min() <= x.min() and x.max() <= grid.x.max()
        assert grid.y.min() <= y.min() and y.max() <= grid.y.max()

    def test_real_eval_grid_interpolates_north_up(self, monkeypatch, tmp_path):
        # A plane survives both the block mean and bilinear interpolation exactly.
        fine = native_grid(1)
        xx, yy = np.meshgrid(fine.x, fine.y[::-1])
        plane = 300.0 + 1e-5 * xx - 2e-5 * yy
        inject_fields(monkeypatch, {("TMP", "850 mb"): plane})
        source = HRRRFieldSource(str(tmp_path), coarsen=8, target="latlon_0p25")
        result = source.fetch_pressure_level("temperature", 850, CYCLE)
        eval_x, eval_y = _eval_points_xy_south_up()
        expected = np.flipud(300.0 + 1e-5 * eval_x - 2e-5 * eval_y)
        assert result.shape == eval_grid_0p25().shape
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert np.abs(result - expected).max() < 1e-3

    @pytest.mark.parametrize(
        ("axis", "edge", "overshoot"),
        [("y", -1, 1.0), ("y", 0, -1.0), ("x", -1, 1.0), ("x", 0, -1.0)],
    )
    def test_guard_refuses_a_node_outside_the_hull(self, monkeypatch, axis, edge, overshoot):
        grid = native_grid(8)
        y_axis = grid.y[::-1]
        x = np.array([[float(grid.x[10])]])
        y = np.array([[float(y_axis[10])]])
        if axis == "x":
            x[0, 0] = grid.x[edge] + overshoot
        else:
            y[0, 0] = y_axis[edge] + overshoot
        monkeypatch.setattr(hrrr, "_eval_points_xy_south_up", lambda: (x, y))
        with pytest.raises(ValueError, match="would have to extrapolate"):
            _interpolate_to_eval(np.full(grid.shape, 280.0), 8)


class TestCache:
    @staticmethod
    def serve(monkeypatch, payload: bytes, idx: str = "1:0:d=2026112012:TMP:850 mb:anl:\n"):
        calls: list[tuple] = []

        def http_get(url, start=None, end=None):
            calls.append((url, start, end))
            if url.endswith(".idx"):
                return idx.encode()
            return payload

        monkeypatch.setattr(hrrr, "_http_get", http_get)
        monkeypatch.setattr(
            hrrr, "_decode_message", lambda _p, _f, _c: np.full((HRRR_NY, HRRR_NX), 280.0)
        )
        return calls

    @staticmethod
    def path(source: HRRRFieldSource) -> str:
        return source._cache_path(TMP_850, CYCLE)

    def test_complete_grib_detection(self):
        good = fake_grib()
        assert _is_complete_grib2(good)
        assert not _is_complete_grib2(b"")
        assert not _is_complete_grib2(good[:-1])
        assert not _is_complete_grib2(b"XXXX" + good[4:])
        assert not _is_complete_grib2(good[:-4] + b"0000")
        assert not _is_complete_grib2(good[:7] + b"\x01" + good[8:])
        assert not _is_complete_grib2(good + good)

    def test_downloads_then_hits_cache(self, monkeypatch, tmp_path):
        payload = fake_grib()
        calls = self.serve(monkeypatch, payload)
        source = HRRRFieldSource(str(tmp_path))
        source.fetch_pressure_level("temperature", 850, CYCLE)
        with open(self.path(source), "rb") as handle:
            assert handle.read() == payload
        assert calls == [
            (f"{HRRR_BASE_URL}/hrrr.20261120/conus/hrrr.t12z.wrfprsf00.grib2.idx", None, None),
            (f"{HRRR_BASE_URL}/hrrr.20261120/conus/hrrr.t12z.wrfprsf00.grib2", 0, None),
        ]
        calls.clear()
        HRRRFieldSource(str(tmp_path)).fetch_pressure_level("temperature", 850, CYCLE)
        assert calls == []

    @pytest.mark.parametrize("bad", [b"", b"GRIB-truncated", fake_grib()[:-1]])
    def test_empty_or_truncated_cache_is_refetched(self, monkeypatch, tmp_path, bad):
        payload = fake_grib()
        calls = self.serve(monkeypatch, payload)
        source = HRRRFieldSource(str(tmp_path))
        path = self.path(source)
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as handle:
            handle.write(bad)
        source.fetch_pressure_level("temperature", 850, CYCLE)
        assert len(calls) == 2
        with open(path, "rb") as handle:
            assert handle.read() == payload

    def test_bad_download_is_not_cached(self, monkeypatch, tmp_path):
        self.serve(monkeypatch, b"")
        source = HRRRFieldSource(str(tmp_path))
        with pytest.raises(ValueError, match="not one complete GRIB2"):
            source.fetch_pressure_level("temperature", 850, CYCLE)
        assert not os.path.exists(self.path(source))

    def test_failed_write_leaves_nothing_behind(self, monkeypatch, tmp_path):
        self.serve(monkeypatch, fake_grib())
        source = HRRRFieldSource(str(tmp_path))

        def fail(_src, _dst):
            raise OSError("disk full")

        monkeypatch.setattr(hrrr.os, "replace", fail)
        with pytest.raises(OSError, match="disk full"):
            source.fetch_pressure_level("temperature", 850, CYCLE)
        assert os.listdir(os.path.dirname(self.path(source))) == []

    def test_cache_key_covers_every_identity_field(self, tmp_path):
        source = HRRRFieldSource(str(tmp_path))
        base = source._cache_path(TMP_850, CYCLE)
        variants = {
            source._cache_path(TMP_850, ForecastCycle("2026-11-21", 12, 0)),
            source._cache_path(TMP_850, ForecastCycle("2026-11-20", 18, 0)),
            source._cache_path(TMP_850, ForecastCycle("2026-11-20", 12, 6)),
            source._cache_path(_GribField("sfc", "TMP", "850 mb"), CYCLE),
            source._cache_path(_GribField("prs", "DPT", "850 mb"), CYCLE),
            source._cache_path(_GribField("prs", "TMP", "700 mb"), CYCLE),
        }
        assert len(variants) == 6
        assert base not in variants

    def test_returned_field_is_read_only(self, monkeypatch, tmp_path):
        self.serve(monkeypatch, fake_grib())
        source = HRRRFieldSource(str(tmp_path))
        result = source.fetch_pressure_level("temperature", 850, CYCLE)
        assert source.fetch_pressure_level("temperature", 850, CYCLE) is result
        with pytest.raises(ValueError):
            result[0, 0] = 0.0


class TestDecodeValidation:
    def test_valid_message_decodes_south_up(self):
        values = np.linspace(250.0, 300.0, HRRR_NY * HRRR_NX)
        decoded = _decode_message(build_grib(values), TMP_850, CYCLE)
        assert decoded.shape == (HRRR_NY, HRRR_NX)
        assert decoded[0, 0] == pytest.approx(250.0, abs=1e-3)
        assert decoded[-1, -1] == pytest.approx(300.0, abs=1e-3)

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"jScansPositively": 0}, "jScansPositively"),
            ({"iScansNegatively": 1}, "iScansNegatively"),
            ({"DxInMetres": 3100.0}, "DxInMetres"),
            ({"LoVInDegrees": 260.0}, "LoVInDegrees"),
            ({"latitudeOfFirstGridPointInDegrees": 22.0}, "latitudeOfFirstGridPoint"),
            ({"dataDate": 20261119}, "message is for"),
            ({"dataTime": 600}, "message is for"),
            ({"parameterNumber": 6}, "GRIB parameter"),
            ({"typeOfFirstFixedSurface": 103}, "surface type"),
            ({"scaledValueOfFirstFixedSurface": 70000}, "surface value"),
        ],
    )
    def test_rejects_wrong_metadata(self, override, message):
        with pytest.raises(ValueError, match=message):
            _decode_message(build_grib(field=TMP_850, **override), TMP_850, CYCLE)

    def test_rejects_wrong_dimensions(self):
        values = np.full((HRRR_NX - 1) * HRRR_NY, 280.0)
        with pytest.raises(ValueError, match="Nx"):
            _decode_message(build_grib(values, Nx=HRRR_NX - 1), TMP_850, CYCLE)

    def test_rejects_non_lambert(self):
        import eccodes

        gid = eccodes.codes_grib_new_from_samples("GRIB2")
        payload = eccodes.codes_get_message(gid)
        eccodes.codes_release(gid)
        with pytest.raises(ValueError, match="gridType"):
            _decode_message(payload, TMP_850, CYCLE)

    def test_rejects_earth_relative_winds(self):
        with pytest.raises(ValueError, match="uvRelativeToGrid"):
            _decode_message(build_grib(field=U_10M, uvRelativeToGrid=0), U_10M, CYCLE)

    def test_rejects_missing_values(self):
        values = np.full(HRRR_NY * HRRR_NX, 280.0)
        values[1234] = 9999.0
        with pytest.raises(ValueError, match="missing values"):
            _decode_message(build_grib(values), TMP_850, CYCLE)


def test_end_to_end_with_real_grib_messages(monkeypatch, tmp_path):
    """Real eccodes messages through idx lookup, byte ranges, rotation, and flip."""
    grid = native_grid(1)
    lat_south_up = grid.lat[::-1]
    u_grid = 10.0 + 0.1 * lat_south_up
    v_grid = -4.0 + 0.05 * lat_south_up
    u_field = _GribField("prs", "UGRD", "850 mb")
    v_field = _GribField("prs", "VGRD", "850 mb")
    u_message = build_grib(u_grid, field=u_field)
    v_message = build_grib(v_grid, field=v_field)
    t_message = build_grib(lat_south_up + 250.0, field=TMP_850)
    blob = u_message + v_message + t_message
    idx = (
        "1:0:d=2026112012:UGRD:850 mb:anl:\n"
        f"2:{len(u_message)}:d=2026112012:VGRD:850 mb:anl:\n"
        f"3:{len(u_message) + len(v_message)}:d=2026112012:TMP:850 mb:anl:\n"
    )

    def http_get(url, start=None, end=None):
        if url.endswith(".idx"):
            return idx.encode()
        return blob[start:] if end is None else blob[start : end + 1]

    monkeypatch.setattr(hrrr, "_http_get", http_get)
    source = HRRRFieldSource(str(tmp_path))

    temperature = source.fetch_pressure_level("temperature", 850, CYCLE)
    assert np.allclose(temperature, grid.lat + 250.0, atol=1e-2)

    u_earth = source.fetch_pressure_level("u_component_of_wind", 850, CYCLE)
    v_earth = source.fetch_pressure_level("v_component_of_wind", 850, CYCLE)
    for row, col in ((10, 10), (10, 1780), (1040, 10), (1040, 1780), (500, 900)):
        lon, lat = float(grid.lon[row, col]), float(grid.lat[row, col])
        expected_u, expected_v = finite_difference_earth_wind(
            10.0 + 0.1 * lat, -4.0 + 0.05 * lat, lon, lat
        )
        assert u_earth[row, col] == pytest.approx(expected_u, abs=1e-3)
        assert v_earth[row, col] == pytest.approx(expected_v, abs=1e-3)


def test_target_steps_are_hourly_and_stay_inside_noaas_publish_length():
    # HRRR's value is the hourly cadence, so the target list steps every hour rather
    # than every 6 h. The lengths are NOAA's, not ours: out-of-range lead hours are a
    # 404 on the .idx, which would silently drop steps from latest.json.
    assert hrrr_target_steps_for_cycle(0) == tuple(range(0, 49))
    assert hrrr_target_steps_for_cycle(12) == tuple(range(0, 49))
    assert hrrr_target_steps_for_cycle(6) == tuple(range(0, 19))
    assert hrrr_target_steps_for_cycle(18) == tuple(range(0, 19))
    assert hrrr_target_steps_for_cycle(3) == tuple(range(0, 19))
    with pytest.raises(ValueError, match="0..23"):
        hrrr_target_steps_for_cycle(24)
