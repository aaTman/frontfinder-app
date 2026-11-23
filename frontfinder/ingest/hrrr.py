"""HRRR CONUS field source, read message by message from the NOAA S3 archive.

Processing order is fixed: decode (south-up), rotate winds at 3 km, block-mean,
optionally interpolate to the 0.25deg comparison grid, flip to north-up. Winds
are rotated before averaging because the rotation angle varies across a block.
"""

from __future__ import annotations

import http.client
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

import numpy as np
import pyproj
from scipy.interpolate import RegularGridInterpolator

from frontfinder.ingest.base import FieldSource, ForecastCycle, GridSpec
from frontfinder.ingest.hrrr_grid import (
    HRRR_CRS,
    HRRR_DX_M,
    HRRR_FIRST_LAT_DEG,
    HRRR_FIRST_LON_DEG,
    HRRR_LATIN_DEG,
    HRRR_LOV_DEG,
    HRRR_NX,
    HRRR_NY,
    eval_grid_0p25,
    native_grid,
)

HRRR_BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HRRR_STATIC_CYCLE = ForecastCycle("2026-11-20", 12, 0)
HRRR_SYNOPTIC_HOURS = (0, 6, 12, 18)


def hrrr_target_steps_for_cycle(run_hour: int) -> tuple[int, ...]:
    """6-hourly forecast lead hours targeted for a HRRR cycle (48h synoptic, 18h otherwise)."""
    if not 0 <= run_hour < 24:
        raise ValueError(f"run_hour must be in 0..23, got {run_hour}")
    max_step = 48 if run_hour in HRRR_SYNOPTIC_HOURS else 18
    return tuple(range(0, max_step + 1, 6))

STANDARD_GRAVITY = 9.80665

_HTTP_ATTEMPTS = 4
_HTTP_TIMEOUT_S = 60
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

# GRIB2 (discipline, category, number). MSLMA is NCEP-local.
_GRIB_PARAMETERS = {
    "TMP": (0, 0, 0),
    "DPT": (0, 0, 6),
    "SPFH": (0, 1, 0),
    "UGRD": (0, 2, 2),
    "VGRD": (0, 2, 3),
    "PRES": (0, 3, 0),
    "HGT": (0, 3, 5),
    "MSLMA": (0, 3, 198),
}
_SURFACE_TYPE_GROUND = 1
_SURFACE_TYPE_MSL = 101
_SURFACE_TYPE_ISOBARIC = 100
_SURFACE_TYPE_HEIGHT = 103


@dataclass(frozen=True)
class _GribField:
    product: Literal["prs", "sfc"]
    var: str
    level: str
    scale: float = 1.0

    @property
    def is_wind(self) -> bool:
        return self.var in ("UGRD", "VGRD")


# variable -> (GRIB var, scale)
_PRESSURE_FIELDS = {
    "temperature": ("TMP", 1.0),
    "specific_humidity": ("SPFH", 1.0),
    "u_component_of_wind": ("UGRD", 1.0),
    "v_component_of_wind": ("VGRD", 1.0),
    "geopotential": ("HGT", STANDARD_GRAVITY),
}
_SINGLE_LEVEL_FIELDS = {
    "10m_u_component_of_wind": _GribField("sfc", "UGRD", "10 m above ground"),
    "10m_v_component_of_wind": _GribField("sfc", "VGRD", "10 m above ground"),
    "2m_temperature": _GribField("sfc", "TMP", "2 m above ground"),
    "2m_dewpoint_temperature": _GribField("sfc", "DPT", "2 m above ground"),
    "surface_pressure": _GribField("sfc", "PRES", "surface"),
    "mean_sea_level_pressure": _GribField("sfc", "MSLMA", "mean sea level"),
}
_STATIC_FIELDS = {
    "geopotential_at_surface": _GribField("sfc", "HGT", "surface", STANDARD_GRAVITY),
}
_NEAR_SURFACE_HUMIDITY = _GribField("sfc", "SPFH", "2 m above ground")


@dataclass(frozen=True)
class _IdxEntry:
    offset: int
    date_hour: str
    var: str
    level: str
    forecast: str


def _parse_idx(text: str) -> list[_IdxEntry]:
    """Parse a wgrib2-style inventory, in file order."""
    entries: list[_IdxEntry] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(":")
        if len(parts) < 6 or not parts[2].startswith("d="):
            raise ValueError(f"malformed idx line: {line!r}")
        entry = _IdxEntry(int(parts[1]), parts[2][2:], parts[3], parts[4], parts[5])
        if entries and entry.offset < entries[-1].offset:
            raise ValueError(f"idx offsets go backwards at: {line!r}")
        entries.append(entry)
    return entries


def _forecast_descriptor(step: int) -> str:
    return "anl" if step == 0 else f"{step} hour fcst"


def _byte_range(
    entries: list[_IdxEntry], field: _GribField, cycle: ForecastCycle
) -> tuple[int, int | None]:
    """Inclusive (start, end) of the one matching message; end None means EOF."""
    forecast = _forecast_descriptor(cycle.step)
    matches = [
        entry
        for entry in entries
        if (entry.var, entry.level, entry.forecast) == (field.var, field.level, forecast)
    ]
    label = f"{field.var}:{field.level}:{forecast}"
    if len(matches) != 1:
        raise ValueError(f"expected exactly one idx match for {label}, found {len(matches)}")
    match = matches[0]
    expected = f"{cycle.date.replace('-', '')}{cycle.run_hour:02d}"
    if match.date_hour != expected:
        raise ValueError(f"idx entry for {label} is for {match.date_hour}, expected {expected}")
    # A shared offset is a GRIB submessage, which a byte range can't isolate.
    if sum(entry.offset == match.offset for entry in entries) != 1:
        raise ValueError(f"idx entry for {label} shares its byte offset with another entry")
    following = [entry.offset for entry in entries if entry.offset > match.offset]
    return match.offset, (following[0] - 1 if following else None)


def _http_get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    """GET a URL or byte range, retrying transient failures a bounded number of times."""
    headers = {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(_HTTP_ATTEMPTS):
        final = attempt == _HTTP_ATTEMPTS - 1
        try:
            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
                status = response.status
                content_range = response.headers.get("Content-Range") or ""
                body = response.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP_STATUS or final:
                raise
        except (OSError, http.client.HTTPException):
            if final:
                raise
        time.sleep(2**attempt)
    if start is not None:
        prefix = f"bytes {start}-" if end is None else f"bytes {start}-{end}/"
        if status != 206 or not content_range.startswith(prefix):
            raise OSError(
                f"{url}: range request not honored (status {status}, Content-Range {content_range!r})"
            )
    return body


def _is_complete_grib2(payload: bytes) -> bool:
    """True for exactly one whole GRIB2 message, by its own declared length."""
    if len(payload) < 20 or payload[:4] != b"GRIB" or payload[7] != 2:
        return False
    return int.from_bytes(payload[8:16], "big") == len(payload) and payload[-4:] == b"7777"


def _atomic_write(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def _expected_surface(field: _GribField) -> tuple[int, float | None]:
    """GRIB (surface type, value in SI units) for an idx level label."""
    if field.level == "surface":
        return _SURFACE_TYPE_GROUND, None
    if field.level == "mean sea level":
        return _SURFACE_TYPE_MSL, None
    if field.level.endswith(" mb"):
        return _SURFACE_TYPE_ISOBARIC, float(field.level[:-3]) * 100.0
    if field.level.endswith(" m above ground"):
        return _SURFACE_TYPE_HEIGHT, float(field.level.split()[0])
    raise ValueError(f"unknown level {field.level!r}")


def _validate_message(gid: int, field: _GribField, cycle: ForecastCycle) -> None:
    """Refuse a message that isn't the requested field on the expected grid."""
    import eccodes

    label = f"{field.var}:{field.level}"
    grid_type = eccodes.codes_get(gid, "gridType")
    if grid_type != "lambert":
        raise ValueError(f"{label}: gridType is {grid_type!r}, expected 'lambert'")
    for key, expected in (
        ("Nx", HRRR_NX),
        ("Ny", HRRR_NY),
        ("iScansNegatively", 0),
        ("jScansPositively", 1),
        ("jPointsAreConsecutive", 0),
        ("alternativeRowScanning", 0),
        ("shapeOfTheEarth", 6),
    ):
        actual = eccodes.codes_get_long(gid, key)
        if actual != expected:
            raise ValueError(f"{label}: {key} is {actual}, expected {expected}")
    for key, expected in (
        ("DxInMetres", HRRR_DX_M),
        ("DyInMetres", HRRR_DX_M),
        ("LoVInDegrees", HRRR_LOV_DEG % 360.0),
        ("LaDInDegrees", HRRR_LATIN_DEG),
        ("Latin1InDegrees", HRRR_LATIN_DEG),
        ("Latin2InDegrees", HRRR_LATIN_DEG),
        ("latitudeOfFirstGridPointInDegrees", HRRR_FIRST_LAT_DEG),
        ("longitudeOfFirstGridPointInDegrees", HRRR_FIRST_LON_DEG),
    ):
        actual = eccodes.codes_get_double(gid, key)
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-5):
            raise ValueError(f"{label}: {key} is {actual}, expected {expected}")

    parameter = tuple(
        eccodes.codes_get_long(gid, key) for key in ("discipline", "parameterCategory", "parameterNumber")
    )
    if parameter != _GRIB_PARAMETERS[field.var]:
        raise ValueError(f"{label}: GRIB parameter is {parameter}, expected {_GRIB_PARAMETERS[field.var]}")

    surface_type, surface_value = _expected_surface(field)
    actual_type = eccodes.codes_get_long(gid, "typeOfFirstFixedSurface")
    if actual_type != surface_type:
        raise ValueError(f"{label}: surface type is {actual_type}, expected {surface_type}")
    if surface_value is not None:
        scale = eccodes.codes_get_long(gid, "scaleFactorOfFirstFixedSurface")
        value = eccodes.codes_get_long(gid, "scaledValueOfFirstFixedSurface") * 10.0 ** (-scale)
        if not np.isclose(value, surface_value):
            raise ValueError(f"{label}: surface value is {value}, expected {surface_value}")

    date = int(cycle.date.replace("-", ""))
    actual_date = eccodes.codes_get_long(gid, "dataDate")
    actual_time = eccodes.codes_get_long(gid, "dataTime")
    if (actual_date, actual_time) != (date, cycle.run_hour * 100):
        raise ValueError(f"{label}: message is for {actual_date} {actual_time:04d}, expected {date} {cycle.run_hour * 100:04d}")
    step_range = eccodes.codes_get_string(gid, "stepRange")
    step_type = eccodes.codes_get_string(gid, "stepType")
    if step_range != str(cycle.step) or step_type != "instant":
        raise ValueError(f"{label}: step is {step_range!r} ({step_type}), expected instant {cycle.step}")

    if field.is_wind and eccodes.codes_get_long(gid, "uvRelativeToGrid") != 1:
        raise ValueError(f"{label}: uvRelativeToGrid must be 1")
    missing = eccodes.codes_get_long(gid, "numberOfMissing")
    if missing != 0:
        raise ValueError(f"{label}: {missing} missing values")


def _decode_message(payload: bytes, field: _GribField, cycle: ForecastCycle) -> np.ndarray:
    """Validated float64 values, shape (Ny, Nx), row 0 at the south edge."""
    import eccodes

    gid = eccodes.codes_new_from_message(payload)
    try:
        _validate_message(gid, field, cycle)
        values = eccodes.codes_get_values(gid)
    finally:
        eccodes.codes_release(gid)
    if values.size != HRRR_NX * HRRR_NY:
        raise ValueError(f"{field.var}:{field.level}: decoded {values.size} values")
    if not np.isfinite(values).all():
        raise ValueError(f"{field.var}:{field.level}: non-finite decoded values")
    return values.reshape(HRRR_NY, HRRR_NX).astype(np.float64)


def rotate_grid_to_earth(
    u_grid: np.ndarray, v_grid: np.ndarray, lon_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate HRRR grid-relative wind components to earth-relative."""
    lon_signed = (np.asarray(lon_deg) + 180.0) % 360.0 - 180.0
    theta = np.sin(np.radians(HRRR_LATIN_DEG)) * np.radians(lon_signed - HRRR_LOV_DEG)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return cos_theta * u_grid + sin_theta * v_grid, -sin_theta * u_grid + cos_theta * v_grid


def _block_mean(field: np.ndarray, coarsen: int) -> np.ndarray:
    """Average `coarsen`-sized blocks of a south-up array, anchored at the SW corner."""
    if coarsen == 1:
        return field
    rows, cols = field.shape[0] // coarsen, field.shape[1] // coarsen
    trimmed = field[: rows * coarsen, : cols * coarsen]
    return trimmed.reshape(rows, coarsen, cols, coarsen).mean(axis=(1, 3))


@lru_cache(maxsize=None)
def _eval_points_xy_south_up() -> tuple[np.ndarray, np.ndarray]:
    """Projected (x, y) of the eval grid's nodes, rows south to north."""
    grid = eval_grid_0p25()
    lon, lat = np.meshgrid((grid.lon + 180.0) % 360.0 - 180.0, grid.lat[::-1])
    to_lcc = pyproj.Transformer.from_crs("EPSG:4326", HRRR_CRS, always_xy=True)
    x, y = to_lcc.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def _interpolate_to_eval(field: np.ndarray, coarsen: int) -> np.ndarray:
    """Bilinear in projected x/y from a south-up native grid to the south-up eval grid."""
    source = native_grid(coarsen)
    y_axis = source.y[::-1]
    x, y = _eval_points_xy_south_up()
    outside = (x < source.x[0]) | (x > source.x[-1]) | (y < y_axis[0]) | (y > y_axis[-1])
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} of {outside.size} eval-grid nodes lie outside the cell-center "
            f"hull of {source.name}; bilinear interpolation would have to extrapolate"
        )
    interpolator = RegularGridInterpolator((y_axis, source.x), field, method="linear", bounds_error=True)
    return interpolator(np.column_stack((y.ravel(), x.ravel()))).reshape(x.shape)


class HRRRFieldSource(FieldSource):
    """HRRR CONUS fields in ERA5-style names and units, north-up on `grid`."""

    source_id = "hrrr"

    def __init__(
        self,
        cache_dir: str,
        coarsen: int = 1,
        wind_frame: Literal["earth", "grid"] = "earth",
        target: Literal["native", "latlon_0p25"] = "native",
        base_url: str = HRRR_BASE_URL,
    ):
        if wind_frame not in ("earth", "grid"):
            raise ValueError(f"wind_frame must be 'earth' or 'grid', got {wind_frame!r}")
        if target not in ("native", "latlon_0p25"):
            raise ValueError(f"target must be 'native' or 'latlon_0p25', got {target!r}")
        if target == "latlon_0p25" and (coarsen != 8 or wind_frame != "earth"):
            raise ValueError("target='latlon_0p25' requires coarsen=8 and wind_frame='earth'")
        # Validates `coarsen` for both targets.
        native = native_grid(coarsen)
        self._grid_spec = native if target == "native" else eval_grid_0p25()
        self._cache_dir = cache_dir
        self._coarsen = coarsen
        self._wind_frame = wind_frame
        self._target = target
        self._base_url = base_url.rstrip("/")
        self._field_cache: dict[tuple, np.ndarray] = {}
        self._idx_cache: dict[str, list[_IdxEntry]] = {}
        os.makedirs(cache_dir, exist_ok=True)

    @property
    def grid(self) -> GridSpec:
        return self._grid_spec

    def is_cycle_available(self, cycle: ForecastCycle) -> bool:
        """HEAD-check that both prs and sfc `.idx` files for `cycle` exist on NOAA S3."""
        for product in ("prs", "sfc"):
            url = f"{self._grib_url(product, cycle)}.idx"
            request = urllib.request.Request(url, method="HEAD")
            try:
                with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
                    if response.status != 200:
                        return False
            except (urllib.error.HTTPError, urllib.error.URLError, OSError, http.client.HTTPException):
                return False
        return True

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: ForecastCycle) -> np.ndarray:
        if variable not in _PRESSURE_FIELDS:
            raise ValueError(f"no HRRR pressure-level mapping for {variable!r}")
        var, scale = _PRESSURE_FIELDS[variable]
        return self._field(_GribField("prs", var, f"{level_hpa} mb", scale), cycle)

    def fetch_single_level(self, variable: str, cycle: ForecastCycle) -> np.ndarray:
        if variable not in _SINGLE_LEVEL_FIELDS:
            raise ValueError(f"no HRRR single-level mapping for {variable!r}")
        return self._field(_SINGLE_LEVEL_FIELDS[variable], cycle)

    def fetch_static(self, variable: str) -> np.ndarray:
        if variable not in _STATIC_FIELDS:
            raise ValueError(f"no HRRR static mapping for {variable!r}")
        return self._field(_STATIC_FIELDS[variable], HRRR_STATIC_CYCLE)

    def _field(self, field: _GribField, cycle: ForecastCycle) -> np.ndarray:
        key = (field, cycle.date, cycle.run_hour, cycle.step)
        if key not in self._field_cache:
            if field.is_wind:
                self._load_wind_pair(field, cycle)
            elif field.var == "SPFH" and field.product == "prs":
                self._field_cache[key] = self._finish(self._load_humidity(field, cycle))
            else:
                self._field_cache[key] = self._finish(self._load_native(field, cycle) * field.scale)
        return self._field_cache[key]

    def _load_humidity(self, field: _GribField, cycle: ForecastCycle) -> np.ndarray:
        """Isobaric q, held at its 2 m value below ground as in ERA5.

        HRRR's post-processor holds RH constant there instead, which puts far
        too much moisture under high terrain. Applied at 3 km, before coarsening.
        """
        q = self._load_native(field, cycle)
        surface_pressure = self._load_native(_SINGLE_LEVEL_FIELDS["surface_pressure"], cycle)
        below_ground = surface_pressure < _expected_surface(field)[1]
        return np.where(below_ground, self._load_native(_NEAR_SURFACE_HUMIDITY, cycle), q)

    def _load_wind_pair(self, field: _GribField, cycle: ForecastCycle) -> None:
        u_field, v_field = replace(field, var="UGRD"), replace(field, var="VGRD")
        u = self._load_native(u_field, cycle)
        v = self._load_native(v_field, cycle)
        if self._wind_frame == "earth":
            u, v = rotate_grid_to_earth(u, v, native_grid(1).lon[::-1])
        for component, values in ((u_field, u), (v_field, v)):
            self._field_cache[(component, cycle.date, cycle.run_hour, cycle.step)] = self._finish(values)

    def _finish(self, south_up: np.ndarray) -> np.ndarray:
        field = _block_mean(south_up, self._coarsen)
        if self._target == "latlon_0p25":
            field = _interpolate_to_eval(field, self._coarsen)
        result = np.flipud(field).astype(np.float32)
        if result.shape != self.grid.shape:
            raise ValueError(f"processed shape {result.shape} != grid shape {self.grid.shape}")
        # Cached and shared between callers, so an in-place edit must fail loudly.
        result.setflags(write=False)
        return result

    def _load_native(self, field: _GribField, cycle: ForecastCycle) -> np.ndarray:
        return _decode_message(self._message_bytes(field, cycle), field, cycle)

    def _message_bytes(self, field: _GribField, cycle: ForecastCycle) -> bytes:
        path = self._cache_path(field, cycle)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                cached = handle.read()
            if _is_complete_grib2(cached):
                return cached
            os.remove(path)
        payload = self._download(field, cycle)
        _atomic_write(path, payload)
        return payload

    def _download(self, field: _GribField, cycle: ForecastCycle) -> bytes:
        url = self._grib_url(field.product, cycle)
        if url not in self._idx_cache:
            self._idx_cache[url] = _parse_idx(_http_get(f"{url}.idx").decode("utf-8"))
        start, end = _byte_range(self._idx_cache[url], field, cycle)
        payload = _http_get(url, start, end)
        if not _is_complete_grib2(payload):
            raise ValueError(f"{url} bytes {start}-{end}: not one complete GRIB2 message")
        return payload

    def _grib_url(self, product: str, cycle: ForecastCycle) -> str:
        date = cycle.date.replace("-", "")
        return (
            f"{self._base_url}/hrrr.{date}/conus/"
            f"hrrr.t{cycle.run_hour:02d}z.wrf{product}f{cycle.step:02d}.grib2"
        )

    def _cache_path(self, field: _GribField, cycle: ForecastCycle) -> str:
        level = field.level.replace(" ", "_")
        name = f"t{cycle.run_hour:02d}z_{field.product}_f{cycle.step:02d}_{field.var}_{level}.grib2"
        return os.path.join(self._cache_dir, cycle.date.replace("-", ""), name)
