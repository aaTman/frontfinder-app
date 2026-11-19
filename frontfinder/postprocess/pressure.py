"""Mean-sea-level-pressure overlays (isobars, H and L centres) for the web viewer.

Writes the optional pressure layer for a published step: one GeoJSON file next to
the step's zarr store,

    <output_root>/<model>/<cycle>_f<step>.pressure.geojson

holding `kind: "isobar"` LineStrings (levels are multiples of `interval_hpa`,
flagged `major` every `major_hpa`) followed by `kind: "L"` / `"H"` Points. The
viewer fetches it only when its pressure option is on.

The field is IFS `msl` in pascals on the pipeline's grid convention (lat
90..-90 descending, lon 0..359.75); ascending latitude and signed lon
(-180..179.75) grids work too. A longitude axis that goes all the way round is
periodic: nothing is cut at the grid's edge, so a low that sits on the seam is
one low and an isobar that crosses the edge is one line. `surface_pressure_pa`
is an optional extra that drops centres over high terrain, where MSLP is an
extrapolation of the pressure field rather than something a barometer reads.

Conventions are the same as `postprocess.front_lines` (the small atomic-write
helper is duplicated rather than shared, so neither module imports the other):
compact GeoJSON, written atomically, and best effort at the write/attach level,
where a bad field is logged and never raised into inference or `latest.json`
publishing. Lines are never cut at the antimeridian either: a line's first
vertex is in [-180, 180) and its later vertices carry on from there, so an
isobar over the date line reads 175, 185. A centre is a local extremum that stands
on its own -- one that owns a closed isobar, in the sense of `pressure_centers` --
because a dot the viewer drops inside a contour this file also draws is the only kind
of centre worth drawing.

`attach_pressure` is the backfill hook: it re-points `latest.json` at pressure
files that already exist. Everything is deterministic -- the same field always
gives the same file.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os

import numpy as np
import scipy.ndimage
from skimage.measure import approximate_polygon, find_contours

logger = logging.getLogger(__name__)

PRESSURE_SUFFIX = ".pressure.geojson"
EARTH_RADIUS_KM = 6371.0
PASCALS_PER_HPA = 100.0

# Contouring runs on a longitude-padded copy of the field, so a closed isobar
# that crosses the grid's seam stays one piece. 20 degrees closes the isobars
# around a synoptic low or high; the padded copies are dropped again by
# `_isobar_features`' seam test. Marching squares only costs a few ms per level,
# so the extra columns are close to free.
PAD_DEGREES = 20.0
# An isobar shorter than this (in degrees of arc) is contour noise, not a
# pressure pattern the viewer should draw.
MIN_ISOBAR_DEG = 5.0
# A footprint that varies by less than this (hPa) is a flat expanse, not a centre:
# every pixel of it is a local extremum of the footprint, tied, and no prominence
# test can tell a flat field from a high. The threshold sits above the noise floor of
# the arithmetic (blurring a field that is constant to float32 pressure resolution,
# 8e-5 hPa, leaves a few millionths of a hPa across a footprint) and far below
# anything a pressure field really does -- a prune set by feel, 1 hPa say, would be as
# blind to a broad system as the ring mean it replaces was, because a high spanning a
# thousand kilometres varies by only a couple of tenths of a hPa across a 3 degree
# footprint and the broadest systems on a real chart by hundredths.
MIN_FOOTPRINT_RANGE_HPA = 1e-3


def pressure_filename(store_name: str) -> str:
    """`2026-10-31T06Z_f006.zarr` -> `2026-10-31T06Z_f006.pressure.geojson`."""
    if not store_name.endswith(".zarr"):
        raise ValueError(f"not a zarr store name: {store_name!r}")
    return store_name[: -len(".zarr")] + PRESSURE_SUFFIX


def _write_json_atomically(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def _is_periodic(lon: np.ndarray) -> bool:
    """Return whether evenly spaced longitudes go all the way around the globe."""
    return len(lon) > 1 and bool(np.isclose(len(lon) * abs(lon[1] - lon[0]), 360.0))


@dataclasses.dataclass(frozen=True)
class _Grid:
    """A smoothed MSLP field in hPa, laid out north to south.

    `field` is the smoothed full-resolution grid (centres come from it);
    `field_s` is the same field thinned by `stride` in both axes, which is what
    gets contoured. The `lat`/`lon` axes stay monotonic (latitude descending,
    longitude increasing west to east), so index arithmetic stands in for
    interpolation of the coordinate arrays.
    """

    lat: np.ndarray
    lon: np.ndarray
    field: np.ndarray
    lat_s: np.ndarray
    lon_s: np.ndarray
    field_s: np.ndarray
    periodic: bool
    stride: int

    @property
    def dlat(self) -> float:
        return float(self.lat[1] - self.lat[0])

    @property
    def dlon(self) -> float:
        return float(self.lon[1] - self.lon[0])


def _prepare(
    mslp_hpa: np.ndarray, lat: np.ndarray, lon: np.ndarray, smoothing_sigma_px: float = 2.0, stride: int = 1
) -> _Grid:
    """Validates the inputs, smooths the field and thins it for contouring.

    Smoothing is not cosmetic: MSLP over terrain is noisy enough that raw
    marching squares gives spaghetti, and a 2-pixel blur costs nothing at the
    scale isobars are drawn. Longitudes wrap on a global grid, so the smoothing
    carries across the seam too.

    Raises:
        ValueError: the field does not match the axes, or an axis is unusable.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    field = np.asarray(mslp_hpa, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(f"lat and lon must be 1-D, got {lat.ndim}-D and {lon.ndim}-D")
    if lat.size < 2 or lon.size < 2:
        raise ValueError(f"need at least a 2x2 grid, got {field.shape}")
    if field.shape != (lat.size, lon.size):
        raise ValueError(f"MSLP has shape {field.shape}, expected {(lat.size, lon.size)}")
    if not np.isfinite(field).all():
        raise ValueError("MSLP has non-finite values")
    if lat[0] < lat[-1]:  # ascending: work north to south like the rest of the pipeline
        lat, field = lat[::-1], field[::-1]
    if lon[0] > lon[-1]:
        raise ValueError("lon must increase west to east")
    periodic = _is_periodic(lon)
    if smoothing_sigma_px > 0:
        mode = ("nearest", "wrap") if periodic else "nearest"
        field = scipy.ndimage.gaussian_filter(field, sigma=smoothing_sigma_px, mode=mode)
    stride = max(1, int(stride))
    return _Grid(
        lat=lat,
        lon=lon,
        field=field,
        lat_s=lat[::stride],
        lon_s=lon[::stride],
        field_s=field[::stride, ::stride],
        periodic=periodic,
        stride=stride,
    )


def _sample(grid: _Grid, field: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Bilinear values at points; longitudes wrap on a global grid, latitudes clamp to the grid."""
    rows = np.clip((lats - grid.lat[0]) / grid.dlat, 0.0, grid.lat.size - 1.0)
    cols = (lons - grid.lon[0]) / grid.dlon
    if grid.periodic:
        # Repeat the first column past the last so points between them interpolate across the seam.
        field = np.concatenate([field, field[:, :1]], axis=1)
        cols = np.clip(((lons - grid.lon[0]) % 360.0) / grid.dlon, 0.0, grid.lon.size)
    else:
        cols = np.clip(cols, 0.0, grid.lon.size - 1.0)
    return scipy.ndimage.map_coordinates(field, [rows, cols], order=1, mode="nearest")


def _longitude_shift(lon: float) -> float:
    """Return the whole number of degrees that moves ``lon`` into [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0 - lon


def _arc_deg(lon: np.ndarray, lat: np.ndarray) -> float:
    """Return the length of a line in degrees of arc (longitudes scaled by cos latitude)."""
    return float(np.sum(np.hypot(np.diff(lon) * np.cos(np.radians(lat[:-1])), np.diff(lat))))


def _haversine_km(
    lon: np.ndarray | float | list[float],
    lat: np.ndarray | float | list[float],
    lon2: np.ndarray | list[float],
    lat2: np.ndarray | list[float],
) -> np.ndarray:
    """Great-circle distances (km) from each point of the first set to each point of the second."""
    lon1, lat1 = np.radians(np.atleast_1d(lon))[:, None], np.radians(np.atleast_1d(lat))[:, None]
    lon2, lat2 = np.radians(np.atleast_1d(lon2))[None, :], np.radians(np.atleast_1d(lat2))[None, :]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _isobar_features(
    grid: _Grid, interval_hpa: float, major_hpa: float, tolerance_deg: float
) -> list[dict]:
    """Marching squares over the (padded, thinned) field, one LineString Feature per isobar.

    A contour is dropped when its minimum longitude falls in the longitude pad,
    because that makes it the copy of a seam-crossing contour kept on the other
    side. A contour that touches both padded edges instead spans the whole
    globe (a zonal isobar, say) and has no other copy, so it is kept.
    """
    field = grid.field_s
    low, high = float(field.min()), float(field.max())
    # Levels strictly inside the data range: a level that equals an edge value
    # makes marching squares trace the array's border.
    levels = np.arange(np.ceil(low / interval_hpa) * interval_hpa, high + 1e-9, interval_hpa)
    levels = [level for level in levels if low < level < high]
    if not levels:  # a (near) constant field has no isobars, and no contours either
        return []

    dlat = float(grid.lat_s[1] - grid.lat_s[0])
    dlon = float(grid.lon_s[1] - grid.lon_s[0])
    pad = 0
    if grid.periodic:
        pad = max(1, int(round(PAD_DEGREES / abs(dlon))))
        if pad >= grid.lon_s.size:  # a grid thinner than the pad cannot be wrapped
            pad = 0
    padded = np.pad(grid.field_s, ((0, 0), (pad, pad)), mode="wrap") if pad else grid.field_s

    features: list[dict] = []
    for level in levels:
        for contour in find_contours(padded, level):
            rows, cols = contour[:, 0], contour[:, 1]
            if grid.periodic:
                min_lon = grid.lon_s[0] + (cols.min() - pad) * dlon
                spans_globe = cols.min() <= 1.0 and cols.max() >= padded.shape[1] - 2.0
                if min_lon < grid.lon_s[0] and not spans_globe:
                    continue  # the padded copy of a contour kept on the other side of the seam
            closed = np.allclose(contour[0], contour[-1])
            points = approximate_polygon(
                np.column_stack([grid.lon_s[0] + (cols - pad) * dlon, grid.lat_s[0] + rows * dlat]), tolerance_deg
            )
            # A straight isobar simplifies to its two ends, which is still a line; a closed
            # ring needs three corners to be a ring (marching squares repeats its first point).
            # Everything short is contour noise, whatever shape it has.
            if points.shape[0] < (4 if closed else 2) or _arc_deg(points[:, 0], points[:, 1]) < MIN_ISOBAR_DEG:
                continue
            lon, lat = points[:, 0] + _longitude_shift(float(points[0, 0])), points[:, 1]
            coordinates = np.round(np.column_stack([lon, lat]), 2)
            if coordinates[0, 0] >= 180.0:  # rounding the first vertex up to 180 is a cut again
                coordinates[:, 0] -= 360.0
            if closed:
                coordinates[-1] = coordinates[0]  # a closed isobar repeats its first coordinate
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coordinates.tolist()},
                    "properties": {
                        "kind": "isobar",
                        "hpa": int(round(level)),
                        "major": bool(np.isclose(level / major_hpa, round(level / major_hpa))),
                    },
                }
            )
    return features


def _prominent(window: np.ndarray, ci: int, cj: int, kind: str, prominence_hpa: float) -> bool:
    """Whether the candidate at (ci, cj) of `window` stands `prominence_hpa` on its own.

    For an L, flood everything in the window within `prominence_hpa` *above* the
    candidate's value and keep the piece the candidate is in. If that piece holds
    no lower pixel, the water around the candidate has nowhere to escape below its
    own level: the candidate has its own closed isobar and so a centre of its own.
    An H mirrors it with the flood coming from below. Pixels outside the window are
    never seen, so a region that reaches the window edge simply stops being judged
    out there: what drains a candidate is ground inside the window, and a candidate
    whose water escapes past the edge of the window stays qualified.
    """
    value = float(window[ci, cj])
    enclosing = window <= value + prominence_hpa if kind == "L" else window >= value - prominence_hpa
    labels, _ = scipy.ndimage.label(enclosing, structure=np.ones((3, 3), dtype=int))
    region = labels == labels[ci, cj]
    if kind == "L":
        return bool(float(window[region].min()) >= value - 1e-6)
    return bool(float(window[region].max()) <= value + 1e-6)


def _centre_features(
    grid: _Grid,
    surface_pressure_hpa: np.ndarray | None,
    footprint_deg: float,
    prominence_hpa: float,
    window_deg: float,
    min_separation_km: float,
    max_abs_lat: float,
    min_surface_hpa: float,
) -> list[dict]:
    """H and L Points from the smoothed full-resolution field, strongest first.

    A candidate is a local extremum of a small footprint (`footprint_deg`, a
    couple of degrees wide) -- that only picks the *shape* of the extrema, and on
    its own it is worthless: it fires at every 2 hPa wiggle of a flat tropical
    field and at every terrain bump. What decides whether a candidate is a centre
    is its **prominence**: how far the field has to rise (for an L) before the
    water poured in around it can escape to a level lower than the candidate
    itself. `prominence_hpa` of at least 4 hPa, one isobar interval, means the
    candidate carries a closed isobar of its own, which is exactly what the viewer
    will draw a centre inside.

    Prominence rather than the older "value below the mean of a ring at radius R"
    test, because a ring mean measures how *pointy* a system is and a real MSLP
    field is full of systems that are broad instead. The subtropical highs are the
    case in point: ~1033 hPa over a thousand kilometres of open ocean, so their
    centre is barely above the surrounding mean (a 5-6 degree ring mean gives a
    depth under the 4 hPa cutoff) and the ring test deletes the most persistent
    features on the map -- while a 2 hPa tropical wiggle, which has no closed
    isobar and no meaning, clears the same test on a locally noisier background.
    Prominence separates the two the way pressure actually works: the broad high
    has a large prominence (its 1032 hPa contour reaches far and only closes on
    itself), the wiggle has none.

    Each candidate is judged in a window of `window_deg` degrees either side: the
    window's longitudes wrap on a periodic grid, its latitudes are clipped to the
    grid, and everything outside it is invisible to the test. A cheap relief test
    comes first -- the field inside the window has to rise (`fall`, for an H) at
    least `prominence_hpa` above (below) the candidate at all, or the candidate has
    no closed isobar anywhere near it -- which is what keeps a flat or noisy field
    from reaching the prominence test once per pixel. Equal-valued plateau pixels can
    all pass the prominence test, so a greedy thinning per kind (strongest first --
    how far the value sits from 1013.25 hPa -- nothing within `min_separation_km`)
    leaves one centre per low or high.
    """
    field = grid.field
    dlat, dlon = abs(grid.dlat), abs(grid.dlon)
    radius_px = max(1, int(round(footprint_deg / dlat)))
    half_rows = max(1, int(round(window_deg / dlat)))
    half_cols = max(1, int(round(window_deg / dlon)))
    window_px = (2 * half_rows + 1, 2 * half_cols + 1)
    mode = ("nearest", "wrap") if grid.periodic else "nearest"
    low = scipy.ndimage.minimum_filter(field, size=2 * radius_px + 1, mode=mode)
    high = scipy.ndimage.maximum_filter(field, size=2 * radius_px + 1, mode=mode)
    # A neighbourhood that is flat to below the noise floor is a footprint extremum at
    # every pixel; drop those wholesale rather than judge a million ties one by one.
    varied = (high - low) > MIN_FOOTPRINT_RANGE_HPA
    # The relief prune: the extreme value in the whole window around a candidate, so a
    # candidate that never stands `prominence_hpa` below (above) it cannot enclose an
    # isobar of its own. A filter rather than a per-candidate loop, which is what makes
    # it cheap on a noisy field, and read at the window's scale rather than the
    # footprint's, which is what makes it safe on a broad system.
    relief_l = scipy.ndimage.maximum_filter(field, size=window_px, mode=mode) - field
    relief_h = field - scipy.ndimage.minimum_filter(field, size=window_px, mode=mode)
    valid = np.abs(grid.lat)[:, None] <= max_abs_lat
    surface = None if surface_pressure_hpa is None else np.asarray(surface_pressure_hpa, dtype=float)
    if surface is not None and surface.shape != field.shape:
        raise ValueError(f"surface pressure has shape {surface.shape}, expected {field.shape}")

    features: list[dict] = []
    for kind, candidate in (
        ("L", (field == low) & varied & (relief_l >= prominence_hpa) & valid),
        ("H", (field == high) & varied & (relief_h >= prominence_hpa) & valid),
    ):
        rows, cols = np.nonzero(candidate)
        if rows.size == 0:
            continue
        lat, lon, centre = grid.lat[rows], grid.lon[cols], field[rows, cols]
        # MSLP over high terrain is an extrapolation of the field, not a reading.
        on_ground = (
            _sample(grid, surface, lat, lon) < min_surface_hpa if surface is not None else np.zeros(rows.size, bool)
        )
        keep = ~on_ground
        for k in np.nonzero(~on_ground)[0]:
            window, ci, cj = _window(field, grid.periodic, rows[k], cols[k], half_rows, half_cols)
            keep[k] = _prominent(window, ci, cj, kind, prominence_hpa)
        lat, lon, centre = lat[keep], lon[keep], centre[keep]
        # Intensity for the thinning order: how far the centre sits from standard
        # atmosphere, so the deepest low and the strongest high are picked first.
        depth = 1013.25 - centre if kind == "L" else centre - 1013.25
        # Greedy thinning, strongest first: one centre per low, plateaus included.
        kept_lon: list[float] = []
        kept_lat: list[float] = []
        for i in np.argsort(-depth, kind="stable"):
            if kept_lon and float(_haversine_km(lon[i], lat[i], kept_lon, kept_lat).min()) < min_separation_km:
                continue
            kept_lon.append(float(lon[i]))
            kept_lat.append(float(lat[i]))
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [_wrap_lon(kept_lon[-1]), round(kept_lat[-1], 2)]},
                    "properties": {"kind": kind, "hpa": int(round(float(centre[i])))},
                }
            )
    return features


def _wrap_lon(lon: float) -> float:
    """Round a longitude into [-180, 180) at the two decimals the file carries."""
    wrapped = round((float(lon) + 180.0) % 360.0 - 180.0, 2)
    return wrapped - 360.0 if wrapped >= 180.0 else wrapped


def _window(
    field: np.ndarray, periodic: bool, row: int, col: int, half_rows: int, half_cols: int
) -> tuple[np.ndarray, int, int]:
    """The window around one pixel, plus the candidate's index inside it.

    Longitudes wrap on a periodic grid (columns are indices taken modulo the axis,
    so the window can straddle the seam), latitudes are clipped to the grid. On a
    non-periodic grid both axes clip. Whatever the window misses is invisible to
    the prominence test, which is the point: only nearby ground can drain it.
    """
    rows = slice(max(0, row - half_rows), min(field.shape[0], row + half_rows + 1))
    ci = row - rows.start
    if periodic:
        cj = half_cols
        cols = np.take(np.arange(field.shape[1]), np.arange(col - half_cols, col + half_cols + 1), mode="wrap")
    else:
        cols = slice(max(0, col - half_cols), min(field.shape[1], col + half_cols + 1))
        cj = col - cols.start
    return field[rows, cols], ci, cj


def isobars(
    mslp_hpa: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    interval_hpa: float = 4.0,
    major_hpa: float = 20.0,
    smoothing_sigma_px: float = 2.0,
    stride: int = 2,
    tolerance_deg: float = 0.03,
) -> list[dict]:
    """Isobars of an MSLP field (hPa) as GeoJSON LineString Features.

    Levels are multiples of `interval_hpa` strictly between the field's min and
    max, `major` marks the multiples of `major_hpa`. Contours come from the
    smoothed field thinned to every `stride`-th point (small files, and the
    smoothing the contours need is the same as the centres get), simplified to
    `tolerance_deg` in degree space. Coordinates are rounded to 2 decimals and
    longitudes are never cut: each line starts in [-180, 180) and carries on.
    """
    grid = _prepare(mslp_hpa, lat, lon, smoothing_sigma_px=smoothing_sigma_px, stride=stride)
    return _isobar_features(grid, interval_hpa, major_hpa, tolerance_deg)


def pressure_centers(
    mslp_hpa: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    surface_pressure_hpa: np.ndarray | None = None,
    footprint_deg: float = 3.0,
    prominence_hpa: float = 4.0,
    window_deg: float = 35.0,
    min_separation_km: float = 1500.0,
    max_abs_lat: float = 85.0,
    min_surface_hpa: float = 850.0,
    smoothing_sigma_px: float = 2.0,
) -> list[dict]:
    """High and low centres of an MSLP field (hPa) as GeoJSON Point Features.

    A candidate local extremum of the `footprint_deg` footprint is a centre once
    its prominence reaches `prominence_hpa`: the field has to rise that far before
    it can fall again below the candidate's own value, looked for inside a window of
    `window_deg` degrees either side (longitudes wrap on a global grid, latitudes
    clip to the grid, and whatever the window misses is invisible). Prominence
    rather than depth against a ring mean, because a ring mean rates a system by how
    pointy it is and throws away the broad ones: the subtropical highs sit some 20
    hPa above the general level of their hemisphere yet barely below the mean of a 6
    degree ring, while a 2 hPa wiggle of a flat tropical field looks deep by the
    same measure. One isobar interval of prominence is the natural cutoff, since it
    means the centre owns a closed isobar.

    L first, then H, each kind strongest first (farthest from 1013.25 hPa), never
    closer together than `min_separation_km`. `hpa` is the smoothed field's value at
    the centre. With `surface_pressure_hpa` given, candidates under
    `min_surface_hpa` are dropped, because MSLP there is an extrapolation off the
    terrain and its "centres" are mostly bumps in the ground.
    """
    grid = _prepare(mslp_hpa, lat, lon, smoothing_sigma_px=smoothing_sigma_px)
    return _centre_features(
        grid,
        surface_pressure_hpa,
        footprint_deg,
        prominence_hpa,
        window_deg,
        min_separation_km,
        max_abs_lat,
        min_surface_hpa,
    )


def build_pressure_geojson(
    mslp_pa: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    valid_time: str | None,
    surface_pressure_pa: np.ndarray | None = None,
) -> dict:
    """The pressure layer for one field: isobars, then centres.

    Args:
        mslp_pa: mean sea level pressure in pascals, shape (len(lat), len(lon)).
        lat: latitude axis, degrees, descending (90..-90) or ascending.
        lon: longitude axis, degrees, evenly spaced, ascending (0..359.75 or
            -180..179.75).
        valid_time: stored at the top level of the collection, or None.
        surface_pressure_pa: optional surface pressure in pascals, same shape,
            for the high-terrain test in `pressure_centers`.

    Raises:
        ValueError: the field or an axis is unusable.
    """
    hpa = np.asarray(mslp_pa, dtype=float) / PASCALS_PER_HPA
    surface_hpa = None if surface_pressure_pa is None else np.asarray(surface_pressure_pa, float) / PASCALS_PER_HPA
    features = isobars(hpa, lat, lon)
    features += pressure_centers(hpa, lat, lon, surface_pressure_hpa=surface_hpa)
    return {"type": "FeatureCollection", "valid_time": valid_time, "features": features}


def write_pressure_layer(
    model_dir: str,
    store_name: str,
    mslp_pa: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    valid_time: str,
    surface_pressure_pa: np.ndarray | None = None,
) -> str | None:
    """Writes `<model_dir>/<store>.pressure.geojson` atomically and returns its filename.

    Returns None (and logs) instead of raising: the pressure layer is an
    optional extra and must not be able to break inference or publishing.
    """
    try:
        name = pressure_filename(store_name)
        payload = build_pressure_geojson(mslp_pa, lat, lon, valid_time, surface_pressure_pa)
        _write_json_atomically(os.path.join(model_dir, name), payload)
    except Exception:
        logger.exception("pressure layer unavailable for %s -- continuing without it", store_name)
        return None
    return name


def attach_pressure(model_dir: str, step_entries: list[dict]) -> None:
    """Sets `entry["pressure"]` on every `latest.json` step entry that has a
    pressure file (and removes a stale `pressure` key from the ones that don't,
    or whose entry is too malformed to name a store). Never raises.
    """
    for entry in step_entries or ():
        try:
            name = pressure_filename(entry["store"])
            found = name if os.path.isfile(os.path.join(model_dir, name)) else None
        except Exception:
            logger.debug("pressure layer skipped for a step entry without a usable store name: %r", entry)
            found = None
        if found is None:
            entry.pop("pressure", None)
        else:
            entry["pressure"] = found
