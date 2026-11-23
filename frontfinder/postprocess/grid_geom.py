"""Pixel <-> lon/lat coordinate adapter for projected (curvilinear) grids.

`postprocess.splines` was written for rectilinear lat/lon grids: it reads
metric distances straight off 1-D `lat`/`lon` coordinate arrays and samples
fields bilinearly in (latitude, longitude) index space. A projected grid --
HRRR's native 3 km Lambert conformal CONUS grid -- has 1-D `x`/`y` coordinate
arrays in CRS metres and only 2-D per-cell `lat`/`lon` tables, so neither of
those reads works directly.

`NativeGridCoords` is the one adapter between the two worlds, defined in both
directions:

- pixel -> lon/lat (forward): a 2-D cell-centre `lat`/`lon` table (rows
  north-to-south, like the grids themselves), bilinearly interpolated for
  fractional pixels. `splines` uses it at whole-pixel positions for crest
  paths, where the table entries are exact; `pixel_to_lonlat` covers the
  fractional case.
- lon/lat -> pixel (inverse): project the point into the grid's CRS with
  pyproj, then interpolate between the monotonic 1-D `x`/`y` arrays.
  `splines.sample` uses it to read probability and orientation fields at
  points along a smoothed line, which live in lon/lat space.

For a tensor-product grid whose `x`/`y` are coordinates in its own CRS (any
projected grid HRRR-style output uses), the inverse is exact: lon/lat ->
(x, y) is the analytic projection inverse and (x, y) -> fractional index is
linear between neighbouring cell centres. Points outside the coordinate
ranges give NaN, matching what the regular 1-D `splines.fractional_index`
path does off-grid, so off-domain sampling points off a line's ends simply
read as missing instead of being clamped to the boundary.

This module deliberately knows nothing about which model produced a grid:
it is built either from an ingest `GridSpec` (forward table comes along for
free) or from just (`crs`, `x`, `y`) (forward table recomputed with pyproj,
bit-identical to how `ingest.hrrr_grid.native_grid` computed it).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pyproj


@lru_cache(maxsize=8)
def _to_crs(crs: str) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)


@lru_cache(maxsize=8)
def _to_lonlat(crs: str) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def _fractional_index(coord: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Where ``values`` fall along a monotonic (ascending or descending) 1-D
    coordinate, in index units; NaN outside the coordinate's range.

    Mirrors `splines.fractional_index`, which only ever sees 1-D lat/lon
    coordinates; this one is local so the adapter stays dependency-free.
    """
    steps = np.arange(len(coord), dtype=float)
    if coord[0] > coord[-1]:
        coord, steps = coord[::-1], steps[::-1]
    index = np.interp(values, coord, steps)
    return np.where((values >= coord[0]) & (values <= coord[-1]), index, np.nan)


@dataclass(frozen=True, eq=False)
class NativeGridCoords:
    """Bidirectional pixel <-> lon/lat mapping for a projected tensor-product grid.

    Attributes:
        crs: Anything `pyproj.CRS.from_user_input` accepts (HRRR's is a proj
            string); the CRS `x`/`y` are expressed in.
        x: 1-D strictly monotonic column coordinates in CRS units (metres).
        y: 1-D strictly monotonic row coordinates in CRS units, ordered like
            the rows (descending for the north-up grids downstream expects).
        lon: 2-D (rows, cols) cell-centre longitudes in degrees, or None when
            only the inverse direction is needed (field sampling).
        lat: 2-D cell-centre latitudes in degrees, or None like `lon`.
    """

    crs: str
    x: np.ndarray
    y: np.ndarray
    lon: np.ndarray | None = None
    lat: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name, axis in (("x", self.x), ("y", self.y)):
            if axis.ndim != 1 or len(axis) < 2:
                raise ValueError(f"{name} must be a 1-D coordinate array of length >= 2, got shape {axis.shape}")
            steps = np.diff(axis)
            if not np.all(steps > 0) and not np.all(steps < 0):
                raise ValueError(f"{name} must be strictly monotonic")
        if (self.lon is None) != (self.lat is None):
            raise ValueError("lon and lat tables must both be given or both be None")
        if self.lon is not None and (self.lon.shape != self.shape or self.lat.shape != self.shape):
            raise ValueError(f"lat/lon tables must match the grid shape {self.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.y), len(self.x))

    @property
    def data_dims(self) -> tuple[str, str]:
        """The dims a field sampled through this adapter is laid out in."""
        return ("y", "x")

    @classmethod
    def from_crs(cls, crs: str, x: np.ndarray, y: np.ndarray) -> "NativeGridCoords":
        """Build with the forward (pixel -> lon/lat) table computed here.

        The table is `pyproj`-deterministic: recomputing it from the same
        (`crs`, `x`, `y`) reproduces the ingest-time tables bit for bit.
        """
        xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        lon, lat = _to_lonlat(crs).transform(xx, yy)
        return cls(crs=crs, x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float), lon=lon, lat=lat)

    @classmethod
    def from_grid_spec(cls, grid) -> "NativeGridCoords":
        """Build from an ingest `GridSpec`, reusing its 2-D lat/lon tables."""
        if grid.lat.ndim != 2:
            raise ValueError("a rectilinear grid needs no adapter; pass geom=None through splines instead")
        return cls(crs=grid.crs, x=np.asarray(grid.x, dtype=float), y=np.asarray(grid.y, dtype=float), lon=grid.lon, lat=grid.lat)

    def lonlat_to_pixel(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fractional (row, col) indices of lon/lat points; NaN off-grid.

        Longitudes are read modulo 360 degrees into [-180, 180), matching the
        signed convention the stores and GeoJSON use, so a smoothed line
        unwrapped past the antimeridian (only reachable by global grids)
        still maps to its own cell.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x_m, y_m = _to_crs(self.crs).transform((lon + 180.0) % 360.0 - 180.0, lat)
        rows = _fractional_index(self.y, y_m)
        cols = _fractional_index(self.x, x_m)
        off = ~np.isfinite(lat)
        rows, cols = np.where(off, np.nan, rows), np.where(off, np.nan, cols)
        return rows, cols

    def pixel_to_lonlat(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Longitudes and latitudes of (possibly fractional) pixel positions;
        NaN outside the grid. Needs the forward table (`lon`/`lat` not None).

        Bilinear in index space; as with the inverse, the linear blend
        between neighbouring cell centres is only an approximation of the
        projection in between them -- good to well under a metre for
        neighbouring 3 km cells, which is why whole-pixel positions read the
        table entries exactly instead.
        """
        if self.lon is None or self.lat is None:
            raise ValueError("this adapter has no forward lat/lon table")
        from scipy.ndimage import map_coordinates

        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)
        inside = (
            np.isfinite(rows) & np.isfinite(cols) & (rows >= 0) & (rows <= len(self.y) - 1) & (cols >= 0) & (cols <= len(self.x) - 1)
        )
        lon = np.full(rows.shape, np.nan)
        lat = np.full(rows.shape, np.nan)
        if inside.any():
            lon[inside] = map_coordinates(self.lon, [rows[inside], cols[inside]], order=1)
            lat[inside] = map_coordinates(self.lat, [rows[inside], cols[inside]], order=1)
        return lon, lat
