"""Model-agnostic contract for forecast field sources.

Everything downstream of ingest (channel assembly, inference, output writing)
should depend on `FieldSource` and `GridSpec` rather than on one NWP model's
grid. A new model is one `FieldSource` subclass that hands back fields in the
ERA5-style variable names and units the trained networks expect.

Real sources return arrays shaped `grid.shape` with rows running north to
south and columns west to east, because that's the orientation the networks
were trained on. Any flip or wind-frame conversion a model's native output
needs happens inside its source, never downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForecastCycle:
    """One forecast init time and lead, e.g. 2026-08-19 12Z f006."""

    date: str  # "YYYY-MM-DD"
    run_hour: int  # 0-23; a source's own cycle type may narrow this
    step: int = 0  # forecast lead time in hours; 0 == analysis/T+0

    def __post_init__(self) -> None:
        if not 0 <= self.run_hour <= 23:
            raise ValueError(f"run_hour must be in 0..23, got {self.run_hour}")
        # raises ValueError if malformed
        datetime.strptime(self.date, "%Y-%m-%d")


@dataclass(frozen=True, eq=False)
class GridSpec:
    """Where a source's array cells sit on the earth.

    `x`/`y` are the 1-D column/row coordinates in `crs` units. `lat`/`lon`
    are 1-D for a rectilinear lat/lon grid and 2-D `(len(y), len(x))` for a
    projected one. `crs` is anything `pyproj.CRS.from_user_input` accepts, so
    output writers and map layers can place a projected grid without knowing
    which model produced it.
    """

    name: str
    crs: str
    x: np.ndarray
    y: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError(f"{self.name}: x and y must be 1-D")
        if self.lat.ndim != self.lon.ndim or self.lat.ndim not in (1, 2):
            raise ValueError(f"{self.name}: lat and lon must both be 1-D or both be 2-D")
        if self.lat.ndim == 1:
            if (len(self.lat), len(self.lon)) != self.shape:
                raise ValueError(f"{self.name}: 1-D lat/lon do not match grid shape {self.shape}")
        elif self.lat.shape != self.shape or self.lon.shape != self.shape:
            raise ValueError(f"{self.name}: 2-D lat/lon do not match grid shape {self.shape}")

    @classmethod
    def regular_latlon(cls, name: str, lat: np.ndarray, lon: np.ndarray) -> "GridSpec":
        return cls(name=name, crs="EPSG:4326", x=lon, y=lat, lat=lat, lon=lon)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.y), len(self.x))

    @property
    def is_rectilinear(self) -> bool:
        return self.lat.ndim == 1

    @property
    def is_global(self) -> bool:
        """True only for a rectilinear grid whose longitudes wrap the full 360deg."""
        if not self.is_rectilinear or len(self.lon) < 2:
            return False
        span = float(self.lon[-1] - self.lon[0]) + float(self.lon[1] - self.lon[0])
        return bool(np.isclose(span, 360.0, atol=1e-6))

    def corners(self) -> dict[str, tuple[float, float]]:
        """(lon, lat) of the four corner cells, longitude signed to -180..180."""
        if self.is_rectilinear:
            lat, lon = np.meshgrid(self.lat, self.lon, indexing="ij")
        else:
            lat, lon = self.lat, self.lon
        cells = {"nw": (0, 0), "ne": (0, -1), "se": (-1, -1), "sw": (-1, 0)}
        return {
            key: (float((lon[cell] + 180.0) % 360.0 - 180.0), float(lat[cell]))
            for key, cell in cells.items()
        }

    def to_metadata(self) -> dict[str, Any]:
        """JSON-safe description, stored next to outputs so a later reader can
        place them without re-deriving the grid from its source."""
        return {
            "name": self.name,
            "crs": self.crs,
            "shape": list(self.shape),
            "x": _axis_metadata(self.x),
            "y": _axis_metadata(self.y),
            "corners": {key: list(value) for key, value in self.corners().items()},
        }


def _axis_metadata(axis: np.ndarray) -> dict[str, float | int]:
    step = float(axis[1] - axis[0]) if len(axis) > 1 else 0.0
    return {"start": float(axis[0]), "step": step, "size": int(len(axis))}


class FieldSource(ABC):
    """A forecast model's fields on one fixed grid."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Short stable identifier, e.g. "ifs" or "hrrr". A subclass can set
        this as a plain class attribute."""

    @property
    @abstractmethod
    def grid(self) -> GridSpec: ...

    @property
    def lat(self) -> np.ndarray:
        return self.grid.lat

    @property
    def lon(self) -> np.ndarray:
        return self.grid.lon

    @abstractmethod
    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: ForecastCycle) -> np.ndarray: ...

    @abstractmethod
    def fetch_single_level(self, variable: str, cycle: ForecastCycle) -> np.ndarray: ...

    @abstractmethod
    def fetch_static(self, variable: str) -> np.ndarray:
        """A physically invariant single-level field, the same for every cycle."""


def field_shape(source: Any) -> tuple[int, int]:
    """(rows, cols) of a source's fields. Falls back to 1-D `lat`/`lon` for
    lightweight wrappers that don't subclass `FieldSource`."""
    if isinstance(source, FieldSource):
        return source.grid.shape
    return (len(source.lat), len(source.lon))
