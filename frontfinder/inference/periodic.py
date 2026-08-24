"""Vendored `PeriodicBoundaryIndex` (from `fronts.utils`, branch
`feat/2.0.0`, https://github.com/aaTman/fronts/blob/feat/2.0.0/src/fronts/utils.py)
plus a small circular-padding helper built on top of it, used to give tiled
inference real context across the antimeridian-of-the-array at the prime
meridian (longitude 0/360).

Why vendor instead of depending on `fronts`: same reasoning as
`keras_compat.py` -- this serving repo intentionally doesn't depend on the
training repo. `PeriodicBoundaryIndex` and `attach_periodic_lon_index` are
copied verbatim (2026-08-23) since they're self-contained (only depend on
xarray's own internals, not on anything else in `fronts`).

2026-08-23 root cause this fixes: `frontfinder/inference/tiling.py`
generates tiles over the assembled (lat, lon, channel) grid as a flat
rectangle -- column 0 (lon=0) and column -1 (lon=359.75) are physically
adjacent on the globe but share no context or blend overlap, since neither
`generate_tiles` nor `stitch` knows the longitude axis wraps. Reported live:
a seam artifact at the prime meridian spanning nearly the full latitude
range. `circularly_pad_longitude` extends the assembled grid by `overlap`
columns on each side, sourced from the opposite edge via
`PeriodicBoundaryIndex`'s wrap-aware `.sel()`, so tiles straddling the seam
get real neighboring data instead of a hard edge; `engine.run_tiled_inference`
crops the padding back off after stitching.
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import xarray as xr
from xarray.core.indexes import IndexSelResult, PandasIndex, _query_slice
from xarray.core.indexing import _expand_slice

_XArray = TypeVar("_XArray", xr.Dataset, xr.DataArray)


class PeriodicBoundaryIndex(PandasIndex):
    """xarray index for a 1-D coordinate that wraps at a period.

    Subclasses PandasIndex and intercepts slice queries so a selection
    that crosses the period boundary (e.g. longitude 350 to 10) is
    returned as two concatenated index arrays rather than an empty
    slice.
    """

    period: float
    _min: float
    _max: float

    __slots__ = ("_max", "_min", "coord_dtype", "dim", "index", "period")

    def __init__(self, *args, period=360, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self._min = self.index.min()
        self._max = self.index.max()

    @classmethod
    def from_variables(cls, variables, options):
        """Construct index from coordinate variables, reading period from options."""
        obj = super().from_variables(variables, options={})
        obj.period = options.get("period", obj.period)  # pyrefly: ignore[missing-attribute]
        return obj

    def _wrap_periodically(self, label_value: float) -> float:
        # Reduce ``label_value`` into ``[_min, _min + period)``. The
        # earlier formulation used ``label - _max`` which silently
        # shifted in-range labels by ``period - (_max - _min)`` (one
        # grid step on a 0-360 ERA5 axis where ``_max=359.75``).
        # ``label - _min`` is the textbook periodic remap and works
        # for both 0-360 and -180/180 axes.
        return self._min + (label_value - self._min) % self.period

    def _split_slice_across_boundary(self, label: slice) -> np.ndarray:
        """Return concatenated integer indices for a slice that wraps."""
        first_slice = slice(label.start, self._max, label.step)
        second_slice = slice(self._min, label.stop, label.step)

        first_as_index_slice = _query_slice(self.index, first_slice)
        second_as_index_slice = _query_slice(self.index, second_slice)

        first_as_indices = _expand_slice(first_as_index_slice, self.index.size)
        second_as_indices = _expand_slice(second_as_index_slice, self.index.size)

        return np.concatenate([first_as_indices, second_as_indices])

    def sel(self, labels: dict[Any, Any], method=None, tolerance=None) -> IndexSelResult:
        """Remap out-of-range labels back into the index range."""
        assert len(labels) == 1
        coord_name, label = next(iter(labels.items()))

        if isinstance(label, slice):
            start, stop, step = label.start, label.stop, label.step
            if start is None or stop is None:
                return super().sel({coord_name: label})
            if stop < start:
                return super().sel({coord_name: []})

            assert self._min < self._max

            wrapped_start = self._wrap_periodically(label.start)
            wrapped_stop = self._wrap_periodically(label.stop)
            wrapped_label = slice(wrapped_start, wrapped_stop, step)

            if wrapped_start < wrapped_stop:
                return super().sel({coord_name: wrapped_label})
            # Slice crosses the wrap boundary; split in two.
            wrapped_indices = self._split_slice_across_boundary(wrapped_label)
            return IndexSelResult({self.dim: wrapped_indices})

        wrapped_label = self._wrap_periodically(label)  # type: ignore
        return super().sel({coord_name: wrapped_label}, method=method, tolerance=tolerance)

    def __repr__(self) -> str:
        """Return string representation showing the period."""
        return f"PeriodicBoundaryIndex(period={self.period})"


def attach_periodic_lon_index(data: _XArray, lon_dim: str = "lon") -> _XArray:
    """Attach a 360deg-period `PeriodicBoundaryIndex` to `lon_dim`.

    Replaces the default `PandasIndex` so wrap-crossing
    `.sel(lon=slice(...))` queries work, e.g. `slice(-8.0, -0.25)` correctly
    wraps to the last few columns near 360deg.
    """
    return data.drop_indexes(lon_dim).set_xindex(lon_dim, index_cls=PeriodicBoundaryIndex, period=360)


def circularly_pad_longitude(grid: np.ndarray, lon_deg: np.ndarray, pad_pixels: int) -> np.ndarray:
    """Extend `grid` (lat, lon, channel) by `pad_pixels` columns on each side
    of the longitude axis, sourced from the opposite edge via
    `PeriodicBoundaryIndex` -- so the returned grid has no seam at the
    lon=0/360 array boundary: it wraps like the sphere it's sampled from.

    `lon_deg` must be the grid's own 1-D longitude coordinate (uniformly
    spaced, ascending, e.g. IFS open-data's 0.0..359.75deg 0.25deg grid).
    Returns an array of shape (lat, lon + 2*pad_pixels, channel); the
    original data occupies columns [pad_pixels : pad_pixels + len(lon_deg)].
    """
    if pad_pixels <= 0:
        return grid
    step = float(lon_deg[1] - lon_deg[0])
    pad_deg = pad_pixels * step

    da = xr.DataArray(grid, dims=("lat", "lon", "channel"), coords={"lon": lon_deg})
    da = attach_periodic_lon_index(da)
    # Negative/overflowing labels wrap via PeriodicBoundaryIndex -- these
    # pull exactly the tail/head bands adjacent to the seam.
    left = da.sel(lon=slice(-pad_deg, -step))
    right = da.sel(lon=slice(360.0, 360.0 + pad_deg - step))
    if left.sizes["lon"] != pad_pixels or right.sizes["lon"] != pad_pixels:
        raise AssertionError(
            f"circular pad produced {left.sizes['lon']}/{right.sizes['lon']} columns, "
            f"expected {pad_pixels} each -- lon_deg may not be a uniform, full 0..360deg grid"
        )
    return np.concatenate([left.values, da.values, right.values], axis=1)
