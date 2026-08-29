"""Builds and writes the multiscale GeoZarr pyramid consumed by the
fronts.espr.ai maplibre viewer, using CarbonPlan's `topozarr` (server-side
pyramid builder, https://carbonplan.github.io/topozarr/) and rendered
client-side with `@carbonplan/zarr-layer`
(see https://carbonplan.org/blog/mapping-ocr-data).

IMPORTANT VERSION NOTE: topozarr is explicitly "Experimental, APIs may
change without notice." The docs site describes a `create_pyramid(...,
layer_hints=...)` API with a `Pyramid.write(store)` method and a
`ZarrLayerVarConfig` colormap/clim hint type -- but the latest PyPI release
at the time this was written (0.0.4) has neither: `create_pyramid()` takes
no `layer_hints` argument, and `Pyramid` only exposes `.datatree` +
`.encoding` (write manually via `datatree.to_zarr(store, encoding=...)`).
This module is written against the *installed* 0.0.4 API. Colormap/clim
styling is instead embedded as plain per-variable zarr attrs (`colormap`,
`clim`) that the frontend reads directly, so nothing here breaks if/when
topozarr's `layer_hints` API lands and this gets upgraded later -- re-check
`pip show topozarr` before bumping the pin in requirements.txt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
import xproj  # noqa: F401 -- registers the `.proj` accessor used below
import zarr
from topozarr import Pyramid, create_pyramid

from frontfinder.config.manifests import ModelManifest

DEFAULT_N_LEVELS = 6
CRS = "EPSG:4326"  # global IFS lat/lon grid; zarr-layer reprojects client-side

# topozarr's default target_chunk_bytes (512 KiB) gives level-0 chunks of
# ~360x361 cells -- fine on its own, but each level-0 chunk request pulls a
# 1/4-globe-wide block even when only a small area is visible (e.g. zoomed
# in on one country), which is wasteful on a slow connection. Smaller chunks
# mean panning/zooming only fetches the region actually on screen.
#
# NOTE, 2026-08-27: this was originally written chasing a theory that
# smaller chunks would also fix a @carbonplan/zarr-layer bug where phones
# (viewport narrower than ~800px) never load any data under MapLibre's
# globe projection. Confirmed live that theory was wrong -- an A/B test
# against a real rebuilt store showed chunk size doesn't move that failure
# threshold at all. That bug is worked around separately, client-side, in
# webapp/index.html (see unstickNarrowViewportDataLoad()). This smaller
# chunk size is kept anyway as a genuine, independent bandwidth win.
#
# topozarr's `get_ideal_dim` floors chunk dimensions at 128 cells regardless
# of how small target_chunk_bytes is asked to go, so 128 is the smallest
# chunk this library can produce -- target_chunk_bytes below is set to hit
# that floor exactly.
SMALL_CHUNK_BYTES = 64 * 1024  # 128*128 float32 cells exactly hits the floor

# Fixed per-class color styling (colormap name + [min, max] clim), embedded
# as zarr variable attrs since the installed topozarr version doesn't yet
# support `layer_hints`. clim=[0, 1] because these are softmax class
# probabilities. Matches conventional synoptic front-chart colors.
CLASS_STYLE: dict[str, dict] = {
    "cold": {"colormap": "blues", "clim": [0.0, 1.0]},
    "warm": {"colormap": "reds", "clim": [0.0, 1.0]},
    "stationary": {"colormap": "greens", "clim": [0.0, 1.0]},
    "occluded": {"colormap": "purples", "clim": [0.0, 1.0]},
}


@dataclass(frozen=True)
class FrontFields:
    """served-class probability grids, one 2D (lat, lon) array per class,
    keys must exactly match `manifest.served_classes`."""

    probabilities: dict[str, np.ndarray]
    lat: np.ndarray
    lon: np.ndarray
    valid_time: str  # ISO8601 -- the forecast valid time
    cycle_time: str  # ISO8601 -- the IFS cycle this run came from

    def __post_init__(self) -> None:
        shapes = {k: v.shape for k, v in self.probabilities.items()}
        expected = (len(self.lat), len(self.lon))
        bad = {k: s for k, s in shapes.items() if s != expected}
        if bad:
            raise ValueError(f"field shape(s) {bad} do not match (lat, lon) = {expected}")


def _roll_lon_0_360_to_signed(lon: np.ndarray, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reindexes a global 0..360-convention longitude axis (IFS-native --
    see `IFSFieldSource._lon = np.linspace(0.0, 359.75, 1440)`) to the
    standard -180..180 convention, ascending.

    2026-08-29: this is a serve-time-only fix for a webapp truncation bug,
    not an inference concern -- inference (tiling, antimeridian padding)
    runs entirely on the native 0..360 grid upstream, unaffected by this.
    @carbonplan/zarr-layer's region-visibility math projects the map
    viewport (always signed -180..180 from maplibregl) against the store's
    declared bounds through a plain identity CRS transform with no
    wraparound. With the store served in 0..360, every negative-longitude
    viewport -- all of the Americas, the Atlantic, western Europe --
    produced negative/out-of-range store-column indices and silently
    fetched zero tiles there: a hard vertical cutoff in the webapp that
    only went away once zoomed out far enough to trip the library's own
    "viewport wraps past +-180" fallback (which just fetches everything).
    Re-serving in -180..180 -- the convention both maplibre and
    @carbonplan/zarr-layer actually assume -- fixes it at every zoom/pan.
    See webapp/index.html's GRID_BOUNDS for the client-side half of this.

    No-op for a regional (non-global) `lon` -- mirrors the `is_global_lon`
    check in `inference.engine.run_tiled_inference`: only a lon axis that
    actually spans the full 360deg globe has a real antimeridian seam to
    roll across; a regional box's edges are just its edges.
    """
    n = len(lon)
    is_global_lon = n > 1 and np.isclose(float(lon[-1] - lon[0]) + float(lon[1] - lon[0]), 360.0, atol=1e-6)
    if not is_global_lon:
        return lon, arrays
    if n % 2 != 0:
        raise ValueError(f"expected an even-length global longitude axis, got {n}")
    shift = n // 2
    rolled_lon = np.roll(lon, shift)
    signed_lon = np.where(rolled_lon >= 180, rolled_lon - 360, rolled_lon)
    rolled_arrays = {k: np.roll(v, shift, axis=1) for k, v in arrays.items()}
    return signed_lon, rolled_arrays


def build_level0_dataset(fields: FrontFields, manifest: ModelManifest) -> xr.Dataset:
    """Assemble the native-resolution, CRS-tagged, style-annotated xr.Dataset
    for one model run."""
    missing = set(manifest.served_classes) - set(fields.probabilities)
    if missing:
        raise ValueError(f"missing served-class fields: {missing}")
    extra = set(fields.probabilities) - set(manifest.served_classes)
    if extra:
        raise ValueError(f"unexpected fields not in manifest.served_classes: {extra}")

    lon, probabilities = _roll_lon_0_360_to_signed(fields.lon, fields.probabilities)

    data_vars = {}
    for cls in manifest.served_classes:
        attrs = dict(CLASS_STYLE.get(cls, {}))
        data_vars[cls] = (("lat", "lon"), probabilities[cls].astype(np.float32), attrs)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"lat": fields.lat, "lon": lon},
        attrs={
            "model": manifest.name,
            "valid_time": fields.valid_time,
            "cycle_time": fields.cycle_time,
            "served_classes": list(manifest.served_classes),
            "source": "ECMWF IFS open-data 0.25deg",
        },
    )
    ds = ds.proj.assign_crs(spatial_ref=CRS)
    return ds


def build_front_pyramid(fields: FrontFields, manifest: ModelManifest, n_levels: int = DEFAULT_N_LEVELS) -> Pyramid:
    """Returns a topozarr `Pyramid` (datatree + encoding) for one model run."""
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    ds = build_level0_dataset(fields, manifest)
    return create_pyramid(
        ds, levels=n_levels, x_dim="lon", y_dim="lat", method="mean", target_chunk_bytes=SMALL_CHUNK_BYTES
    )


def write_front_pyramid(pyramid: Pyramid, store_path: str) -> None:
    """Write a topozarr pyramid to a zarr v3 store.

    2026-08-22: followed by an explicit `zarr.consolidate_metadata()` call.
    `datatree.to_zarr` already writes a `consolidated_metadata` block on
    every group node, but leaves each one's `metadata` dict EMPTY --
    confirmed live by inspecting a real written store's root `zarr.json`.
    That's the same "not part of the Zarr v3 spec yet" limitation the
    ZarrUserWarning at write time already flags (see test_run_cycle.py's
    warnings), and it means the frontend's zarr client (`zarrita`, via
    `@carbonplan/zarr-layer`'s `withMaybeConsolidatedMetadata`) can't
    actually use it to skip per-array metadata fetches -- it still issues
    one `zarr.json` GET per array per pyramid level it opens (up to 6
    levels x 4 classes) before it can even start fetching data, all on the
    critical path for first paint. Calling `consolidate_metadata()`
    explicitly (confirmed live against a real store) correctly populates
    the root's `consolidated_metadata.metadata` with every descendant
    array's real shape/chunks/codecs, keyed by path (e.g. "0/cold"), which
    the client library specifically looks for -- collapsing those N
    metadata round-trips into the one root `zarr.json` fetch it makes
    anyway.
    """
    pyramid.datatree.to_zarr(store_path, encoding=pyramid.encoding, mode="w")
    zarr.consolidate_metadata(store_path)
