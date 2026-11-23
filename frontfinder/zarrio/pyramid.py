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

import os
from dataclasses import dataclass

import numpy as np
import xarray as xr
import xproj  # noqa: F401 -- registers the `.proj` accessor used below
import zarr
from topozarr import Pyramid, create_pyramid

from frontfinder.config.manifests import ModelManifest
from frontfinder.ingest.base import GridSpec

DEFAULT_N_LEVELS = 6
REGIONAL_N_LEVELS = 3  # a 132x224 grid has no use for a coarser overview
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

# Packed uint8 probability grids (2026-11-16). Class probabilities are
# float32 on the wire, but zstd compresses them badly: one level-0 shard per
# class was ~3.27 MB, and first paint pulls ~7 of them. Rounding to uint8 is
# visually lossless here -- the viewer's display bins are 0.1 wide and it
# discards below ~0.01 -- and re-encodes ~35x smaller (measured: 94 KB vs
# 3.27 MB on a real shard).
#
# `scale_factor`/`add_offset`/`_FillValue` are plain CF packing attrs on each
# served-class array; @carbonplan/zarr-layer reads them straight out of
# `getLevelMetadata`/consolidated metadata and unpacks client-side (standard
# CF convention: value = packed * scale_factor + add_offset). Valid
# probabilities are packed to 0..254 with scale_factor 1/254, so max abs
# error is 0.5/254 = 0.00197 (well under the <= 0.004 budget and half the
# error of 1/255), and 1.0 still round-trips exactly. 255 is reserved as the
# uint8 NaN sentinel (`_FillValue`) so NaNs stay distinguishable from a real
# 1.0 -- with scale_factor 1/255 that sentinel would collide with 255 and
# force clipping 1.0 down to 254, wasting a code point and adding error
# exactly at the most visible end of the ramp. Downsampling (topozarr's mean
# resampling) still runs on the float32 grid; packing happens only at write
# time, after every pyramid level is built.
PACKED_DTYPE = np.dtype(np.uint8)
PACKED_SCALE_FACTOR = 1.0 / 254.0
PACKED_ADD_OFFSET = 0.0
PACKED_FILL_VALUE = 255

# 2026-11-16: packing is OPT-IN on the live write path. The production
# webapp (webapp/index.html) predates the uint8 handling in webapp/dev/ --
# it doesn't pin zarr-layer's fillValue to the 255 sentinel and its tooltip
# drops decoded-zero rows -- so flipping every new run to uint8 before that
# client is promoted would show a saturated blob at any NaN and lose
# zero-probability tooltip rows. Set this env var (or pass pack_uint8=True)
# once the dev viewer has been promoted. Offline packing for the /dev data
# tree (scripts/refresh_dev_data.sh) is unaffected: it calls
# write_quantized_datatree directly.
PACK_UINT8_ENV = "FRONTFINDER_PACK_UINT8"

# Fixed per-class color styling (colormap name + [min, max] clim), embedded
# as zarr variable attrs since the installed topozarr version doesn't yet
# support `layer_hints`. clim=[0, 1] because these are softmax class
# probabilities. Matches conventional synoptic front-chart colors.
CLASS_STYLE: dict[str, dict] = {
    "cold": {"colormap": "blues", "clim": [0.0, 1.0]},
    "warm": {"colormap": "reds", "clim": [0.0, 1.0]},
    "stationary": {"colormap": "greens", "clim": [0.0, 1.0]},
    "occluded": {"colormap": "purples", "clim": [0.0, 1.0]},
    "dryline": {"colormap": "oranges", "clim": [0.0, 1.0]},
    "trough": {"colormap": "greys", "clim": [0.0, 1.0]},
    "tropical_trough": {"colormap": "ylorbr", "clim": [0.0, 1.0]},
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


@dataclass(frozen=True)
class RegionalFrontFields:
    """Served-class grids on a projected `GridSpec`, north-up."""

    probabilities: dict[str, np.ndarray]
    grid: GridSpec
    valid_time: str
    cycle_time: str
    source: str


def build_regional_pyramid(
    fields: RegionalFrontFields, manifest: ModelManifest, n_levels: int = REGIONAL_N_LEVELS
) -> Pyramid:
    """Pyramid for a regional grid, kept in its own CRS with (y, x) dims so the
    viewer reprojects it client-side instead of us resampling it to lat/lon."""
    if set(fields.probabilities) != set(manifest.served_classes):
        raise ValueError(f"fields {sorted(fields.probabilities)} != served classes {sorted(manifest.served_classes)}")
    bad = {k: v.shape for k, v in fields.probabilities.items() if v.shape != fields.grid.shape}
    if bad:
        raise ValueError(f"field shape(s) {bad} do not match grid shape {fields.grid.shape}")
    data_vars = {
        cls: (("y", "x"), fields.probabilities[cls].astype(np.float32), dict(CLASS_STYLE.get(cls, {})))
        for cls in manifest.served_classes
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"y": fields.grid.y, "x": fields.grid.x},
        attrs={
            "model": manifest.name,
            "valid_time": fields.valid_time,
            "cycle_time": fields.cycle_time,
            "served_classes": list(manifest.served_classes),
            "source": fields.source,
            "grid": fields.grid.name,
        },
    )
    ds = ds.proj.assign_crs(spatial_ref=fields.grid.crs)
    return create_pyramid(
        ds, levels=n_levels, x_dim="x", y_dim="y", method="mean", target_chunk_bytes=SMALL_CHUNK_BYTES
    )


def _served_classes_for_level(node: xr.DataTree) -> list[str]:
    """The served-class names carried in a pyramid level's group attrs."""
    served = node.attrs.get("served_classes") or node.to_dataset().attrs.get("served_classes")
    if not served:
        raise ValueError(f"pyramid level {node.path!r} has no served_classes attrs")
    return [str(cls) for cls in served]


def _pack_probability_variable(var: xr.DataArray) -> xr.DataArray:
    """Round one float32 [0, 1] probability field onto the packed uint8 grid.

    Clips before casting so a stray out-of-range value lands on 0/254
    instead of wrapping around in the uint8 cast, and fills NaN with
    PACKED_FILL_VALUE (255) -- see the PACKED_* comment at the top of this
    module for why 1/254 + a 255 sentinel is the encoding used. All
    operations stay lazy on dask-backed (downsampled) fields.
    """
    packed = (
        (var.clip(0.0, 1.0) * 254.0)
        .round()
        .fillna(float(PACKED_FILL_VALUE))
        .astype(PACKED_DTYPE)
    )
    attrs = dict(var.attrs)
    attrs.update(
        scale_factor=PACKED_SCALE_FACTOR,
        add_offset=PACKED_ADD_OFFSET,
        _FillValue=PACKED_FILL_VALUE,
    )
    packed.attrs = attrs
    return packed


def quantize_datatree(datatree: xr.DataTree, base_encoding: dict | None = None) -> dict:
    """Pack every served-class variable at every pyramid level to uint8, in place.

    Mutates the served-class data vars of `datatree` and returns the
    `datatree.to_zarr(..., encoding=...)` mapping. Coordinates and the
    pyramid's CRS variable (lat/lon/spatial_ref) are left untouched -- only
    the actual served-class grids are packed.

    Shared by write_front_pyramid (fresh pyramids straight out of topozarr)
    and scripts/requantize_store.py (already-published float32 stores), so
    both paths get byte-identical packing/metadata. `base_encoding` is
    topozarr's per-level `chunks`/`shards` mapping for a fresh pyramid; when
    omitted (the requantize path) the layout is taken from each variable's
    own read-time `.encoding`.

    2026-11-16: deliberately not specifying a compressor here. The current
    default (zstd level 0, which zstd itself treats as "default" == level 3)
    produced byte-identical output to an explicit ZstdCodec(level=3) on a
    real quantized store -- 1,003,437 bytes either way for a whole step --
    with identical decode speed, so the explicit level is pure noise.
    """
    encoding: dict[str, dict[str, dict]] = {}
    for path in datatree.groups:
        if path in ("", "/"):
            continue
        node = datatree[path]
        level_encoding: dict[str, dict] = {}
        for cls in _served_classes_for_level(node):
            var = node[cls]
            base = (base_encoding or {}).get(path, {}).get(cls, {})
            if not base:
                base = var.encoding
            # Keep topozarr's/the source store's existing chunk + shard
            # layout; zarr defaults the rest (zstd inner codec, bytes codec).
            level_encoding[cls] = {k: base[k] for k in ("chunks", "shards") if k in base}
            # Packing in place is fine: both callers build (or load) the tree
            # for a one-shot write and never hand it on.
            node[cls] = _pack_probability_variable(var)
        if level_encoding:
            encoding[path] = level_encoding
    return encoding


def write_quantized_datatree(
    datatree: xr.DataTree, store_path: str, base_encoding: dict | None = None
) -> None:
    """Write a pyramid datatree with every served class uint8-packed, then
    consolidate metadata (see write_front_pyramid's docstring for why the
    explicit consolidate call matters)."""
    encoding = quantize_datatree(datatree, base_encoding)
    datatree.to_zarr(store_path, encoding=encoding, mode="w")
    zarr.consolidate_metadata(store_path)


def pack_uint8_enabled(pack_uint8: bool | None = None) -> bool:
    """Whether a written store should be uint8-packed: an explicit
    `pack_uint8` wins, else the FRONTFINDER_PACK_UINT8 env var (default off)."""
    if pack_uint8 is None:
        pack_uint8 = os.environ.get(PACK_UINT8_ENV, "").strip().lower() in ("1", "true", "yes", "on")
    return pack_uint8


def write_front_pyramid(pyramid: Pyramid, store_path: str, pack_uint8: bool | None = None) -> None:
    """Write a topozarr pyramid to a zarr v3 store.

    float32 by default (unchanged behaviour). With `pack_uint8=True` -- or
    `pack_uint8=None` and the FRONTFINDER_PACK_UINT8 env var set to a truthy
    value -- every served-class variable at every level is uint8-packed with CF
    scale_factor/add_offset/_FillValue attrs (see the PACKED_* comment);
    lat/lon/spatial_ref are written unchanged either way.

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
    levels x 5 classes) before it can even start fetching data, all on the
    critical path for first paint. Calling `consolidate_metadata()`
    explicitly (confirmed live against a real store) correctly populates
    the root's `consolidated_metadata.metadata` with every descendant
    array's real shape/chunks/codecs, keyed by path (e.g. "0/cold"), which
    the client library specifically looks for -- collapsing those N
    metadata round-trips into the one root `zarr.json` fetch it makes
    anyway.
    """
    if pack_uint8_enabled(pack_uint8):
        write_quantized_datatree(pyramid.datatree, store_path, base_encoding=pyramid.encoding)
        return
    pyramid.datatree.to_zarr(store_path, encoding=pyramid.encoding, mode="w")
    zarr.consolidate_metadata(store_path)
