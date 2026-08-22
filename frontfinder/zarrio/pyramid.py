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


def build_level0_dataset(fields: FrontFields, manifest: ModelManifest) -> xr.Dataset:
    """Assemble the native-resolution, CRS-tagged, style-annotated xr.Dataset
    for one model run."""
    missing = set(manifest.served_classes) - set(fields.probabilities)
    if missing:
        raise ValueError(f"missing served-class fields: {missing}")
    extra = set(fields.probabilities) - set(manifest.served_classes)
    if extra:
        raise ValueError(f"unexpected fields not in manifest.served_classes: {extra}")

    data_vars = {}
    for cls in manifest.served_classes:
        attrs = dict(CLASS_STYLE.get(cls, {}))
        data_vars[cls] = (("lat", "lon"), fields.probabilities[cls].astype(np.float32), attrs)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"lat": fields.lat, "lon": fields.lon},
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
    return create_pyramid(ds, levels=n_levels, x_dim="lon", y_dim="lat", method="mean")


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
