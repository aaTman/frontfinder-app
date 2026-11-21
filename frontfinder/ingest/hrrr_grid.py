"""Grid definitions for HRRR CONUS: the native 3 km Lambert conformal grid,
block-coarsened versions of it, and the 0.25deg lat/lon box inscribed in its
domain that HRRR and IFS output are compared on.

Constants were read from a live HRRR GRIB message (2026-11-20 12Z). Projecting
that message's own lat/lon array through `HRRR_CRS` reproduces a uniform
3000 m grid to better than a micrometer.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyproj

from frontfinder.ingest.base import GridSpec

HRRR_CRS = "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +R=6371229 +units=m +no_defs"
HRRR_NX = 1799
HRRR_NY = 1059
HRRR_DX_M = 3000.0
HRRR_LOV_DEG = -97.5
HRRR_LATIN_DEG = 38.5
# First grid point in GRIB scan order, which is the domain's southwest corner.
HRRR_FIRST_LAT_DEG = 21.138123
HRRR_FIRST_LON_DEG = 237.280472

SUPPORTED_COARSEN = (1, 2, 4, 8)

# Largest 0.25deg lat/lon rectangle, on IFS grid nodes, fully inside the HRRR
# domain with a 2-cell margin, minus its 50.25N row: that row's corner nodes
# fall outside the 24 km cell centers, where bilinear interpolation would
# extrapolate. North to south and 0..360, like IFS.
EVAL_LAT_DEG = np.linspace(50.0, 24.5, 103)
EVAL_LON_DEG = np.linspace(236.75, 288.25, 207)


@lru_cache(maxsize=None)
def native_grid(coarsen: int = 1) -> GridSpec:
    """The HRRR grid after block-averaging by `coarsen`, rows north to south.

    Blocks are anchored at the southwest corner, so cells that don't fill a
    whole block along the north and east edges are dropped.
    """
    if coarsen not in SUPPORTED_COARSEN:
        raise ValueError(f"coarsen must be one of {SUPPORTED_COARSEN}, got {coarsen}")
    to_lcc = pyproj.Transformer.from_crs("EPSG:4326", HRRR_CRS, always_xy=True)
    x0, y0 = to_lcc.transform(HRRR_FIRST_LON_DEG - 360.0, HRRR_FIRST_LAT_DEG)
    spacing = HRRR_DX_M * coarsen
    offset = HRRR_DX_M * (coarsen - 1) / 2.0
    x = x0 + offset + spacing * np.arange(HRRR_NX // coarsen)
    y = (y0 + offset + spacing * np.arange(HRRR_NY // coarsen))[::-1]
    xx, yy = np.meshgrid(x, y)
    to_geo = pyproj.Transformer.from_crs(HRRR_CRS, "EPSG:4326", always_xy=True)
    lon, lat = to_geo.transform(xx, yy)
    return GridSpec(
        name=f"hrrr-conus-{int(spacing / 1000)}km",
        crs=HRRR_CRS,
        x=x,
        y=y,
        lat=np.asarray(lat),
        lon=np.asarray(lon),
    )


@lru_cache(maxsize=None)
def eval_grid_0p25() -> GridSpec:
    """The common 0.25deg comparison grid (103 rows, 207 columns)."""
    return GridSpec.regular_latlon("hrrr-eval-0p25", EVAL_LAT_DEG, EVAL_LON_DEG)
