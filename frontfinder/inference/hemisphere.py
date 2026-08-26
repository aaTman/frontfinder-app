"""Southern-hemisphere latitude flip for tiled inference.

The training data's fronts are all Coriolis-consistent with the northern
hemisphere's rotation sense; the model has never seen a genuinely southern
-hemisphere flow pattern. Mirroring the southern-hemisphere rows across the
equator before inference (`flip_southern_hemisphere`) makes that half of the
grid look, row-order-wise, like a northern-hemisphere pattern to the model;
the *same* function applied again afterwards (an involution) restores the
original north-to-south row order before the result is written out.
"""

from __future__ import annotations

import numpy as np


def flip_southern_hemisphere(grid: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Reverse the row-order of `grid`'s southern-hemisphere (lat < 0) rows.

    `grid` is (lat, lon, ...); `lat_deg` is its 1-D latitude coordinate,
    same length as `grid`'s first axis. Rows with lat >= 0 are left
    untouched; rows with lat < 0 are reversed in place (same row indices,
    reversed order) -- so this is its own inverse: calling it twice with the
    same `lat_deg` returns the original array.
    """
    lat_deg = np.asarray(lat_deg)
    if grid.shape[0] != len(lat_deg):
        raise ValueError(
            f"grid has {grid.shape[0]} rows, lat_deg has {len(lat_deg)}"
        )
    south_mask = lat_deg < 0
    out = grid.copy()
    out[south_mask] = grid[south_mask][::-1]
    return out
