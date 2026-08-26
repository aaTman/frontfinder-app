"""Southern-hemisphere Coriolis correction for tiled inference.

The training data's fronts are all Coriolis-consistent with the northern
hemisphere's rotation sense; the model has never seen a genuinely southern
-hemisphere flow pattern (cyclones circulate clockwise there, not
counterclockwise). `negate_southern_hemisphere_meridional_wind` corrects for
this by flipping the sign of every meridional-wind channel (`v_component_of_wind`,
including the single-level `10m_v_component_of_wind`) on southern-hemisphere
(lat < 0) rows only -- the same sign flip a true mirror-image reflection
about the equator would apply to a north-south vector component -- which
converts the southern hemisphere's clockwise-consistent flow into the
counterclockwise-consistent flow the model was trained on.

This module used to instead reverse the row *order* of the southern
hemisphere before inference (an attempt at the same fix via spatial
mirroring rather than a value-level sign flip). That reversed each row's
position relative to the *middle of the southern hemisphere block*, not the
equator -- so the row immediately adjacent to the equator ended up holding
the south pole's data (and vice versa), stitching two physically unrelated
air masses together at lat=0. Real production output confirmed this: served
front probabilities spiked approaching the equator from the north (up to
0.42 summed class probability, versus ~0.001 with no hemisphere handling at
all) then fell off a cliff exactly at the lat=0/-0.25 row boundary -- an
order-of-magnitude drop in a single 0.25deg step, visible in the webapp as
spurious weak frontal boundaries banding the equator. Negating only the
meridional wind leaves every row in its original, physically continuous
position, so no such seam is possible; a direct comparison run (real
2026-08-25 18Z cycle, real weights) confirmed the equator crossing is smooth
under this fix, while southern-hemisphere storm-track front probabilities
(e.g. 40-60S) are still measurably higher than with no hemisphere handling
at all (mean summed probability 0.0060 vs 0.0026), i.e. the correction still
does real work, just without the seam.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def negate_southern_hemisphere_meridional_wind(
    grid: np.ndarray, lat_deg: np.ndarray, channel_names: Sequence[str]
) -> np.ndarray:
    """Flip the sign of meridional-wind channels on southern-hemisphere rows.

    `grid` is (lat, lon, channel); `lat_deg` is its 1-D latitude coordinate,
    same length as `grid`'s first axis; `channel_names` names each of
    `grid`'s channels in order (e.g. `ModelManifest.channel_names()`).
    Channels matched by the substring "v_component_of_wind" -- this catches
    both pressure-level channels (`v_component_of_wind_850`) and the
    single-level `10m_v_component_of_wind` without matching
    `u_component_of_wind`. Rows with lat >= 0, and every non-matching
    channel, are returned unchanged.
    """
    lat_deg = np.asarray(lat_deg)
    if grid.shape[0] != len(lat_deg):
        raise ValueError(
            f"grid has {grid.shape[0]} rows, lat_deg has {len(lat_deg)}"
        )
    if grid.shape[2] != len(channel_names):
        raise ValueError(
            f"grid has {grid.shape[2]} channels, channel_names has {len(channel_names)}"
        )
    v_indices = [i for i, name in enumerate(channel_names) if "v_component_of_wind" in name]
    if not v_indices:
        return grid

    south_mask = lat_deg < 0
    out = grid.copy()
    for idx in v_indices:
        out[south_mask, :, idx] *= -1.0
    return out
