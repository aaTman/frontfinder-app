import numpy as np
import pytest

from frontfinder.inference.periodic import circularly_pad_longitude


def _lon_tagged_grid(lon: np.ndarray, height: int = 4) -> np.ndarray:
    """(height, len(lon), 1) grid where every pixel's value is its own
    longitude -- makes wrap correctness trivial to assert on."""
    grid = np.zeros((height, len(lon), 1), dtype=np.float32)
    grid[:, :, 0] = lon[None, :]
    return grid


def test_circularly_pad_longitude_is_a_noop_for_zero_pad():
    lon = np.linspace(0.0, 359.75, 1440)
    grid = _lon_tagged_grid(lon)
    out = circularly_pad_longitude(grid, lon, pad_pixels=0)
    assert out is grid


def test_circularly_pad_longitude_pulls_real_data_from_the_opposite_edge():
    lon = np.linspace(0.0, 359.75, 1440)
    grid = _lon_tagged_grid(lon)
    padded = circularly_pad_longitude(grid, lon, pad_pixels=32)

    assert padded.shape == (4, 1440 + 64, 1)
    # left pad == the grid's own tail (lon 352.0..359.75)
    np.testing.assert_allclose(padded[:, :32, 0], np.broadcast_to(lon[-32:], (4, 32)))
    # interior is untouched, at the shifted offset
    np.testing.assert_allclose(padded[:, 32:32 + 1440, 0], grid[:, :, 0])
    # right pad == the grid's own head (lon 0.0..7.75)
    np.testing.assert_allclose(padded[:, 32 + 1440:, 0], np.broadcast_to(lon[:32], (4, 32)))


def test_circularly_pad_longitude_rejects_a_non_global_grid():
    # a regional box (doesn't span the full 360deg) has no real wraparound
    # to pull padding from -- production callers must check for this before
    # asking for padding (see engine.run_tiled_inference's is_global_lon
    # check) rather than getting silently-wrong padding here.
    lon = np.linspace(228.0, 299.75, 64)
    grid = _lon_tagged_grid(lon)
    with pytest.raises(AssertionError):
        circularly_pad_longitude(grid, lon, pad_pixels=8)
