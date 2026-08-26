import numpy as np
import pytest

from frontfinder.inference.hemisphere import flip_southern_hemisphere


def test_flip_southern_hemisphere_reverses_only_negative_lat_rows():
    # lat descending N->S, like the real IFS grid (90 .. -90).
    lat = np.array([2.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    grid = np.arange(6, dtype=np.float32)[:, None, None] * np.ones((1, 1, 1))
    out = flip_southern_hemisphere(grid, lat)

    # northern (lat >= 0) rows 0,1,2 untouched
    np.testing.assert_allclose(out[:3, 0, 0], [0, 1, 2])
    # southern (lat < 0) rows 3,4,5 reversed in place
    np.testing.assert_allclose(out[3:, 0, 0], [5, 4, 3])


def test_flip_southern_hemisphere_is_a_noop_when_no_southern_rows():
    lat = np.array([2.0, 1.0, 0.0])
    grid = np.arange(3, dtype=np.float32)[:, None, None] * np.ones((1, 1, 1))
    out = flip_southern_hemisphere(grid, lat)
    np.testing.assert_allclose(out[:, 0, 0], [0, 1, 2])


def test_flip_southern_hemisphere_is_its_own_inverse():
    lat = np.linspace(90.0, -90.0, 721)
    rng = np.random.default_rng(0)
    grid = rng.standard_normal((721, 4, 3)).astype(np.float32)
    flipped = flip_southern_hemisphere(grid, lat)
    restored = flip_southern_hemisphere(flipped, lat)
    np.testing.assert_allclose(restored, grid)
    # sanity: flipping actually changed something (southern rows moved)
    assert not np.allclose(flipped, grid)


def test_flip_southern_hemisphere_preserves_lon_and_channel_axes():
    lat = np.linspace(90.0, -90.0, 8)
    grid = np.zeros((8, 5, 2), dtype=np.float32)
    grid[:, :, 0] = np.arange(5)[None, :]  # tag every column with its lon index
    out = flip_southern_hemisphere(grid, lat)
    # lon tagging along the column axis must be unchanged for every row
    np.testing.assert_allclose(out[:, :, 0], grid[:, :, 0])


def test_flip_southern_hemisphere_rejects_shape_mismatch():
    lat = np.array([1.0, 0.0, -1.0])
    grid = np.zeros((4, 2, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        flip_southern_hemisphere(grid, lat)
