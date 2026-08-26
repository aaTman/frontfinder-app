import numpy as np
import pytest

from frontfinder.inference.hemisphere import negate_southern_hemisphere_meridional_wind


def test_negates_only_v_component_channels_on_southern_rows():
    # lat descending N->S, like the real IFS grid (90 .. -90).
    lat = np.array([2.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    channel_names = ["equivalent_potential_temperature_850", "u_component_of_wind_850", "v_component_of_wind_850"]
    grid = np.ones((6, 1, 3), dtype=np.float32)
    out = negate_southern_hemisphere_meridional_wind(grid, lat, channel_names)

    # non-v channels untouched everywhere
    np.testing.assert_allclose(out[:, 0, 0], 1.0)
    np.testing.assert_allclose(out[:, 0, 1], 1.0)
    # v channel: northern (lat >= 0) rows 0,1,2 untouched, southern rows 3,4,5 negated
    np.testing.assert_allclose(out[:3, 0, 2], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out[3:, 0, 2], [-1.0, -1.0, -1.0])


def test_matches_single_level_10m_v_component_but_not_u_component():
    lat = np.array([1.0, -1.0])
    channel_names = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
    grid = np.ones((2, 1, 2), dtype=np.float32)
    out = negate_southern_hemisphere_meridional_wind(grid, lat, channel_names)
    np.testing.assert_allclose(out[:, 0, 0], [1.0, 1.0])  # u untouched
    np.testing.assert_allclose(out[:, 0, 1], [1.0, -1.0])  # v negated south of equator


def test_is_a_noop_when_no_southern_rows():
    lat = np.array([2.0, 1.0, 0.0])
    channel_names = ["v_component_of_wind_850"]
    grid = np.ones((3, 1, 1), dtype=np.float32)
    out = negate_southern_hemisphere_meridional_wind(grid, lat, channel_names)
    np.testing.assert_allclose(out[:, 0, 0], [1.0, 1.0, 1.0])


def test_is_a_noop_when_manifest_has_no_v_component_channel():
    lat = np.array([1.0, -1.0])
    channel_names = ["equivalent_potential_temperature_850", "u_component_of_wind_850"]
    grid = np.ones((2, 1, 2), dtype=np.float32)
    out = negate_southern_hemisphere_meridional_wind(grid, lat, channel_names)
    np.testing.assert_allclose(out, grid)


def test_preserves_row_order_and_lon_axis():
    # No row reordering happens at all -- unlike the old row-reversal
    # implementation, this is a pure per-pixel value transform.
    lat = np.linspace(90.0, -90.0, 8)
    channel_names = ["v_component_of_wind_850"]
    grid = np.zeros((8, 5, 1), dtype=np.float32)
    grid[:, :, 0] = np.arange(8)[:, None]  # tag every row with its own index
    out = negate_southern_hemisphere_meridional_wind(grid, lat, channel_names)
    south = lat < 0
    expected = np.where(south, -np.arange(8), np.arange(8))
    np.testing.assert_allclose(out[:, 0, 0], expected)
    # untouched lon broadcast
    np.testing.assert_allclose(out[:, :, 0], out[:, 0:1, 0] * np.ones((1, 5)))


def test_rejects_row_shape_mismatch():
    lat = np.array([1.0, 0.0, -1.0])
    grid = np.zeros((4, 2, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        negate_southern_hemisphere_meridional_wind(grid, lat, ["v_component_of_wind_850"])


def test_rejects_channel_names_length_mismatch():
    lat = np.array([1.0, 0.0, -1.0])
    grid = np.zeros((3, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        negate_southern_hemisphere_meridional_wind(grid, lat, ["v_component_of_wind_850"])
