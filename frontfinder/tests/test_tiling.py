import numpy as np
import pytest

from frontfinder.inference.tiling import (
    Tile,
    blend_weight,
    generate_tiles,
    pad_to_multiple,
    stitch,
)


def test_pad_to_multiple_already_aligned():
    assert pad_to_multiple(256, 16) == 256


def test_pad_to_multiple_rounds_up():
    assert pad_to_multiple(721, 16) == 736  # global IFS 0.25deg lat dim
    assert pad_to_multiple(1440, 16) == 1440  # lon dim already aligned


def test_pad_to_multiple_rejects_nonpositive():
    with pytest.raises(ValueError):
        pad_to_multiple(0, 16)


def test_generate_tiles_covers_full_grid_with_no_gaps():
    tiles = generate_tiles(height=736, width=1440, patch_size=256, overlap=32)
    # every point in the grid must be covered by at least one tile
    covered = np.zeros((736, 1440), dtype=bool)
    for t in tiles:
        covered[t.row_start:t.row_end, t.col_start:t.col_end] = True
    assert covered.all()


def test_generate_tiles_all_tiles_are_multiple_of_16():
    tiles = generate_tiles(height=736, width=1440, patch_size=256, overlap=32, multiple=16)
    for t in tiles:
        assert t.height % 16 == 0
        assert t.width % 16 == 0


def test_generate_tiles_rejects_patch_size_not_multiple_of_16():
    with pytest.raises(ValueError):
        generate_tiles(height=736, width=1440, patch_size=250, overlap=32)


def test_generate_tiles_rejects_overlap_too_large():
    with pytest.raises(ValueError):
        generate_tiles(height=736, width=1440, patch_size=256, overlap=256)


def test_generate_tiles_small_grid_single_tile():
    tiles = generate_tiles(height=200, width=200, patch_size=256, overlap=32)
    assert len(tiles) == 1
    assert tiles[0] == Tile(0, 256, 0, 256)


def test_blend_weight_interior_is_one():
    w = blend_weight(patch_size=64, overlap=16)
    assert w[32, 32] == pytest.approx(1.0)


def test_blend_weight_no_overlap_is_uniform_one():
    w = blend_weight(patch_size=64, overlap=0)
    assert np.all(w == 1.0)


def test_stitch_recovers_constant_field_exactly():
    # a constant field, tiled and stitched, should come back out constant --
    # this is the key correctness property for the blend (no seam artifacts).
    height, width, n_classes = 736, 1440, 4
    tiles = generate_tiles(height, width, patch_size=256, overlap=32)
    predictions = [np.full((t.height, t.width, n_classes), 0.5, dtype=np.float32) for t in tiles]
    stitched = stitch(tiles, predictions, out_height=height, out_width=width, overlap=32)
    assert stitched.shape == (height, width, n_classes)
    np.testing.assert_allclose(stitched, 0.5, atol=1e-5)


def test_stitch_crops_to_original_unpadded_size():
    height, width, n_classes = 721, 1440, 4  # unpadded IFS global lat dim
    tiles = generate_tiles(pad_to_multiple(height, 16), width, patch_size=256, overlap=32)
    predictions = [np.zeros((t.height, t.width, n_classes), dtype=np.float32) for t in tiles]
    stitched = stitch(tiles, predictions, out_height=height, out_width=width, overlap=32)
    assert stitched.shape == (height, width, n_classes)


def test_stitch_rejects_mismatched_tile_and_prediction_count():
    tiles = generate_tiles(height=256, width=256, patch_size=256, overlap=32)
    with pytest.raises(ValueError):
        stitch(tiles, predictions=[], out_height=256, out_width=256, overlap=32)


def test_stitch_no_wide_flat_band_at_squeezed_edge_tiles():
    # Regression test for the 2026-08-22 Bermuda seam: on the real global
    # 721x1440 grid at patch_size=256/overlap=32, generate_tiles snaps the
    # last column tile to the edge with only a 64px gap from its neighbor
    # (not the nominal 224px stride) -- col_starts end in [..., 1120, 1184].
    # A fixed 32px blend ramp left a ~128-column band where both tiles
    # reported full (1.0) weight simultaneously, silently 50/50-averaging
    # two differently-windowed predictions and smearing any real front
    # structure into a wide, flat, structureless bar sitting at ~0.5 across
    # that whole band. With per-tile ramps sized to the actual overlap, two
    # tiles that disagree completely should blend as a smooth, monotonic
    # gradient across their real overlap -- never plateau at the halfway
    # point for more than a few columns.
    height, width, n_classes = 721, 1440, 4
    padded_h = pad_to_multiple(height, 16)
    tiles = generate_tiles(padded_h, width, patch_size=256, overlap=32)

    col_starts = sorted(set(t.col_start for t in tiles))
    assert col_starts[-2:] == [1120, 1184]  # confirms the squeeze this grid produces

    predictions = []
    for t in tiles:
        value = 0.0 if t.col_start == 1120 else 1.0
        predictions.append(np.full((t.height, t.width, n_classes), value, dtype=np.float32))

    stitched = stitch(tiles, predictions, out_height=height, out_width=width, overlap=32)
    profile = stitched[0, :, 0]

    # Columns covered by only the 0.0-valued tile (before its neighbor
    # starts) must reproduce that value exactly.
    assert np.allclose(profile[1152:1184], 0.0, atol=1e-3)
    # Columns covered by only the 1.0-valued tile (after the other tile's
    # footprint ends) must reproduce that value exactly.
    assert np.allclose(profile[1376:1440], 1.0, atol=1e-3)

    # The old bug's signature: a wide run of columns stuck at ~0.5. The
    # fixed blend should be monotonic through the transition, with no more
    # than a handful of columns sitting near the halfway point.
    near_half = np.abs(profile[1152:1376] - 0.5) < 0.02
    assert near_half.sum() <= 10, "wide flat band at the seam -- blend ramp not localized to real overlap"
    transition = profile[1152:1376]
    assert np.all(np.diff(transition) >= -1e-6), "blend across the seam is not monotonic"
