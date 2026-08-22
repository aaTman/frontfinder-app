"""Patch-based tiling for CPU-only global-grid inference.

mandelhub has no GPU and a fixed RAM budget, so a single whole-globe forward
pass through a U-Net-style model is avoided. Instead the grid is split into
overlapping patches (each patch's H and W divisible by 16, per the models'
architecture constraint), inference runs patch-by-patch, and results are
stitched back together with linear-ramp blending across the overlap region
so class-probability seams don't appear at patch boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    row_start: int
    row_end: int  # exclusive, in the *padded* grid
    col_start: int
    col_end: int  # exclusive, in the *padded* grid

    @property
    def height(self) -> int:
        return self.row_end - self.row_start

    @property
    def width(self) -> int:
        return self.col_end - self.col_start


def pad_to_multiple(size: int, multiple: int) -> int:
    """Smallest size' >= size such that size' % multiple == 0."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    remainder = size % multiple
    return size if remainder == 0 else size + (multiple - remainder)


def generate_tiles(
    height: int,
    width: int,
    patch_size: int,
    overlap: int,
    multiple: int = 16,
) -> list[Tile]:
    """Cover a `height` x `width` grid with overlapping tiles.

    Each tile is `patch_size` x `patch_size` (except possibly the last row/col
    of tiles, which are still snapped to a multiple of `multiple`). Stride is
    `patch_size - overlap`, so overlap >= multiple is generally recommended to
    give the blend ramp room to work.
    """
    if patch_size % multiple != 0:
        raise ValueError(
            f"patch_size ({patch_size}) must be a multiple of {multiple}"
        )
    if overlap < 0 or overlap >= patch_size:
        raise ValueError(f"overlap ({overlap}) must be in [0, patch_size)")

    stride = patch_size - overlap
    tiles: list[Tile] = []

    row_starts = list(range(0, max(height - patch_size, 0) + 1, stride))
    if not row_starts or row_starts[-1] + patch_size < height:
        row_starts.append(max(height - patch_size, 0))
    row_starts = sorted(set(row_starts))

    col_starts = list(range(0, max(width - patch_size, 0) + 1, stride))
    if not col_starts or col_starts[-1] + patch_size < width:
        col_starts.append(max(width - patch_size, 0))
    col_starts = sorted(set(col_starts))

    for r in row_starts:
        for c in col_starts:
            tiles.append(
                Tile(
                    row_start=r,
                    row_end=r + patch_size,
                    col_start=c,
                    col_end=c + patch_size,
                )
            )
    return tiles


def blend_weight(patch_size: int, overlap: int) -> np.ndarray:
    """2D ramp weight for a patch, 1.0 in the interior, linearly ramping to a
    small nonzero floor at the outer edge of the overlap band, so neighboring
    tiles' contributions blend smoothly instead of hard-cutting.
    """
    if overlap == 0:
        return np.ones((patch_size, patch_size), dtype=np.float32)

    ramp = np.ones(patch_size, dtype=np.float32)
    edge = np.linspace(1.0 / overlap, 1.0, overlap, dtype=np.float32)
    ramp[:overlap] = edge
    ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


def _ramp_vector(patch_size: int, left: int, right: int) -> np.ndarray:
    """1D ramp for one tile along one axis: 1.0 in the interior, ramping down
    to a small nonzero floor over `left`/`right` pixels at each edge -- where
    `left`/`right` are this tile's *actual* overlap with its previous/next
    neighbor along that axis (not a fixed assumed overlap), so the ramp
    always exactly matches the real overlap band no matter how the tiling
    snapped to the grid edge.
    """
    # Clamp so opposing ramps can never claim the same pixel (which would
    # otherwise happen when a neighbor is snapped much closer than the
    # nominal stride, e.g. the final edge tile on a non-evenly-divisible
    # grid -- see the 2026-08-22 Bermuda-seam bug this replaces).
    left = max(0, min(left, patch_size // 2))
    right = max(0, min(right, patch_size - left))

    ramp = np.ones(patch_size, dtype=np.float32)
    if left > 0:
        ramp[:left] = np.linspace(1.0 / left, 1.0, left, dtype=np.float32)
    if right > 0:
        ramp[patch_size - right:] = np.linspace(1.0, 1.0 / right, right, dtype=np.float32)
    return ramp


def _axis_ramps(starts: list[int], patch_size: int) -> dict[int, tuple[int, int]]:
    """For each unique tile start along one axis, the actual (left, right)
    overlap in pixels with its previous/next neighbor -- 0 at the outer
    edges of the grid, `patch_size - gap` wherever tiles are `gap` apart.
    """
    ordered = sorted(set(starts))
    ramps: dict[int, tuple[int, int]] = {}
    for i, s in enumerate(ordered):
        left = max(0, patch_size - (s - ordered[i - 1])) if i > 0 else 0
        right = max(0, patch_size - (ordered[i + 1] - s)) if i < len(ordered) - 1 else 0
        ramps[s] = (left, right)
    return ramps


def stitch(
    tiles: list[Tile],
    predictions: list[np.ndarray],
    out_height: int,
    out_width: int,
    overlap: int,
) -> np.ndarray:
    """Weighted-average stitch of per-tile predictions back onto the full grid.

    `predictions[i]` has shape (patch_size, patch_size, n_classes) and
    corresponds to `tiles[i]`. Returns an (out_height, out_width, n_classes)
    array cropped to the original (unpadded) grid size.

    Each tile's blend ramp is sized to its *actual* overlap with its
    neighbors (see `_axis_ramps`), not the nominal `overlap` parameter --
    `generate_tiles` snaps the final row/column of tiles to the grid edge,
    which can leave a much smaller gap than the nominal stride (e.g. this
    global 721x1440 grid's last two column tiles are only 64px apart, not
    the nominal 224px stride). Using a fixed 32px ramp there left a
    ~128-column band where both tiles report full (1.0) weight
    simultaneously, so the stitch silently 50/50-averaged two independently
    -run, differently-windowed predictions across a ~32deg-wide longitude
    band -- visible as a wide, flat, structureless bar smeared over real
    front structure (reported 2026-08-22: a bar straddling Bermuda, right on
    that exact seam, ~64.75W). Sizing each tile's ramp to its real neighbor
    distance guarantees adjacent tiles' full-weight interiors never
    coincide, regardless of how the grid snaps.
    """
    if len(tiles) != len(predictions):
        raise ValueError("tiles and predictions must be the same length")
    if not tiles:
        raise ValueError("no tiles to stitch")

    patch_size = tiles[0].height
    n_classes = predictions[0].shape[-1]
    padded_h = max(t.row_end for t in tiles)
    padded_w = max(t.col_end for t in tiles)

    row_ramps = _axis_ramps([t.row_start for t in tiles], patch_size)
    col_ramps = _axis_ramps([t.col_start for t in tiles], tiles[0].width)

    accum = np.zeros((padded_h, padded_w, n_classes), dtype=np.float64)
    weight_sum = np.zeros((padded_h, padded_w, 1), dtype=np.float64)

    for tile, pred in zip(tiles, predictions):
        if pred.shape[:2] != (tile.height, tile.width):
            raise ValueError(
                f"prediction shape {pred.shape[:2]} does not match tile "
                f"shape ({tile.height}, {tile.width})"
            )
        row_left, row_right = row_ramps[tile.row_start]
        col_left, col_right = col_ramps[tile.col_start]
        w = np.outer(
            _ramp_vector(tile.height, row_left, row_right),
            _ramp_vector(tile.width, col_left, col_right),
        )[..., None]
        accum[tile.row_start:tile.row_end, tile.col_start:tile.col_end, :] += (
            pred.astype(np.float64) * w
        )
        weight_sum[tile.row_start:tile.row_end, tile.col_start:tile.col_end, :] += w

    weight_sum[weight_sum == 0] = 1.0  # guard: shouldn't happen if tiles cover the grid
    stitched = (accum / weight_sum).astype(np.float32)
    return stitched[:out_height, :out_width, :]
