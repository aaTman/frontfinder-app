"""Whole-domain, single-pass inference for regional (limited-area) grids.

`run_tiled_inference` pads with raw zeros, which sit far outside the model's
normalized input range and create false edges inside a regional domain. This
path reflect-pads instead, so the model never sees a zero-padded cell.
"""

from __future__ import annotations

import numpy as np

from frontfinder.config.manifests import ModelManifest
from frontfinder.inference.engine import Predictor
from frontfinder.inference.tiling import pad_to_multiple


def _split_padding(size: int, multiple: int) -> tuple[int, int]:
    """(before, after) padding to reach the next multiple, split evenly."""
    total = pad_to_multiple(size, multiple) - size
    before = total // 2
    return before, total - before


def run_regional_inference(
    predictor: Predictor,
    input_grid: np.ndarray,
    manifest: ModelManifest,
    multiple: int = 8,
) -> np.ndarray:
    """(H, W, C) -> (H, W, len(manifest.served_classes)), float32."""
    if input_grid.ndim != 3:
        raise ValueError(f"input_grid must be (H, W, C), got shape {input_grid.shape}")
    height, width, n_channels = input_grid.shape
    if n_channels != manifest.n_channels:
        raise ValueError(
            f"input_grid has {n_channels} channels, manifest {manifest.name!r} "
            f"expects {manifest.n_channels}"
        )
    if not np.isfinite(input_grid).all():
        raise ValueError("input_grid contains non-finite values")

    top, bottom = _split_padding(height, multiple)
    left, right = _split_padding(width, multiple)
    padded = np.pad(input_grid, ((top, bottom), (left, right), (0, 0)), mode="reflect")

    output = predictor.predict_batch(padded[np.newaxis, ...])
    if output.shape[:3] != (1, *padded.shape[:2]):
        raise ValueError(
            f"predictor output shape {output.shape} does not match input "
            f"shape {(1, *padded.shape[:2])}"
        )

    cropped = output[0, top : top + height, left : left + width, :]
    return cropped[..., manifest.served_class_indices()].astype(np.float32)
