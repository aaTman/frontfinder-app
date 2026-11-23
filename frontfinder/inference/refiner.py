"""C1 3 km HRRR residual U-Net refiner inference.

Predicts residual logits for `frontal` and `trough` on the native 3 km HRRR grid
using standardized 3 km HRRR inputs and bilinearly upsampled 24 km B0 baseline
probabilities, then reconstructs the 7 served classes at 3 km by scaling B0's
four frontal class shares proportionally to C1's refined `frontal` probability.

TensorFlow and Keras are imported lazily inside `load_refiner_model` so dry runs
and non-TF test suites never trigger a TensorFlow import.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from frontfinder.config.manifests import (
    NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST,
    ModelManifest,
)
from frontfinder.ingest.hrrr_grid import native_grid

REFINER_WEIGHTS_FILENAME = "hrrr_c1_refiner.keras"
REFINER_NORM_FILENAME = "hrrr_c1_norm.npz"

FRONTAL_CLASSES: tuple[str, ...] = ("cold", "warm", "stationary", "occluded")
TROUGH_CLASS = "trough"
PASSTHROUGH_CLASSES: tuple[str, ...] = ("dryline", "tropical_trough")
REFINED_GROUPS: tuple[str, ...] = ("frontal", "trough")

_BASE_LOGIT_CLIP = 1e-4
_LOGIT_EXP_CLIP = 60.0
_SHARE_EPS = 1e-12


class RefinerModel(Protocol):
    """Protocol implemented by Keras and synthetic C1 refiner predictors."""

    def predict_logits(self, hrrr_norm_batch: np.ndarray, b0_probs_batch: np.ndarray) -> np.ndarray:
        """Return `(N, H, W, 2)` residual-adjusted logits for `(frontal, trough)`."""


def load_refiner_norm(norm_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate per-channel `(mean, std)` normalization arrays."""
    with np.load(norm_path, allow_pickle=False) as handle:
        if "mean" not in handle.files or "std" not in handle.files:
            raise ValueError(f"{norm_path}: expected 'mean' and 'std' arrays")
        mean = np.asarray(handle["mean"], dtype=np.float32)
        std = np.asarray(handle["std"], dtype=np.float32)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape or mean.size == 0:
        raise ValueError(f"{norm_path}: expected non-empty matching 1-D mean/std, got {mean.shape} and {std.shape}")
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError(f"{norm_path}: normalization arrays contain non-finite values")
    if not (std > 0.0).all():
        raise ValueError(f"{norm_path}: normalization std must be strictly positive")
    return mean, std


def standardize_hrrr(input_grid: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize `(H, W, C)` HRRR channels with `(x - mean) / std` in float32."""
    arr = np.asarray(input_grid, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"input_grid must be (H, W, C), got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("input_grid contains non-finite values")
    n_channels = arr.shape[-1]
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    if mean_arr.shape != (n_channels,) or std_arr.shape != (n_channels,):
        raise ValueError(
            f"normalization shapes {mean_arr.shape}/{std_arr.shape} do not match {n_channels} channels"
        )
    if not (np.isfinite(mean_arr).all() and np.isfinite(std_arr).all() and (std_arr > 0.0).all()):
        raise ValueError("normalization arrays must be finite with strictly positive std")
    return ((arr - mean_arr) / std_arr).astype(np.float32)


def _axis_weights(fine: np.ndarray, coarse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower coarse index and upper weight evaluating `fine` on ascending `coarse` nodes.

    Outside the coarse node span, `np.interp` clamps to the endpoints, giving nearest-edge
    extension outside the coarse cell-center hull.
    """
    if len(coarse) < 2:
        raise ValueError(f"need at least 2 coarse nodes, got {len(coarse)}")
    pos = np.interp(fine, coarse, np.arange(len(coarse), dtype=np.float64))
    j0 = np.clip(np.floor(pos), 0, len(coarse) - 2).astype(np.intp)
    return j0, (pos - j0).astype(np.float32)


def upsample_24km_to_3km(
    probs_24km: np.ndarray,
    target_shape: tuple[int, int] = (1059, 1799),
    block: int = 8,
) -> np.ndarray:
    """Bilinearly upsample `(Hc, Wc, C)` or `(Hc, Wc)` probabilities from 24 km to 3 km.

    HRRR coarsening averages SW-anchored `block x block` cells, so coarse node `j` sits at
    fine index `block * j + (block - 1) / 2` measured from the southwest corner. Cells
    outside the coarse cell-center hull clamp to the nearest coarse edge value.
    """
    if block < 1:
        raise ValueError(f"block must be positive, got {block}")
    arr = np.asarray(probs_24km, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"probs_24km must be 2-D or 3-D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("probs_24km contains non-finite values")
    rows, cols = target_shape
    if rows < 1 or cols < 1:
        raise ValueError(f"target_shape dimensions must be positive, got {target_shape}")
    coarse_rows, coarse_cols = arr.shape[:2]
    if coarse_rows < 2 or coarse_cols < 2:
        raise ValueError(f"probs_24km needs at least 2 rows and 2 cols, got {arr.shape[:2]}")

    squeeze = arr.ndim == 2
    field = arr[..., np.newaxis] if squeeze else arr

    # Work south-up so both x and y node coordinates ascend from the SW anchor, matching
    # `refine.data.upsample_probs` exactly for native 24 km -> 3 km grids.
    south = field[::-1]
    fine_native = native_grid(1)
    coarse_native = native_grid(8)
    if block == 8 and target_shape == fine_native.shape and (coarse_rows, coarse_cols) == coarse_native.shape:
        jx, wx = _axis_weights(fine_native.x, coarse_native.x)
        jy, wy = _axis_weights(fine_native.y[::-1], coarse_native.y[::-1])
    else:
        offset = (block - 1) / 2.0
        coarse_x = offset + block * np.arange(coarse_cols, dtype=np.float64)
        coarse_y = offset + block * np.arange(coarse_rows, dtype=np.float64)
        jx, wx = _axis_weights(np.arange(cols, dtype=np.float64), coarse_x)
        jy, wy = _axis_weights(np.arange(rows, dtype=np.float64), coarse_y)

    wx_3d = wx[np.newaxis, :, np.newaxis]
    wy_3d = wy[:, np.newaxis, np.newaxis]
    across = south[:, jx] * (1.0 - wx_3d) + south[:, jx + 1] * wx_3d
    north_up = (across[jy] * (1.0 - wy_3d) + across[jy + 1] * wy_3d)[::-1]
    out = np.ascontiguousarray(north_up, dtype=np.float32)
    return out[..., 0] if squeeze else out


def pad_south_east_to_multiple(arr: np.ndarray, multiple: int = 8) -> np.ndarray:
    """Reflect-pad `(H, W, C)` along south and east edges so sides divide `multiple`."""
    if arr.ndim != 3:
        raise ValueError(f"expected (H, W, C) array, got shape {arr.shape}")
    if multiple < 1:
        raise ValueError(f"multiple must be positive, got {multiple}")
    rows, cols = arr.shape[:2]
    pad_rows = (-rows) % multiple
    pad_cols = (-cols) % multiple
    if pad_rows == 0 and pad_cols == 0:
        return np.asarray(arr, dtype=np.float32)
    return np.ascontiguousarray(
        np.pad(arr, ((0, pad_rows), (0, pad_cols), (0, 0)), mode="reflect"),
        dtype=np.float32,
    )


def combine_c1_served_probabilities(
    probs_24km_up: np.ndarray,
    c1_group_probs: np.ndarray,
    served_classes: tuple[str, ...] = NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST.served_classes,
) -> np.ndarray:
    """Reconstruct `(H, W, 7)` 3 km probabilities from B0 upsampled + C1 `(frontal, trough)`."""
    b0 = np.asarray(probs_24km_up, dtype=np.float32)
    c1 = np.asarray(c1_group_probs, dtype=np.float32)
    if b0.ndim != 3 or b0.shape[-1] != len(served_classes):
        raise ValueError(f"probs_24km_up must be (H, W, {len(served_classes)}), got {b0.shape}")
    if c1.shape != (*b0.shape[:2], len(REFINED_GROUPS)):
        raise ValueError(f"c1_group_probs must be {(*b0.shape[:2], len(REFINED_GROUPS))}, got {c1.shape}")
    if not (np.isfinite(b0).all() and np.isfinite(c1).all()):
        raise ValueError("input probabilities contain non-finite values")

    b0 = np.clip(b0, 0.0, 1.0)
    c1 = np.clip(c1, 0.0, 1.0)
    out = b0.copy()

    frontal_indices = [served_classes.index(cls) for cls in FRONTAL_CLASSES if cls in served_classes]
    if frontal_indices:
        p_frontal = c1[..., 0]
        b0_frontal_parts = b0[..., frontal_indices]
        b0_frontal_sum = np.sum(b0_frontal_parts, axis=-1, keepdims=True)
        uniform_share = np.full_like(b0_frontal_parts, 1.0 / len(frontal_indices))
        safe_denom = np.where(b0_frontal_sum > _SHARE_EPS, b0_frontal_sum, 1.0)
        shares = np.where(b0_frontal_sum > _SHARE_EPS, b0_frontal_parts / safe_denom, uniform_share)
        out[..., frontal_indices] = p_frontal[..., np.newaxis] * shares

    if TROUGH_CLASS in served_classes:
        out[..., served_classes.index(TROUGH_CLASS)] = c1[..., 1]

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_refiner_model(weights_path: str) -> Any:
    """Load C1 `.keras` refiner weights lazily, registering `GroupBaseLogit`."""
    import keras
    from keras import layers, ops

    @keras.saving.register_keras_serializable(package="refine")
    class GroupBaseLogit(layers.Layer):
        """Served-class probabilities -> clipped logit of `(frontal, trough)`."""

        def __init__(
            self,
            frontal_channels: tuple[int, ...] = (0, 1, 2, 3),
            trough_channel: int = 5,
            clip: float = _BASE_LOGIT_CLIP,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.frontal_channels = tuple(int(c) for c in frontal_channels)
            self.trough_channel = int(trough_channel)
            self.clip = float(clip)

        def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
            return (*input_shape[:-1], len(REFINED_GROUPS))

        def call(self, probs: Any) -> Any:
            channels = list(self.frontal_channels)
            frontal = ops.sum(ops.take(probs, channels, axis=-1), axis=-1)
            trough = ops.take(probs, [self.trough_channel], axis=-1)[..., 0]
            groups = ops.stack([frontal, trough], axis=-1)
            groups = ops.clip(groups, self.clip, 1.0 - self.clip)
            return ops.log(groups) - ops.log1p(-groups)

        def get_config(self) -> dict[str, Any]:
            config = super().get_config()
            config.update(
                {
                    "frontal_channels": list(self.frontal_channels),
                    "trough_channel": self.trough_channel,
                    "clip": self.clip,
                }
            )
            return config

    return keras.saving.load_model(
        weights_path,
        compile=False,
        custom_objects={
            "GroupBaseLogit": GroupBaseLogit,
            "refine>GroupBaseLogit": GroupBaseLogit,
        },
    )


class RefinerPredictor:
    """Wraps the trained Keras C1 refiner model for batch logit prediction."""

    def __init__(self, weights_path: str) -> None:
        self.model = load_refiner_model(weights_path)

    def predict_logits(self, hrrr_norm_batch: np.ndarray, b0_probs_batch: np.ndarray) -> np.ndarray:
        logits = self.model([hrrr_norm_batch, b0_probs_batch], training=False)
        return np.asarray(logits, dtype=np.float32)


class FakeRefinerPredictor:
    """Pure-NumPy C1 predictor for `--dry-run` and non-TensorFlow tests."""

    def __init__(
        self,
        frontal_channels: tuple[int, ...] = (0, 1, 2, 3),
        trough_channel: int = 5,
        clip: float = _BASE_LOGIT_CLIP,
    ) -> None:
        self._frontal_channels = frontal_channels
        self._trough_channel = trough_channel
        self._clip = clip

    def predict_logits(self, hrrr_norm_batch: np.ndarray, b0_probs_batch: np.ndarray) -> np.ndarray:
        frontal = np.sum(b0_probs_batch[..., list(self._frontal_channels)], axis=-1)
        trough = b0_probs_batch[..., self._trough_channel]
        groups = np.clip(np.stack([frontal, trough], axis=-1), self._clip, 1.0 - self._clip).astype(np.float64)
        base_logits = np.log(groups) - np.log1p(-groups)
        residual = 0.05 * np.tanh(hrrr_norm_batch[..., :2].astype(np.float64))
        return (base_logits + residual).astype(np.float32)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -_LOGIT_EXP_CLIP, _LOGIT_EXP_CLIP)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def run_c1_refinement(
    refiner: RefinerModel,
    input_3km: np.ndarray,
    probs_24km: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    manifest: ModelManifest = NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST,
    multiple: int = 8,
    block: int = 8,
) -> np.ndarray:
    """Refine upsampled 24 km B0 probabilities with C1 on `(H, W, C)` 3 km HRRR inputs."""
    arr_3km = np.asarray(input_3km, dtype=np.float32)
    if arr_3km.ndim != 3:
        raise ValueError(f"input_3km must be (H, W, C), got shape {arr_3km.shape}")
    height, width, n_channels = arr_3km.shape
    if n_channels != manifest.n_channels:
        raise ValueError(
            f"input_3km has {n_channels} channels, manifest {manifest.name!r} expects {manifest.n_channels}"
        )

    b0_24km = np.asarray(probs_24km, dtype=np.float32)
    if b0_24km.ndim != 3 or b0_24km.shape[-1] != len(manifest.served_classes):
        raise ValueError(
            f"probs_24km must be (Hc, Wc, {len(manifest.served_classes)}), got shape {b0_24km.shape}"
        )

    probs_24km_up = upsample_24km_to_3km(b0_24km, target_shape=(height, width), block=block)
    hrrr_norm = standardize_hrrr(arr_3km, mean, std)

    hrrr_pad = pad_south_east_to_multiple(hrrr_norm, multiple=multiple)
    probs_pad = pad_south_east_to_multiple(probs_24km_up, multiple=multiple)
    if not (math.isfinite(float(hrrr_pad.max())) and math.isfinite(float(probs_pad.max()))):
        raise ValueError("padded refiner inputs contain non-finite values")

    logits = np.asarray(
        refiner.predict_logits(hrrr_pad[np.newaxis, ...], probs_pad[np.newaxis, ...]),
        dtype=np.float32,
    )
    expected_shape = (1, *hrrr_pad.shape[:2], len(REFINED_GROUPS))
    if logits.shape != expected_shape:
        raise ValueError(f"refiner output shape {logits.shape} does not match {expected_shape}")
    if not np.isfinite(logits).all():
        raise ValueError("refiner output contains non-finite logits")

    c1_group_probs = _sigmoid(logits[0, :height, :width, :])
    return combine_c1_served_probabilities(probs_24km_up, c1_group_probs, manifest.served_classes)
