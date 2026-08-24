"""Tiled inference engine: runs a Keras model over a full grid patch-by-patch
via `frontfinder.inference.tiling`, and reduces the softmax output down to the
served front classes per `ModelManifest.served_class_indices()`.

The Keras model itself is accessed through a narrow `Predictor` protocol
(`predict_batch(patches) -> class_probs`) so tests can swap in a fake model
and never need TensorFlow installed or a real `.keras`/`.h5` file on disk.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from frontfinder.config.manifests import ModelManifest
from frontfinder.inference.periodic import circularly_pad_longitude
from frontfinder.inference.tiling import generate_tiles, pad_to_multiple, stitch


class Predictor(Protocol):
    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        """patches: (n, patch, patch, n_channels) -> (n, patch, patch, n_classes)"""
        ...


class KerasPredictor:
    """Thin adapter around a loaded `keras.Model`. Not covered by unit tests --
    exercise this in an integration smoke test on the Proxmox VM where
    TensorFlow/Keras and the real weights files are available.
    """

    def __init__(self, model_path: str):
        import keras  # local import: keeps TensorFlow out of the unit-test path

        # Importing this module runs its @register_keras_serializable
        # decorators as a side effect, registering vendored stand-ins for
        # fronts.model's SharedTargetModel/TemperatureScaledModel *before*
        # load_model() runs -- without it, load_model() raises
        # `TypeError: Could not locate class 'SharedTargetModel'` because
        # the saved config's registered_name ("fronts>SharedTargetModel")
        # was never registered in this process. See keras_compat.py's
        # module docstring for why vendoring instead of depending on the
        # real `fronts` training repo is safe here.
        from frontfinder.inference import keras_compat

        keras_compat.ensure_registered()
        self._model = keras.models.load_model(model_path, compile=False)

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        output = self._model.predict(patches, verbose=0)
        # 2026-08-22 correction: this model's deep-supervision heads are
        # named sup1..sup4 and were WRONGLY assumed to go
        # coarsest-to-finest, with the finest ("full-resolution") head
        # last -- so this used to take output[-1]. Directly measuring each
        # head's actual spatial resolution against real assembled input
        # (counting exact-duplicate rows/cols -- a nearest-neighbor
        # upsample from a coarser decoder stage leaves whole rows/cols
        # byte-identical) showed the opposite: sup1/sup2 are true full
        # resolution (0 duplicate rows/cols out of 255), sup3 is
        # half-resolution (2x2 blocks), and sup4 -- the one being served
        # -- is quarter-resolution (4x4 blocks, i.e. ~1deg on this 0.25deg
        # grid). That 4x4 block quantization is exactly the "wide, thin,
        # flat bar" artifact reported near Bermuda: a real front gradient
        # falling across one of sup4's crude 4x4 upsample block edges.
        # UNet3+'s convention (this model's architecture, per its module
        # name `unet_3plus_2D`) is sup1 = the main/finest decoder output,
        # sup2..sup4 = auxiliary coarser heads added only to supervise
        # intermediate decoder stages during training -- so output[0], not
        # output[-1], is the one to serve.
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output


def run_tiled_inference(
    predictor: Predictor,
    input_grid: np.ndarray,
    manifest: ModelManifest,
    patch_size: int = 256,
    overlap: int = 32,
    batch_size: int = 8,
    lon_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Run `predictor` over `input_grid` (H, W, n_channels) patch-by-patch and
    return served-class probabilities on the original (unpadded) grid, shape
    (H, W, len(manifest.served_classes)).

    `lon_deg`: the grid's 1-D longitude coordinate (e.g. `source.lon`).
    Circular padding is only applied when this actually spans the full
    360deg globe (checked below) -- when given, the grid is circularly
    extended by `overlap` columns on each side (see
    `periodic.circularly_pad_longitude`) before tiling, so patches straddling
    the lon=0/360 array edge get real context from the opposite side instead
    of a hard, seam-producing edge -- see that module's docstring for the
    2026-08-23 prime-meridian artifact this fixes. Safe to pass a regional
    (non-global) grid's lon coordinate too -- it's a no-op there, since a
    regional box has no real wraparound to give it.
    """
    if input_grid.ndim != 3:
        raise ValueError(f"input_grid must be (H, W, C), got shape {input_grid.shape}")
    height, width, n_channels = input_grid.shape
    if n_channels != manifest.n_channels:
        raise ValueError(
            f"input_grid has {n_channels} channels, manifest {manifest.name!r} "
            f"expects {manifest.n_channels}"
        )

    is_global_lon = (
        lon_deg is not None
        and len(lon_deg) > 1
        and np.isclose(float(lon_deg[-1] - lon_deg[0]) + float(lon_deg[1] - lon_deg[0]), 360.0, atol=1e-6)
    )
    lon_pad = overlap if is_global_lon else 0
    working_grid = circularly_pad_longitude(input_grid, lon_deg, lon_pad) if lon_pad else input_grid
    working_width = working_grid.shape[1]

    padded_h = pad_to_multiple(height, manifest.patch_multiple)
    padded_w = pad_to_multiple(working_width, manifest.patch_multiple)
    # patch_size itself must also be a multiple of patch_multiple, and large
    # enough to cover the padded grid in at least one tile.
    padded_h = max(padded_h, patch_size)
    padded_w = max(padded_w, patch_size)

    padded = np.zeros((padded_h, padded_w, n_channels), dtype=input_grid.dtype)
    padded[:height, :working_width, :] = working_grid

    tiles = generate_tiles(padded_h, padded_w, patch_size, overlap, multiple=manifest.patch_multiple)

    served_idx = manifest.served_class_indices()
    predictions: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start:start + batch_size]
        batch = np.stack(
            [padded[t.row_start:t.row_end, t.col_start:t.col_end, :] for t in batch_tiles],
            axis=0,
        )
        out = predictor.predict_batch(batch)
        for i in range(out.shape[0]):
            predictions.append(out[i][..., served_idx])

    stitched = stitch(tiles, predictions, out_height=height, out_width=working_width, overlap=overlap)
    if lon_pad:
        stitched = stitched[:, lon_pad:lon_pad + width, :]
    return stitched
