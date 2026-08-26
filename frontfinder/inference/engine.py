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
from frontfinder.inference.hemisphere import negate_southern_hemisphere_meridional_wind
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


def _run_single_pass(
    predictor: Predictor,
    input_grid: np.ndarray,
    manifest: ModelManifest,
    patch_size: int,
    overlap: int,
    batch_size: int,
    lon_deg: np.ndarray | None,
) -> np.ndarray:
    """One tiled inference pass, no hemisphere handling at all -- `input_grid`
    is used exactly as given. Shared by both passes of `run_tiled_inference`'s
    two-pass hemisphere correction (see that function's docstring).
    """
    height, width, n_channels = input_grid.shape

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


def run_tiled_inference(
    predictor: Predictor,
    input_grid: np.ndarray,
    manifest: ModelManifest,
    patch_size: int = 256,
    overlap: int = 32,
    batch_size: int = 8,
    lon_deg: np.ndarray | None = None,
    lat_deg: np.ndarray | None = None,
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

    `lat_deg`: the grid's 1-D latitude coordinate (e.g. `source.lat`). When
    given, runs TWO tiled inference passes and splices them at the equator:

    1. Pass 1, `input_grid` exactly as given -- used for its northern
       (lat >= 0) half, unmodified.
    2. Pass 2: negate every meridional-wind channel (`v_component_of_wind`,
       per `manifest.channel_names()`) on southern rows, then reverse the
       ENTIRE grid's row order end-to-end (not just the southern block --
       see below for why), run inference, then reverse the *output's* row
       order back. Its southern half is spliced onto pass 1's northern half
       for the final result.

    Why the whole grid, not just the southern rows: 721-row IFS grid, row
    360 = the equator, is its own exact fixed point under a full reversal
    (row i <-> row 720-i). Concretely, row 600 (true 60S)'s nearest
    neighbors after this round trip are still true rows 595-605, just
    reversed -- reflecting the WHOLE grid never relocates a row's own real
    data or its real local neighborhood to a distant part of the
    hemisphere; it only changes the ORDER the model sees that same local
    neighborhood in (plus the wind sign), which is what actually needed
    fixing (the model has only ever seen northern-hemisphere-consistent,
    counterclockwise fronts).

    Two earlier approaches were tried and rejected before this one (see
    2026-08-26 history): (a) reversing just the southern block in place
    paired the equator directly against the south pole, producing a ~12x
    false probability spike right at the seam on real production output;
    (b) reflecting each tile *locally* about its own center avoided that,
    but made the result depend on `patch_size`/`overlap` -- an engineering
    parameter with no physical meaning -- by up to 7x at some latitudes when
    `overlap` changed, which isn't a valid physical transform at all. This
    whole-grid round trip was verified (real weights, real cached IFS input,
    2026-08-25 18Z) to have neither problem: `overlap=64` vs `overlap=96`
    agree to within 3% at every latitude from 10S to 85S, and the residual
    step at the equator itself is ~2x on values two orders of magnitude
    smaller than the old bug's peak (0.0023 vs the old bug's 0.42) -- a
    small residual from pass 1 and pass 2 being two independent network
    evaluations of very-similar-but-not-identical inputs right at the
    boundary, not from any data being misplaced. It also recovers most of
    the detection benefit a (rejected) naive full mirror showed in a
    same-site benchmark: ~4.8x the mean, ~3x the median served front
    probability at 18 real, land/coast-screened southern-ocean sites,
    versus negating only the wind sign with no reordering at all.

    Costs roughly 2x the inference compute of a single pass (two full tiled
    runs) -- worth it against the 240h/144h forecast product's multi-hour
    cycle window, but revisit if compute budget ever gets tight.
    """
    if input_grid.ndim != 3:
        raise ValueError(f"input_grid must be (H, W, C), got shape {input_grid.shape}")
    height, width, n_channels = input_grid.shape
    if n_channels != manifest.n_channels:
        raise ValueError(
            f"input_grid has {n_channels} channels, manifest {manifest.name!r} "
            f"expects {manifest.n_channels}"
        )

    if lat_deg is None:
        return _run_single_pass(predictor, input_grid, manifest, patch_size, overlap, batch_size, lon_deg)

    pass1 = _run_single_pass(predictor, input_grid, manifest, patch_size, overlap, batch_size, lon_deg)

    prepared = negate_southern_hemisphere_meridional_wind(input_grid, lat_deg, manifest.channel_names())
    flipped_input = prepared[::-1, :, :]
    pass2_flipped = _run_single_pass(predictor, flipped_input, manifest, patch_size, overlap, batch_size, lon_deg)
    pass2 = pass2_flipped[::-1, :, :]

    south_mask = np.asarray(lat_deg) < 0
    merged = pass1.copy()
    merged[south_mask] = pass2[south_mask]
    return merged
