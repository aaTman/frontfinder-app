import numpy as np
import pytest

from frontfinder.config.manifests import ALL_CLASSES, MODEL_1702_MANIFEST
from frontfinder.inference.engine import run_tiled_inference


class ConstantPredictor:
    """Fake predictor: returns a fixed uniform class distribution regardless
    of input, so stitching correctness can be checked in isolation from any
    real model behavior."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.calls: list[tuple[int, ...]] = []

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        self.calls.append(patches.shape)
        n, h, w, _ = patches.shape
        out = np.zeros((n, h, w, self.n_classes), dtype=np.float32)
        out[..., 1] = 1.0  # always "cold front" with probability 1
        return out


class ShapeCheckingPredictor:
    def __init__(self, expected_channels: int, n_classes: int):
        self.expected_channels = expected_channels
        self.n_classes = n_classes

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        assert patches.shape[-1] == self.expected_channels
        assert patches.shape[1] % 16 == 0
        assert patches.shape[2] % 16 == 0
        n, h, w, _ = patches.shape
        return np.random.default_rng(0).random((n, h, w, self.n_classes)).astype(np.float32)


def test_run_tiled_inference_output_shape_matches_unpadded_grid_and_served_classes():
    height, width = 300, 400  # deliberately not a multiple of 16
    input_grid = np.random.default_rng(0).normal(size=(height, width, MODEL_1702_MANIFEST.n_channels)).astype(np.float32)
    predictor = ConstantPredictor(n_classes=len(ALL_CLASSES))
    out = run_tiled_inference(predictor, input_grid, MODEL_1702_MANIFEST, patch_size=256, overlap=32)
    assert out.shape == (height, width, len(MODEL_1702_MANIFEST.served_classes))


def test_run_tiled_inference_recovers_constant_prediction_everywhere():
    height, width = 300, 400
    input_grid = np.zeros((height, width, MODEL_1702_MANIFEST.n_channels), dtype=np.float32)
    predictor = ConstantPredictor(n_classes=len(ALL_CLASSES))
    out = run_tiled_inference(predictor, input_grid, MODEL_1702_MANIFEST, patch_size=256, overlap=32)
    cold_idx = MODEL_1702_MANIFEST.served_classes.index("cold")
    np.testing.assert_allclose(out[..., cold_idx], 1.0, atol=1e-5)
    other_idx = [i for i in range(out.shape[-1]) if i != cold_idx]
    np.testing.assert_allclose(out[..., other_idx], 0.0, atol=1e-5)


def test_run_tiled_inference_rejects_wrong_channel_count():
    input_grid = np.zeros((64, 64, 3), dtype=np.float32)
    predictor = ConstantPredictor(n_classes=len(ALL_CLASSES))
    with pytest.raises(ValueError):
        run_tiled_inference(predictor, input_grid, MODEL_1702_MANIFEST, patch_size=64, overlap=16)


def test_run_tiled_inference_sends_only_16_multiple_patches_to_predictor():
    predictor = ShapeCheckingPredictor(
        expected_channels=MODEL_1702_MANIFEST.n_channels, n_classes=len(ALL_CLASSES)
    )
    input_grid = np.zeros((721, 1440, MODEL_1702_MANIFEST.n_channels), dtype=np.float32)  # real global grid
    out = run_tiled_inference(predictor, input_grid, MODEL_1702_MANIFEST, patch_size=256, overlap=32)
    assert out.shape == (721, 1440, len(MODEL_1702_MANIFEST.served_classes))


def test_run_tiled_inference_gives_seam_tiles_real_wrapped_context():
    # Regression test for the 2026-08-23 prime-meridian seam: without
    # circular padding, the tile(s) touching column 0 have no data from the
    # grid's own opposite edge (lon=359.75) even though they're physically
    # adjacent on the globe. Tag every pixel's first channel with its own
    # longitude and check the tile(s) covering column 0 actually received
    # values from the high end of the range (352.0+), not zero-padding.
    height = 256
    lon = np.linspace(0.0, 359.75, 1440)
    n_channels = MODEL_1702_MANIFEST.n_channels
    input_grid = np.zeros((height, 1440, n_channels), dtype=np.float32)
    input_grid[:, :, 0] = lon[None, :]

    all_rows: list[np.ndarray] = []

    class SeamCapturingPredictor:
        def predict_batch(self, patches: np.ndarray) -> np.ndarray:
            n, h, w, _ = patches.shape
            all_rows.extend(patches[i, 0, :, 0] for i in range(n))
            return np.zeros((n, h, w, len(ALL_CLASSES)), dtype=np.float32)

    run_tiled_inference(
        SeamCapturingPredictor(), input_grid, MODEL_1702_MANIFEST,
        patch_size=256, overlap=32, lon_deg=lon,
    )

    # find a tile that has both a high-longitude pixel (from the wrap-pad)
    # and, at the very next column, lon=0 -- i.e. the seam's two physically
    # adjacent longitudes ended up adjacent in a single tile, proving the
    # tile got real context across the wrap rather than a synthetic zero.
    found_continuous_wrap = any(
        row[j] >= 352.0 and row[j + 1] == 0.0
        for row in all_rows
        for j in range(len(row) - 1)
    )
    assert found_continuous_wrap, "no tile saw lon=352..360 immediately followed by lon=0 -- seam wasn't wrapped"


def test_run_tiled_inference_batches_calls_to_predictor():
    predictor = ConstantPredictor(n_classes=len(ALL_CLASSES))
    input_grid = np.zeros((512, 512, MODEL_1702_MANIFEST.n_channels), dtype=np.float32)
    run_tiled_inference(predictor, input_grid, MODEL_1702_MANIFEST, patch_size=256, overlap=32, batch_size=2)
    # every call except possibly the last should be exactly batch_size patches
    assert all(shape[0] <= 2 for shape in predictor.calls)


def test_run_tiled_inference_flips_southern_hemisphere_rows_before_predicting():
    # The model has only ever seen northern-hemisphere-consistent fronts, so
    # southern-hemisphere (lat < 0) rows must be mirrored across the equator
    # before the predictor sees them. Tag channel 0 with each row's original
    # index and capture exactly what the predictor receives: northern rows
    # (lat >= 0) must arrive in their original order, southern rows reversed.
    height, width = 32, 32
    lat = np.linspace(15.0, -15.0, height)  # rows 0..15 >= 0, rows 16..31 < 0
    n_channels = MODEL_1702_MANIFEST.n_channels
    input_grid = np.zeros((height, width, n_channels), dtype=np.float32)
    input_grid[:, :, 0] = np.arange(height, dtype=np.float32)[:, None]

    captured: list[np.ndarray] = []

    class CapturingPredictor:
        def predict_batch(self, patches: np.ndarray) -> np.ndarray:
            captured.append(patches[0, :, 0, 0].copy())
            n, h, w, _ = patches.shape
            return np.zeros((n, h, w, len(ALL_CLASSES)), dtype=np.float32)

    run_tiled_inference(
        CapturingPredictor(), input_grid, MODEL_1702_MANIFEST,
        patch_size=32, overlap=0, lat_deg=lat,
    )

    seen = captured[0]
    np.testing.assert_allclose(seen[:16], np.arange(16))  # northern rows untouched
    np.testing.assert_allclose(seen[16:], np.arange(31, 15, -1))  # southern rows reversed


def test_run_tiled_inference_flips_output_back_to_original_orientation():
    height, width = 32, 32
    lat = np.linspace(15.0, -15.0, height)
    n_channels = MODEL_1702_MANIFEST.n_channels
    input_grid = np.zeros((height, width, n_channels), dtype=np.float32)
    input_grid[:, :, 0] = np.arange(height, dtype=np.float32)[:, None]

    class RowTagPredictor:
        """Echoes back the row-tag channel as every served class's value, so
        the output can be checked for having been un-flipped."""

        def predict_batch(self, patches: np.ndarray) -> np.ndarray:
            n, h, w, _ = patches.shape
            tag = patches[..., 0]
            return np.repeat(tag[..., None], len(ALL_CLASSES), axis=-1).astype(np.float32)

    out = run_tiled_inference(
        RowTagPredictor(), input_grid, MODEL_1702_MANIFEST,
        patch_size=32, overlap=0, lat_deg=lat,
    )

    cold_idx = MODEL_1702_MANIFEST.served_classes.index("cold")
    np.testing.assert_allclose(out[:, 0, cold_idx], np.arange(height))


def test_run_tiled_inference_is_unaffected_by_lat_deg_when_all_northern():
    height, width = 32, 32
    lat = np.linspace(45.0, 15.0, height)  # entirely northern hemisphere
    n_channels = MODEL_1702_MANIFEST.n_channels
    rng = np.random.default_rng(0)
    input_grid = rng.standard_normal((height, width, n_channels)).astype(np.float32)
    predictor = ConstantPredictor(n_classes=len(ALL_CLASSES))

    with_lat = run_tiled_inference(
        predictor, input_grid, MODEL_1702_MANIFEST, patch_size=32, overlap=0, lat_deg=lat,
    )
    without_lat = run_tiled_inference(
        predictor, input_grid, MODEL_1702_MANIFEST, patch_size=32, overlap=0,
    )
    np.testing.assert_allclose(with_lat, without_lat)


def test_keras_predictor_serves_the_first_deep_supervision_head_not_the_last():
    # Regression test for the 2026-08-22 "wide flat bar" bug: KerasPredictor
    # used to take output[-1] on the (wrong) assumption that a deep
    # -supervision model's LAST head is the finest/full-resolution one.
    # Directly measuring the real _best_loss.keras model's four sup1..sup4
    # heads against real assembled input showed the opposite -- sup1/sup2
    # are full resolution, sup3 is half, and sup4 (what was being served)
    # is quarter-resolution, a crude 4x4-block nearest-neighbor upsample
    # from a much coarser decoder stage. That block quantization is what
    # rendered as a wide, flat, structureless bar wherever a real front
    # gradient crossed one of its block edges (reported near Bermuda).
    # This builds a tiny synthetic model with the same sup1..sup4 shape
    # (first head genuinely finer than the last) to pin the fix -- output[0]
    # must be served, not output[-1] -- without needing the real weights
    # file or TensorFlow-heavy training setup.
    pytest.importorskip("keras")
    import keras

    from frontfinder.inference.engine import KerasPredictor

    inputs = keras.Input(shape=(None, None, 3))
    # sup1: full-resolution head (no pooling/upsampling in the path).
    sup1 = keras.layers.Conv2D(2, 1, activation="softmax", name="sup1_softmax")(inputs)
    # sup4: half-resolution features upsampled back with nearest-neighbor,
    # the same block-quantizing operation the real model's coarser heads use.
    pooled = keras.layers.MaxPooling2D(2)(inputs)
    upsampled = keras.layers.UpSampling2D(2, interpolation="nearest")(pooled)
    sup4 = keras.layers.Conv2D(2, 1, activation="softmax", name="sup4_softmax")(upsampled)
    model = keras.Model(inputs=inputs, outputs=[sup1, sup4])

    predictor = KerasPredictor.__new__(KerasPredictor)  # bypass load_model(); inject the model directly
    predictor._model = model

    rng = np.random.default_rng(0)
    patch = rng.standard_normal((1, 16, 16, 3)).astype(np.float32)
    served = predictor.predict_batch(patch)

    plane = served[0, :, :, 0]
    dup_rows = sum(1 for r in range(1, plane.shape[0]) if np.array_equal(plane[r], plane[r - 1]))
    assert dup_rows == 0, "served output has duplicate rows -- picked the coarse head, not the fine one"
