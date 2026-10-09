import numpy as np
import pytest

from frontfinder.config.manifests import ALL_CLASSES, MODEL_1702_MANIFEST, ModelManifest, VariableSpec
from frontfinder.inference.engine import VolumetricKerasPredictor, run_tiled_inference


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


def test_run_tiled_inference_negates_meridional_wind_on_southern_rows_in_final_output():
    # The model has only ever seen northern-hemisphere-consistent
    # (counterclockwise) fronts, so the final, row-aligned output must show
    # southern-hemisphere (lat < 0) rows with their meridional wind sign
    # flipped, in their ORIGINAL row order -- the two-pass whole-grid flip
    # (see engine.run_tiled_inference's docstring) must restore correct row
    # alignment on the way out, not leave the south still reversed.
    height, width = 32, 32
    lat = np.linspace(15.0, -15.0, height)  # rows 0..15 >= 0, rows 16..31 < 0
    n_channels = MODEL_1702_MANIFEST.n_channels
    v_idx = MODEL_1702_MANIFEST.channel_names().index("v_component_of_wind_1000")
    input_grid = np.zeros((height, width, n_channels), dtype=np.float32)
    input_grid[:, :, v_idx] = np.arange(height, dtype=np.float32)[:, None]

    class RowTagPredictor:
        """Echoes the v-channel's own value back as every served class, so
        the final stitched output reveals exactly what value each row's
        v-channel carried all the way through (both passes, both flips)."""

        def predict_batch(self, patches: np.ndarray) -> np.ndarray:
            tag = patches[..., v_idx]
            return np.repeat(tag[..., None], len(ALL_CLASSES), axis=-1).astype(np.float32)

    out = run_tiled_inference(
        RowTagPredictor(), input_grid, MODEL_1702_MANIFEST,
        patch_size=32, overlap=0, lat_deg=lat,
    )

    cold_idx = MODEL_1702_MANIFEST.served_classes.index("cold")
    np.testing.assert_allclose(out[:16, 0, cold_idx], np.arange(16))         # northern rows untouched
    np.testing.assert_allclose(out[16:, 0, cold_idx], -np.arange(16, 32))    # southern rows negated, order preserved


def test_run_tiled_inference_does_not_negate_zonal_wind_or_other_channels():
    height, width = 32, 32
    lat = np.linspace(15.0, -15.0, height)
    n_channels = MODEL_1702_MANIFEST.n_channels
    u_idx = MODEL_1702_MANIFEST.channel_names().index("u_component_of_wind_1000")
    input_grid = np.zeros((height, width, n_channels), dtype=np.float32)
    input_grid[:, :, u_idx] = np.arange(height, dtype=np.float32)[:, None]

    class RowTagPredictor:
        def predict_batch(self, patches: np.ndarray) -> np.ndarray:
            tag = patches[..., u_idx]
            return np.repeat(tag[..., None], len(ALL_CLASSES), axis=-1).astype(np.float32)

    out = run_tiled_inference(
        RowTagPredictor(), input_grid, MODEL_1702_MANIFEST,
        patch_size=32, overlap=0, lat_deg=lat,
    )

    cold_idx = MODEL_1702_MANIFEST.served_classes.index("cold")
    np.testing.assert_allclose(out[:, 0, cold_idx], np.arange(height))  # untouched, and correctly row-aligned


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


def test_volumetric_keras_predictor_reshapes_flat_channels_into_level_variable_volume():
    # VolumetricKerasPredictor must turn the SAME flat, level-major channel
    # layout every other model gets (see ModelManifest.channel_names()) into
    # (level, variable) axes: the pressure-level block is a plain reshape
    # (level-major-then-variable-inner IS (level, variable) in row-major
    # order already), and the single-level channels get broadcast identically
    # across every level -- matching how the training pipeline built its
    # volume batches (see SPIX_WCE_DILATION_MANIFEST's comment).
    manifest = ModelManifest(
        name="fake_volumetric",
        weights_filename="fake.keras",
        variables=(
            VariableSpec("a", levels=(1000, 500)),
            VariableSpec("b", levels=(1000, 500)),
            VariableSpec("c", levels=None),
            VariableSpec("d", levels=None),
        ),
        volumetric=True,
    )
    # channel_names(): a_1000, b_1000, a_500, b_500, c, d
    assert manifest.channel_names() == ["a_1000", "b_1000", "a_500", "b_500", "c", "d"]

    predictor = VolumetricKerasPredictor.__new__(VolumetricKerasPredictor)  # bypass load_model()
    predictor._device = None
    predictor._n_levels = 2
    predictor._n_pressure_vars = 2
    predictor._n_single_vars = 2

    captured = {}

    class FakeModel:
        def predict(self, x, verbose=0):
            captured["input"] = x
            return np.zeros((x.shape[0], x.shape[1], x.shape[2], len(ALL_CLASSES)), dtype=np.float32)

    predictor._model = FakeModel()

    n, h, w = 1, 2, 2
    patches = np.zeros((n, h, w, 6), dtype=np.float32)
    patches[..., 0] = 10  # a_1000
    patches[..., 1] = 20  # b_1000
    patches[..., 2] = 11  # a_500
    patches[..., 3] = 21  # b_500
    patches[..., 4] = 100  # c (single-level)
    patches[..., 5] = 200  # d (single-level)

    predictor.predict_batch(patches)

    volume = captured["input"]
    assert volume.shape == (n, h, w, 2, 4)
    np.testing.assert_allclose(volume[0, 0, 0, 0], [10, 20, 100, 200])  # level 1000: a,b,c,d
    np.testing.assert_allclose(volume[0, 0, 0, 1], [11, 21, 100, 200])  # level 500: single-level vars unchanged


def test_keras_predictor_wraps_load_and_predict_in_the_given_device(monkeypatch):
    # 2026-09-17 (two-3090 host): cli.py pins each model to its own GPU via
    # `device=`. Uses a real "/CPU:0" (present on every machine, unlike a
    # GPU) so this actually exercises tf.device()'s real context manager --
    # not just a mock -- while still proving KerasPredictor requests THAT
    # device for both load_model() and predict(), which is the part that
    # would misroute inference onto the wrong physical device if it
    # regressed.
    pytest.importorskip("keras")
    import keras
    import tensorflow as tf

    from frontfinder.inference.engine import KerasPredictor

    inputs = keras.Input(shape=(None, None, 2))
    outputs = keras.layers.Conv2D(2, 1, activation="softmax")(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model_path = "/tmp/frontfinder_test_tiny_device_model.keras"
    model.save(model_path)

    requested_devices: list[str] = []
    real_tf_device = tf.device

    def recording_device(name):
        requested_devices.append(name)
        return real_tf_device(name)

    monkeypatch.setattr(tf, "device", recording_device)

    predictor = KerasPredictor(model_path, device="/CPU:0")
    rng = np.random.default_rng(0)
    patch = rng.standard_normal((1, 8, 8, 2)).astype(np.float32)
    predictor.predict_batch(patch)

    assert requested_devices == ["/CPU:0", "/CPU:0"]  # once for load_model, once for predict


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
    predictor._device = None

    rng = np.random.default_rng(0)
    patch = rng.standard_normal((1, 16, 16, 3)).astype(np.float32)
    served = predictor.predict_batch(patch)

    plane = served[0, :, :, 0]
    dup_rows = sum(1 for r in range(1, plane.shape[0]) if np.array_equal(plane[r], plane[r - 1]))
    assert dup_rows == 0, "served output has duplicate rows -- picked the coarse head, not the fine one"
