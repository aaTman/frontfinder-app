import numpy as np
import pytest

from frontfinder.config.manifests import (
    ALL_CLASSES,
    NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST,
    ModelManifest,
    VariableSpec,
)
from frontfinder.inference.refiner import (
    FakeRefinerPredictor,
    combine_c1_served_probabilities,
    run_c1_refinement,
    upsample_24km_to_3km,
)
from frontfinder.inference.regional import run_regional_inference

HRRR_SHAPES = [(1059, 1799), (529, 899), (264, 449), (132, 224), (103, 207)]

MANIFEST = ModelManifest(
    name="fake_regional",
    weights_filename="fake.keras",
    variables=(VariableSpec("a", levels=None), VariableSpec("b", levels=None)),
)


class EchoPredictor:
    """Returns channel 0 as every class and records what it was given."""

    def __init__(self, n_classes: int = len(ALL_CLASSES)):
        self.n_classes = n_classes
        self.batches: list[np.ndarray] = []

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        self.batches.append(patches)
        return np.repeat(patches[..., :1], self.n_classes, axis=-1)


def _positive_grid(height: int, width: int) -> np.ndarray:
    rows = np.arange(1, height + 1, dtype=np.float32)[:, None]
    cols = np.arange(1, width + 1, dtype=np.float32)[None, :]
    return np.stack([rows * 10_000 + cols, rows + cols], axis=-1)


@pytest.mark.parametrize(("height", "width"), HRRR_SHAPES)
def test_round_trip_shape_alignment_and_padded_input(height: int, width: int):
    grid = _positive_grid(height, width)
    predictor = EchoPredictor()

    out = run_regional_inference(predictor, grid, MANIFEST)

    assert out.shape == (height, width, len(MANIFEST.served_classes))
    assert out.dtype == np.float32
    for class_idx in range(out.shape[-1]):
        np.testing.assert_array_equal(out[..., class_idx], grid[..., 0])

    assert len(predictor.batches) == 1
    batch = predictor.batches[0]
    assert batch.shape[0] == 1
    assert batch.shape[1] % 8 == 0
    assert batch.shape[2] % 8 == 0
    assert batch.shape[1] - height < 8
    assert batch.shape[2] - width < 8
    assert batch.min() > 0  # every cell in this grid is positive, so a zero is padding


def test_padding_is_an_even_reflection_on_both_sides_of_each_axis():
    # 13 -> 16 rows (1 before, 2 after); 6 -> 8 columns (1 before, 1 after).
    rows = np.arange(1, 14, dtype=np.float32)
    cols = np.arange(1, 7, dtype=np.float32)
    grid = np.stack(
        [np.broadcast_to(rows[:, None], (13, 6)), np.broadcast_to(cols[None, :], (13, 6))],
        axis=-1,
    )
    predictor = EchoPredictor()

    run_regional_inference(predictor, grid, MANIFEST)

    batch = predictor.batches[0][0]
    np.testing.assert_array_equal(batch[:, 0, 0], [2, *range(1, 14), 12, 11])
    np.testing.assert_array_equal(batch[0, :, 1], [2, 1, 2, 3, 4, 5, 6, 5])


def test_grid_smaller_than_the_padding_is_still_reflected():
    grid = _positive_grid(3, 5)
    predictor = EchoPredictor()

    out = run_regional_inference(predictor, grid, MANIFEST)

    assert predictor.batches[0].shape == (1, 8, 8, 2)
    assert predictor.batches[0].min() > 0
    np.testing.assert_array_equal(out[..., 0], grid[..., 0])


def test_already_divisible_grid_is_not_padded():
    grid = _positive_grid(16, 24)
    predictor = EchoPredictor()

    run_regional_inference(predictor, grid, MANIFEST)

    np.testing.assert_array_equal(predictor.batches[0][0], grid)


def test_custom_multiple_is_respected():
    grid = _positive_grid(132, 224)
    predictor = EchoPredictor()

    out = run_regional_inference(predictor, grid, MANIFEST, multiple=16)

    assert predictor.batches[0].shape[1:3] == (144, 224)
    np.testing.assert_array_equal(out[..., 0], grid[..., 0])


def test_selects_served_classes_in_manifest_order():
    manifest = ModelManifest(
        name="fake_served",
        weights_filename="fake.keras",
        variables=MANIFEST.variables,
        served_classes=("warm", "cold"),
    )

    class ClassTagPredictor:
        def predict_batch(self, patches: np.ndarray) -> np.ndarray:
            n, h, w, _ = patches.shape
            tags = np.arange(len(ALL_CLASSES), dtype=np.float64)
            return np.broadcast_to(tags, (n, h, w, len(ALL_CLASSES))).copy()

    out = run_regional_inference(ClassTagPredictor(), _positive_grid(13, 6), manifest)

    assert out.shape == (13, 6, 2)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[..., 0], ALL_CLASSES.index("warm"))
    np.testing.assert_array_equal(out[..., 1], ALL_CLASSES.index("cold"))


def test_input_grid_is_not_modified():
    grid = _positive_grid(13, 6)
    original = grid.copy()

    run_regional_inference(EchoPredictor(), grid, MANIFEST)

    np.testing.assert_array_equal(grid, original)


@pytest.mark.parametrize("shape", [(16, 16), (1, 16, 16, 2)])
def test_rejects_non_3d_grid(shape: tuple[int, ...]):
    predictor = EchoPredictor()
    with pytest.raises(ValueError, match="must be"):
        run_regional_inference(predictor, np.ones(shape, dtype=np.float32), MANIFEST)
    assert predictor.batches == []


def test_rejects_wrong_channel_count():
    predictor = EchoPredictor()
    with pytest.raises(ValueError, match="channels"):
        run_regional_inference(predictor, np.ones((16, 16, 3), dtype=np.float32), MANIFEST)
    assert predictor.batches == []


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_input(bad_value: float):
    grid = _positive_grid(16, 16)
    grid[7, 9, 1] = bad_value
    predictor = EchoPredictor()
    with pytest.raises(ValueError, match="non-finite"):
        run_regional_inference(predictor, grid, MANIFEST)
    assert predictor.batches == []


def test_rejects_predictor_output_with_the_wrong_spatial_shape():
    class DownsamplingPredictor:
        def predict_batch(self, patches: np.ndarray) -> np.ndarray:
            n, h, w, _ = patches.shape
            return np.zeros((n, h // 2, w // 2, len(ALL_CLASSES)), dtype=np.float32)

    with pytest.raises(ValueError, match="output shape"):
        run_regional_inference(DownsamplingPredictor(), _positive_grid(16, 16), MANIFEST)


def test_upsample_24km_to_3km_reproduces_linear_field_and_clamps_hull():
    fine = (1059, 1799)
    coarse = (132, 224)
    # In south-up index space, coarse node j sits at 8*j + 3.5.
    cy = 8.0 * np.arange(coarse[0], dtype=np.float32) + 3.5
    cx = 8.0 * np.arange(coarse[1], dtype=np.float32) + 3.5
    yy_c, xx_c = np.meshgrid(cy[::-1], cx, indexing="ij")
    coarse_field = np.stack([2.0 * yy_c + 3.0 * xx_c, yy_c - xx_c], axis=-1).astype(np.float32)

    out = upsample_24km_to_3km(coarse_field, target_shape=fine, block=8)

    assert out.shape == (*fine, 2)
    assert out.dtype == np.float32
    fy = np.arange(fine[0], dtype=np.float32)[::-1]
    fx = np.arange(fine[1], dtype=np.float32)
    yy_f, xx_f = np.meshgrid(fy, fx, indexing="ij")
    expected_0 = 2.0 * np.clip(yy_f, 3.5, cy[-1]) + 3.0 * np.clip(xx_f, 3.5, cx[-1])
    expected_1 = np.clip(yy_f, 3.5, cy[-1]) - np.clip(xx_f, 3.5, cx[-1])
    np.testing.assert_allclose(out[..., 0], expected_0, atol=2e-3)
    np.testing.assert_allclose(out[..., 1], expected_1, atol=2e-3)


def test_combine_c1_served_probabilities_scales_frontal_shares_and_preserves_passthrough():
    served = NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST.served_classes
    b0 = np.zeros((2, 2, len(served)), dtype=np.float32)
    # Cell (0, 0): cold=0.3, warm=0.1, stationary=0.0, occluded=0.0 (frontal=0.4) -> shares (0.75, 0.25, 0, 0)
    b0[0, 0, served.index("cold")] = 0.3
    b0[0, 0, served.index("warm")] = 0.1
    # Cell (0, 1): all frontal classes zero -> uniform 0.25 share each
    b0[:, :, served.index("dryline")] = 0.42
    b0[:, :, served.index("tropical_trough")] = 0.17
    b0[:, :, served.index("trough")] = 0.05

    c1 = np.array([[[0.8, 0.25], [0.6, 0.9]], [[0.0, 0.0], [1.0, 0.5]]], dtype=np.float32)
    out = combine_c1_served_probabilities(b0, c1, served)

    assert out.shape == b0.shape and out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0, [served.index(c) for c in ("cold", "warm", "stationary", "occluded")]], [0.6, 0.2, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[0, 1, [served.index(c) for c in ("cold", "warm", "stationary", "occluded")]], [0.15, 0.15, 0.15, 0.15], atol=1e-6)
    np.testing.assert_allclose(out[..., served.index("trough")], c1[..., 1], atol=1e-6)
    np.testing.assert_allclose(out[..., served.index("dryline")], 0.42, atol=1e-6)
    np.testing.assert_allclose(out[..., served.index("tropical_trough")], 0.17, atol=1e-6)


def test_run_c1_refinement_end_to_end_with_fake_refiner():
    manifest = NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST
    fine_shape = (19, 25)
    coarse_shape = (2, 3)
    rng = np.random.default_rng(7)
    input_3km = rng.normal(size=(*fine_shape, manifest.n_channels)).astype(np.float32)
    probs_24km = rng.uniform(0.0, 0.2, size=(*coarse_shape, len(manifest.served_classes))).astype(np.float32)
    mean = np.zeros(manifest.n_channels, dtype=np.float32)
    std = np.full(manifest.n_channels, 2.0, dtype=np.float32)

    class RecordingFakeRefiner(FakeRefinerPredictor):
        def __init__(self) -> None:
            super().__init__()
            self.shapes: list[tuple[int, ...]] = []

        def predict_logits(self, hrrr_norm_batch: np.ndarray, b0_probs_batch: np.ndarray) -> np.ndarray:
            self.shapes.append(hrrr_norm_batch.shape)
            assert hrrr_norm_batch.shape[:3] == b0_probs_batch.shape[:3]
            return super().predict_logits(hrrr_norm_batch, b0_probs_batch)

    refiner = RecordingFakeRefiner()
    out = run_c1_refinement(refiner, input_3km, probs_24km, mean, std, manifest)

    assert refiner.shapes == [(1, 24, 32, manifest.n_channels)]
    assert out.shape == (*fine_shape, len(manifest.served_classes))
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
