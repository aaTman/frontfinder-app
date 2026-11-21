import numpy as np
import pytest

from frontfinder.config.manifests import ALL_CLASSES, ModelManifest, VariableSpec
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
