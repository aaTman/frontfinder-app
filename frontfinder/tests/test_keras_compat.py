"""Tests for keras_compat.py's registration side effect -- the fix for the
`TypeError: Could not locate class 'SharedTargetModel'` error hit during the
first real end-to-end smoke test on mandelhub (2026-08-20).

Skipped automatically in the sandbox (no `keras`/TensorFlow installed
there, same as KerasPredictor itself -- see engine.py and the README's
"what's not covered by the test suite" note). Run on the Proxmox container
with `uv run --group dev pytest -q` where TensorFlow is actually installed.
"""

import pytest

keras = pytest.importorskip("keras")


def test_shared_target_model_is_registered_under_the_saved_models_name():
    # This is the exact registered_name string that appeared in the
    # TypeError's "Full object config" dump from the real _best_loss.keras
    # load failure -- if this lookup fails, load_model() will too.
    from frontfinder.inference import keras_compat

    keras_compat.ensure_registered()
    cls = keras.saving.get_registered_object("fronts>SharedTargetModel")
    assert cls is keras_compat.SharedTargetModel


def test_temperature_scaled_model_is_registered_under_the_saved_models_name():
    from frontfinder.inference import keras_compat

    keras_compat.ensure_registered()
    cls = keras.saving.get_registered_object("fronts>TemperatureScaledModel")
    assert cls is keras_compat.TemperatureScaledModel


def test_shared_target_model_broadcasts_single_target_across_list_outputs():
    from frontfinder.inference.keras_compat import SharedTargetModel

    # SharedTargetModel adds no forward-pass behavior -- constructing one
    # around a trivial functional graph and checking compute_loss doesn't
    # raise on a list of outputs (deep-supervision shape) is enough to
    # confirm the broadcast logic itself, without needing a real trained
    # model.
    inputs = keras.Input(shape=(4,))
    out1 = keras.layers.Dense(2, activation="softmax", name="sup1")(inputs)
    out2 = keras.layers.Dense(2, activation="softmax", name="sup2")(inputs)
    model = SharedTargetModel(inputs=inputs, outputs=[out1, out2])
    model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")

    import numpy as np

    x = np.zeros((3, 4), dtype="float32")
    y = np.zeros((3,), dtype="int64")  # single target, NOT a list -- the whole point
    model.fit(x, y, epochs=1, verbose=0)  # would raise if broadcasting were missing


def test_keras_predictor_loads_real_best_loss_weights_if_present(tmp_path, monkeypatch):
    """Guards the actual bug: only runs against a real weights file, which
    won't exist in CI -- deliberately skips rather than failing when absent
    so this suite stays green off the Proxmox container, while still
    catching a regression the moment it's run somewhere the weights are.
    """
    import os

    model_dir = os.environ.get("FRONTFINDER_TEST_MODEL_DIR")
    if not model_dir:
        pytest.skip("set FRONTFINDER_TEST_MODEL_DIR to a directory containing _best_loss.keras to run this")

    from frontfinder.config.manifests import BEST_LOSS_MANIFEST
    from frontfinder.inference.engine import KerasPredictor

    weights_path = os.path.join(model_dir, BEST_LOSS_MANIFEST.weights_filename)
    if not os.path.exists(weights_path):
        pytest.skip(f"{weights_path} not found")

    predictor = KerasPredictor(weights_path)
    assert predictor._model is not None
