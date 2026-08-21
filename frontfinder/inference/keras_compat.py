"""Registers the tiny custom Keras classes the trained models were saved
with, so `keras.models.load_model` can deserialize `_best_loss.keras`
without this serving repo depending on the training repo (`fronts`).

Where this comes from: `_best_loss.keras`'s saved config references
`{'module': 'fronts.model', 'class_name': 'SharedTargetModel', ...
'registered_name': 'fronts>SharedTargetModel'}`. Keras 3 looks that
registered_name up in its process-global registry *before* trying to import
the original module -- so anything that runs the same
`@register_keras_serializable(package="fronts")` decorator on a
same-shaped class, in this process, before `load_model()` is called,
satisfies the deserializer. It does not need to be literally the same
class object as the one in the `fronts` repo.

Confirmed safe to vendor rather than import from `fronts` by reading
fronts/src/fronts/model.py and fronts/src/fronts/layers/modules.py
(2026-08-20):
  - `SharedTargetModel` and `TemperatureScaledModel` are the ONLY two
    `register_keras_serializable`-decorated classes anywhere in
    `fronts.model`.
  - Both only override training-time hooks (SharedTargetModel:
    compute_loss/compute_metrics; TemperatureScaledModel: those plus a
    call()/get_config()/from_config() for its temperature wrapper) --
    `KerasPredictor.predict_batch` only ever calls `.predict()`, which
    never touches compute_loss/compute_metrics.
  - The actual UNet3+ architecture (`fronts.layers.modules`) is built
    entirely out of standard `tf.keras.layers` (Conv2D, BatchNormalization,
    etc.) via plain Python functions, not custom `Layer` subclasses -- so
    nothing from that module needs registering here. Deserializing the
    saved graph only needs Keras's own built-in layer classes.

Mirrored, not imported, from fronts/src/fronts/model.py's `SharedTargetModel`
(lines 13-46) and `TemperatureScaledModel` (lines 49-86) as of 2026-08-20. If
those classes' bodies change -- a new `__init__`/`call` override, or a new
`@register_keras_serializable` class gets added to `fronts.model` -- this
file needs to be re-synced by hand, since there is no automated link between
the two repos.
"""

from __future__ import annotations

from typing import Any

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="fronts")
class SharedTargetModel(keras.Model):
    """Deserialization stand-in for fronts.model.SharedTargetModel.

    Training-time hooks are reproduced for completeness, but nothing in the
    serving pipeline calls them -- `.predict()` only exercises the forward
    pass, which is built entirely from standard Keras layers.
    """

    def compute_loss(
        self,
        x: tf.Tensor | None = None,
        y: tf.Tensor | list[tf.Tensor] | None = None,
        y_pred: tf.Tensor | list[tf.Tensor] | None = None,
        sample_weight: tf.Tensor | None = None,
        training: bool = True,
    ) -> tf.Tensor:
        y_in = [y] * len(y_pred) if isinstance(y_pred, (list, tuple)) else y
        return super().compute_loss(x=x, y=y_in, y_pred=y_pred, sample_weight=sample_weight, training=training)

    def compute_metrics(
        self,
        x: tf.Tensor | None = None,
        y: tf.Tensor | list[tf.Tensor] | None = None,
        y_pred: tf.Tensor | list[tf.Tensor] | None = None,
        sample_weight: tf.Tensor | None = None,
    ) -> dict[str, tf.Tensor]:
        y_in = [y] * len(y_pred) if isinstance(y_pred, (list, tuple)) else y
        return super().compute_metrics(x=x, y=y_in, y_pred=y_pred, sample_weight=sample_weight)


@keras.saving.register_keras_serializable(package="fronts")
class TemperatureScaledModel(keras.Model):
    """Deserialization stand-in for fronts.model.TemperatureScaledModel.

    Not referenced by `_best_loss.keras`'s config (only SharedTargetModel
    was) -- vendored anyway in case a calibrated model gets served later,
    since it costs nothing to register up front.
    """

    def __init__(self, logit_model: keras.Model, temperature: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.logit_model = logit_model
        self.temperature = float(temperature)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor | list[tf.Tensor]:
        logits = self.logit_model(x, training=training)
        if isinstance(logits, (list, tuple)):
            return [tf.nn.softmax(logit / self.temperature) for logit in logits]
        return tf.nn.softmax(logits / self.temperature)

    def get_config(self) -> dict[str, Any]:
        return {
            "logit_model": tf.keras.layers.serialize(self.logit_model),
            "temperature": self.temperature,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TemperatureScaledModel":
        config = dict(config)
        logit_model = tf.keras.layers.deserialize(config.pop("logit_model"))
        return cls(logit_model=logit_model, **config)


def ensure_registered() -> None:
    """No-op call whose only purpose is to make importing this module
    explicit at the `KerasPredictor` call site, so the registration-via-
    import-side-effect isn't a silent/implicit dependency someone could
    accidentally optimize away.
    """
