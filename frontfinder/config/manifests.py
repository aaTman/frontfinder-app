"""Model manifests: declares exactly which variables/pressure levels each
frontfinder Keras model expects as input, in channel order.

Sourced from Taylor's `fronts` repo configs:
  - theta-e_uv_q: configs/sooner_ablations.yaml (originally n_channels: 30,
                 levels confirmed by Taylor as [1000, 925, 850, 700, 500,
                 300]). potential_vorticity dropped 2026-08-20 -- no native
                 IFS open-data PV field exists to fetch (confirmed live,
                 scripts/probe_ifs_native_pv.py), and the isobaric-PV
                 approximation in ingest/derive.py had no verified
                 relationship to the ERA5-native PV field the model was
                 actually trained on. equivalent_potential_temperature
                 stays -- Taylor confirmed (2026-08-20) the retrained
                 _best_loss.keras still uses it, just not PV. Now 24
                 channels (4 vars x 6 levels): theta-e/u/v/specific_humidity.
                 If Taylor's retrained model changed anything else
                 (different levels, a variable not listed here), Keras will
                 raise a shape mismatch the moment KerasPredictor.predict_batch
                 runs, rather than silently misaligning channels again --
                 scripts/diagnose_model_saturation.py exercises this
                 directly and is the fastest way to surface that.
  - model_1702:  configs/model_1702/generate_conus.yaml (pressure_levels:
                 [1000, 950, 900, 850])

Both source configs list a CONUS bounding box (`coordinates:
[25.0, 56.75, 228.0, 299.75]`) as the training/eval domain. Taylor has
confirmed frontfinder should nonetheless run true global inference on the
IFS grid -- these models have not been validated outside CONUS, so the
serving pipeline should be treated as producing a global *extrapolation*,
not a validated global product, until accuracy outside CONUS is checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


# Front classes as predicted by the model output layer (softmax over 6
# classes, matching `model_config.n_classes: 6` in sooner_ablations.yaml:
# background + 5 front types). Class order follows the AIES FrontFinder
# convention used across both configs.
ALL_CLASSES: tuple[str, ...] = (
    "background",
    "cold",
    "warm",
    "stationary",
    "occluded",
    "dryline",
)

# Per Taylor: the model predicts drylines but frontfinder will not serve
# them. "background" is never served either.
SERVED_CLASSES: tuple[str, ...] = ("cold", "warm", "stationary", "occluded")


@dataclass(frozen=True)
class VariableSpec:
    """One input variable. `levels=None` means a single-level/surface field."""

    name: str
    levels: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.levels is not None and len(self.levels) == 0:
            raise ValueError(f"{self.name}: levels must be non-empty or None, not ()")
        if self.levels is not None and len(set(self.levels)) != len(self.levels):
            raise ValueError(f"{self.name}: duplicate pressure levels in {self.levels}")

    @property
    def n_channels(self) -> int:
        return len(self.levels) if self.levels is not None else 1

    def channel_names(self) -> list[str]:
        if self.levels is None:
            return [self.name]
        return [f"{self.name}_{lvl}" for lvl in self.levels]


@dataclass(frozen=True)
class ModelManifest:
    name: str
    weights_filename: str
    variables: tuple[VariableSpec, ...]
    patch_multiple: int = 16
    all_classes: tuple[str, ...] = field(default=ALL_CLASSES)
    served_classes: tuple[str, ...] = field(default=SERVED_CLASSES)

    def __post_init__(self) -> None:
        if len(self.variables) == 0:
            raise ValueError(f"{self.name}: manifest has no variables")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate variable names in manifest: {names}")
        missing = set(self.served_classes) - set(self.all_classes)
        if missing:
            raise ValueError(f"{self.name}: served_classes not in all_classes: {missing}")
        # fronts/src/fronts/data/inputs.py's inputs_ds_to_dataarray() -- the
        # function that actually built the training input tensor -- stacks
        # every pressure-level variable against ONE shared `level` dim
        # (`ds[level_vars].to_array(...).stack(channel=("level","variable"))`).
        # That only produces a well-defined channel order if every
        # pressure-level variable in the manifest was trained against the
        # exact same level list; if they differed, level-major stacking
        # would be ambiguous about whose levels win.
        level_tuples = {v.levels for v in self.variables if v.levels is not None}
        if len(level_tuples) > 1:
            raise ValueError(
                f"{self.name}: pressure-level variables must share one levels tuple "
                f"for level-major channel ordering to be well-defined, got {level_tuples}"
            )

    @property
    def n_channels(self) -> int:
        return sum(v.n_channels for v in self.variables)

    def pressure_level_variables(self) -> list[VariableSpec]:
        """Pressure-level variables, in the order they appear in `variables`."""
        return [v for v in self.variables if v.levels is not None]

    def single_level_variables(self) -> list[VariableSpec]:
        """Single-level variables, in the order they appear in `variables`."""
        return [v for v in self.variables if v.levels is None]

    @property
    def shared_levels(self) -> Optional[tuple[int, ...]]:
        """The one levels tuple shared by every pressure-level variable, or
        None if the manifest has no pressure-level variables."""
        pressure_vars = self.pressure_level_variables()
        return pressure_vars[0].levels if pressure_vars else None

    def channel_names(self) -> list[str]:
        """Channel names in the model's actual trained input order.

        Matches fronts/src/fronts/data/inputs.py's inputs_ds_to_dataarray():
        pressure-level channels come first, ordered level-outer/variable-inner
        (every variable at level[0], then every variable at level[1], ...),
        followed by single-level channels in the order given in `variables`.
        This is NOT simply each VariableSpec's own levels grouped together --
        see the 2026-08-20 postmortem in ecmwf_ifs.py's assemble_model_input
        docstring for why that variable-major grouping was a real bug (it fed
        the model's baked-in per-channel normalization stats against the
        wrong channels, silently producing garbage predictions).
        """
        pressure_vars = self.pressure_level_variables()
        names: list[str] = []
        if pressure_vars:
            for level in pressure_vars[0].levels:
                for v in pressure_vars:
                    names.append(f"{v.name}_{level}")
        for v in self.single_level_variables():
            names.append(v.name)
        return names

    def served_class_indices(self) -> list[int]:
        return [self.all_classes.index(c) for c in self.served_classes]


THETA_E_UV_Q_MANIFEST = ModelManifest(
    name="theta-e_uv_q",
    weights_filename="_best_loss.keras",
    variables=(
        VariableSpec("equivalent_potential_temperature", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("u_component_of_wind", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("v_component_of_wind", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("specific_humidity", levels=(1000, 925, 850, 700, 500, 300)),
        # potential_vorticity removed 2026-08-20 -- see module docstring.
    ),
)

MODEL_1702_MANIFEST = ModelManifest(
    name="model_1702",
    weights_filename="model_1702.h5",
    variables=(
        VariableSpec("geopotential", levels=(1000, 950, 900, 850)),
        VariableSpec("temperature", levels=(1000, 950, 900, 850)),
        VariableSpec("u_component_of_wind", levels=(1000, 950, 900, 850)),
        VariableSpec("v_component_of_wind", levels=(1000, 950, 900, 850)),
        VariableSpec("specific_humidity", levels=(1000, 950, 900, 850)),
        VariableSpec("surface_pressure", levels=None),
        VariableSpec("2m_temperature", levels=None),
        VariableSpec("2m_dewpoint_temperature", levels=None),
        VariableSpec("10m_u_component_of_wind", levels=None),
        VariableSpec("10m_v_component_of_wind", levels=None),
    ),
)

MANIFESTS: dict[str, ModelManifest] = {
    THETA_E_UV_Q_MANIFEST.name: THETA_E_UV_Q_MANIFEST,
    MODEL_1702_MANIFEST.name: MODEL_1702_MANIFEST,
}


def get_manifest(model_name: str) -> ModelManifest:
    try:
        return MANIFESTS[model_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model {model_name!r}; known models: {sorted(MANIFESTS)}"
        ) from exc
