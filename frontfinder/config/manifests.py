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

# Drylines are served as of 2026-10-02 (previously excluded per Taylor).
# "background" is never served.
SERVED_CLASSES: tuple[str, ...] = ("cold", "warm", "stationary", "occluded", "dryline")


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
    # True for a model trained on volume-shaped input (batch, H, W, n_levels,
    # n_variables) via Conv3D, keeping the level axis intact instead of
    # flattening level*variable into one channel dim -- e.g.
    # spix_wce_loss_modified_dilation.keras (see fronts'
    # configs/sooner_train_3d.yaml, `volume_inputs: true`). `variables` is
    # still declared the normal (flat) way; `inference.engine.
    # VolumetricKerasPredictor` is what actually reshapes the assembled flat
    # (H, W, n_channels) grid into (H, W, n_levels, n_variables) right before
    # calling the model, so every other stage (tiling, hemisphere flip,
    # longitude padding, zarr writing) never needs to know the difference.
    volumetric: bool = False

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

# Trained order confirmed against Taylor's `fronts` repo config
# (configs/sooner_train_3d.yaml, feat/2.0.0 branch, run_name
# "spix_wce_loss_modified_dilation_3d_forming_dissipating_and_troughs"):
# same 4 pressure-level vars/levels as theta-e_uv_q, plus 4 single-level
# fields the training pipeline broadcasts as constant columns across every
# level to fill the (level, variable) volume -- see `variables:` and
# `volume_inputs: true` in that yaml. Verified against the saved
# .keras file's own baked-in Rescaling (normalization) layer: 4 of its 8
# per-(level, variable) scale/offset columns vary by level (the pressure
# vars, in this order) and the other 4 are constant across all 6 levels
# (the broadcast single-level vars, in this order).
#
# geopotential_at_surface (terrain) is NOT available as a native IFS
# open-data field -- it's invariant/static, not part of the real-time "oper"
# forecast stream (confirmed against ECMWF's published oper param table: no
# surface geopotential/orography among its single-level params). Per
# Taylor's call, this is instead sourced once from the public ARCO-ERA5
# zarr archive (gs://gcp-public-data-arco-era5, `geopotential_at_surface`,
# same 0.25deg/721x1440 grid as IFS open-data) and cached locally as a
# static file -- see ecmwf_ifs.py's STATIC_SINGLE_LEVEL_VARIABLES and
# EcmwfOpenDataSource.fetch_static.
SPIX_WCE_DILATION_MANIFEST = ModelManifest(
    name="spix_wce_loss_modified_dilation",
    weights_filename="spix_wce_loss_modified_dilation.keras",
    variables=(
        VariableSpec("equivalent_potential_temperature", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("u_component_of_wind", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("v_component_of_wind", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("specific_humidity", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("10m_u_component_of_wind", levels=None),
        VariableSpec("10m_v_component_of_wind", levels=None),
        VariableSpec("2m_dewpoint_temperature", levels=None),
        VariableSpec("geopotential_at_surface", levels=None),
    ),
    volumetric=True,
)

# Output class order of nbs_square_dilation_trof_tt_no_inst.keras, read off
# the saved model's own compile_config: its PerClassContingencyMetric entries
# carry each class's `class_index` (hss_CF=1, hss_WF=2, hss_SF=3, hss_OF=4,
# hss_DL=5, hss_TROF=6, hss_TT=7, hss_INST=8; index 0 is background).
# Dryline sits at index 5, same as ALL_CLASSES above.
TROF_TT_CLASSES: tuple[str, ...] = (
    "background",
    "cold",
    "warm",
    "stationary",
    "occluded",
    "dryline",
    "trough",
    "tropical_trough",
    "instability",
)

# Instability lines ("no_inst" in the filename) are not served.
TROF_TT_SERVED_CLASSES: tuple[str, ...] = (
    "cold",
    "warm",
    "stationary",
    "occluded",
    "dryline",
    "trough",
    "tropical_trough",
)

# Added 2026-10-08. Same 8 inputs, in the same order, as
# SPIX_WCE_DILATION_MANIFEST, but a 2D UNet3+ (flat 28-channel input, not a
# Conv3D volume). Verified against the .keras file's baked-in Rescaling
# layer: 6 groups of 4 level-varying scale/offset columns (theta-e, u, v, q;
# theta-e's implied minimum rises 214 K -> 283 K from 1000 to 300 hPa), then
# 4 single-level columns whose implied ranges match 10m u, 10m v, 2m
# dewpoint (~206-304 K) and surface geopotential (~-900 to ~34000 m2/s2).
NBS_SQUARE_DILATION_TROF_TT_MANIFEST = ModelManifest(
    name="nbs_square_dilation_trof_tt_no_inst",
    weights_filename="nbs_square_dilation_trof_tt_no_inst.keras",
    variables=SPIX_WCE_DILATION_MANIFEST.variables,
    all_classes=TROF_TT_CLASSES,
    served_classes=TROF_TT_SERVED_CLASSES,
)

MANIFESTS: dict[str, ModelManifest] = {
    THETA_E_UV_Q_MANIFEST.name: THETA_E_UV_Q_MANIFEST,
    MODEL_1702_MANIFEST.name: MODEL_1702_MANIFEST,
    SPIX_WCE_DILATION_MANIFEST.name: SPIX_WCE_DILATION_MANIFEST,
    NBS_SQUARE_DILATION_TROF_TT_MANIFEST.name: NBS_SQUARE_DILATION_TROF_TT_MANIFEST,
}


def get_manifest(model_name: str) -> ModelManifest:
    try:
        return MANIFESTS[model_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model {model_name!r}; known models: {sorted(MANIFESTS)}"
        ) from exc
