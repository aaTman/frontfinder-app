"""Maps a ModelManifest onto fetch calls against any `FieldSource`: derives
theta-e and PV where needed and stacks channels in the trained order. Pure
and unit-testable against a fake source.
"""

from __future__ import annotations

import numpy as np

from frontfinder.config.manifests import ModelManifest
from frontfinder.ingest.base import FieldSource, ForecastCycle, field_shape
from frontfinder.ingest.derive import equivalent_potential_temperature, potential_temperature, potential_vorticity_isobaric

# Base variables every FieldSource serves directly at pressure levels or
# single-level, keyed by the ERA5-style name used in the model configs.
DIRECT_PRESSURE_LEVEL_VARIABLES = {
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
}
DIRECT_SINGLE_LEVEL_VARIABLES = {
    "surface_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
}
# Variables the model configs ask for that sources don't publish directly -- computed in `derive.py` from the direct variables above.
DERIVED_VARIABLES = {"equivalent_potential_temperature", "potential_vorticity"}

# Single-level variables that are physically invariant (never change cycle to
# cycle) and therefore have no real-time IFS open-data field at all -- IFS's
# "oper" fc stream only publishes actual forecast fields (confirmed against
# ECMWF's published oper single-level param table: no surface
# geopotential/orography among them). Sourced once from a static local file
# instead of a per-cycle fetch -- see `EcmwfOpenDataSource.fetch_static` and
# `STATIC_FIELD_FILENAMES` below.
STATIC_SINGLE_LEVEL_VARIABLES = {"geopotential_at_surface"}


def _resolve_direct_variable_at_level(
    var_name: str, level_hpa: int, source: FieldSource, cycle: ForecastCycle
) -> np.ndarray:
    return source.fetch_pressure_level(var_name, level_hpa, cycle)


def _resolve_direct_single_level_variable(
    var_name: str, source: FieldSource, cycle: ForecastCycle
) -> np.ndarray:
    return source.fetch_single_level(var_name, cycle)


def _resolve_derived_variable_at_level(
    var_name: str, level_hpa: int, source: FieldSource, cycle: ForecastCycle
) -> np.ndarray:
    # potential_vorticity is NOT handled here -- it needs every level's
    # theta/u/v at once for its vertical derivative, so it's resolved
    # separately via _resolve_potential_vorticity_all_levels and never
    # routed through this per-level function (see assemble_model_input).
    if var_name == "equivalent_potential_temperature":
        t = source.fetch_pressure_level("temperature", level_hpa, cycle)
        q = source.fetch_pressure_level("specific_humidity", level_hpa, cycle)
        return equivalent_potential_temperature(t, q, pressure_hpa=level_hpa)

    raise ValueError(f"no per-level derivation defined for {var_name!r}")


def _resolve_potential_vorticity_all_levels(
    levels: tuple[int, ...], source: FieldSource, cycle: ForecastCycle
) -> dict[int, np.ndarray]:
    """potential_vorticity needs every level's theta/u/v at once for its
    vertical derivative -- resolved together and cached by level rather than
    one level at a time like the other variables, then indexed into as the
    level-major assembly loop reaches each level."""
    if np.ndim(source.lat) != 1:
        raise ValueError("potential_vorticity needs a rectilinear lat/lon grid")
    theta_by_level = {
        lvl: potential_temperature(source.fetch_pressure_level("temperature", lvl, cycle), lvl) for lvl in levels
    }
    u_by_level = {lvl: source.fetch_pressure_level("u_component_of_wind", lvl, cycle) for lvl in levels}
    v_by_level = {lvl: source.fetch_pressure_level("v_component_of_wind", lvl, cycle) for lvl in levels}
    return {
        lvl: potential_vorticity_isobaric(u_by_level, v_by_level, theta_by_level, source.lat, source.lon, level_hpa=lvl)
        for lvl in levels
    }


def assemble_model_input(
    manifest: ModelManifest, source: FieldSource, cycle: ForecastCycle
) -> np.ndarray:
    """Fetch + derive + stack every channel `manifest` needs, in the model's
    actual trained channel order.

    2026-08-20 postmortem: this function used to stack channels
    VARIABLE-major (every level of variable[0], then every level of
    variable[1], ...) -- matching how ModelManifest.variables happens to be
    grouped, but NOT how fronts/src/fronts/data/inputs.py's
    inputs_ds_to_dataarray() actually built the training input tensor. That
    function stacks pressure-level channels LEVEL-major (every variable at
    level[0], then every variable at level[1], ...) via
    `ds[level_vars].to_array(...).stack(channel=("level","variable"))`
    (level listed before variable in the stack call, so level varies
    slowest). The mismatch meant the model's baked-in per-channel
    normalization (`input_normalization` Rescaling layer -- see
    scripts/diagnose_model_saturation.py) was applied to the wrong channels:
    e.g. a specific-humidity channel's [~0, ~0.02]-range min/max stats
    getting applied to a 250-390K temperature-like channel, driving the
    rescaled value to thousands of standard-deviations off. That saturated
    every downstream layer into predicting background with ~1.0 confidence
    regardless of the real input -- confirmed live: `diagnose_best_loss_output.py`
    showed background at a bit-identical 0.99999994 across every sampled
    pixel, and `diagnose_model_saturation.py`'s per-channel comparison showed
    rescaled ranges like 8755..13310 or -2091..3206 instead of the expected
    ~0..1 for a minmax Rescaling layer. Fixed by stacking in the same
    level-major order `ModelManifest.channel_names()` now documents.

    Returns an array of shape (rows, cols, manifest.n_channels) on the source's grid,
    with channels ordered exactly as `manifest.channel_names()` describes.
    """
    known_direct = DIRECT_PRESSURE_LEVEL_VARIABLES | DIRECT_SINGLE_LEVEL_VARIABLES | STATIC_SINGLE_LEVEL_VARIABLES
    for var in manifest.variables:
        if var.name not in known_direct and var.name not in DERIVED_VARIABLES:
            raise ValueError(
                f"variable {var.name!r} is neither a known direct field nor a "
                "known derived field -- add a mapping in ingest/assembly.py before running"
            )

    pressure_vars = manifest.pressure_level_variables()
    channels: list[np.ndarray] = []

    if pressure_vars:
        levels = manifest.shared_levels
        # potential_vorticity's cross-level dependency means it has to be
        # resolved for all its levels up front rather than one level at a
        # time inside the level-major loop below.
        pv_by_level: dict[int, np.ndarray] | None = None
        if any(v.name == "potential_vorticity" for v in pressure_vars):
            pv_by_level = _resolve_potential_vorticity_all_levels(levels, source, cycle)

        for level in levels:
            for var in pressure_vars:
                if var.name == "potential_vorticity":
                    channels.append(pv_by_level[level])
                elif var.name == "equivalent_potential_temperature":
                    channels.append(_resolve_derived_variable_at_level(var.name, level, source, cycle))
                else:
                    channels.append(_resolve_direct_variable_at_level(var.name, level, source, cycle))

    for var in manifest.single_level_variables():
        if var.name in STATIC_SINGLE_LEVEL_VARIABLES:
            channels.append(source.fetch_static(var.name))
        else:
            channels.append(_resolve_direct_single_level_variable(var.name, source, cycle))

    stacked = np.stack(channels, axis=-1)
    expected_shape = (*field_shape(source), manifest.n_channels)
    if stacked.shape != expected_shape:
        raise AssertionError(f"assembled input shape {stacked.shape} != expected {expected_shape}")
    return stacked
