"""Derived-variable formulas for fields the models need that IFS open-data
does not archive directly.

Both models were trained on ERA5. `equivalent_potential_temperature` is not
a standard IFS open-data pressure-level param, so it's computed here from
base fields (temperature, specific humidity) that IFS *does* publish.

2026-08-20 correctness fix: this formula used to be a simplified Bolton
(1980) approximation that skipped the LCL-temperature correction term
(`T_L ~= T`). Reading fronts/src/fronts/data/derived.py -- the actual
training-time derivation -- showed the real pipeline computes the FULL
Bolton (1980) eq. 43 with the LCL temperature chain (dewpoint -> LCL temp),
and clamps specific humidity to a 1e-9 kg/kg floor before any log-based step
(ERA5 spectral-truncation artifacts can make q <= 0 in cold, dry upper-level
air, which makes log(e) NaN). This module now mirrors that exactly
(`_R_D`/`_C_PD`/`_L_V`/`_EPSILON` constants included) rather than
approximating it, so serving feeds the model the same theta-e it was
trained on.

`potential_vorticity` is a DIFFERENT kind of gap, not fixed by a formula
change: reading fronts/src/fronts/data/sources.py showed `potential_vorticity`
maps directly to Arraylake ERA5's native `"pv"` short name -- it was fetched
as a raw ERA5 field at training time, never derived at all.
`potential_vorticity_isobaric` below is this repo's own isobaric-PV
approximation (relative vorticity + static stability, centered finite
differences), invented because IFS open-data doesn't obviously publish PV on
pressure levels the way it publishes t/u/v/q -- it has no real relationship
to whatever ERA5's own PV field actually is. This is a bigger, unresolved
science-validation gap than the theta-e one was: see
scripts/probe_ifs_native_pv.py, which checks whether IFS open-data actually
has a native pressure-level PV parameter frontfinder should switch to
fetching directly instead of approximating.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
OMEGA = 7.292115e-5  # Earth's rotation rate, rad/s
G = 9.80665  # m/s^2

# Matches fronts/src/fronts/data/derived.py's constants exactly (names kept
# distinct from this module's existing RD/CP/etc. below since those still
# back potential_vorticity_isobaric's own, separately-approximated formula).
_R_D = 287.05  # dry air gas constant, J kg-1 K-1
_C_PD = 1004.0  # specific heat of dry air at constant pressure, J kg-1 K-1
_L_V = 2.501e6  # latent heat of vaporization at 0 degC, J kg-1
_EPSILON = 0.622  # ratio of molar masses of water vapour to dry air
_MIN_SPECIFIC_HUMIDITY = 1e-9  # kg/kg -- same floor fronts uses, same reason (avoids log(<=0))

RD = 287.05  # J/(kg K), dry air gas constant -- used by potential_vorticity_isobaric only
CP = 1004.0  # J/(kg K), specific heat of dry air at constant pressure
P0 = 1000.0  # hPa reference pressure


def potential_temperature(temperature_k: np.ndarray, pressure_hpa: float) -> np.ndarray:
    """theta = T * (P0/P)^(Rd/Cp)"""
    return temperature_k * (P0 / pressure_hpa) ** (RD / CP)


def mixing_ratio(specific_humidity: np.ndarray) -> np.ndarray:
    """r = q / (1 - q), specific_humidity in kg/kg."""
    return specific_humidity / (1.0 - specific_humidity)


def _clamped_specific_humidity(specific_humidity: np.ndarray) -> np.ndarray:
    """Floor specific humidity at 1e-9 kg/kg, matching
    fronts/data/derived.py's _clamped_specific_humidity exactly -- ERA5 (and
    presumably IFS) upper-level dry air can be <=0 from spectral truncation,
    which makes the log-based vapour-pressure/LCL steps below NaN."""
    return np.clip(specific_humidity, _MIN_SPECIFIC_HUMIDITY, None)


def equivalent_potential_temperature(
    temperature_k: np.ndarray,
    specific_humidity: np.ndarray,
    pressure_hpa: float,
) -> np.ndarray:
    """Equivalent potential temperature via the full Bolton (1980) eq. 43,
    mirroring fronts/src/fronts/data/derived.py's
    _compute_equivalent_potential_temperature exactly (same formula, same
    constants, same specific-humidity floor) so serving feeds the model the
    same theta-e it was trained on.

    Reference: Bolton, D. (1980). Mon. Wea. Rev., 108, 1046-1053.
    """
    p_pa = pressure_hpa * 100.0
    q = _clamped_specific_humidity(specific_humidity)
    r = q / (1.0 - q)
    # actual vapour pressure (hPa) from specific humidity + pressure
    e = (r / (_EPSILON + r)) * pressure_hpa
    log_e = np.log(e / 6.112)
    t_d = 243.5 * log_e / (17.67 - log_e) + 273.15  # dewpoint temperature
    t_l = 1.0 / (1.0 / (t_d - 56.0) + np.log(temperature_k / t_d) / 800.0) + 56.0  # LCL temperature
    theta = temperature_k * (100000.0 / p_pa) ** (_R_D / _C_PD)
    return theta * np.exp((_L_V * r) / (_C_PD * t_l))


def coriolis_parameter(lat_deg: np.ndarray) -> np.ndarray:
    """f = 2*Omega*sin(lat)"""
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat_deg))


def relative_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> np.ndarray:
    """zeta = dv/dx - du/dy on a regular lat/lon grid.

    `u`, `v` are 2D (lat, lon) arrays. `lat_deg` and `lon_deg` are 1D
    coordinate arrays matching those axes. Uses centered finite differences
    with spherical map factors; edges use one-sided differences.
    """
    if u.shape != v.shape:
        raise ValueError(f"u and v shape mismatch: {u.shape} vs {v.shape}")
    if u.shape != (len(lat_deg), len(lon_deg)):
        raise ValueError(
            f"u/v shape {u.shape} does not match (len(lat), len(lon)) = "
            f"({len(lat_deg)}, {len(lon_deg)})"
        )

    lat_rad = np.deg2rad(lat_deg)
    dlat = np.gradient(lat_rad)
    dlon = np.gradient(np.deg2rad(lon_deg))

    dy = EARTH_RADIUS_M * dlat  # meters per grid step, per row
    cos_lat = np.cos(lat_rad)
    cos_lat_safe = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
    dx = EARTH_RADIUS_M * cos_lat_safe[:, None] * dlon[None, :]  # meters per grid step, per (row, col)

    dv_dx = np.gradient(v, axis=1) / dx
    du_dy = np.gradient(u, axis=0) / dy[:, None]
    return dv_dx - du_dy


def potential_vorticity_isobaric(
    u_by_level: dict[int, np.ndarray],
    v_by_level: dict[int, np.ndarray],
    theta_by_level: dict[int, np.ndarray],
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    level_hpa: int,
) -> np.ndarray:
    """Isobaric potential vorticity at `level_hpa` (in PVU, 1e-6 K m^2 kg^-1 s^-1):

        PV = -g * (zeta + f) * (d theta / d p)

    `d theta / d p` is estimated from the nearest available levels above and
    below `level_hpa` in `theta_by_level` (falls back to a one-sided
    difference at the top/bottom of the column).
    """
    levels = sorted(theta_by_level.keys())
    if level_hpa not in levels:
        raise ValueError(f"level {level_hpa} not present in theta_by_level: {levels}")

    idx = levels.index(level_hpa)
    if idx == 0:
        lower, upper = levels[0], levels[1]
    elif idx == len(levels) - 1:
        lower, upper = levels[-2], levels[-1]
    else:
        lower, upper = levels[idx - 1], levels[idx + 1]

    # pressure decreases with altitude; d theta/d p computed with p in Pa
    dtheta = theta_by_level[upper] - theta_by_level[lower]
    dp = (upper - lower) * 100.0  # hPa -> Pa
    dtheta_dp = dtheta / dp

    zeta = relative_vorticity(u_by_level[level_hpa], v_by_level[level_hpa], lat_deg, lon_deg)
    f = coriolis_parameter(lat_deg)[:, None]
    pv_si = -G * (zeta + f) * dtheta_dp
    return pv_si * 1e6  # SI -> PVU
