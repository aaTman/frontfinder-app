"""ECMWF IFS open-data source abstraction + model-input assembly.

Design goal: the assembly logic (mapping a ModelManifest's variables/levels
to fetch calls, deriving theta-e and PV where needed, stacking channels in
manifest order) is pure and unit-testable against a fake data source. The
real network-backed source (`EcmwfOpenDataSource`) is a thin adapter around
the `ecmwf-opendata` client and is not exercised by unit tests -- it needs
an integration/smoke test run on the Proxmox VM where network access and
the real package are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from frontfinder.config.manifests import ModelManifest, VariableSpec
from frontfinder.ingest.derive import equivalent_potential_temperature, potential_vorticity_isobaric, potential_temperature

# Native IFS open-data base variables directly available at pressure levels
# or single-level, keyed by the ERA5-style name used in the model configs.
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
# Variables the model configs ask for that IFS open-data doesn't publish
# directly -- computed in `derive.py` from the direct variables above.
DERIVED_VARIABLES = {"equivalent_potential_temperature", "potential_vorticity"}

# ERA5-style variable name -> IFS open-data GRIB shortname, for the real
# network client (see EcmwfOpenDataSource).
ERA5_NAME_TO_IFS_SHORTNAME = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "surface_pressure": "sp",
    "2m_temperature": "2t",
    "2m_dewpoint_temperature": "2d",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
}


@dataclass(frozen=True)
class IFSCycle:
    """One IFS operational forecast cycle, e.g. 2026-08-19 12Z."""

    date: str  # "YYYY-MM-DD"
    run_hour: int  # one of 0, 6, 12, 18
    step: int = 0  # forecast lead time in hours; 0 == analysis/T+0

    def __post_init__(self) -> None:
        if self.run_hour not in (0, 6, 12, 18):
            raise ValueError(f"run_hour must be one of 0/6/12/18, got {self.run_hour}")
        # raises ValueError if malformed
        datetime.strptime(self.date, "%Y-%m-%d")


# Desired lead-time grid for the "every 6 hours out to 240 hours" forecast
# product (Taylor's call, 2026-08-21): 0, 6, 12, ..., 240 -- 41 steps.
SIX_HOURLY_STEPS_TO_240H: tuple[int, ...] = tuple(range(0, 241, 6))


def available_forecast_steps(run_hour: int) -> tuple[int, ...]:
    """Which forecast lead times (hours) IFS open-data's 0.25deg "oper" HRES
    stream actually publishes for a given cycle run hour.

    2026-08-22 correction (second pass, per Taylor): the prior version of
    this function capped 00Z/12Z at 240h -- live-checking the AWS bucket
    (and the official ecmwf source) confirms 00Z/12Z actually run out to
    360h, 6-hourly from 150h on (confirmed live for 2026-08-20, steps
    150-360h all 200). 06Z/18Z remain capped at 144h -- confirmed live
    across two dates and both the `aws` and `ecmwf` sources that every
    step from 150h-240h 404s for those cycles; per Taylor, this is because
    IFS Cycle 50R1 folded the former `stream=scda` (short cutoff, 144h
    ceiling) into `stream=oper` for 06Z/18Z, rather than 06Z/18Z gaining
    scda's short range under oper -- there is no extended tail for them.
    Real, live-confirmed shape: 00Z/12Z publish 0-144h at 3h steps then
    150-360h at 6h steps (360h total); 06Z/18Z publish 0-144h at 3h steps
    only (144h total). `target_steps_for_cycle` below intersects the
    desired 6-hourly grid against this per-run-hour ceiling rather than
    requesting steps that will 404 -- re-verify live (see
    scripts/probe_ifs_native_pv.py-style direct bucket checks) before
    trusting either boundary again; ECMWF has changed this shape more
    than once already.
    """
    if run_hour in (0, 12):
        return tuple(range(0, 145, 3)) + tuple(range(150, 361, 6))
    if run_hour in (6, 18):
        return tuple(range(0, 145, 3))
    raise ValueError(f"run_hour must be one of 0/6/12/18, got {run_hour}")


def target_steps_for_cycle(
    run_hour: int, desired: tuple[int, ...] = SIX_HOURLY_STEPS_TO_240H
) -> tuple[int, ...]:
    """The subset of `desired` (default: every 6h to 240h) that this cycle's
    run hour actually publishes. For 00Z/12Z this is the full 41-step grid
    (every multiple of 6 up to 240h is a subset of what's published); for
    06Z/18Z it's capped at 144h -- 25 steps (0, 6, ..., 144), never 240h."""
    published = set(available_forecast_steps(run_hour))
    return tuple(s for s in desired if s in published)


class IFSFieldSource(Protocol):
    """Anything that can hand back IFS fields on the global 0.25deg grid."""

    @property
    def lat(self) -> np.ndarray: ...

    @property
    def lon(self) -> np.ndarray: ...

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: IFSCycle) -> np.ndarray: ...

    def fetch_single_level(self, variable: str, cycle: IFSCycle) -> np.ndarray: ...


class FakeIFSFieldSource:
    """Deterministic synthetic field source for tests -- no network."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, seed: int = 0):
        self._lat = lat
        self._lon = lon
        self._rng = np.random.default_rng(seed)
        self._cache: dict[tuple, np.ndarray] = {}

    @property
    def lat(self) -> np.ndarray:
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        return self._lon

    def _grid(self, base: float, spread: float) -> np.ndarray:
        return base + spread * self._rng.standard_normal((len(self._lat), len(self._lon)))

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: IFSCycle) -> np.ndarray:
        key = ("pl", variable, level_hpa, cycle.date, cycle.run_hour)
        if key not in self._cache:
            if variable == "temperature":
                base = 288.0 - 0.05 * (1000 - level_hpa)  # cooler aloft
                self._cache[key] = self._grid(base, 5.0)
            elif variable == "geopotential":
                self._cache[key] = self._grid(9.8 * (44330.0 * (1 - (level_hpa / 1013.25) ** 0.1903)), 50.0)
            elif variable == "specific_humidity":
                self._cache[key] = np.clip(self._grid(0.005, 0.002), 1e-6, 0.03)
            else:
                self._cache[key] = self._grid(0.0, 5.0)
        return self._cache[key]

    def fetch_single_level(self, variable: str, cycle: IFSCycle) -> np.ndarray:
        key = ("sl", variable, cycle.date, cycle.run_hour)
        if key not in self._cache:
            if variable == "surface_pressure":
                self._cache[key] = self._grid(101325.0, 500.0)
            elif "temperature" in variable or "dewpoint" in variable:
                self._cache[key] = self._grid(288.0, 5.0)
            else:
                self._cache[key] = self._grid(0.0, 5.0)
        return self._cache[key]


class EcmwfOpenDataSource:
    """Real network-backed IFSFieldSource, using the `ecmwf-opendata` client
    to pull the global 0.25deg operational IFS grid.

    NOT covered by unit tests -- needs `ecmwf-opendata`, `cfgrib`/`eccodes`,
    and live network access, none of which are available in this sandbox.
    Treat this class as a first-pass implementation to smoke-test on the
    Proxmox VM before it runs unattended: verify each shortname in
    `ERA5_NAME_TO_IFS_SHORTNAME` actually resolves against a real IFS
    open-data request (some fields, e.g. `q` on pressure levels, are only
    published on a subset of levels for the 0.25deg open-data feed -- check
    against https://www.ecmwf.int/en/forecasts/datasets/open-data before
    relying on 300hPa specific humidity being present).

    CONFIRMED via live smoke test (scripts/smoke_test_ecmwf.py, 2026-08-20):
    IFS open-data's "oper" stream is forecast-only and does not serve
    `type="an"` -- a request built that way 404s. `_fetch_grib` always
    requests `type="fc"`, with `step=0` standing in for an analysis field.
    Also confirmed: requesting a single `levelist` value makes cfgrib
    decode `isobaricInhPa` as a scalar coordinate, not an indexable
    dimension -- `fetch_pressure_level` handles both shapes rather than
    assuming `.sel()` always works.

    Defaults to `source="aws"` (the ecmwf-opendata client's built-in AWS
    Open Data replica, https://registry.opendata.aws/ecmwf-forecasts/,
    resolving to the `ecmwf-forecasts` S3 bucket) rather than ECMWF's own
    `data.ecmwf.int` portal, which caps concurrent connections at 500 and
    explicitly recommends the cloud replicas (AWS/Azure/Google, all
    supported by name via this same `source=` argument) for reliability.
    This pipeline's own request volume is trivial either way (a handful of
    fields, 4x/day) -- the reason to prefer AWS isn't our load, it's not
    contending with everyone else's.
    """

    def __init__(
        self,
        cache_dir: str = "/tmp/frontfinder_ifs_cache",
        source: str = "aws",
        maximum_retries: int = 5,
        retry_after: int = 30,
    ):
        import os

        os.makedirs(cache_dir, exist_ok=True)
        self._cache_dir = cache_dir
        self._source = source
        # 2026-09-01 postmortem: the ecmwf-opendata Client's own default
        # (maximum_retries=500, retry_after=120) was built for a client that
        # blocks until the data shows up, not for a systemd oneshot with
        # TimeoutStartSec=3h -- a step that gets a sustained run of S3 "503
        # Slow Down" (observed for ~35+ min straight on a freshly-published
        # step, most likely a thundering-herd of every other open-data
        # consumer hitting the same brand-new key at once, not anything on
        # our end: our own request volume is trivial, per the class
        # docstring) could retry for up to 500*120s = ~16.7h, which the
        # systemd timeout kills mid-retry -- SIGTERM, no partial progress
        # saved beyond whatever steps already landed. Bounded to a few
        # minutes per step instead: a step that's still hot after this
        # budget gets skipped (run_cycle already treats a failed step as
        # "log and move on", same as an unpublished 404), and the next
        # --poll firing (every 5 min) or a later step's own request has a
        # fresh shot rather than this process camping on one throttled key.
        # cli.py's poll loop now resumes a cycle missing steps rather than
        # treating "any output" as done, so nothing is lost by giving up
        # early here.
        self._maximum_retries = maximum_retries
        self._retry_after = retry_after
        self._client = None  # lazy: constructed on first fetch, see _get_client
        self._lat = np.linspace(90.0, -90.0, 721)  # IFS open-data 0.25deg grid, N->S
        self._lon = np.linspace(0.0, 359.75, 1440)
        self._field_cache: dict[tuple, np.ndarray] = {}

    def _get_client(self):
        if self._client is None:
            from ecmwf.opendata import Client  # local import: not a unit-test dependency

            self._client = Client(
                source=self._source,
                maximum_retries=self._maximum_retries,
                retry_after=self._retry_after,
            )
        return self._client

    def is_cycle_available(self, cycle: IFSCycle, probe_param: str = "2t") -> bool:
        """HEAD-checks whether this cycle/step has been published on the
        configured IFS open-data source, without downloading any data.
        Used by the scheduler's `--poll` mode (scheduler/cli.py) to trigger
        a run event-driven off actual availability instead of guessing a
        fixed publish lag -- see the 2026-08-22 postmortem where a run
        fired at exactly 7h and 404'd on every single step because the
        whole cycle was still ~30min from landing.

        Probes a single small single-level field (2m temperature) rather
        than the fields this run actually needs: ECMWF publishes all params
        for a given cycle/step together, so any one field's presence is a
        reliable proxy for the whole step's.

        Reuses the `ecmwf-opendata` client's own URL-building
        (`_get_urls`), the same private method `Client.latest()` itself
        HEAD-checks against -- so this works across whichever `source`
        (aws/azure/google/ecmwf) the pipeline is configured for, rather
        than hardcoding one replica's URL layout.
        """
        client = self._get_client()
        request = dict(
            date=cycle.date.replace("-", ""),
            time=cycle.run_hour,
            step=cycle.step,
            stream="oper",
            type="fc",
            param=probe_param,
        )
        result = client._get_urls(request, use_index=False)
        if not result.urls:
            return False
        return all(
            client._robust(client.session.head)(url, verify=client.verify).status_code == 200
            for url in result.urls
        )

    @property
    def lat(self) -> np.ndarray:
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        return self._lon

    def _fetch_grib(self, param: str, cycle: IFSCycle, levelist: list[int] | None) -> "xr.Dataset":
        import os

        import xarray as xr

        key = (param, tuple(levelist or ()), cycle.date, cycle.run_hour, cycle.step)
        # 2026-08-21: the target filename didn't used to include cycle.step
        # at all, which was a latent cache-collision bug -- harmless while
        # every request was step=0, but the moment the pipeline started
        # fetching multiple lead times per cycle (see
        # scheduler/run_cycle.py's multi-step orchestration), two different
        # steps of the same cycle/param/levelist would silently overwrite
        # the same cache file on disk, so whichever step wasn't fetched
        # first would end up serving the WRONG lead time's data through a
        # stale cache hit rather than actually fetching. Caught while
        # scoping the multi-step feature, before it ever ran live.
        target = os.path.join(
            self._cache_dir,
            f"{cycle.date}_{cycle.run_hour:02d}z_f{cycle.step:03d}_{param}_{'-'.join(map(str, levelist or []))}.grib2",
        )
        if not os.path.exists(target):
            request = dict(
                date=cycle.date.replace("-", ""),
                time=cycle.run_hour,
                step=cycle.step,
                stream="oper",
                # IFS open-data's HRES "oper" stream is forecast-only -- it
                # does not serve type="an" (analysis) at all, confirmed by a
                # live 404 during smoke-testing. step=0 is the closest
                # equivalent to an analysis field and is still requested as
                # type="fc".
                type="fc",
                param=param,
                target=target,
            )
            if levelist:
                request["levelist"] = levelist
            self._get_client().retrieve(**request)
        ds = xr.open_dataset(target, engine="cfgrib")
        # 2026-08-22 fix (Taylor's Seattle-front report): the raw GRIB's
        # native grid starts at longitude 180.0 (confirmed live via
        # eccodes: longitudeOfFirstGridPointInDegrees=180.0, scanning
        # positively), so cfgrib decodes `longitude` as -180..179.75, NOT
        # the 0..359.75 this class's `self._lon` declares. Every
        # fetch_pressure_level/fetch_single_level call used to hand back
        # `da.values` POSITIONALLY -- i.e. column 0 of the array, silently
        # assumed to be lon=0 (matching self._lon), actually held the data
        # for lon=-180/180. That's a silent half-globe (720-column, 180deg)
        # roll between the values every channel was assembled from and the
        # `lon` coordinate the whole pipeline (and the webapp) tags them
        # with -- e.g. a pixel labeled 237.7degE (Seattle) was actually
        # populated from 57.75degE (Kazakhstan). Normalizing to 0..359.75
        # and sorting here makes the returned dataset's `longitude` axis
        # actually match `self._lon` before any `.values` positional read
        # happens downstream.
        if "longitude" in ds.coords:
            ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby("longitude")
        return ds

    def fetch_pressure_level(self, variable: str, level_hpa: int, cycle: IFSCycle) -> np.ndarray:
        key = ("pl", variable, level_hpa, cycle.date, cycle.run_hour, cycle.step)
        if key not in self._field_cache:
            shortname = ERA5_NAME_TO_IFS_SHORTNAME[variable]
            ds = self._fetch_grib(shortname, cycle, levelist=[level_hpa])
            data_var = next(iter(ds.data_vars))
            da = ds[data_var]
            # Requesting a single levelist means the decoded GRIB has exactly
            # one level, so cfgrib gives back `isobaricInhPa` as a 0-d scalar
            # coordinate rather than an indexable dimension -- .sel() has
            # nothing to select over and raises. Confirmed live via smoke
            # test (scripts/smoke_test_ecmwf.py, 2026-08-20): "Could not
            # automatically create PandasIndex for coord 'isobaricInhPa'
            # with 0 dimensions." Handle both shapes defensively rather than
            # assume cfgrib always collapses it.
            if "isobaricInhPa" in da.dims:
                da = da.sel(isobaricInhPa=level_hpa)
            elif "isobaricInhPa" in da.coords:
                actual_level = float(da.coords["isobaricInhPa"].values)
                if actual_level != level_hpa:
                    raise ValueError(
                        f"requested {variable}@{level_hpa}hPa but decoded GRIB has "
                        f"isobaricInhPa={actual_level} -- levelist request may have been ignored"
                    )
            self._field_cache[key] = da.values
        return self._field_cache[key]

    def fetch_single_level(self, variable: str, cycle: IFSCycle) -> np.ndarray:
        key = ("sl", variable, cycle.date, cycle.run_hour, cycle.step)
        if key not in self._field_cache:
            shortname = ERA5_NAME_TO_IFS_SHORTNAME[variable]
            ds = self._fetch_grib(shortname, cycle, levelist=None)
            data_var = next(iter(ds.data_vars))
            self._field_cache[key] = ds[data_var].values
        return self._field_cache[key]


def _resolve_direct_variable_at_level(
    var_name: str, level_hpa: int, source: IFSFieldSource, cycle: IFSCycle
) -> np.ndarray:
    return source.fetch_pressure_level(var_name, level_hpa, cycle)


def _resolve_direct_single_level_variable(
    var_name: str, source: IFSFieldSource, cycle: IFSCycle
) -> np.ndarray:
    return source.fetch_single_level(var_name, cycle)


def _resolve_derived_variable_at_level(
    var_name: str, level_hpa: int, source: IFSFieldSource, cycle: IFSCycle
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
    levels: tuple[int, ...], source: IFSFieldSource, cycle: IFSCycle
) -> dict[int, np.ndarray]:
    """potential_vorticity needs every level's theta/u/v at once for its
    vertical derivative -- resolved together and cached by level rather than
    one level at a time like the other variables, then indexed into as the
    level-major assembly loop reaches each level."""
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
    manifest: ModelManifest, source: IFSFieldSource, cycle: IFSCycle
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

    Returns an array of shape (len(source.lat), len(source.lon), manifest.n_channels),
    with channels ordered exactly as `manifest.channel_names()` describes.
    """
    known_direct = DIRECT_PRESSURE_LEVEL_VARIABLES | DIRECT_SINGLE_LEVEL_VARIABLES
    for var in manifest.variables:
        if var.name not in known_direct and var.name not in DERIVED_VARIABLES:
            raise ValueError(
                f"variable {var.name!r} is neither a known direct IFS field nor a "
                "known derived field -- add a mapping in ecmwf_ifs.py before running"
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
        channels.append(_resolve_direct_single_level_variable(var.name, source, cycle))

    stacked = np.stack(channels, axis=-1)
    expected_shape = (len(source.lat), len(source.lon), manifest.n_channels)
    if stacked.shape != expected_shape:
        raise AssertionError(f"assembled input shape {stacked.shape} != expected {expected_shape}")
    return stacked
