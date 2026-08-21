"""Manual smoke test for EcmwfOpenDataSource -- run this on the Proxmox
container, NOT in CI/pytest. It hits real ECMWF servers and needs
`ecmwf-opendata` + `cfgrib`/`eccodes` + live network, none of which exist in
the sandbox this pipeline was originally built in.

Deliberately staged rather than all-or-nothing: it tries the cheapest,
most-likely-to-work request first (one single-level field) and only moves on
to pricier/riskier requests (multi-level, then the full per-model assembly)
once the previous stage passes. If something fails, you'll know exactly
which layer broke instead of guessing from one big traceback.

Usage:
    cd /srv/frontfinder
    uv run python scripts/smoke_test_ecmwf.py

Known risk areas going in (see EcmwfOpenDataSource's docstring):
  1. `type="an"` (analysis) may not actually be valid for IFS open-data's
     0.25deg pressure-level fields -- open-data is primarily a *forecast*
     product. If stage 2 fails with something like "no data found" or a
     4xx from the ECMWF API, try step 3/type="fc" instead of step 0/type="an"
     (edit CYCLE below) before assuming anything else is wrong.
  2. Not every variable is published at every pressure level in the 0.25deg
     feed -- specific humidity in particular sometimes drops out at the
     uppermost levels. Stage 3 checks this per-level for best_loss's full
     6-level list.
  3. cfgrib needs a fresh `target` filename per distinct request or it will
     silently reuse a stale cached .grib2/.idx file -- if you rerun this
     after changing CYCLE, clear /tmp/frontfinder_ifs_cache first.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np

from frontfinder.config.manifests import BEST_LOSS_MANIFEST  # MODEL_1702_MANIFEST: disabled, see stage4
from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource, assemble_model_input
from frontfinder.scheduler.cli import most_recent_completed_cycle

# Reuses the same cycle-selection logic scheduler/cli.py uses in production,
# so this test is always checking "the cycle the timer would actually fetch
# right now" rather than a date that goes stale the day after you write it.
CYCLE = most_recent_completed_cycle(datetime.now(timezone.utc))


def describe(name: str, arr: np.ndarray) -> None:
    n_nan = int(np.isnan(arr).sum())
    print(
        f"    {name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f} mean={np.nanmean(arr):.3f} "
        f"nan_count={n_nan}"
    )
    if n_nan > 0:
        print(f"    !! {name} has {n_nan} NaN values -- investigate before trusting this field")


def stage1_single_level(source: EcmwfOpenDataSource) -> None:
    print("\n=== stage 1: one single-level field (surface_pressure) ===")
    arr = source.fetch_single_level("surface_pressure", CYCLE)
    describe("surface_pressure", arr)
    assert arr.shape == (721, 1440), f"expected global 0.25deg grid, got {arr.shape}"
    # surface pressure should be roughly 50000-105000 Pa virtually everywhere
    # (lower over high terrain, e.g. Tibetan Plateau/Andes) -- a wildly wrong
    # value here (e.g. still in hPa, or zero-filled) means the shortname
    # mapping or unit handling is off before we even get to pressure levels.
    plausible = (arr > 40000) & (arr < 110000)
    frac_plausible = plausible.mean()
    print(f"    fraction of grid in plausible surface-pressure range: {frac_plausible:.3f}")
    assert frac_plausible > 0.95, "too much of the grid outside a plausible pressure range"
    print("stage 1 PASSED")


def stage2_one_pressure_level(source: EcmwfOpenDataSource) -> None:
    print("\n=== stage 2: one pressure-level field (temperature @ 850hPa) ===")
    arr = source.fetch_pressure_level("temperature", 850, CYCLE)
    describe("temperature_850", arr)
    assert arr.shape == (721, 1440)
    # 850hPa temperature should be broadly 200-320K
    plausible = (arr > 200) & (arr < 320)
    frac_plausible = plausible.mean()
    print(f"    fraction of grid in plausible temperature range: {frac_plausible:.3f}")
    assert frac_plausible > 0.95, "too much of the grid outside a plausible temperature range"
    print("stage 2 PASSED")


def stage3_all_levels_one_variable(source: EcmwfOpenDataSource, levels: list[int]) -> None:
    print(f"\n=== stage 3: specific_humidity at every best_loss level {levels} ===")
    for lvl in levels:
        arr = source.fetch_pressure_level("specific_humidity", lvl, CYCLE)
        describe(f"specific_humidity_{lvl}", arr)
        if np.allclose(arr, 0.0):
            print(f"    !! specific_humidity_{lvl} is all zeros -- likely not published at this level")
    print("stage 3 done (inspect output above for zero-filled/missing levels before trusting this)")


def stage4_full_manifest_assembly(source: EcmwfOpenDataSource) -> None:
    print("\n=== stage 4: full assemble_model_input (active manifests only) ===")
    # model_1702 deliberately excluded: it's disabled pipeline-wide because
    # its trained levels [1000, 950, 900, 850] include 950/900hPa, which
    # IFS open-data's 0.25deg feed doesn't publish (confirmed live,
    # 2026-08-20) -- this would fail identically every run until that gap
    # has a chosen resolution. See frontfinder/scheduler/cli.py.
    for manifest in (BEST_LOSS_MANIFEST,):
        print(f"  assembling {manifest.name} ({manifest.n_channels} channels)...")
        arr = assemble_model_input(manifest, source, CYCLE)
        describe(f"{manifest.name}_input", arr)
        n_bad = int((~np.isfinite(arr)).sum())
        assert n_bad == 0, f"{manifest.name}: {n_bad} non-finite values in assembled input"
        print(f"  {manifest.name} PASSED")
    print("stage 4 PASSED")


def main() -> int:
    print(f"smoke-testing EcmwfOpenDataSource against cycle: {CYCLE}")
    source = EcmwfOpenDataSource(cache_dir="/tmp/frontfinder_ifs_cache")
    try:
        stage1_single_level(source)
        stage2_one_pressure_level(source)
        stage3_all_levels_one_variable(source, [1000, 925, 850, 700, 500, 300])
        stage4_full_manifest_assembly(source)
    except Exception:
        print("\n!!! smoke test FAILED -- see traceback below and the risk areas in this file's docstring")
        raise
    print("\nall stages passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
