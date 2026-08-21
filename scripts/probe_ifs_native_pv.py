"""Probe: does IFS open-data publish potential vorticity ("pv") as a native
pressure-level parameter, the same way it publishes t/u/v/q?

Why this matters (2026-08-20 finding, prompted by cross-checking serving's
input assembly against fronts/src/fronts/data/sources.py): `potential_vorticity`
in the training config maps directly to Arraylake ERA5's native `"pv"`
variable -- it was fetched as a raw ERA5 field, never derived from anything.
frontfinder's own `potential_vorticity_isobaric` (ingest/derive.py) is a
hand-rolled isobaric-PV approximation (relative vorticity + static
stability via centered finite differences) invented because nobody had
checked whether IFS open-data has a real PV field to fetch instead. That
approximation has no verified relationship to ERA5's actual PV -- it's a
bigger, more open science-validation gap than the equivalent_potential_temperature
mismatch was (that one was fixed by matching the real formula; there's no
formula to match here if IFS just doesn't have this field, or has it on a
different level set than t/u/v/q).

This script checks the live IFS open-data index the same way pressure
levels were confirmed earlier (scripts/smoke_test_ecmwf.py's stage 3 /
the earlier live-index level check): request `pv` on a pressure level and
see whether it 404s or comes back with real data, on the same level set
`best_loss` uses.

Usage:
    cd /srv/frontfinder
    uv run python scripts/probe_ifs_native_pv.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np

from frontfinder.ingest.ecmwf_ifs import IFSCycle
from frontfinder.scheduler.cli import most_recent_completed_cycle

BEST_LOSS_LEVELS = [1000, 925, 850, 700, 500, 300]
CANDIDATE_SHORTNAMES = ["pv"]  # ECMWF's standard GRIB shortName for potential vorticity


def main() -> int:
    from ecmwf.opendata import Client
    import os
    import xarray as xr

    cycle = most_recent_completed_cycle(datetime.now(timezone.utc))
    cache_dir = "/tmp/frontfinder_ifs_cache"
    os.makedirs(cache_dir, exist_ok=True)
    client = Client(source="aws")

    print(f"probing IFS open-data for a native PV pressure-level parameter, cycle: {cycle}")
    found_any = False
    for shortname in CANDIDATE_SHORTNAMES:
        for level in BEST_LOSS_LEVELS:
            target = os.path.join(cache_dir, f"probe_pv_{shortname}_{level}.grib2")
            try:
                if not os.path.exists(target):
                    client.retrieve(
                        date=cycle.date.replace("-", ""),
                        time=cycle.run_hour,
                        step=cycle.step,
                        stream="oper",
                        type="fc",
                        param=shortname,
                        levelist=[level],
                        target=target,
                    )
                ds = xr.open_dataset(target, engine="cfgrib")
                data_var = next(iter(ds.data_vars))
                arr = ds[data_var].values
                print(
                    f"    param={shortname!r} level={level}hPa: FOUND, shape={arr.shape} "
                    f"min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g}"
                )
                found_any = True
            except Exception as exc:
                print(f"    param={shortname!r} level={level}hPa: NOT AVAILABLE ({type(exc).__name__}: {exc})")

    print()
    if found_any:
        print(
            "IFS open-data DOES publish a native PV-like parameter on at least one of best_loss's "
            "levels. Next step: switch ecmwf_ifs.py to fetch it directly (add 'potential_vorticity' "
            "to DIRECT_PRESSURE_LEVEL_VARIABLES + ERA5_NAME_TO_IFS_SHORTNAME, drop the "
            "potential_vorticity_isobaric approximation) instead of approximating it -- this would "
            "match training exactly instead of guessing at it."
        )
    else:
        print(
            "IFS open-data does NOT appear to publish a native PV pressure-level parameter frontfinder "
            "can fetch directly. potential_vorticity_isobaric's approximation stays as the only option -- "
            "worth flagging to Taylor as an open science-validation risk (unlike the theta-e formula, "
            "there's no 'real formula' to match here without either a different data source for PV or "
            "accepting the approximation's error)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
