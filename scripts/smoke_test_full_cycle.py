"""Manual end-to-end smoke test: real IFS data -> real _best_loss.keras
inference -> real zarr pyramid write. Run this on the Proxmox container,
NOT in CI/pytest -- it needs live network, the real weight file, and
TensorFlow actually loaded.

This is the test that scripts/smoke_test_ecmwf.py deliberately stopped
short of: that one proved the *input assembly* is correct against live
data; this one proves the *whole pipeline* works and, just as importantly,
tells you how long CPU-only tiled inference actually takes and how much
RAM it actually uses on mandelhub -- both of which you want to know before
trusting the systemd timer to run this unattended, not find out from a
3am failure.

Usage:
    cd /srv/frontfinder
    uv run python scripts/smoke_test_full_cycle.py --model-dir /srv/frontfinder/models

Writes into --output-root (default: a throwaway /tmp dir, NOT
/srv/frontfinder/output) so a smoke test never gets mistaken for a real
published run by the webapp's latest.json lookup.
"""

from __future__ import annotations

import argparse
import logging
import os
import resource
import sys
import time

import numpy as np
import xarray as xr

from frontfinder.config.manifests import BEST_LOSS_MANIFEST
from frontfinder.inference.engine import KerasPredictor
from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource
from frontfinder.scheduler.cli import most_recent_completed_cycle
from frontfinder.scheduler.run_cycle import ModelRunConfig, run_one_model
from datetime import datetime, timezone


def peak_rss_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS -- this runs on the Linux
    # container, so KB -> MB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-cycle smoke test: real data, real model, real zarr write.")
    parser.add_argument("--model-dir", required=True, help="directory containing _best_loss.keras")
    parser.add_argument("--output-root", default="/tmp/frontfinder_smoke_output")
    parser.add_argument("--ifs-cache-dir", default="/tmp/frontfinder_ifs_cache")
    parser.add_argument("--ifs-source", default="aws", choices=["aws", "azure", "google", "ecmwf"])
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    weights_path = os.path.join(args.model_dir, BEST_LOSS_MANIFEST.weights_filename)
    if not os.path.exists(weights_path):
        print(f"!!! weights file not found: {weights_path}")
        print("    make sure _best_loss.keras is actually in --model-dir before running this")
        return 1

    cycle = most_recent_completed_cycle(datetime.now(timezone.utc))
    print(f"cycle: {cycle}")
    print(f"weights: {weights_path}")
    print(f"output root: {args.output_root} (throwaway -- separate from the real /srv/frontfinder/output)")

    t_start = time.monotonic()

    print("\n=== loading model ===")
    t0 = time.monotonic()
    predictor = KerasPredictor(weights_path)
    print(f"    model load: {time.monotonic() - t0:.1f}s, peak RSS so far: {peak_rss_mb():.0f} MB")

    run_config = ModelRunConfig(
        manifest=BEST_LOSS_MANIFEST,
        predictor=predictor,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    source = EcmwfOpenDataSource(cache_dir=args.ifs_cache_dir, source=args.ifs_source)

    print("\n=== running full cycle (ingest -> tiled inference -> zarr write) ===")
    t0 = time.monotonic()
    store_path = run_one_model(run_config, source, cycle, args.output_root)
    t_run = time.monotonic() - t0
    print(f"    full run: {t_run:.1f}s, peak RSS: {peak_rss_mb():.0f} MB")
    print(f"    wrote: {store_path}")

    print("\n=== verifying the written pyramid is actually readable ===")
    dt = xr.open_datatree(store_path, engine="zarr")
    lvl0 = dt["0"].to_dataset()
    print(f"    level 0 variables: {list(lvl0.data_vars)}")
    print(f"    level 0 dims: {dict(lvl0.sizes)}")
    for cls in BEST_LOSS_MANIFEST.served_classes:
        arr = lvl0[cls].values
        n_nan = int(np.isnan(arr).sum())
        print(
            f"    {cls}: min={np.nanmin(arr):.4f} max={np.nanmax(arr):.4f} "
            f"mean={np.nanmean(arr):.4f} nan_count={n_nan}"
        )
        assert n_nan == 0, f"{cls} has NaNs in the written pyramid"
        assert 0.0 <= np.nanmin(arr) and np.nanmax(arr) <= 1.0, f"{cls} outside [0,1] -- not a valid probability"

    n_levels = len([g for g in dt.groups if g not in ("", "/")])
    print(f"    pyramid levels written: {n_levels}")

    total = time.monotonic() - t_start
    print(f"\nall checks passed. total wall time: {total:.1f}s ({total / 60:.1f} min), peak RSS: {peak_rss_mb():.0f} MB")
    print(
        "\nCompare peak RSS against your container's memory allocation -- if this is uncomfortably close "
        "to the limit, that's the moment to revisit the 32GB RAM upgrade or shrink --patch-size."
    )
    print(
        "Compare total wall time against the ~6h window between synoptic cycles -- if this took anywhere "
        "close to that, the timer needs a hard look before you trust it to keep up."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
