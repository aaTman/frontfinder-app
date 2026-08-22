"""Diagnostic for the "all four served classes are exactly 0.0000 everywhere"
result from smoke_test_full_cycle.py (2026-08-19T18Z cycle, 2026-08-20).

That result alone doesn't say whether the model is genuinely, confidently
predicting "no front" almost everywhere on this globe (plausible -- fronts
are a small fraction of the grid, and the model's out-of-domain outside
CONUS) or whether something upstream (channel order, units, normalization)
is producing degenerate input that saturates to background regardless of
what's actually happening in the atmosphere. smoke_test_full_cycle.py's
existing check (`0 <= min and max <= 1`) can't distinguish these -- 0.0000
passes it either way.

This script runs the SAME assembled input through the SAME model as the
real pipeline, but stops before served-class extraction/stitching so it can
report all 6 raw softmax classes (not just the 4 served ones), confirm
softmax actually sums to 1 per pixel (catches a broken/garbage output before
blaming the data), and report where in [0,1] the *front* classes' mass
actually sits with more precision than 4 decimals.

Usage:
    cd /srv/frontfinder-app
    uv run python scripts/diagnose_best_loss_output.py --model-dir /srv/frontfinder-app/models
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from frontfinder.config.manifests import ALL_CLASSES, THETA_E_UV_Q_MANIFEST
from frontfinder.inference.engine import KerasPredictor
from frontfinder.inference.tiling import generate_tiles, pad_to_multiple
from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource, assemble_model_input
from frontfinder.scheduler.cli import most_recent_completed_cycle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ifs-cache-dir", default="/tmp/frontfinder_ifs_cache")
    parser.add_argument("--ifs-source", default="aws", choices=["aws", "azure", "google", "ecmwf"])
    parser.add_argument("--patch-size", type=int, default=256)
    args = parser.parse_args()

    import os

    weights_path = os.path.join(args.model_dir, THETA_E_UV_Q_MANIFEST.weights_filename)
    predictor = KerasPredictor(weights_path)

    cycle = most_recent_completed_cycle(datetime.now(timezone.utc))
    source = EcmwfOpenDataSource(cache_dir=args.ifs_cache_dir, source=args.ifs_source)

    print(f"cycle: {cycle}")
    print("assembling input...")
    input_grid = assemble_model_input(THETA_E_UV_Q_MANIFEST, source, cycle)
    print(f"input_grid shape: {input_grid.shape}, dtype: {input_grid.dtype}")

    # Per-channel input stats: catches "one channel is garbage/wrong units"
    # before it even reaches the model.
    print(f"\n=== per-channel input stats (first/last 3 of {input_grid.shape[-1]}) ===")
    names = THETA_E_UV_Q_MANIFEST.channel_names()
    for i in list(range(3)) + list(range(len(names) - 3, len(names))):
        ch = input_grid[..., i]
        print(f"    [{i:2d}] {names[i]:45s} min={np.nanmin(ch):.4g} max={np.nanmax(ch):.4g} mean={np.nanmean(ch):.4g}")

    height, width, n_channels = input_grid.shape
    padded_h = max(pad_to_multiple(height, 16), args.patch_size)
    padded_w = max(pad_to_multiple(width, 16), args.patch_size)
    padded = np.zeros((padded_h, padded_w, n_channels), dtype=input_grid.dtype)
    padded[:height, :width, :] = input_grid

    # Just run one full-size tile through predict_batch directly so we see
    # the model's RAW 6-class softmax output, not the served-4-class,
    # stitched-and-blended-across-tiles version smoke_test_full_cycle.py
    # inspects.
    tiles = generate_tiles(padded_h, padded_w, args.patch_size, overlap=0, multiple=16)
    print(f"\nrunning raw predict on {len(tiles)} non-overlapping tiles (overlap=0, for a clean look)...")
    batch = np.stack(
        [padded[t.row_start:t.row_end, t.col_start:t.col_end, :] for t in tiles[:8]], axis=0
    )
    out = predictor.predict_batch(batch)
    print(f"raw output shape: {out.shape}  (expect (n, {args.patch_size}, {args.patch_size}, 6))")

    sums = out.sum(axis=-1)
    print(f"\nsoftmax sum-per-pixel: min={sums.min():.6f} max={sums.max():.6f} (should be ~1.0 everywhere)")

    print("\n=== all 6 raw classes, full precision ===")
    for i, cls in enumerate(ALL_CLASSES):
        ch = out[..., i]
        print(
            f"    {cls:12s} min={ch.min():.8f} max={ch.max():.8f} mean={ch.mean():.8f} "
            f"frac>0.01={float((ch > 0.01).mean()):.4%} frac>0.5={float((ch > 0.5).mean()):.4%}"
        )

    argmax = out.argmax(axis=-1)
    print("\n=== argmax class distribution across sampled tiles ===")
    for i, cls in enumerate(ALL_CLASSES):
        frac = float((argmax == i).mean())
        print(f"    {cls:12s} {frac:.4%} of pixels are argmax'd to this class")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
