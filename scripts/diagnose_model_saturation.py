"""Diagnostic #2: diagnose_best_loss_output.py proved the model's output is
suspicious in a specific way -- not just "mostly background" but *bit-
identical* background=0.99999994 across every pixel of 8 different 256x256
tiles sampled from different parts of the globe. Real atmospheric input
varies enough that a real conv net would never produce spatially-constant
output like that. Something is saturating the model regardless of input.

This script isolates where:
  1. Feeds the model a genuinely different input (all-zeros vs the real
     assembled grid) and checks whether the output actually changes AT ALL.
     If not, the model is provably ignoring its input entirely -- points at
     something structural (bad weight load, or every conv layer saturated).
  2. Inspects the baked-in `input_normalization` Rescaling layer's per-
     channel scale/offset against the real input's actual per-channel
     range -- if the training-time stats and the live IFS data are in
     wildly different units/ranges, the rescaled values could be pushed
     deep into a saturating region of the first activation.
  3. Spot-checks a handful of the model's own weight arrays for all-zero or
     NaN kernels, which would indicate a weight-loading problem rather than
     a normalization mismatch.

Usage:
    cd /srv/frontfinder
    uv run python scripts/diagnose_model_saturation.py --model-dir /srv/frontfinder/models
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np

from frontfinder.config.manifests import BEST_LOSS_MANIFEST
from frontfinder.inference.engine import KerasPredictor
from frontfinder.ingest.ecmwf_ifs import EcmwfOpenDataSource, assemble_model_input
from frontfinder.scheduler.cli import most_recent_completed_cycle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ifs-cache-dir", default="/tmp/frontfinder_ifs_cache")
    parser.add_argument("--ifs-source", default="aws", choices=["aws", "azure", "google", "ecmwf"])
    parser.add_argument("--patch-size", type=int, default=256)
    args = parser.parse_args()

    weights_path = os.path.join(args.model_dir, BEST_LOSS_MANIFEST.weights_filename)
    predictor = KerasPredictor(weights_path)
    model = predictor._model

    print("=== 1. does the model's output change at all with the input? ===")
    p = args.patch_size
    n = BEST_LOSS_MANIFEST.n_channels
    zeros_input = np.zeros((1, p, p, n), dtype=np.float32)
    random_input = np.random.default_rng(0).normal(size=(1, p, p, n)).astype(np.float32) * 50 + 100
    out_zeros = predictor.predict_batch(zeros_input)
    out_random = predictor.predict_batch(random_input)
    print(f"    all-zero input  -> background mean: {out_zeros[..., 0].mean():.8f}")
    print(f"    random input    -> background mean: {out_random[..., 0].mean():.8f}")
    same = np.allclose(out_zeros, out_random, atol=1e-6)
    print(f"    outputs identical regardless of input: {same}")
    if same:
        print("    !! model output does NOT depend on its input at all -- this is a structural")
        print("       problem (weight loading, or every layer saturated), not a normalization")
        print("       range mismatch that only shows up on real data.")

    print("\n=== 2. input_normalization layer: baked-in stats vs real IFS data range ===")
    norm_layer = None
    for layer in model.layers:
        if "normali" in layer.name.lower() or "rescal" in layer.name.lower():
            norm_layer = layer
            break
    if norm_layer is None:
        print("    could not find a layer with 'normali'/'rescal' in its name -- listing all layer names:")
        for layer in model.layers[:20]:
            print(f"      {layer.name}  ({type(layer).__name__})")
    else:
        print(f"    found layer: {norm_layer.name} ({type(norm_layer).__name__})")
        # Per fronts/src/fronts/model.py's build() (the "minmax" branch,
        # ~line 462): this is a plain tf.keras.layers.Rescaling built with
        # scale=1/(stat_b-stat_a), offset=-stat_a*scale, baked in as
        # constructor args -- NOT trainable weights (get_weights() is empty
        # for it; that was the bug in this script's first run). Keras stores
        # those constructor args as instance attributes on the layer object
        # itself (`.scale` / `.offset`), which is the same thing
        # get_config() would return but read directly off the live layer as
        # requested, rather than through the config round-trip.
        raw_scale = norm_layer.scale
        raw_offset = norm_layer.offset
        scale_arr = np.asarray(raw_scale, dtype=np.float64).flatten()
        offset_arr = np.asarray(raw_offset, dtype=np.float64).flatten()
        print(f"      scale:  shape_in_config={np.asarray(raw_scale).shape} n={scale_arr.size} "
              f"min={scale_arr.min():.6g} max={scale_arr.max():.6g} mean={scale_arr.mean():.6g} "
              f"n_zero={int((scale_arr == 0).sum())} n_nan={int(np.isnan(scale_arr).sum())}")
        print(f"      offset: shape_in_config={np.asarray(raw_offset).shape} n={offset_arr.size} "
              f"min={offset_arr.min():.6g} max={offset_arr.max():.6g} mean={offset_arr.mean():.6g} "
              f"n_zero={int((offset_arr == 0).sum())} n_nan={int(np.isnan(offset_arr).sum())}")

    print("\n=== 3. fetching real input and comparing per-channel range vs normalization ===")
    cycle = most_recent_completed_cycle(datetime.now(timezone.utc))
    source = EcmwfOpenDataSource(cache_dir=args.ifs_cache_dir, source=args.ifs_source)
    real_input = assemble_model_input(BEST_LOSS_MANIFEST, source, cycle)
    names = BEST_LOSS_MANIFEST.channel_names()
    if norm_layer is not None and scale_arr.size == len(names) and offset_arr.size == len(names):
        scale, offset = scale_arr, offset_arr
        # rescaled = (raw - stat_a) / (stat_b - stat_a) per the minmax formula
        # above, so a channel whose live range falls inside its training-time
        # [min, max] should land roughly in [0, 1] -- values wildly outside
        # that (very negative, or way past 1) mean live IFS data is outside
        # the range this channel's stats were computed from at train time.
        print("    channel: real_min real_max | rescaled_min rescaled_max (should be roughly 0..1 for minmax normalization)")
        for i in range(len(names)):
            ch = real_input[..., i]
            rescaled = ch * scale[i] + offset[i]
            print(
                f"    [{i:2d}] {names[i]:45s} real=({np.nanmin(ch):.4g}, {np.nanmax(ch):.4g}) "
                f"rescaled=({np.nanmin(rescaled):.4g}, {np.nanmax(rescaled):.4g})"
            )
    elif norm_layer is not None:
        print(
            f"    scale/offset don't have one entry per channel (scale n={scale_arr.size}, "
            f"offset n={offset_arr.size}, expected {len(names)}) -- can't do a per-channel "
            "comparison; the raw values printed in section 2 are still worth checking by hand."
        )

    print("\n=== 4. spot-check a few weight arrays for all-zero / NaN kernels ===")
    checked = 0
    for w in model.weights:
        arr = w.numpy() if hasattr(w, "numpy") else np.asarray(w)
        if arr.size < 4:
            continue
        n_nan = int(np.isnan(arr).sum())
        all_zero = bool(np.all(arr == 0))
        flag = ""
        if n_nan > 0:
            flag = "  !! HAS NaN"
        elif all_zero:
            flag = "  !! ALL ZERO"
        if flag or checked < 10:
            print(f"    {w.name:60s} shape={arr.shape} std={arr.std():.6g}{flag}")
            checked += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
