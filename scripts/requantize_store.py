"""Re-encode already-published float32 front pyramids as uint8 in place.

`write_front_pyramid` now packs served-class probabilities to uint8 at write
time (see the PACKED_* comment in frontfinder/zarrio/pyramid.py), but every
store published before that change is still float32. Those compress badly:
one level-0 shard per class is ~3.27 MB and first paint pulls ~7 of them,
whereas the same shard re-encoded as uint8 + zstd is ~94 KB (35x smaller,
max abs error ~0.002 for display bins 0.1 wide).

This is the offline backfill tool for those stores. It takes one model's
published directory exactly as the webapp sees it --

    <src>/latest.json
    <src>/<cycle>Z_f<NNN>.zarr/
    <src>/<cycle>Z_f<NNN>.fronts.geojson

-- re-encodes every store through the SAME shared helper
(`write_quantized_datatree`, via `quantize_datatree`) that
`write_front_pyramid` uses, so a backfilled store gets identical packing,
chunk/shard layout and metadata to one produced by a fresh inference run,
and copies latest.json and the geojson files through unchanged. No network
access; works against a local fixture.

Usage:
    uv run python scripts/requantize_store.py SRC_MODEL_DIR DST_MODEL_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import xarray as xr

from frontfinder.zarrio.pyramid import write_quantized_datatree


def _directory_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(root, name)) for root, _, names in os.walk(path) for name in names)


def _is_already_packed(datatree: xr.DataTree) -> bool:
    """True when every served-class variable in level 0 is already uint8.

    Makes the script idempotent: re-running it over a directory it already
    quantized copies those stores through untouched instead of trying to
    pack packed data (which would double-decode and corrupt the values).
    """
    level0 = datatree["0"]
    served_classes = level0.attrs.get("served_classes") or []
    return bool(served_classes) and all(level0[cls].dtype == np.uint8 for cls in served_classes)


def _store_complete(dst_store: str) -> bool:
    """True when a previously written destination store finished writing.

    `write_quantized_datatree` calls zarr.consolidate_metadata() LAST, so a
    non-empty consolidated listing in the root zarr.json means every array and
    shard before it landed; a crash mid-write leaves it empty/absent and the
    store gets rewritten. Stores are immutable per cycle+step, so a complete
    one never needs redoing.
    """
    try:
        with open(os.path.join(dst_store, "zarr.json")) as f:
            return bool(json.load(f).get("consolidated_metadata", {}).get("metadata"))
    except (OSError, ValueError):
        return False


def _requantize_store(src_store: str, dst_store: str) -> str:
    # mask_and_scale=False so `_is_already_packed` sees the on-disk dtype
    # (a re-run over our own uint8 output would otherwise be decoded back to
    # float and silently packed a second time).
    datatree = xr.open_datatree(src_store, engine="zarr", mask_and_scale=False)

    if _is_already_packed(datatree):
        shutil.rmtree(dst_store, ignore_errors=True)
        shutil.copytree(src_store, dst_store)
        return f"already uint8 -> copied unchanged ({_directory_bytes(dst_store):,} bytes)"

    levels = [path for path in datatree.groups if path not in ("", "/")]
    served_classes = datatree["0"].attrs.get("served_classes") or []
    before_bytes = _directory_bytes(src_store)

    # mode="w" overwrites arrays but leaves any stale shard files behind, so
    # clear the destination first -- otherwise a store that shrank can still
    # carry dead shards from the previous write.
    shutil.rmtree(dst_store, ignore_errors=True)
    write_quantized_datatree(datatree, dst_store)

    after_bytes = _directory_bytes(dst_store)
    ratio = before_bytes / after_bytes if after_bytes else 0.0
    return (
        f"{len(levels)} levels x {len(served_classes)} classes -> uint8, "
        f"{before_bytes:,} -> {after_bytes:,} bytes ({ratio:.1f}x smaller)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", help="source model directory (latest.json + *.zarr + *.fronts.geojson)")
    parser.add_argument("dst", help="destination model directory; created if missing")
    parser.add_argument("--skip-existing", action="store_true", help="leave already-complete destination stores alone (stores are immutable per cycle+step)")
    parser.add_argument("--prune", action="store_true", help="delete destination stores/geojson that no longer exist in the source (mirrors the source's retention)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.src):
        print(f"error: source directory not found: {args.src}", file=sys.stderr)
        return 1

    os.makedirs(args.dst, exist_ok=True)

    # sorted(): "<cycle>Z_f<NNN>..." entries sort before "latest.json", so the
    # pointer is copied LAST and never references a store that isn't there yet.
    for name in sorted(os.listdir(args.src)):
        src_path = os.path.join(args.src, name)
        dst_path = os.path.join(args.dst, name)

        if name.endswith(".zarr") and os.path.isdir(src_path):
            if args.skip_existing and _store_complete(dst_path):
                print(f"{name}: already written, skipped")
                continue
            print(f"{name}: {_requantize_store(src_path, dst_path)}")
            continue

        # latest.json and the *.fronts.geojson sidecars are pointer/vector
        # files the webapp reads directly -- copy them byte-for-byte.
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"{name}: copied unchanged")

    if args.prune:
        for name in sorted(set(os.listdir(args.dst)) - set(os.listdir(args.src))):
            dst_path = os.path.join(args.dst, name)
            shutil.rmtree(dst_path) if os.path.isdir(dst_path) else os.remove(dst_path)
            print(f"{name}: pruned (no longer in source)")

    print(f"\nwrote quantized model directory: {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
