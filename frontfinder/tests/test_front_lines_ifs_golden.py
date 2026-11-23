"""Byte-identical regression test for the IFS (regular lat/lon) front-line path.

The curvilinear (HRRR native-grid) support in `postprocess.splines` /
`postprocess.front_lines` was added additively: every new code path is behind
an optional grid-adapter argument, and the regular 1-D lat/lon path never
constructs one. This test pins the IFS side of that promise: it replays a
fixed synthetic two-step cycle (fixture `data/ifs_golden/inputs.npz`, built by
`build_fixture_inputs` below from a deterministic formula -- no RNG) through
`front_lines.write_front_lines` and compares every emitted GeoJSON file to the
frozen golden bytes in `data/ifs_golden/` byte for byte.

The goldens were generated from the code at the commit BEFORE curvilinear
support landed (see SPLINES_HRRR.md). If a change here makes this test fail,
either IFS front-line output changed (regression) or the goldens were
deliberately regenerated -- say which in the PR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_MANIFEST as MANIFEST
from frontfinder.postprocess import front_lines
from frontfinder.zarrio.pyramid import FrontFields, build_front_pyramid, write_front_pyramid

DATA = Path(__file__).parent / "data" / "ifs_golden"
FIXTURE = DATA / "inputs.npz"

STEPS = [
    ("2026-10-31T06Z_f000.zarr", "2026-10-31T06:00:00"),
    ("2026-10-31T06Z_f006.zarr", "2026-10-31T12:00:00"),
]


def build_fixture_inputs() -> dict[str, np.ndarray]:
    """Deterministic two-step probability/field fixture on a regional IFS-style
    0.5 degree grid: a mixed CF/WF band running west-east with a wavy crest, a
    wavy dryline band running north-south that crosses it (a junction), and the
    whole system drifting 1 degree north between the two steps."""
    lat = np.linspace(55.0, 25.0, 61)  # descending, like the IFS stores
    lon = np.linspace(228.0, 298.0, 141)  # store convention: 0..360 minus nothing; signed below
    lon_signed = lon - 360.0  # -132..-62, what the viewer-facing grids use
    lat2d, lon2d = np.meshgrid(lat, lon_signed, indexing="ij")

    def ridge(distance_deg: np.ndarray, width_deg: float) -> np.ndarray:
        return np.exp(-0.5 * (distance_deg / width_deg) ** 2)

    inputs: dict[str, np.ndarray] = {"lat": lat, "lon": lon_signed}
    for index, drift in enumerate((0.0, 1.0)):
        crest_lat = 40.0 + 6.0 * np.sin((lon2d + 97.0) / 18.0) + drift
        cf = ridge(lat2d - crest_lat, 2.2) * 0.95
        wf = ridge(lat2d - (crest_lat + 2.0), 2.2) * 0.90
        dl = ridge(lon2d - (-85.0 + 4.0 * np.sin((lat2d - 38.0) / 9.0)), 2.0) * 0.90
        for cls, array in (("cold", cf), ("warm", wf), ("dryline", dl)):
            inputs[f"step{index}_{cls}"] = array.astype(np.float32)
        for cls in ("stationary", "occluded", "trough", "tropical_trough"):
            inputs[f"step{index}_{cls}"] = np.zeros_like(cf, dtype=np.float32)
        # 850 hPa theta-e warm toward the south, plus a gentle cross-front wind.
        inputs[f"step{index}_theta_e"] = (312.0 - 0.6 * lat2d).astype(np.float32)
        inputs[f"step{index}_u"] = np.full_like(lat2d, 6.0, dtype=np.float32)
        inputs[f"step{index}_v"] = np.full_like(lat2d, -2.0, dtype=np.float32)
    return inputs


def publish_fixture_cycle(model_dir: Path) -> list[dict]:
    """Writes the fixture's two stores (unpacked) + the float32 probs and
    orientation sidecars, exactly as `run_one_model` would, and returns the
    `latest.json` step entries."""
    with np.load(FIXTURE) as z:
        arrays = {key: z[key] for key in z.files}
    entries = []
    for index, (store_name, valid_time) in enumerate(STEPS):
        probabilities = {cls: arrays[f"step{index}_{cls}"] for cls in MANIFEST.served_classes}
        fields = FrontFields(
            probabilities=probabilities,
            lat=arrays["lat"],
            lon=arrays["lon"],
            valid_time=valid_time,
            cycle_time="2026-10-31T06:00:00",
        )
        pyramid = build_front_pyramid(fields, MANIFEST, n_levels=2)
        front_lines.save_probabilities_sidecar(
            pyramid.datatree["0"].to_dataset(), str(model_dir / front_lines.probabilities_filename(store_name))
        )
        write_front_pyramid(pyramid, str(model_dir / store_name))
        front_lines.save_orientation_fields(
            {
                "temperature": arrays[f"step{index}_theta_e"],
                "u": arrays[f"step{index}_u"],
                "v": arrays[f"step{index}_v"],
                "lat": arrays["lat"],
                "lon": arrays["lon"],
            },
            str(model_dir / front_lines.fields_filename(store_name)),
        )
        entries.append({"step_hours": 6 * index, "valid_time": valid_time, "store": store_name})
    return entries


def test_ifs_front_lines_are_byte_identical(tmp_path):
    assert FIXTURE.exists(), "committed fixture data missing"
    entries = publish_fixture_cycle(tmp_path)
    written = front_lines.write_front_lines(str(tmp_path), entries)
    assert list(written) == [store for store, _ in STEPS]
    for (store_name, _), (store, name) in zip(STEPS, written.items()):
        got = (tmp_path / name).read_bytes()
        golden = (DATA / f"{store_name[: -len('.zarr')]}.fronts.geojson").read_bytes()
        assert got == golden, f"{name} differs from the frozen golden"
    # keep the fixture honest: it must actually produce a multi-line, typed, oriented cycle
    first = json.loads((tmp_path / written[STEPS[0][0]]).read_text())
    assert first["oriented"] is True
    assert len(first["features"]) > 2
