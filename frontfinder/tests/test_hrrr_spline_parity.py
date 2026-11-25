"""Measures and threshold rule of `scripts/hrrr_spline_parity.py`.

Hand-built lines only: the parity tool's numbers have to be checkable without
the 48 MB sidecars it usually reads, and a geometry measure is easiest to pin
down on shapes whose answer is known on paper (a straight run turns nowhere, a
200 km-radius arc turns 25/200 radians per 25 km step, a 1 degree meridian
segment is 111.19 km).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

from frontfinder.postprocess import splines

KM_PER_DEG = float(np.radians(splines.EARTH_RADIUS_KM))


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "hrrr_spline_parity.py"
    spec = importlib.util.spec_from_file_location("hrrr_spline_parity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parity = _load()


def boundary(*lines: tuple[np.ndarray, np.ndarray]) -> splines.FrontBoundary:
    """A one-line-per-argument boundary, all sections typed cold so they merge."""
    return splines.FrontBoundary(
        lines=[[splines.FrontSection("CF", lon, lat, np.ones_like(lat))] for lon, lat in lines]
    )


def polyline(**points: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """`(lon, lat)` through named (lon, lat) points, in argument order."""
    pairs = list(points.values())
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def metrics_of(*boundaries: splines.FrontBoundary) -> dict[str, float]:
    return parity.line_metrics(list(boundaries))


def test_straight_line_never_turns() -> None:
    """A run due east has no heading change anywhere, so no hook either."""
    lon, lat = polyline(a=(0.0, 40.0), b=(1.0, 40.0), c=(2.0, 40.0), d=(3.0, 40.0), e=(4.0, 40.0))
    metrics = metrics_of(boundary((lon, lat)))
    assert metrics["n_lines"] == 1
    assert metrics["turn_mean"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["turn_p90"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["hook150_med"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["hook150_p90"] == pytest.approx(0.0, abs=1e-9)


def test_meridian_degree_measures_its_great_circle_length() -> None:
    """One degree of latitude along a meridian is 111.19 km on the haversine sphere."""
    lon, lat = polyline(a=(-100.0, 30.0), b=(-100.0, 30.5), c=(-100.0, 30.75), d=(-100.0, 31.0))
    metrics = metrics_of(boundary((lon, lat)))
    assert metrics["total_km"] == pytest.approx(111.2, abs=0.3)
    assert metrics["len_med_km"] == pytest.approx(111.2, abs=0.3)
    assert metrics["frac_lt_300km"] == 1.0


def test_semicircle_hook_is_arc_over_radius() -> None:
    """A 200 km-radius arc sampled every 25 km accumulates 25/200 rad per segment.

    Laid out at the equator so the metric's `cos(lat)` longitude shrink is
    exactly the one used to place the points. Six of those turns -- the
    `hook150` window -- is 150/200 rad = 43 deg.
    """
    radius_km, step_km = 200.0, 25.0
    theta = np.arange(0.0, math.pi + 1e-9, step_km / radius_km)
    x_km = radius_km * np.sin(theta)
    y_km = radius_km * (1.0 - np.cos(theta))
    lon = -100.0 + x_km / KM_PER_DEG
    lat = y_km / KM_PER_DEG
    metrics = metrics_of(boundary((lon, lat)))
    expected = math.degrees(6.0 * step_km / radius_km)
    assert metrics["hook150_med"] == pytest.approx(expected, abs=3.0)
    assert metrics["hook150_p90"] == pytest.approx(expected, abs=3.0)
    assert metrics["turn_mean"] == pytest.approx(math.degrees(step_km / radius_km), abs=0.5)


def test_hook_cancels_a_wiggle_but_scores_a_hook() -> None:
    """Signed turns cancel along a zig-zag and add up along a hairpin."""
    wiggle = polyline(a=(0.0, 40.0), b=(1.0, 40.0), c=(2.0, 40.1), d=(3.0, 40.0), e=(4.0, 40.1))
    hairpin = polyline(
        a=(0.0, 40.0), b=(2.0, 40.0), c=(3.0, 40.0), d=(3.0, 40.5), e=(2.0, 40.5), f=(1.0, 40.5)
    )
    assert parity.hook_degrees(parity.turn_degrees(*wiggle)) < 60.0
    # East, east, north, west, west: two 90 deg turns the window adds up.
    assert parity.hook_degrees(parity.turn_degrees(*hairpin)) == pytest.approx(180.0)


def test_corner_scores_high_on_its_own_line_and_dilutes_when_pooled() -> None:
    """turn_p90 is a pooled statistic, so one corner is only visible per line.

    On the cornered line alone the 90 deg turn is the top decile (interpolated
    between the one 90 and the one 0 it sits next to: 0.9 * 90 = 81). Pooled
    with a long straight run, nine zero turns outvote it and the p90 collapses
    to 9 -- which is exactly why the mean and p90 of the pool are reported next
    to the per-line hook rather than instead of it.
    """
    cornered = polyline(a=(-100.0, 30.0), b=(-99.0, 30.0), c=(-98.0, 30.0), d=(-98.0, 31.0))
    straight = polyline(**{f"p{i}": (-90.0 + i, 35.0) for i in range(10)})
    alone = metrics_of(boundary(cornered))
    assert alone["turn_p90"] == pytest.approx(81.0, abs=1.0)
    assert alone["turn_mean"] == pytest.approx(45.0)

    pooled = metrics_of(boundary(cornered), boundary(straight))
    assert pooled["n_lines"] == 2
    assert pooled["turn_mean"] == pytest.approx(90.0 / 10.0)
    assert pooled["turn_p90"] == pytest.approx(9.0)


def test_a_type_change_does_not_add_phantom_turns() -> None:
    """A type change repeats the junction vertex; measuring must drop the repeat.

    Left in, the zero-length segment has heading arctan2(0, 0) = 0 (due east),
    so a straight north-going line with one type change would score two
    +/-90 deg turns at the junction -- an artifact that scales with the number
    of sections, so a config would "smooth" a line just by merging types.
    """
    lon = np.array([-100.0, -100.0, -100.0, -100.0])
    lat = np.array([30.0, 31.0, 32.0, 33.0])
    joined = splines.FrontBoundary(
        lines=[
            [
                splines.FrontSection("CF", lon[:3], lat[:3], np.ones(3)),
                splines.FrontSection("WF", lon[2:], lat[2:], np.ones(2)),  # repeats lon[2], lat[2]
            ]
        ]
    )
    (measured_lon, measured_lat), = parity.line_points([joined])
    assert measured_lon.size == 4, "the shared junction vertex is counted once"
    metrics = parity.line_metrics([joined])
    assert metrics["total_km"] == pytest.approx(splines.line_length_km(lon, lat), abs=1e-6)
    assert parity.turn_degrees(measured_lon, measured_lat) == pytest.approx([0.0, 0.0])
    assert metrics["turn_mean"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["hook150_med"] == pytest.approx(0.0, abs=1e-9)


def test_stubs_are_dropped_and_longitudes_wrap() -> None:
    """Under four vertices measures nothing; lon goes to [-180, 180)."""
    stub = polyline(a=(0.0, 40.0), b=(1.0, 40.0), c=(2.0, 40.0))
    empty = metrics_of(boundary(stub))
    assert empty["n_lines"] == 0 and empty["total_km"] == 0.0
    assert math.isnan(empty["turn_mean"]) and math.isnan(empty["hook150_med"])

    crossing = splines.FrontBoundary(
        lines=[
            [
                splines.FrontSection(
                    "CF", np.array([179.0, 179.5, 180.5, 181.0]), np.array([40.0, 40.0, 40.0, 40.0]), np.ones(4)
                )
            ]
        ]
    )
    (lon, lat), = parity.line_points([crossing])
    assert np.all(lon >= -180.0) and np.all(lon < 180.0)
    assert np.allclose(lon, [179.0, 179.5, -179.5, -179.0])
    assert np.allclose(lat, 40.0)


def test_short_lines_are_counted_by_length() -> None:
    """`frac_lt_300km` is the share of lines under 300 km, `len_med_km` their median."""
    short = polyline(a=(0.0, 40.0), b=(0.5, 40.0), c=(1.0, 40.0), d=(1.5, 40.0))
    long_ = polyline(a=(10.0, 40.0), b=(15.0, 40.0), c=(20.0, 40.0), d=(25.0, 40.0))
    short_km, long_km = metrics_of(boundary(short))["total_km"], metrics_of(boundary(long_))["total_km"]
    parallel_km = KM_PER_DEG * math.cos(math.radians(40.0))
    assert short_km == pytest.approx(1.5 * parallel_km, abs=1.0)
    assert long_km == pytest.approx(15.0 * parallel_km, abs=1.0)

    metrics = metrics_of(boundary(short), boundary(long_))
    assert metrics["n_lines"] == 2
    assert metrics["total_km"] == pytest.approx(short_km + long_km, abs=1.0)
    assert metrics["len_med_km"] == pytest.approx((short_km + long_km) / 2, abs=1.0)
    assert metrics["frac_lt_300km"] == pytest.approx(0.5)


def test_parse_thresholds_includes_both_ends() -> None:
    assert parity.parse_thresholds("0.16:0.24:0.01") == [round(0.16 + 0.01 * i, 4) for i in range(9)]
    assert parity.parse_thresholds("0.2:0.2:0.01") == [0.2]
    with pytest.raises(ValueError):
        parity.parse_thresholds("0.24:0.16:0.01")
    with pytest.raises(ValueError):
        parity.parse_thresholds("0.16:0.24:0")


def test_threshold_rule_takes_the_arg_min_over_the_worst_date() -> None:
    """Each candidate scores its worst date; the arg-min over those wins."""
    worst = parity.worst_case_deviation(
        {0.18: [1.37, 0.98, 1.15], 0.19: [1.05, 0.93, 0.88], 0.20: [1.06, 1.07, 0.93]}
    )
    assert worst == pytest.approx({0.18: 0.37, 0.19: 0.12, 0.20: 0.07})
    assert parity.select_threshold(worst) == 0.20
    assert parity.select_threshold({}) is None
    assert parity.worst_case_deviation({0.2: [float("nan"), 1.02]}) == pytest.approx({0.2: 0.02})
    assert parity.worst_case_deviation({0.2: [float("nan")]}) == {}


def test_threshold_selection_reads_the_measured_rows() -> None:
    """The rule runs over the printed rows, per-date, ignoring the baseline itself."""

    def result(ratios: dict[float, float]) -> dict:
        return {
            "rows": {"baseline": {"threshold": None, "km_ratio": 1.0}}
            | {f"native {t:.2f}": {"threshold": t, "km_ratio": r} for t, r in ratios.items()}
        }

    dates = [result({0.18: 0.9, 0.2: 1.04}), result({0.18: 1.5, 0.2: 0.97})]
    selected, worst = parity.threshold_selection(dates)
    assert worst[0.18] == pytest.approx(0.5)
    assert worst[0.2] == pytest.approx(0.04)
    assert selected == 0.2
    assert parity.threshold_selection([{"rows": {"baseline": {"threshold": None, "km_ratio": 1.0}}}]) == (None, {})


def test_canvas_projection_and_clipping() -> None:
    x, y = parity.project(np.array([-125.0, -66.0]), np.array([50.0, 24.0]))
    assert np.allclose(x, [0.0, parity.CANVAS_PX[0] - 1])
    assert np.allclose(y, [0.0, parity.CANVAS_PX[1] - 1])

    assert parity.clip_segment(10.0, 10.0, 20.0, 20.0) == pytest.approx((10.0, 10.0, 20.0, 20.0))
    # A segment starting off the west edge and ending inside enters at x = 0.
    xs, ys = parity.project(np.array([-130.0, -100.0]), np.array([40.0, 40.0]))
    clipped = parity.clip_segment(float(xs[0]), float(ys[0]), float(xs[1]), float(ys[1]))
    assert clipped is not None
    assert clipped[0] == pytest.approx(0.0) and clipped[2] == pytest.approx(float(xs[1]))
    assert clipped[1] == pytest.approx(float(ys[0])) and clipped[3] == pytest.approx(float(ys[1]))
    assert parity.clip_segment(-50.0, 400.0, -10.0, 500.0) is None
    assert parity.clip_segment(2000.0, 400.0, 3000.0, 500.0) is None
    assert parity.clip_segment(200.0, 400.0, 201.0, 400.0) == pytest.approx((200.0, 400.0, 201.0, 400.0))


def test_render_writes_lines_inside_the_canvas(tmp_path) -> None:
    Image = pytest.importorskip("PIL.Image")
    latitudes = np.arange(50.0, 23.75, -0.25)
    longitudes = np.arange(-125.0, -65.75, 0.25)
    probabilities = xr.Dataset(
        {"CF": (("latitude", "longitude"), np.tile(np.linspace(0.0, 0.9, longitudes.size), (latitudes.size, 1)))},
        coords={"latitude": latitudes, "longitude": longitudes},
    )
    lon = np.array([-120.0, -110.0, -100.0, -90.0])
    lat = np.array([40.0, 41.0, 40.0, 39.0])
    path = tmp_path / "frame.png"
    parity.render_png(str(path), [boundary((lon, lat))], probabilities, None)

    canvas = Image.open(path)
    canvas.load()
    assert canvas.size == parity.CANVAS_PX
    pixels = np.array(canvas.convert("RGB"))
    blue = np.array(parity.TYPE_COLORS["CF"], dtype=np.int16)
    painted = np.abs(pixels.astype(np.int16) - blue).sum(axis=-1) < 60
    assert painted.sum() > 500, "the cold front should be drawn, several pixels wide"
    rows, cols = np.nonzero(painted)
    assert rows.min() > 0 and rows.max() < parity.CANVAS_PX[1] - 1
    assert cols.min() > 0 and cols.max() < parity.CANVAS_PX[0] - 1
    # The underlay fades white to grey with the probability ramp, east-west.
    assert pixels[parity.CANVAS_PX[1] - 3, 5].mean() > pixels[parity.CANVAS_PX[1] - 3, -6].mean()
