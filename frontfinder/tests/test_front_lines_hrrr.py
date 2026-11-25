"""Front lines on HRRR's native (curvilinear) grid.

Covers the pieces `postprocess.front_lines` / `postprocess.splines` grew for
the `hrrr-3km` model: extracting lines ON the native projected grid through
the `NativeGridCoords` adapter (never a lat/lon reprojection of the 3 km
probabilities), the native `x`/`y`/`crs` flavour of the probs/fields sidecars,
and the scheduler wiring that publishes `<step>.fronts.geojson` + the
`fronts` keys in `hrrr-3km/latest.json` -- where a failing step is always
logged and skipped, never raised into inference or publishing.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pyproj
import xarray as xr

from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_NO_DEWPOINT_MANIFEST as NBS
from frontfinder.config.manifests import NBS_SQUARE_DILATION_TROF_TT_MANIFEST as MANIFEST
from frontfinder.ingest.base import GridSpec
from frontfinder.ingest.hrrr_grid import HRRR_CRS, native_grid
from frontfinder.postprocess import front_lines
from frontfinder.postprocess.grid_geom import NativeGridCoords
from frontfinder.zarrio.pyramid import RegionalFrontFields, build_regional_pyramid, write_front_pyramid

CELL_M = 12000.0
NX, NY = 80, 60


def lcc_patch() -> GridSpec:
    """A 960 x 708 km patch of HRRR's own Lambert conformal projection at
    12 km cells -- same grid family as the 3 km production grid, small enough
    for a fast end-to-end run. Rows north-to-south like every grid here."""
    x = np.arange(NX, dtype=float) * CELL_M
    y = (NY - 1 - np.arange(NY, dtype=float)) * CELL_M  # descending: row 0 is the north edge
    to_geo = pyproj.Transformer.from_crs(HRRR_CRS, "EPSG:4326", always_xy=True)
    xx, yy = np.meshgrid(x, y)
    lon, lat = to_geo.transform(xx, yy)
    return GridSpec(name="test-lcc-12km", crs=HRRR_CRS, x=x, y=y, lat=np.asarray(lat), lon=np.asarray(lon))


def cold_band(center_row: float) -> np.ndarray:
    """A straight east-west CF probability band along `center_row`, spanning
    the full grid width (~950 km of crest)."""
    rows = np.arange(NY, dtype=float)[:, None]
    return (0.95 * np.exp(-0.5 * ((rows - center_row) / 1.2) ** 2) * np.ones((1, NX))).astype(np.float32)


def zeros_like_grid() -> np.ndarray:
    return np.zeros((NY, NX), dtype=np.float32)


def publish_native_step(
    model_dir,
    store_name: str,
    valid_time: str,
    grid: GridSpec,
    cold: np.ndarray,
    temperature: np.ndarray | None = None,
) -> dict:
    """Writes one unpacked native-grid store + both sidecars exactly as
    `run_one_hrrr_step` does (probs BEFORE the write, native x/y/crs keys)."""
    probabilities = {cls: cold if cls == "cold" else zeros_like_grid() for cls in MANIFEST.served_classes}
    pyramid = build_regional_pyramid(
        RegionalFrontFields(
            probabilities=probabilities, grid=grid, valid_time=valid_time, cycle_time="2026-08-19T12:00:00", source="test"
        ),
        MANIFEST,
        n_levels=2,
    )
    coords = NativeGridCoords.from_grid_spec(grid)
    front_lines.save_probabilities_sidecar(
        pyramid.datatree["0"].to_dataset(),
        str(model_dir / front_lines.probabilities_filename(store_name)),
        coords=coords,
    )
    write_front_pyramid(pyramid, str(model_dir / store_name))
    if temperature is not None:
        front_lines.save_orientation_fields(
            {
                "temperature": temperature.astype(np.float32),
                "u": np.zeros_like(temperature, dtype=np.float32),
                "v": np.zeros_like(temperature, dtype=np.float32),
                "x": grid.x,
                "y": grid.y,
                "crs": grid.crs,
            },
            str(model_dir / front_lines.fields_filename(store_name)),
        )
    return {"step_hours": 0, "valid_time": valid_time, "store": store_name}


def warm_north_field() -> np.ndarray:
    """Theta-e falling 0.4 K per row toward the south pole side (~10 K across
    the band's +/-150 km sampling offsets): an unmistakable warm-north
    gradient, identical in index space on the curvilinear and its 1-D twin."""
    return (310.0 - 0.4 * np.arange(NY, dtype=float)[:, None] * np.ones((1, NX))).astype(np.float32)


# --- end to end on a synthetic curvilinear grid ---


def test_write_front_lines_on_a_curvilinear_grid_emits_wgs84_lines_in_grid_bounds(tmp_path):
    grid = lcc_patch()
    entries = [
        publish_native_step(tmp_path, "2026-08-19T12Z_f000.zarr", "2026-08-19T12:00:00", grid, cold_band(30.0), warm_north_field()),
        publish_native_step(tmp_path, "2026-08-19T12Z_f001.zarr", "2026-08-19T13:00:00", grid, cold_band(28.0), warm_north_field()),
    ]
    written = front_lines.write_front_lines(str(tmp_path), entries)
    assert list(written) == ["2026-08-19T12Z_f000.zarr", "2026-08-19T12Z_f001.zarr"]

    lon_min, lon_max = float(grid.lon.min()) - 1.0, float(grid.lon.max()) + 1.0
    lat_min, lat_max = float(grid.lat.min()) - 1.0, float(grid.lat.max()) + 1.0
    ids = []
    for name in written.values():
        collection = json.loads((tmp_path / name).read_text())
        assert collection["type"] == "FeatureCollection"
        assert collection["oriented"] is True
        names = {f["properties"]["name"] for f in collection["features"]}
        assert names == {"cold_front"}
        ids.append({f["properties"]["event_id"] for f in collection["features"]})
        for feature in collection["features"]:
            for line in feature["geometry"]["coordinates"]:
                for lon, lat in line:
                    assert lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

    # the front drifted 2 rows (~24 km) north and kept its event id
    assert ids[0] == ids[1]
    # and the line actually sits on the band, not somewhere else on the grid
    # (compared against row 30's own lat/lon table: a constant-y LCC row is
    # an arc, so a straight crest is NOT a line of constant latitude)
    first = json.loads((tmp_path / written["2026-08-19T12Z_f000.zarr"]).read_text())
    (line,) = first["features"][0]["geometry"]["coordinates"]
    line_lon = np.array([p[0] for p in line])
    line_lat = np.array([p[1] for p in line])
    expected = np.interp(line_lon, grid.lon[30, :], grid.lat[30, :])
    assert np.all(np.abs(line_lat - expected) < 0.05)


def test_a_published_native_store_without_a_sidecar_still_gets_lines(tmp_path):
    """The backfill case: an already-published (packed) hrrr-3km store. The
    store carries only x/y, so its geometry comes from the ingest grid
    definition matched by the store's own grid attr."""
    grid = native_grid(8)
    cold = (0.95 * np.exp(-0.5 * ((np.arange(grid.shape[0], dtype=float)[:, None] - 66.0) / 1.5) ** 2)).astype(
        np.float32
    ) * np.ones((1, grid.shape[1]), dtype=np.float32)
    probabilities = {cls: (cold if cls == "cold" else np.zeros(grid.shape, np.float32)) for cls in MANIFEST.served_classes}
    pyramid = build_regional_pyramid(
        RegionalFrontFields(
            probabilities=probabilities,
            grid=grid,
            valid_time="2026-08-19T12:00:00",
            cycle_time="2026-08-19T12:00:00",
            source="test",
        ),
        MANIFEST,
        n_levels=1,
    )
    write_front_pyramid(pyramid, str(tmp_path / "2026-08-19T12Z_f000.zarr"), pack_uint8=True)

    written = front_lines.write_front_lines(
        str(tmp_path), [{"step_hours": 0, "valid_time": "2026-08-19T12:00:00", "store": "2026-08-19T12Z_f000.zarr"}]
    )
    collection = json.loads((tmp_path / written["2026-08-19T12Z_f000.zarr"]).read_text())
    assert collection["features"]
    assert collection["oriented"] is False  # no fields sidecar here -- motion-only, no symbols drawn
    lons = np.array([p[0] for f in collection["features"] for line in f["geometry"]["coordinates"] for p in line])
    lats = np.array([p[1] for f in collection["features"] for line in f["geometry"]["coordinates"] for p in line])
    assert lons.min() >= grid.lon.min() - 1.0 and lons.max() <= grid.lon.max() + 1.0
    assert lats.min() >= grid.lat.min() - 1.0 and lats.max() <= grid.lat.max() + 1.0


# --- orientation sampling on the curvilinear grid vs the 1-D equivalent ---


def rectified_twin(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """1-D lat/lon coordinates for the same cell array: the patch's central
    column's latitudes and central row's longitudes -- the flat-grid reading
    of the same cells."""
    return grid.lat[:, NX // 2].copy(), grid.lon[NY // 2, :].copy()


def test_sampling_a_native_field_at_pixel_centers_is_exact_and_matches_the_1d_twin():
    grid = lcc_patch()
    coords = NativeGridCoords.from_grid_spec(grid)
    values = (np.arange(NY, dtype=float)[:, None] * 10.0 + np.arange(NX, dtype=float)[None, :]).astype(np.float32)
    field = xr.DataArray(values, dims=("y", "x"), coords={"y": grid.y, "x": grid.x})
    rows, cols = np.mgrid[5 : NY - 5 : 7, 5 : NX - 5 : 9]
    lon, lat = coords.pixel_to_lonlat(rows.ravel(), cols.ravel())

    from frontfinder.postprocess import splines

    via_geom = splines.sample(field, lon, lat, geom=coords)
    assert np.allclose(via_geom, values[rows.ravel(), cols.ravel()], atol=1e-4)

    lat1d, lon1d = rectified_twin(grid)
    twin = xr.DataArray(values, dims=("latitude", "longitude"), coords={"latitude": lat1d, "longitude": lon1d})
    via_1d = splines.sample(twin, lon, lat)
    inside = np.isfinite(via_1d)
    assert inside.sum() > 50
    # the twin reads the same field bilinearly in the same index space; the
    # two agree wherever both are on-grid, up to the projection's curvature
    # between the patch's central column/row and each sampled cell -- on this
    # patch under 0.41 deg latitude / 0.56 deg longitude, so within ~5 rows
    # or columns of the ramp: far less than the +/-12-cell left/right offsets
    # symbol sampling actually uses.
    assert np.nanmax(np.abs(via_geom[inside] - via_1d[inside])) < 60.0
    assert np.nanmax(np.abs(via_geom[inside] - via_1d[inside])) > 0.0  # genuinely off-row somewhere


def test_symbol_sides_on_the_curvilinear_grid_match_the_1d_case_for_a_known_gradient():
    from frontfinder.postprocess import splines

    grid = lcc_patch()
    coords = NativeGridCoords.from_grid_spec(grid)
    probs = cold_band(30.0)
    temp = warm_north_field()

    curv = xr.Dataset({"CF": (("y", "x"), probs)}, coords={"y": grid.y, "x": grid.x})
    curv_fields = {
        name: xr.DataArray(array, dims=("y", "x"), coords={"y": grid.y, "x": grid.x})
        for name, array in (("temperature", temp), ("u", np.zeros_like(temp)), ("v", np.zeros_like(temp)))
    }
    config = front_lines.default_config()
    curv_fronts = splines.extract_fronts(
        curv, config, geom=coords, **{k: v for k, v in curv_fields.items()}
    )

    lat1d, lon1d = rectified_twin(grid)
    rect = xr.Dataset({"CF": (("latitude", "longitude"), probs)}, coords={"latitude": lat1d, "longitude": lon1d})
    rect_fields = {
        name: xr.DataArray(array, dims=("latitude", "longitude"), coords={"latitude": lat1d, "longitude": lon1d})
        for name, array in (("temperature", temp), ("u", np.zeros_like(temp)), ("v", np.zeros_like(temp)))
    }
    rect_fronts = splines.extract_fronts(rect, config, **rect_fields)

    def sections(fronts):
        return [s for b in fronts for line in b.lines for s in line]

    curv_sections, rect_sections = sections(curv_fronts), sections(rect_fronts)
    assert len(curv_sections) == 1 and len(rect_sections) == 1
    (cs,) = curv_sections
    (rs,) = rect_sections

    # theta-e increases north (warm side = north), no wind, no previous step:
    # a CF's symbols go on the warm side, so walking the line east they are on
    # the left, walking west on the right -- identical rule for both grids.
    def expected(side_section) -> str:
        return "left" if side_section.lon[-1] > side_section.lon[0] else "right"

    assert cs.symbol_side == rs.symbol_side == expected(cs)


# --- coarsening before crest extraction: config + a native-grid run ---


def test_default_config_stays_the_ifs_config_and_native_config_coarsens():
    """`default_config()` keeps its pre-coarsening values (it is what the IFS path and the golden test
    use); `native_config()` is the same config with the native-grid coarsening turned on."""
    default = front_lines.default_config()
    assert (default.threshold, default.match_threshold) == (0.25, 0.5)
    assert default.native_block == 1 and default.native_blur_cells == 0.0

    native = front_lines.native_config()
    assert native.native_block == front_lines.NATIVE_BLOCK == 3
    assert native.native_blur_cells == front_lines.NATIVE_BLUR_CELLS == 3.5
    assert native.threshold == front_lines.NATIVE_FRONT_LINE_THRESHOLD
    assert native.match_threshold == 0.5  # everything else carries over from default_config()
    assert front_lines.native_config(0.3).threshold == 0.3


def trough_temperature_field() -> np.ndarray:
    """A theta-e field with a trough at row 15: `310 + 0.5 * |row - 15|` K. The line at row 30 sees a
    negative (left-minus-right) contrast through the native adapter's real 150 km offsets (rows ~17.5
    and ~42.5), but samples read through a wrongly bound COARSE-grid adapter -- whose index space puts
    them near rows ~5.5 and ~13.8, across the trough's north flank -- see a positive one. A mis-bound
    `field_geom` therefore flips the symbol side, which a monotonic gradient (like `warm_north_field`)
    would hide."""
    rows = np.abs(np.arange(NY, dtype=float)[:, None] - 15.0)
    return (310.0 + 0.5 * rows * np.ones((1, NX))).astype(np.float32)


def test_extract_fronts_coarsens_the_native_grid_without_moving_the_symbol_sides():
    """The straight synthetic band still yields one front when the 12 km grid is block-averaged by 3 and
    blurred, every point stays inside the grid, and -- the pitfall -- the orientation fields stay mapped
    through the ORIGINAL (native) adapter: their symbol sides match the uncoarsened run exactly (and
    `trough_temperature_field` makes that equality fail if the fields were re-bound to the coarse grid)."""
    from frontfinder.postprocess import splines

    grid = lcc_patch()
    coords = NativeGridCoords.from_grid_spec(grid)
    temp = trough_temperature_field()
    curv = xr.Dataset({"CF": (("y", "x"), cold_band(30.0))}, coords={"y": grid.y, "x": grid.x})
    fields = {
        name: xr.DataArray(array, dims=("y", "x"), coords={"y": grid.y, "x": grid.x})
        for name, array in (("temperature", temp), ("u", np.zeros_like(temp)), ("v", np.zeros_like(temp)))
    }
    config = splines.SplineConfig(native_block=3, native_blur_cells=1.0)
    coarse = splines.extract_fronts(curv, config, geom=coords, **fields)
    plain = splines.extract_fronts(curv, splines.SplineConfig(), geom=coords, **fields)

    def sections(fronts):
        return [s for b in fronts for line in b.lines for s in line]

    coarse_sections = sections(coarse)
    assert len(coarse) >= 1 and coarse_sections
    lon_min, lon_max = float(grid.lon.min()) - 1.0, float(grid.lon.max()) + 1.0
    lat_min, lat_max = float(grid.lat.min()) - 1.0, float(grid.lat.max()) + 1.0
    for section in coarse_sections:
        assert np.all((section.lon >= lon_min) & (section.lon <= lon_max))
        assert np.all((section.lat >= lat_min) & (section.lat <= lat_max))

    # compare the sides in one frame: the tracer may walk the two crests in opposite directions, which
    # flips the raw left/right strings without moving the symbols on the earth
    def side_facing_east(section) -> str:
        side = section.symbol_side
        if section.lon[-1] < section.lon[0]:  # walked westward: left and right swap
            side = "right" if side == "left" else "left"
        return side

    # the real 150 km offsets see a negative contrast (the trough's far flank), so a CF's symbols sit
    # on its right facing east; samples squeezed through a coarse-grid adapter would sit across the
    # trough and land on the left instead
    assert [side_facing_east(s) for s in coarse_sections] == [side_facing_east(s) for s in sections(plain)]
    assert [side_facing_east(s) for s in coarse_sections] == ["right"]


# --- failing steps are logged and skipped, never raised ---


def test_a_corrupt_sidecar_or_missing_store_skips_only_its_step(tmp_path):
    grid = lcc_patch()
    good0 = publish_native_step(tmp_path, "2026-08-19T12Z_f000.zarr", "2026-08-19T12:00:00", grid, cold_band(30.0))
    (tmp_path / front_lines.probabilities_filename("2026-08-19T12Z_f000.zarr")).write_bytes(b"not an npz")
    # store still loads through the ingest-grid fallback... but this patch is
    # not one of the ingest grids, so its step is skipped rather than guessed
    missing = {"step_hours": 1, "valid_time": "2026-08-19T13:00:00", "store": "2026-08-19T12Z_f001.zarr"}
    good2 = publish_native_step(tmp_path, "2026-08-19T12Z_f002.zarr", "2026-08-19T14:00:00", grid, cold_band(28.0))
    entries = [good0, missing, good2]
    written = front_lines.write_front_lines(str(tmp_path), entries)
    assert list(written) == ["2026-08-19T12Z_f002.zarr"]  # steps 0 and 1 skipped, step 2 still gets lines

    # and with a READABLE sidecar the corrupt file is not even consulted
    publish_native_step(tmp_path, "2026-08-19T12Z_f000.zarr", "2026-08-19T12:00:00", grid, cold_band(30.0))
    (tmp_path / front_lines.fields_filename("2026-08-19T12Z_f000.zarr")).write_bytes(b"not an npz")
    written = front_lines.write_front_lines(str(tmp_path), [good0])
    collection = json.loads((tmp_path / written[good0["store"]]).read_text())
    assert collection["oriented"] is False  # orientation sidecar unreadable -> motion-only
    assert collection["features"]  # lines themselves unaffected


# --- scheduler wiring: hrrr-3km publishes fronts like IFS does ---


def _cycle_fixtures():
    from frontfinder.inference.refiner import FakeRefinerPredictor
    from frontfinder.scheduler.run_cycle import HRRRRunConfig
    from frontfinder.tests.test_run_cycle import _FakeHRRRSource

    class NbsPredictor:
        def predict_batch(self, patches: np.ndarray) -> np.ndarray:
            out = np.full((*patches.shape[:3], len(NBS.all_classes)), 0.01, dtype=np.float32)
            return out / out.sum(axis=-1, keepdims=True)

    run_config = HRRRRunConfig(
        manifest=NBS,
        base_predictor=NbsPredictor(),
        refiner=FakeRefinerPredictor(),
        refiner_mean=np.zeros(NBS.n_channels, dtype=np.float32),
        refiner_std=np.ones(NBS.n_channels, dtype=np.float32),
    )
    return run_config, _FakeHRRRSource(native_grid(1)), _FakeHRRRSource(native_grid(8))


def test_run_hrrr_cycle_publishes_sidecars_geojson_and_fronts_keys(tmp_path):
    from frontfinder.scheduler.run_cycle import HRRR_3KM_MODEL_KEY, run_hrrr_cycle

    run_config, source_3km, source_24km = _cycle_fixtures()
    paths = run_hrrr_cycle(
        run_config,
        source_3km,
        source_24km,
        "2026-08-19",
        12,
        str(tmp_path),
        steps=(0, 6),
        front_lines=True,
        front_line_threshold=0.9,  # the fake model's flat probabilities must not skeletonize the whole grid
    )
    assert len(paths) == 2
    model_dir = tmp_path / HRRR_3KM_MODEL_KEY
    for store in ["2026-08-19T12Z_f000.zarr", "2026-08-19T12Z_f006.zarr"]:
        with np.load(model_dir / front_lines.probabilities_filename(store)) as z:
            assert {"CF", "WF", "SF", "OF", "DL", "TROF", "TT", "x", "y", "crs"} <= set(z.files)
            assert str(z["crs"].item()) == native_grid(1).crs
        with np.load(model_dir / front_lines.fields_filename(store)) as z:
            assert {"temperature", "u", "v", "x", "y", "crs"} <= set(z.files)
        collection = json.loads((model_dir / front_lines.fronts_filename(store)).read_text())
        assert collection["type"] == "FeatureCollection"
        assert collection["oriented"] is True  # the 850 hPa sidecar was found
    pointer = json.loads((model_dir / "latest.json").read_text())
    assert [s["fronts"] for s in pointer["steps"]] == [
        "2026-08-19T12Z_f000.fronts.geojson",
        "2026-08-19T12Z_f006.fronts.geojson",
    ]


def test_run_hrrr_cycle_hands_the_native_config_to_front_lines(tmp_path, monkeypatch):
    """The scheduler wires `run_hrrr_cycle` to the coarsened config (thresholds aside), where `run_cycle`
    (IFS) keeps the default one."""
    from frontfinder.scheduler.run_cycle import run_hrrr_cycle

    seen = []
    real = front_lines.write_front_lines

    def spy(model_dir, step_entries, config=None):
        seen.append(config)
        return real(model_dir, step_entries, config)

    monkeypatch.setattr(front_lines, "write_front_lines", spy)
    run_config, source_3km, source_24km = _cycle_fixtures()
    run_hrrr_cycle(
        run_config, source_3km, source_24km, "2026-08-19", 12, str(tmp_path), steps=(0,), front_lines=True,
        front_line_threshold=0.9,
    )
    assert seen and all(
        config.native_block == 3 and config.native_blur_cells == 3.5 and config.threshold == 0.9
        for config in seen
    )


def test_a_failing_hrrr_step_is_skipped_and_latest_json_stays_intact(tmp_path, monkeypatch):
    from frontfinder.scheduler import run_cycle as run_cycle_module
    from frontfinder.scheduler.run_cycle import HRRR_3KM_MODEL_KEY, run_hrrr_cycle

    run_config, source_3km, source_24km = _cycle_fixtures()
    real = run_cycle_module.run_one_hrrr_step

    def flaky(config, s3, s24, cycle, *args, **kwargs):
        if cycle.step == 6:
            raise RuntimeError("GRIB download exploded")
        return real(config, s3, s24, cycle, *args, **kwargs)

    monkeypatch.setattr(run_cycle_module, "run_one_hrrr_step", flaky)
    paths = run_hrrr_cycle(
        run_config, source_3km, source_24km, "2026-08-19", 12, str(tmp_path), steps=(0, 6, 12), front_lines=True, front_line_threshold=0.9
    )
    assert [os.path.basename(p) for p in paths] == ["2026-08-19T12Z_f000.zarr", "2026-08-19T12Z_f012.zarr"]
    pointer = json.loads((tmp_path / HRRR_3KM_MODEL_KEY / "latest.json").read_text())
    assert [s["step_hours"] for s in pointer["steps"]] == [0, 12]
    assert all("fronts" in s for s in pointer["steps"])


def test_a_front_lines_crash_never_stops_the_hrrr_publish(tmp_path, monkeypatch):
    from frontfinder.scheduler.run_cycle import HRRR_3KM_MODEL_KEY, run_hrrr_cycle

    def boom(*args, **kwargs):
        raise RuntimeError("scikit-image exploded")

    monkeypatch.setattr(front_lines, "write_front_lines", boom)
    run_config, source_3km, source_24km = _cycle_fixtures()
    paths = run_hrrr_cycle(
        run_config, source_3km, source_24km, "2026-08-19", 12, str(tmp_path), steps=(0,), front_lines=True, front_line_threshold=0.9
    )
    assert len(paths) == 1  # store landed
    pointer = json.loads((tmp_path / HRRR_3KM_MODEL_KEY / "latest.json").read_text())
    assert all("fronts" not in s for s in pointer["steps"])  # and published without the optional key
