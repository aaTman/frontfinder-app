# Front lines on HRRR's native grid (`hrrr-3km`)

Makes the webapp's optional "Front lines" overlay work for the HRRR 3 km model
the same way it works for IFS, without changing IFS output. Lines are
extracted **on the native 1059x1799 Lambert-conformal 3 km grid** -- never
from a lat/lon reprojection of the probabilities -- and emitted as plain
WGS84 GeoJSON like the IFS path.

## What changed

- `frontfinder/postprocess/grid_geom.py` (new): `NativeGridCoords`, the one
  pixel <-> lon/lat coordinate adapter for projected tensor-product grids.
  Knows nothing about HRRR specifically (built from a `GridSpec` or from
  `crs + x + y`).
- `frontfinder/postprocess/splines.py`: curvilinear support, threaded purely
  additively. `is_periodic` reports False for 2-D longitude tables;
  `skeleton_graph`/`path_points` take a 2-D (row, col) lat/lon cell table
  when given one (1-D branch byte-identical); `sample`/`symbol_side`/
  `extract_fronts` gained optional `geom` / `field_geom` arguments. No metric
  helper (`haversine_km`, `motion_km`, `turn_angle_deg`, `left_normals`,
  `offset_points`, `project_to_polyline`, `pin_end`/`pin_junctions`,
  `to_geojson`, `track_fronts`) changed -- they all already worked on lon/lat
  *points*, which both paths produce.
- `frontfinder/postprocess/front_lines.py`: `save_probabilities_sidecar`
  gained a `coords` argument (native sidecars carry `x`/`y`/`crs` keys
  instead of `lat`/`lon`); loaders return `(dataset, geom)` internally and
  hand the adapter to `extract_fronts`; new
  `extract_native_orientation_fields` writes the 850 hPa theta-e/wind
  sidecar thinned to every other native cell (symbol sides are sampled
  150 km off the line -- 6 km resolution is invisible at that offset); a
  published native-grid store with no sidecar is placed via
  `ingest.hrrr_grid.native_grid` matched by its own `grid` attr/shape, so
  already-published stores can be backfilled (packed values, warned).
- `frontfinder/scheduler/run_cycle.py`: `run_one_hrrr_step` takes
  `save_orientation_fields` and writes both sidecars (probs BEFORE the
  always-on uint8 packing mutates the pyramid); `run_hrrr_cycle` takes
  `front_lines` / `front_line_threshold`, mirrors `run_cycle`'s pattern, and
  calls `attach_front_lines` after each step lands -- `hrrr-3km/latest.json`
  step entries get the same `"fronts"` key semantics as IFS. Everything
  front-line-related is logged-and-skipped on failure (per step *and* per
  cycle: `attach_front_lines` never raises, so neither inference nor
  `latest.json` publishing can be broken by it).
- `frontfinder/scheduler/cli.py`: passes the existing `--front-lines` /
  `--front-line-threshold` flags through to `run_hrrr_cycle`.
- Tests: `test_front_lines_ifs_golden.py` (byte-identical IFS goldens),
  `test_front_lines_hrrr.py` (end-to-end curvilinear publish, native-store
  backfill without a sidecar, sampling correctness + 1-D twin parity,
  symbol-side parity/analytic check, corrupt-sidecar and failing-step skip,
  full scheduler wiring incl. crash safety).

## How the curvilinear mapping works

`NativeGridCoords(crs, x, y, lon, lat)` maps a grid whose cells are a tensor
product of monotonic 1-D CRS coordinates (`x` columns, `y` rows, `lat`/`lon`
the 2-D cell-centre tables in degrees).

- **Forward, pixel -> lon/lat** (`splines` reading crest geometry): the 2-D
  cell-centre tables. Crest paths are whole pixels, so `path_points` /
  `skeleton_graph` index `lat[r, c]` / `lon[r, c]` exactly;
  `pixel_to_lonlat(rows, cols)` covers fractional pixels by bilinear index
  interpolation (error vs the projection between neighbouring 3 km cells is
  < 1 mm here, but the tables are free, so whole-pixel reads never go
  through it). `from_grid_spec` reuses ingest tables; `from_crs` recomputes
  them with pyproj, bit-identically to how `ingest.hrrr_grid.native_grid`
  built them from the same `(crs, x, y)`.
- **Inverse, lon/lat -> pixel** (`sample` reading probability/orientation
  fields at points along a smoothed line): project the point with pyproj
  (`EPSG:4326 -> crs`, longitudes wrapped to [-180, 180)), then invert the
  two monotonic 1-D arrays (`np.interp` between cell-centre coordinates,
  NaN outside their ranges -- the same off-grid contract as the 1-D
  `fractional_index`). For a tensor-product projected grid this is exact:
  lon/lat -> (x, y) is the analytic projection inverse, and (x, y) ->
  fractional index is linear between cell centres. `sample` then does
  `map_coordinates(order=1)` in index space -- the identical operation the
  regular path does in (latitude, longitude) index space.

Everything downstream of `extract_fronts`' pixel->point conversion is pure
lon/lat math (smoothing, junction pinning, tracking, GeoJSON), so the IFS and
HRRR paths share it untouched.

## Why IFS output is provably unchanged

1. **Every new behaviour is behind an optional argument** (`geom`,
   `field_geom`, `coords`) or an `ndim`/key branch that a regular lat/lon
   grid cannot take (`lat.ndim == 2`, `"lat" not in z.files`). The IFS call
   chain passes none of them, so it executes exactly the pre-existing
   statements.
2. **Byte-identical golden test** (`test_front_lines_ifs_golden.py`): a
   fixed 2-step IFS-style fixture (deterministic multi-front field with a
   junction, drifted between steps -> tracking + mixed CF/WF sections +
   orientation) is replayed through `write_front_lines` and the emitted
   GeoJSON is compared **byte for byte** against goldens generated from the
   code at the parent commit `e16d0a7` before any change landed.
3. The full pre-existing suite (448 tests, incl. all of `test_splines.py`
   and `test_front_lines.py`) passes unmodified.

## Measured cost of the HRRR front-line pass

Measured on this box (CPU-only, 8 cores shared -- foreign load inflated the
pipeline stages below; the front-line stages are single-threaded and stable),
using the REAL production pipeline offline: real weights + the pilot's cached
12Z GRIB for 2026-11-16/18/20, f000 each, `run_hrrr_cycle(..., front_lines=True)`,
scratch output root.

Per-step additions to the scheduler's cycle:

| stage | s/step |
| --- | --- |
| probs sidecar (float32, 7 x 1059 x 1799, compressed npz) | 1.8 |
| orientation fields sidecar | 0.2 |
| `write_front_lines` for one step (extract + track + GeoJSON) | 1.2 |

`write_front_lines` re-derives all published steps on every publish (the
tracking design `run_cycle` already uses for IFS), so cost grows with the
number of steps already published: 1 entry = 1.2 s, 8 = 8.1 s, 16 = 15.3 s
(marginal 0.91 s/step), 49 = 41.7 s for the final publish. Summing the
progressive pattern over a 49-step hourly cycle: **~1125 s (~19 min) added
per cycle, ~23 s average per published step** (~13 min for the 19-step 06/18Z
cycles). The real 3 km field yields 5-7 boundaries / 4.5-7.2 Mm of fronts per
step and ~9 KB of GeoJSON per step.

Context: a full HRRR cycle is ~2.2 min for 9 steps today (~15 s/step), so a
49-step cycle is ~12 min of pipeline plus ~19 min of front lines = ~31 min --
comfortably inside the 3 h cycle budget, but it is now the dominant stage.
If it ever needs to shrink, the quadratic re-derive is the lever (e.g.
tracking state keyed by step), not the extraction (1.2 s/step is cheap).

## Threshold recommendation for hourly steps

Constants are `FRONT_LINE_THRESHOLD = 0.25` (crest cutoff),
`FRONT_MATCH_THRESHOLD = 0.5` and `SplineConfig.max_motion_km = 500` /
`motion_scale_km = 50` / `match_distance_km = 500`, tuned for 6-hourly steps.

Front displacement was measured on real published fields (extracted with the
new code). Only 00-hour fields exist offline (three 12Z cycles, 48 h apart;
no network allowed), so per-step displacement is by measured mean advection:

| pair (48 h apart) | median | mean | p90 | p99 | implied km/h | est. 1 h | est. 6 h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-16 -> 11-18 | 202 km | 276 km | 543 km | 1114 km | 4.2 | ~4 km | ~25 km |
| 11-18 -> 11-20 | 258 km | 305 km | 634 km | 939 km | 5.4 | ~5 km | ~32 km |

Conclusion: **no HRRR-specific SplineConfig; shared constants unchanged.**

- `FRONT_LINE_THRESHOLD` is a probability cutoff -- independent of cadence.
- At 1 h steps fronts move ~4-5 km (median): 100x below `max_motion_km` = 500
  and far below `match_distance_km` = 500, so tracking/matching only gets
  *easier* at hourly cadence -- `FRONT_MATCH_THRESHOLD` = 0.5 stays.
- The one cadence-sensitive knob is `motion_scale_km` = 50: with ~5 km of
  true motion per step the tanh-squashed motion clue contributes ~0.1, i.e.
  the symbol sides end up decided almost entirely by the 850 hPa theta-e and
  wind clues (weight 2/3, both from the production sidecars). That is a mild
  quality loss, not a correctness one -- and at 1 h the true motion is the
  same size as line-position noise (25 km output spacing on a 3 km grid), so
  scaling the clue *up* would amplify noise as much as signal. If hourly
  orientation ever looks weak, the measured displacement above says a
  *smaller* `motion_scale_km` (~10-15) is the only defensible hourly-specific
  change; nothing measured here justifies adding one now.

## Sidecar key conventions (note for the "same filenames/keys" criterion)

Filenames, the `fronts` key, and the `oriented` semantics are identical to
IFS. Inside the sidecars the coordinate keys differ by necessity -- a
projected grid has no 1-D lat/lon: `probs.npz` = class codes + `x`/`y`/`crs`
(instead of + `lat`/`lon`), `fields.npz` = `temperature`/`u`/`v` +
`x`/`y`/`crs`. The loaders dispatch on the presence of `lat`.
