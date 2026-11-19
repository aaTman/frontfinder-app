# fronts.espr.ai webapp

Single self-contained `index.html` -- maplibre-gl + `@carbonplan/zarr-layer`,
loaded via CDN import maps (no build step). Served as static files by Caddy
on the mandelhub Proxmox VM (see `deploy/Caddyfile`).

## How it finds data

`ZARR_ROOT` (`/data` by default) is expected to be reverse-proxied by Caddy
to the same directory the scheduler (`frontfinder.scheduler.run_cycle`)
writes into: `<output_root>/<model>/<cycle>_f<step>.zarr` plus a
`<output_root>/<model>/latest.json` pointer file written after each
successful run. The frontend fetches `latest.json` for whichever model is
selected, rather than trying to list the zarr directory or guess the most
recent cycle -- this also means a failed model run for one cycle just leaves
the previous `latest.json` in place instead of showing a broken/partial map.

**2026-08-21: `latest.json` now lists every published forecast step for the
current cycle**, not just one store -- see the "Multi-step forecast
product" section in the top-level README for the full design (every 6h out
to 240h, capped at 90h for 06Z/18Z cycles). Shape:
```json
{"cycle_time": "2026-08-21T12:00:00",
 "steps": [{"step_hours": 0, "valid_time": "2026-08-21T12:00:00", "store": "2026-08-21T12Z_f000.zarr"}, ...]}
```
`resolveLatestRun()` fetches this once per model load and caches the full
`steps` array in memory (`activeSteps`) -- the time slider (`#time-control`)
just indexes into it, so dragging it doesn't re-fetch `latest.json` on every
move. The slider is hidden when a cycle only has one published step so far
(e.g. a 06Z/18Z run, or a 00Z/12Z run the timer caught mid-publish before
its longer lead times were available).

## Front lines overlay (optional)

An **Overlays -> Front lines** checkbox (off by default; `?lines=1` opens the
page with it on) draws smooth front lines over the probability layers. It is
independent of the zarr layers: its own GeoJSON source plus five layers
(`front-lines-casing`, `-solid`, `-dashed`, and two symbol layers), kept above
the raster layers, and it never touches their visibility, opacity or caching.

Data: each `latest.json` step entry may carry `"fronts": "<cycle>_f<step>.fronts.geojson"`,
written by `frontfinder/postprocess/front_lines.py` (see the top-level README).
The viewer fetches the on-screen step's file (primary model only; the last 12
are kept in memory) when the box is on and a step is revealed, and again when
new steps are polled in. A step with no `fronts` shows "(none for this step)".
The class checkboxes apply to the lines and symbols too. Lines are colored like
the probability layers; troughs and drylines are dashed, and tropical troughs are solid.

**Symbols** follow the WPC convention: cold = triangles, warm = semicircles,
occluded = both alternating on one side, stationary = blue triangles and orange
semicircles alternating on opposite sides. Each feature is split into one
LineString per part, tagged `cls` and `side` (`left`/`right` of the line walking
first point to last, from `summary.symbol_sides`); a symbol is drawn on that
side, pointing off the line. They are repeating tiles drawn on a canvas
(`drawSymbolTile`, patterns in `lib.js` `FRONT_SYMBOL_PATTERNS`), registered as
map images, and placed with `symbol-placement: line`. `symbol-spacing` is not
data-driven, so each tile width (26 px for cold/warm, 52 px for
occluded/stationary) has its own layer; `icon-keep-upright` is off so a tile's
"up" always stays the line's left. Longitudes are never cut at 180 degrees (a
front can read 175, 185), which MapLibre draws as one line.

`lib.js` holds the pure parts (`frontLinesUrl`, `withLineClasses`,
`lineColorExpression`, `lineFilter`, the symbol patterns and filters), covered by
`lib.test.js`. The
map-side code in `index.html` was checked in headless Chromium (globe
projection, slider, class toggles, on/off), which caught a bug the unit tests
cannot: MapLibre rejects `zoom` inside a `+` expression.

## Pressure overlays (optional)

Two more checkboxes in the same **Overlays** group, each off by default and
each shareable as a URL param: **Highs & lows** (`?hl=1`) draws a bold blue `H`
or red `L` at every pressure centre with its central pressure in hPa under it,
and **Isobars** (`?isobars=1`) draws the mean-sea-level isobars as thin slate
lines -- every 4 hPa, with the 20 hPa multiples (`major: true`) darker,
thicker and more opaque -- labelled with hPa along the lines. Nothing is
fetched until one of them is on; either box on fetches, both off clears the
source and stops fetching, and both read the **same** file, so they share one
GeoJSON source, one cache (the last 12 steps), one superseding fetch token and
one status line (`#pressure-status`). The two layers of isobars and the H/L
marks each take their `visibility` from their own box, so checking one shows
only that one even though the shared source holds both. Like the front lines,
only the PRIMARY model's visible step is ever drawn, and the class checkboxes do
not apply (an isobar is not a front class).

Data: each `latest.json` step entry may carry
`"pressure": "<cycle>_f<step>.pressure.geojson"`, served at
`${ZARR_ROOT}/${modelKey}/${step.pressure}` -- a sibling of the `fronts` key.
The file is a `FeatureCollection` of isobar `LineString`s
(`{"kind":"isobar","hpa":1008,"major":false}`) and centre `Point`s
(`{"kind":"L","hpa":1003}`). Isobar longitudes are never cut at 180 degrees (a
low over the pole reads 175, 185), which MapLibre draws as one line. A step
with no `pressure` key shows "(none for this step)"; a file that arrives but
has nothing for one of the two boxes says which.

Three layers on the `pressure` source, in this bottom-up order:
`pressure-isobars` (line), `pressure-isobar-labels` (symbol, `symbol-placement:
line`; below zoom 4 only the 20 hPa multiples are labelled, because one label
every 4 hPa is a smear at world scale), and `pressure-centers` (symbol, one
`format` text with the letter and the hPa stacked). They sit above the
probability rasters and below the front lines: `raiseOverlays()` (which replaces
the old `raiseFrontLines()` call after every new zarr layer) raises the pressure
layers first and the front lines second, and `ensurePressureLayers()` inserts
with `beforeId` so checking a pressure box while the front lines are already
drawn cannot put the isobars on top of them. Both symbol layers use the
basemap's own glyph stacks (`Noto Sans Bold` / `Noto Sans Regular`); a stack the
OpenFreeMap glyph server does not have 404s and the text never draws.

`lib.js` holds the pure parts (`pressureUrl`, `withPressureKinds`,
`isobarFilter`, `centerFilter`, `isobarLabelFilter`, `centerColorExpression`,
`pressureStatus`, `PRESSURE_COLORS`/`PRESSURE_FONT*`), covered by
`lib.test.js`.

## Colors, bin math, and tests (`lib.js`)

2026-08-21: `CLASS_STYLE` (the per-class hex colors), the `FRONT_*` bin
constants, and the pure helper functions (`binForBandOpacity`, `binLabel`,
`frontBinBoundaries`, `labeledTickIndices`, `formatTickLabel`,
`estimateClassAlpha`, `hexToRgb01`) moved out of `index.html`'s inline
module script into `lib.js`, a plain ES module with no DOM/browser
dependency. `index.html` imports it via a relative `./lib.js` import (no
build step, so this has to resolve as a real static file next to
`index.html` on whatever serves this directory -- Caddy does that for free).

`lib.test.js` covers `lib.js` with Node's built-in test runner: `node --test
webapp/lib.test.js` (Node >= 18.17, no dependencies to install). Covers the
color palette's shape (4 distinct hex classes, locked to the specific
colorblind-safe order below) and the bin/tick/alpha-estimation math: bin
round-tripping, tick-boundary/label generation, and the hover tooltip's
pixel-to-alpha inversion.

The 4 class colors (`#3987e5` cold / `#d95926` warm / `#199e70` stationary /
`#9085e9` occluded) replaced the original blue/red/green/purple set, which
put warm=red next to stationary=green -- a classic deuteranopia collision.
These are dark-mode slots 1, 2, 3, and 7 of the Claude Code `dataviz`
skill's reference categorical palette (occluded uses the violet slot so it
reads as purple), validated with that skill's `validate_palette.js` against
this app's dark legend surface: `node scripts/validate_palette.js
"#3987e5,#d95926,#199e70,#9085e9" --mode dark --surface "#0f121a"` passes
every check on the default adjacent-pair gate. It fails the stricter
all-pairs gate (documented in the skill's own `palette.md` as expected past
3 slots) -- accepted here because every class also carries a legend swatch,
a text label, and an independent checkbox toggle, which is the "secondary
encoding" the skill requires to accept that. Re-run the validator if these
colors ever change.

## Known API-version caveat (read before touching colors)

topozarr's docs site describes an upcoming `layer_hints` API for embedding
colormap/clim styling directly in the zarr store as `ZarrLayerVarConfig`
objects (colormap given as a *name*, e.g. `"blues"`). The pinned/installed
topozarr version (0.0.4, see `requirements.txt`) does **not** have that
API yet -- `frontfinder/zarrio/pyramid.py` instead writes `colormap`/`clim`
as plain informational zarr attrs.

Separately, `@carbonplan/zarr-layer`'s own README example takes `colormap`
as an **array of hex/rgb color stops**, not a name string. So this
frontend does not try to read the colormap name out of the zarr attrs at
all -- `CLASS_STYLE` in `index.html` hardcodes one solid hex color per
front class. If topozarr's `layer_hints` API lands and gets adopted here,
revisit this: either keep the hardcoded colors (recommended, since they're
also used for the panel swatch legend) or wire them from the zarr attrs
and drop the duplication.

**The colormap array itself has no alpha channel** -- confirmed against the
real (unminified) `dist/index.js` from the npm package, not just the
README: the shader only ever reads `.rgb` out of the colormap texture, and
per-pixel alpha comes from a single flat `opacity` uniform, the same value
for every pixel regardless of the data value there. The original version of
this file tried `colormap: ["rgba(0,0,0,0)", hex]` to fade probability=0 to
invisible and it threw live (`Invalid hex color: rgba(0,0,0,0)` --
`ZarrLayer`'s constructor validates every colormap entry as hex/`[r,g,b]`
only). Real per-pixel transparency needs `customFrag` -- see
`customClassFrag()` in `index.html`, which reimplements the library's
default single-band fragment shader body (also read out of the unminified
bundle) with `alpha = opacity * rescaled` instead of a flat `opacity`, so
probability 0 is genuinely transparent and probability 1 is fully opaque.

## Before deploying

~~Everything above the "Known API-version caveat" section was written
against real, fetched documentation... has not been opened in an actual
browser against a real zarr-layer store.~~ **Done, 2026-08-21.** Checked
live in Chrome DevTools against the real fronts.espr.ai deployment and
found two real bugs in the process, neither of them things a README could
have caught:

1. A Caddy directive-ordering bug (`try_files` silently swallowing
   `/data/*` requests before `handle_path` ever saw them) that made the
   frontend's `fetch("/data/theta-e_uv_q/latest.json")` come back with the
   webapp's own `index.html` instead of JSON -- see `deploy/Caddyfile`'s
   comment on the `handle {}` wrapping for the full story.
2. The `colormap`/alpha issue described above.

Both are now fixed and the map renders. `raster-opacity` toggling was
**not** what ended up wired up for the checkboxes -- `customClassFrag`'s
alpha ramp lives inside the shader, outside maplibre's own paint-property
system, so the checkbox handler's `map.setPaintProperty?.(...,
"raster-opacity", ...)` call is a no-op for these layers (harmless, just
dead code -- `ZarrLayer`'s own per-instance `.opacity` setter, used in the
same handler, is what actually works). Worth cleaning up the dead
`setPaintProperty` branch next time this file is touched.

## Globe projection + maplibre-gl v5 (2026-08-21)

Switched to a globe view via `map.setProjection({type: "globe"})`, per
Taylor's reference (https://hazard.degreeday.org/?layer=wildfire). This
required bumping the import map's `maplibre-gl` pin from v4 to v5 --
checked the real, installed npm package's `dist/maplibre-gl.d.ts` for both
versions rather than assume: v4.7.1 has no `setProjection` method or
`ProjectionSpecification` type at all; v5.24.0 has both. There's no
`projection` field in the `Map` constructor's `MapOptions` in either
version -- `setProjection()` after construction is the only way in.

**First deploy broke the page entirely** -- confirmed live, 2026-08-21:
calling `map.setProjection(...)` immediately after `new maplibregl.Map(...)`
(before the style finishes loading) throws synchronously:
`Error: Style is not done loading.` Since that was an uncaught exception at
module top level, it aborted the REST of the script too -- not just the
globe switch failed silently, `map.on("load", ...)` never even got
registered, so the page sat on "loading..." forever with the panel/
checkboxes rendering (they ran before the throw) but nothing else ever
happening. Fixed by moving `setProjection` inside the `"load"` event
handler, right before `loadModel()`.

## Globe projection + maplibre-gl v5 (2026-08-21)

Switched to a globe view via `map.setProjection({type: "globe"})`, per
Taylor's reference (https://hazard.degreeday.org/?layer=wildfire). This
required bumping the import map's `maplibre-gl` pin from v4 to v5 --
checked the real, installed npm package's `dist/maplibre-gl.d.ts` for both
versions rather than assume: v4.7.1 has no `setProjection` method or
`ProjectionSpecification` type at all; v5.24.0 has both. There's no
`projection` field in the `Map` constructor's `MapOptions` in either
version -- `setProjection()` after construction is the only way in.
Not yet checked live in a browser (this was implemented alongside the
multi-step backend work in the same sandbox session, and the frontend
verification loop that caught the two bugs above hasn't been re-run against
this specific change) -- do that before considering this done.

## Startup latency + uint8 stores (promoted 2026-11-16)

`index.html` carries the startup-latency work: pinned `maplibre-gl`/`zarr-layer`
versions with a flat `modulepreload` list (incl. the zstd decoder, previously
fetched only after the first chunk), `latest.json` and the first step's
metadata fetched at HTML-parse time, and data started on `style.load` instead
of the full map `load` (which waits on ~2 MB of glyphs). First data request
went from ~3.5 s to ~1.3 s on an unthrottled link.

It also renders uint8-packed probability stores (scale 1/254, 255 = NaN
sentinel; ~35x smaller) as well as float32 ones, per store, so a cycle that
mixes both is fine. Two things make that work and must stay: the layer is built
with `fillValue: 255` (zarr-layer ignores the `_FillValue` attribute for
integer dtypes and would otherwise treat the array's own fill_value 0 as
missing), and the click tooltip keeps decoded zeros (`value >= 0`).

The preload URLs pin the `esm.sh` resolution of the current dependency graph;
if `esm.sh` re-resolves a dependency, some preloads are wasted until the list
is refreshed.

Writing uint8 is opt-in on the pipeline side: `FRONTFINDER_PACK_UINT8=1` (or
`write_front_pyramid(..., pack_uint8=True)`). Note `postprocess/front_lines.py`
reads each published store back, so packing also changes the inputs to the
spline extraction (see the PR that enabled it).
