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
`#c98500` occluded) replaced the original blue/red/green/purple set, which
put warm=red next to stationary=green -- a classic deuteranopia collision.
These are dark-mode slots 1-4 of the Claude Code `dataviz` skill's
reference categorical palette, validated with that skill's
`validate_palette.js` against this app's dark legend surface:
`node scripts/validate_palette.js "#3987e5,#d95926,#199e70,#c98500" --mode
dark --surface "#0f121a"` passes every check on the default adjacent-pair
gate. It fails the stricter all-pairs gate (documented in the skill's own
`palette.md` as expected past 3 slots) -- accepted here because every class
also carries a legend swatch, a text label, and an independent checkbox
toggle, which is the "secondary encoding" the skill requires to accept that.
Re-run the validator if these colors ever change.

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
   frontend's `fetch("/data/best_loss/latest.json")` come back with the
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
