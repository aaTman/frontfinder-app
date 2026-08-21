# fronts.espr.ai webapp

Single self-contained `index.html` -- maplibre-gl + `@carbonplan/zarr-layer`,
loaded via CDN import maps (no build step). Served as static files by Caddy
on the mandelhub Proxmox VM (see `deploy/Caddyfile`).

## How it finds data

`ZARR_ROOT` (`/data` by default) is expected to be reverse-proxied by Caddy
to the same directory the scheduler (`frontfinder.scheduler.run_cycle`)
writes into: `<output_root>/<model>/<cycle>.zarr` plus a
`<output_root>/<model>/latest.json` pointer file written after each
successful run. The frontend fetches `latest.json` for whichever model is
selected, rather than trying to list the zarr directory or guess the most
recent cycle -- this also means a failed model run for one cycle just leaves
the previous `latest.json` in place instead of showing a broken/partial map.

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
