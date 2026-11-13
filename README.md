# frontfinder serving (fronts.espr.ai)

A batch inference and zarr serving pipeline for weather front detection. It runs
Keras models against ECMWF IFS open-data global forecasts, predicts the
probability of cold, warm, occluded, and stationary fronts, and writes the
results to a GeoZarr pyramid. A maplibre-gl web viewer at fronts.espr.ai reads
that pyramid and displays the forecast on a map, with a time slider for
stepping through lead times.

Inference runs on a schedule tied to the IFS forecast cycle (00/06/12/18Z),
not per request. Each cycle produces a forecast out to 240 hours, with a new
zarr store per lead time.

## Package layout

```
frontfinder/
  config/manifests.py    per-model variable/pressure-level manifests
  ingest/derive.py       derived-variable formulas (theta-e, isobaric PV)
  ingest/ecmwf_ifs.py    IFS field source and model-input assembly
  inference/tiling.py    patch generation and blended stitching
  inference/engine.py    tiled inference runner
  zarrio/pyramid.py      GeoZarr pyramid builder
  scheduler/run_cycle.py per-cycle, per-model orchestration
  scheduler/cli.py       systemd timer entrypoint
  scheduler/retention.py deletes old zarr stores and cached downloads
  postprocess/splines.py      front-line (spline) extraction, vendored from ai2es/fronts
  postprocess/front_lines.py  per-step GeoJSON front lines for the viewer's optional overlay
webapp/
  index.html             maplibre-gl viewer
deploy/
  Caddyfile, systemd/*   deployment config for the Proxmox host
```

## Front lines (optional overlay)

The viewer has an off-by-default **Front lines** checkbox that draws smooth,
typed front lines, with the usual front symbols, over the probability layers. They are fitted to the
probability bands by `postprocess/splines.py` (vendored from the
`feature/front-splines` branch of [ai2es/fronts](https://github.com/ai2es/fronts)
without its command-line entry point, so serving does not need the training
stack): each band is thinned to its crest, ordered into paths, smoothed with a
spline, and cut into sections by most likely type. Fronts keep one event id
across lead times.

After each step is published, `run_cycle` writes
`<model>/<cycle>_f<step>.fronts.geojson` next to the zarr store and lists it as
`"fronts"` on that step's `latest.json` entry. The zarr stores are unchanged,
and the viewer fetches a step's file only while the checkbox is on. About 0.4 s
and 100 kB per step.

- Failures here are logged and never affect inference or publishing.
- `--no-front-lines` turns generation off; `--front-line-threshold` changes the
  summed-probability cutoff for a front (default 0.25; event-matching threshold 0.5; the probabilities are
  uncalibrated, so tune it by eye against the overlay).
- Retention prunes the files with their zarr stores.
- **Symbols.** The viewer draws the standard front symbols along each line:
  cold = triangles, warm = semicircles, occluded = triangles and semicircles
  alternating, stationary = blue triangles and orange semicircles alternating on
  opposite sides. Troughs and drylines are dashed. The side of each symbol is
  chosen per line section by `splines` from the 850 hPa equivalent potential
  temperature contrast across the line, the cross-front wind, and motion since
  the previous step: cold-front triangles point into the warm air, warm-front
  semicircles into the cold air, occluded symbols the way the front moves.
- **Orientation fields.** The 850 hPa theta-e and wind are saved at inference
  time as `<model>/<cycle>_f<step>.fields.npz` (0.5 degree, ~1.5 MB each, so
  ~60 MB for a 41-step cycle; pruned with the stores), so lines can be re-derived later. Without a sidecar a
  step falls back to motion alone, which is a weak clue and does not exist for
  the first step -- so symbols there may sit on the wrong side.
- To add lines to a cycle published without them or without sidecars (the
  scheduler does both itself from the next cycle it runs), run it while no cycle
  is being published, because it rewrites `latest.json`:
  `uv run python -m frontfinder.postprocess.front_lines --output-root output
  --rebuild-fields --cache-dir ifs_cache --model-dir models`
  (about 1 s per step when the GRIBs are still in the cache; it downloads
  anything missing).
  Add `--fields-only` to write just the missing sidecars without touching
  `latest.json`: safe while a cycle is running, and the next scheduler process
  to publish a step re-derives the lines from them.

## Setup

Package management is [`uv`](https://docs.astral.sh/uv/), driven by
`pyproject.toml` and `uv.lock`. From this directory:

```
uv run --group dev pytest -q
uv run python -m frontfinder.scheduler.cli --model-dir ... --output-root ...
```

## Deployment

The pipeline runs as a systemd timer. See `deploy/systemd/` for the unit
files and `deploy/Caddyfile` for the reverse proxy config that serves the
zarr stores and web viewer.
