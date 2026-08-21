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
webapp/
  index.html             maplibre-gl viewer
deploy/
  Caddyfile, systemd/*   deployment config for the Proxmox host
```

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
