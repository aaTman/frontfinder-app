# frontfinder serving (fronts.espr.ai)

Batch inference + zarr serving pipeline: runs Keras front-detection models
(`_best_loss.keras`, and `model_1702.h5` once re-enabled -- see below)
against ECMWF IFS open-data global 0.25deg fields, writes cold/warm/
occluded/stationary front probabilities to a GeoZarr pyramid (via
`topozarr`), and serves them to a maplibre-gl + `@carbonplan/zarr-layer`
viewer at fronts.espr.ai, hosted on mandelhub via Proxmox.

**model_1702 is currently disabled**, pipeline-side and on the webapp.
Live smoke-testing (2026-08-20) confirmed IFS open-data's 0.25deg feed only
publishes pressure levels `[10, 50, 100, 150, 200, 250, 300, 400, 500,
600, 700, 850, 925, 1000]` -- model_1702 was trained on ERA5's `[1000,
950, 900, 850]`, and 950hPa/900hPa simply aren't in that list. Taylor's
call: don't approximate (interpolation or nearest-level substitution) yet,
leave it off until there's a real answer. See
`frontfinder/scheduler/cli.py`'s module docstring for exactly where it
plugs back in.

**`potential_vorticity` was dropped from `best_loss`'s input variables**
(2026-08-20), per Taylor's call after `scripts/probe_ifs_native_pv.py`
confirmed live that IFS open-data has no native PV pressure-level
parameter to fetch -- and training never derived PV from anything, it came
straight from ERA5's own native `"pv"` field (see
`frontfinder/config/manifests.py`'s module docstring), so the isobaric-PV
approximation that used to fill this gap had no verified relationship to
what the model actually saw. Taylor confirmed the retrained
`_best_loss.keras` still uses `equivalent_potential_temperature` (theta-e)
-- just not PV -- so `BEST_LOSS_MANIFEST` now declares 4 variables
(theta-e/u/v/specific_humidity) x 6 levels = 24 channels, down from 30. If
it turns out the retraining changed anything else (a different level set,
an added variable), Keras will raise a shape mismatch the moment
`KerasPredictor.predict_batch` runs rather than silently misaligning
channels again -- `scripts/diagnose_model_saturation.py` exercises this
directly and is the fastest way to surface that.

## Architecture decision log (read this first)

Before any code was written, mandelhub's spec (i7-4770, 4c/8t, 16GB->32GB
RAM, no GPU, 8gbps internet) was checked against this workload. Verdict:
buildable, contingent on three things Taylor confirmed:

1. **Scheduled batch, not on-demand.** Inference runs on the IFS cycle
   cadence (00/06/12/18Z), not per user request -- CPU-only inference has
   hours of slack, not milliseconds.
2. **Most/all of mandelhub's CPU+RAM dedicated to this VM/CT.**
3. **Plenty of disk (500GB+)** for GRIB inputs + pyramid storage.

Two things surfaced during config review that change the shape of this
system and are worth restating here:

- Both training configs (`sooner_ablations.yaml`, `generate_conus.yaml`)
  use a **CONUS bounding box** (`[25.0, 56.75, 228.0, 299.75]`), not
  global. Taylor explicitly chose **true global inference** anyway --
  meaning both models are being run well outside the domain they were
  validated on. This is a modeling risk, not an engineering one: expect to
  sanity-check output quality outside CONUS before trusting it, especially
  near the poles and over open ocean far from any training analog.
- `_best_loss.keras`'s pressure levels weren't in the config
  (`n_channels: 30` implied 6 levels but didn't list them) -- Taylor
  confirmed `[1000, 925, 850, 700, 500, 300]`.

## Why no GPU is fine here, and where the CPU constraint actually shows up

`frontfinder/inference/tiling.py` + `engine.py` never run one whole-globe
forward pass. The grid is split into overlapping, 16-divisible patches
(the models' architecture constraint), inference runs patch-by-patch, and
results are blended back together -- bounding peak RAM regardless of how
large the global grid is, at the cost of some wall-clock time that's fine
given the multi-hour window between IFS cycles.

## Package layout

```
frontfinder/
  config/manifests.py     # per-model variable/pressure-level manifests (TESTED)
  ingest/derive.py         # theta-e (Bolton 1980) + isobaric PV formulas (TESTED)
  ingest/ecmwf_ifs.py       # IFS field source + model-input assembly (TESTED via fake source;
                             # EcmwfOpenDataSource itself is untested, network-only)
  inference/tiling.py       # patch generation + blended stitching (TESTED)
  inference/engine.py       # tiled inference runner (TESTED via fake predictor;
                             # KerasPredictor itself is untested, needs real weights)
  zarrio/pyramid.py         # topozarr-based GeoZarr pyramid builder (TESTED, real topozarr)
  scheduler/run_cycle.py    # per-cycle, per-model orchestration + latest.json (TESTED end-to-end
                             # with fakes)
  scheduler/cli.py          # systemd timer entrypoint (untested wiring; smoke-test on mandelhub)
  scheduler/retention.py    # deletes zarr stores + cached GRIB downloads older than N days (TESTED)
webapp/
  index.html                # maplibre-gl + @carbonplan/zarr-layer viewer, model-swap toggle
deploy/
  Caddyfile, systemd/*      # Proxmox VM deployment config
```

## Setup

Package management is [`uv`](https://docs.astral.sh/uv/), driven entirely by
`pyproject.toml` + `uv.lock` (no `requirements.txt`). From this directory:

```
uv run --group dev pytest -q   # installs into .venv on first run, then tests
uv run python -m frontfinder.scheduler.cli --model-dir ... --output-root ...
```

`uv lock` regenerates `uv.lock` after editing dependencies in
`pyproject.toml`; commit the updated lockfile alongside. `requires-python =
">=3.11"` is set by `topozarr==0.0.4`'s own requirement, not an arbitrary
choice -- `uv lock` caught this when it was first set to `>=3.10`.

87 tests (86 run + 1 skipped in this sandbox, needs real TensorFlow), all
green, `uv run --group dev pytest -q` from this directory. TDD was used
throughout: tiling, the derived-variable formulas, manifest validation, and
the full assembly/inference/pyramid/scheduler chain are all exercised
against fakes before touching real models or real network calls -- the only
things NOT covered by the test suite are the two integration points this
sandbox genuinely cannot exercise: `EcmwfOpenDataSource` (needs live
network + `cfgrib`/`eccodes`) and `KerasPredictor` (needs the real
`.keras`/`.h5` weight files and TensorFlow). Both are thin, isolated
adapters specifically so the untested surface area is as small as possible.

## Installing the systemd timer on mandelhub

```
sudo cp deploy/systemd/frontfinder-run-cycle.service deploy/systemd/frontfinder-run-cycle.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now frontfinder-run-cycle.timer
```

(no `sudo` on CT108 as set up so far -- drop it and run these as root via
`pct enter` instead, same as everything else on this container.)

Verify it's scheduled and check the next run time:
```
systemctl list-timers frontfinder-run-cycle.timer
```

Trigger one run immediately without waiting for the schedule, to confirm
the unit itself works before trusting the timer:
```
systemctl start frontfinder-run-cycle.service
journalctl -u frontfinder-run-cycle.service -f
```

The service passes `--retention-days 10` (see `scheduler/retention.py`):
after every cycle, published zarr stores and cached GRIB downloads whose
IFS cycle date is more than 10 days old are deleted automatically, so
`/srv/frontfinder/output` and `/srv/frontfinder/ifs_cache` don't grow
unbounded across months of unattended runs. `latest.json` (the webapp's
pointer to the most recent run) is never touched by pruning. Change the
`--retention-days` value in the service file if 10 days isn't the right
window once you see real disk usage.

## What's NOT done yet / needs your input before this goes live

1. ~~Smoke-test `EcmwfOpenDataSource` for real.~~ **Done, 2026-08-20**
   (`scripts/smoke_test_ecmwf.py`, run against a live cycle on mandelhub).
   Found and fixed three real issues in the process: IFS open-data's `oper`
   stream is forecast-only (`type="fc"`, never `"an"`); a single-`levelist`
   request makes cfgrib return `isobaricInhPa` as a scalar coordinate, not
   an indexable dimension; and `model_1702`'s trained levels aren't fully
   published, which is why it's disabled (see the note at the top of this
   file). `best_loss`'s full input assembly (stage 4) now passes end to end
   against real IFS data. Also switched the default source from ECMWF's
   own rate-limited portal to the AWS open-data replica (`--ifs-source`
   flag to override).
2. ~~Verify the `@carbonplan/zarr-layer` frontend against a real pyramid.~~
   **Done, 2026-08-21** -- and found a real bug in the process, not a slow
   load. Live in Chrome DevTools: the "Fronts" control panel rendered fine,
   but the map stayed black with meta text stuck on "loading...", and the
   console showed `SyntaxError: Unexpected token '<', "<!doctype "... is
   not valid JSON` -- the fetch for `/data/best_loss/latest.json` was
   getting back `webapp/index.html`'s HTML instead of JSON. Root cause:
   Caddy's directive execution order is fixed and NOT source order --
   `try_files` (an "incoming request manipulation" directive) always runs
   *before* `handle_path` ("routing & dispatching"), no matter where each
   appears in the Caddyfile. So the old config's bare top-level
   `try_files {path} /index.html` ran on every request including
   `/data/...`, found no matching file under the webapp root, and silently
   rewrote it to `/index.html` before `handle_path /data/*` ever got a
   chance to claim it. Fixed in `deploy/Caddyfile` by moving the webapp's
   `root`/`file_server`/`try_files` into its own `handle {}` block --
   `handle` and `handle_path` share one mutually-exclusive, first-match-wins
   group evaluated in source order, so `/data/*` now reliably reaches the
   zarr store root. **Needs `caddy reload` (or a copy + restart) on CT108
   to take effect** -- see the deploy note below.
3. ~~Get the model loading and producing real predictions.~~ **Done, 2026-08-20**,
   after finding and fixing three real bugs via live end-to-end
   smoke-testing (`scripts/smoke_test_full_cycle.py`,
   `scripts/diagnose_best_loss_output.py`,
   `scripts/diagnose_model_saturation.py`):
   - `_best_loss.keras`'s config referenced a custom `SharedTargetModel`
     class Keras couldn't deserialize without it being registered in this
     process first (`frontfinder/inference/keras_compat.py` vendors a
     stand-in, since only training-time hooks are overridden -- see its
     docstring for why that's safe rather than an approximation).
   - **The big one**: `assemble_model_input` stacked channels
     variable-major (all 6 levels of theta-e, then all 6 of u, ...), but
     `fronts/src/fronts/data/inputs.py`'s `inputs_ds_to_dataarray()` --
     what actually built the training tensor -- stacks level-major (every
     variable at 1000hPa, then every variable at 925hPa, ...). This fed the
     model's baked-in per-channel normalization stats to the wrong
     channels and silently saturated every prediction to background
     probability ~1.0 regardless of the real input. Fixed in
     `ModelManifest.channel_names()` / `assemble_model_input()` -- see
     their docstrings for the full postmortem.
   - `equivalent_potential_temperature` was a simplified Bolton (1980)
     approximation missing the LCL-temperature correction term; the real
     training-time formula (`fronts/src/fronts/data/derived.py`) computes
     the full eq. 43 with a dewpoint->LCL chain and a `1e-9` specific-humidity
     floor. `ingest/derive.py` now mirrors it exactly. This one still
     matters -- Taylor confirmed the retrained `_best_loss.keras` still
     uses theta-e (see the note near the top of this file), just not PV.
4. **`potential_vorticity`: resolved by removal, not by a fix.**
   `fronts/src/fronts/data/sources.py` showed PV was fetched as ERA5's own
   native `"pv"` field at training time, never derived from anything --
   unlike theta-e, there was no "real formula" for
   `potential_vorticity_isobaric` (`ingest/derive.py`'s isobaric-PV
   approximation from relative vorticity + static stability) to match
   against. `scripts/probe_ifs_native_pv.py` confirmed live (2026-08-20)
   that IFS open-data has no native PV pressure-level parameter to fetch
   either, so Taylor's call was to drop `potential_vorticity` from
   `best_loss`'s manifest entirely rather than keep serving an
   unvalidated approximation. `potential_vorticity_isobaric` itself is
   still in `ingest/derive.py`, tested, unused by either active manifest.
5. **32GB RAM upgrade** on mandelhub -- not a blocker (peak RSS during the
   2026-08-20 full-cycle smoke test was ~2.4GB), but still recommended
   before this runs unattended long-term.
6. Model weight files (`_best_loss.keras`, `model_1702.h5`) need to actually
   land on the Proxmox VM at the path `scheduler/cli.py --model-dir` points
   to -- not included in this repo.
