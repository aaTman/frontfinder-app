# frontfinder-app (fronts.espr.ai)

## Goal
Batch inference and zarr serving pipeline for ESPR's Fronts product. Every IFS cycle it runs the FrontFinder model on ECMWF open data, writes immutable zarr pyramids plus `latest.json` pointers, and a static MapLibre web app shows them live at fronts.espr.ai. Used by Taylor and forecasters. The training code is a separate repo (`aaTman/fronts`).

## Stack
- Language/framework: Python >=3.11 (`uv`), TensorFlow/Keras 3, xarray, zarr 3, `topozarr==0.0.4` (pinned, pre-1.0), `ecmwf-opendata`, cfgrib/eccodes. Frontend: static `webapp/` (`index.html`, `lib.js`), no build step.
- Data: ECMWF IFS open data cached in `ifs_cache/`; outputs in `output/` as `<cycle>Z_f<step>.zarr` stores, immutable once written and pruned wholesale by retention. Model weights in `models/`.
- Compute: runs on LXC 108 `frontfinder` (192.168.0.19), a systemd timer (`frontfinder-run-cycle.timer`, every 5 min, `uv run ... --poll`). Caddy serves `webapp/` and `/data/*`; Cloudflare Tunnel publishes it.

## Commands
- Setup: do not run a plain `uv sync` on LXC 108 (see Do not). In CI/dev: `uv sync --frozen --group dev`.
- Test: `just test` (pytest, `frontfinder/tests`). Frontend: `node --test webapp/lib.test.js` needs Node 22.
- Lint: none configured yet.
- Train/run: `uv run` the scheduler module with `--poll` (what the timer does); smoke tests in `scripts/smoke_test_*.py`.
- Deploy: `just deploy` (stub).

## Conventions
- Python style: type hints, short docstrings. Branching: one worktree per task, PRs into main, CI must pass.
- Never commit data, keys, model weights, `output/` or `ifs_cache/`.
- A running cycle keeps the module versions it loaded; new code applies to the next scheduler process.
- `test_run_cycle_dispatches_models_concurrently_when_devices_differ` is wall-clock based and flakes under live load; rerun it alone before suspecting a regression.

## Do not
- Run plain `uv sync` on LXC 108: it is exact and uninstalls hand-installed packages (tensorflow-cpu, matplotlib, ...). Use `uv sync --inexact` or `uv pip install`. `uv run` is safe.
- `pkill caddy` there: it kills the production Caddy. Kill test instances by exact PID.
- Commit, revert or overwrite the uncommitted edits in the LXC 108 working tree unless Taylor asks.
- Interrupt a running inference process (a killed mid-write zarr directory is trusted as complete); check `systemctl is-active frontfinder-run-cycle.service` before touching `latest.json` or the venv.
- Use CARTO basemaps; use MapLibre + OpenFreeMap positron.
- Deploy directly; open a PR and let CI gate it.

## Current state
See docs/decisions.md for decisions and open questions. Known gap: the live tree on LXC 108 is ahead of `main` (uncommitted front-lines/splines work).
