#!/usr/bin/env bash
# Rebuild the uint8 copy of the served zarr tree that backs
# https://fronts.espr.ai/dev/?data=dev (Caddy route /data-dev/*, see
# deploy/Caddyfile). Production /data is only READ; nothing under it changes.
#
#   scripts/refresh_dev_data.sh                      # default model
#   APP_DIR=/srv/frontfinder-app MODEL=... scripts/refresh_dev_data.sh
#
# Idempotent and cheap to re-run (e.g. from cron / after each run_cycle):
# stores already written are skipped, stores pruned from /data by retention
# are pruned here too, and latest.json is copied last.
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/frontfinder-app}"
MODEL="${MODEL:-nbs_square_dilation_trof_tt_no_inst_no_dewpoint}"

cd "$APP_DIR"
exec uv run python scripts/requantize_store.py --skip-existing --prune \
  "output/$MODEL" "output-dev/$MODEL"
