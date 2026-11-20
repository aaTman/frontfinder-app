#!/usr/bin/env bash
# Fast-forward the live checkout to origin/main when it is safe to do so.
# Run by frontfinder-auto-deploy.timer on LXC 108. Never modifies uncommitted
# work, never touches a running inference cycle, never runs `uv sync`.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/srv/frontfinder-app}"
GH_REPO="${GH_REPO:-aaTman/frontfinder-app}"
BRANCH=main
CYCLE_UNIT=frontfinder-run-cycle.service

log() { echo "auto-deploy: $*"; }

exec 9>/run/frontfinder-auto-deploy.lock
flock -n 9 || { log "another run in progress"; exit 0; }

cd "$REPO_DIR"
git fetch -q origin "$BRANCH"
head=$(git rev-parse HEAD)
target=$(git rev-parse "origin/$BRANCH")
[ "$head" = "$target" ] && exit 0

# Only ever fast-forward, on main, with no local edits to tracked files.
if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
  log "checkout is not on $BRANCH; skipping"; exit 0
fi
if ! git merge-base --is-ancestor "$head" "$target"; then
  log "local $BRANCH has commits not on origin; skipping"; exit 0
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  log "uncommitted changes in working tree; skipping (never overwritten)"; exit 0
fi

# CI gate: every check run on the target commit must have succeeded.
read -r total bad < <(gh api "repos/$GH_REPO/commits/$target/check-runs" --jq \
  '"\(.total_count) \([.check_runs[] | select(.status != "completed" or (.conclusion != "success" and .conclusion != "skipped"))] | length)"' \
  2>/dev/null) || { log "could not query CI for ${target:0:7}; skipping"; exit 0; }
if [ "${total:-0}" -eq 0 ] || [ "${bad:-1}" -ne 0 ]; then
  log "CI not green for ${target:0:7} (checks=${total:-0}, not-passing=${bad:-?}); skipping"; exit 0
fi

# Don't change code or the venv under a running inference cycle.
# A oneshot unit reports "activating" while running, which `is-active` calls inactive.
state=$(systemctl show -p ActiveState --value "$CYCLE_UNIT")
if [ "$state" != "inactive" ]; then
  log "$CYCLE_UNIT is $state; deferring ${target:0:7}"; exit 0
fi

changed=$(git diff --name-only "$head" "$target")
git merge --ff-only -q "$target"
log "deployed ${head:0:7} -> ${target:0:7} (webapp is static, live immediately; scheduler code applies on next cycle)"
if grep -qE '^deploy/(Caddyfile|systemd/)' <<<"$changed"; then
  log "WARNING: deploy/ config changed; Caddyfile/systemd units are NOT auto-applied, review and install manually"
fi
if grep -qE '^(pyproject.toml|uv.lock)$' <<<"$changed"; then
  log "dependencies changed; next 'uv run' resolves them (never run plain 'uv sync' here)"
fi
