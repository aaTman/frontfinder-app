"""Disk retention: deletes old published zarr stores and cached GRIB
downloads so frontfinder doesn't slowly fill mandelhub's disk across months
of unattended systemd timer runs.

Age is derived from each entry's OWN name where possible (the IFS cycle
date embedded in a store/cache filename -- see run_cycle.py's `store_name`
and ecmwf_ifs.py's `_fetch_grib` target naming), not filesystem mtime.
mtime can be misleading: a file re-copied/rsynced/touched by something else
looks "fresh" even though the cycle it represents is old, and a directory's
top-level mtime doesn't reliably reflect "how old is the cycle it holds"
once consolidated-metadata writes touch it after creation. Anything whose
age can't be confidently determined from its name is left alone rather than
guessed at from mtime -- a retention bug that keeps too much is recoverable
disk pressure; one that deletes the wrong thing is not.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)

# "2026-08-20T18Z_f006.zarr" -> 2026-08-20 (output store directories, see
# run_cycle.py's _store_name). The `_f<NNN>` step suffix is optional in this
# pattern so that pre-2026-08-21 stores (written before the multi-step
# forecast product existed, one store per cycle rather than one per
# cycle+step) are still recognized and eventually pruned rather than
# silently orphaned as "unrecognized" forever.
_STORE_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T\d{2}Z(?:_f\d{3})?\.zarr$")
# "2026-08-20_18z_f006_t_1000.grib2" -> 2026-08-20 (GRIB cache files, see
# EcmwfOpenDataSource._fetch_grib's target naming -- the `_f<NNN>` step
# segment was added 2026-08-21 alongside the multi-step forecast product,
# but the wildcard here already covered it with no regex change needed).
_CACHE_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_\d{2}z_.*\.grib2$")


def _parse_date(name: str, pattern: re.Pattern) -> date | None:
    m = pattern.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def prune_old_output_stores(output_root: str, max_age_days: int, now: datetime) -> list[str]:
    """Deletes model output directories under `output_root/<model>/` whose
    store name encodes a cycle date older than `max_age_days`.

    Never touches `latest.json` (the webapp's pointer file -- see
    run_cycle.py's `_write_latest_pointer`) or anything whose name doesn't
    match the expected `<date>T<hour>Z.zarr` pattern.

    Returns the list of deleted store paths, for logging by the caller.
    """
    if max_age_days < 0:
        raise ValueError(f"max_age_days must be >= 0, got {max_age_days}")
    cutoff = now.date() - timedelta(days=max_age_days)
    deleted: list[str] = []
    if not os.path.isdir(output_root):
        return deleted

    for model_name in sorted(os.listdir(output_root)):
        model_dir = os.path.join(output_root, model_name)
        if not os.path.isdir(model_dir):
            continue
        for entry in sorted(os.listdir(model_dir)):
            entry_path = os.path.join(model_dir, entry)
            if entry == "latest.json" or not os.path.isdir(entry_path):
                continue
            cycle_date = _parse_date(entry, _STORE_NAME_RE)
            if cycle_date is None:
                logger.warning(
                    "retention: skipping %s -- name doesn't match the expected store pattern", entry_path
                )
                continue
            if cycle_date < cutoff:
                shutil.rmtree(entry_path)
                deleted.append(entry_path)
                logger.info(
                    "retention: deleted %s (cycle date %s, older than %d days)",
                    entry_path, cycle_date, max_age_days,
                )
    return deleted


def prune_old_cache_files(cache_dir: str, max_age_days: int, now: datetime) -> list[str]:
    """Deletes cached GRIB downloads under `cache_dir` whose embedded cycle
    date is older than `max_age_days`. Same never-guess-from-mtime policy
    as `prune_old_output_stores` -- a file whose name doesn't match the
    expected GRIB cache pattern (e.g. a cfgrib-generated `.idx` sidecar
    file) is left alone rather than pruned by mtime.

    Returns the list of deleted file paths, for logging by the caller.
    """
    if max_age_days < 0:
        raise ValueError(f"max_age_days must be >= 0, got {max_age_days}")
    cutoff = now.date() - timedelta(days=max_age_days)
    deleted: list[str] = []
    if not os.path.isdir(cache_dir):
        return deleted

    for entry in sorted(os.listdir(cache_dir)):
        entry_path = os.path.join(cache_dir, entry)
        if not os.path.isfile(entry_path):
            continue
        cycle_date = _parse_date(entry, _CACHE_NAME_RE)
        if cycle_date is None:
            continue  # not a recognized GRIB cache filename -- leave it alone
        if cycle_date < cutoff:
            os.remove(entry_path)
            deleted.append(entry_path)
            logger.info(
                "retention: deleted cached %s (cycle date %s, older than %d days)",
                entry_path, cycle_date, max_age_days,
            )
    return deleted
