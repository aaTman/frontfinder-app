import os
from datetime import datetime, timezone

import pytest

from frontfinder.scheduler.retention import prune_old_cache_files, prune_old_output_stores


def _touch_store(output_root: str, model: str, store_name: str) -> str:
    path = os.path.join(output_root, model, store_name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        f.write("{}")
    return path


def _write_latest_pointer(output_root: str, model: str) -> str:
    path = os.path.join(output_root, model, "latest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("{}")
    return path


NOW = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)


def test_prune_old_output_stores_deletes_stores_older_than_cutoff(tmp_path):
    root = str(tmp_path)
    old_store = _touch_store(root, "best_loss", "2026-08-01T00Z.zarr")  # 20 days old
    new_store = _touch_store(root, "best_loss", "2026-08-20T18Z.zarr")  # 1 day old

    deleted = prune_old_output_stores(root, max_age_days=10, now=NOW)

    assert deleted == [old_store]
    assert not os.path.exists(old_store)
    assert os.path.exists(new_store)


def test_prune_old_output_stores_never_deletes_latest_pointer(tmp_path):
    root = str(tmp_path)
    pointer = _write_latest_pointer(root, "best_loss")
    _touch_store(root, "best_loss", "2026-08-01T00Z.zarr")

    prune_old_output_stores(root, max_age_days=10, now=NOW)

    assert os.path.exists(pointer)


def test_prune_old_output_stores_skips_unrecognized_names(tmp_path):
    root = str(tmp_path)
    weird = os.path.join(root, "best_loss", "not_a_store_dir")
    os.makedirs(weird)

    deleted = prune_old_output_stores(root, max_age_days=10, now=NOW)

    assert deleted == []
    assert os.path.exists(weird)


def test_prune_old_output_stores_handles_multiple_models_independently(tmp_path):
    root = str(tmp_path)
    old_best_loss = _touch_store(root, "best_loss", "2026-08-01T00Z.zarr")
    old_model_1702 = _touch_store(root, "model_1702", "2026-08-01T00Z.zarr")
    new_model_1702 = _touch_store(root, "model_1702", "2026-08-20T18Z.zarr")

    deleted = prune_old_output_stores(root, max_age_days=10, now=NOW)

    assert sorted(deleted) == sorted([old_best_loss, old_model_1702])
    assert os.path.exists(new_model_1702)


def test_prune_old_output_stores_handles_missing_output_root(tmp_path):
    missing = os.path.join(str(tmp_path), "does_not_exist")
    assert prune_old_output_stores(missing, max_age_days=10, now=NOW) == []


def test_prune_old_output_stores_rejects_negative_max_age():
    with pytest.raises(ValueError):
        prune_old_output_stores("/tmp/whatever", max_age_days=-1, now=NOW)


def test_prune_old_output_stores_boundary_is_exclusive_of_cutoff_day(tmp_path):
    # exactly 10 days old (cutoff == 2026-08-11) should NOT be deleted --
    # only strictly older than the cutoff is pruned.
    root = str(tmp_path)
    boundary_store = _touch_store(root, "best_loss", "2026-08-11T00Z.zarr")
    deleted = prune_old_output_stores(root, max_age_days=10, now=NOW)
    assert deleted == []
    assert os.path.exists(boundary_store)


def _touch_grib(cache_dir: str, name: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, name)
    with open(path, "w") as f:
        f.write("fake grib bytes")
    return path


def test_prune_old_cache_files_deletes_old_gribs(tmp_path):
    cache_dir = str(tmp_path)
    old = _touch_grib(cache_dir, "2026-08-01_18z_t_1000.grib2")
    new = _touch_grib(cache_dir, "2026-08-20_18z_t_1000.grib2")

    deleted = prune_old_cache_files(cache_dir, max_age_days=10, now=NOW)

    assert deleted == [old]
    assert not os.path.exists(old)
    assert os.path.exists(new)


def test_prune_old_cache_files_leaves_unrecognized_files_alone(tmp_path):
    cache_dir = str(tmp_path)
    idx_file = _touch_grib(cache_dir, "2026-08-01_18z_t_1000.grib2.923a8.idx")

    deleted = prune_old_cache_files(cache_dir, max_age_days=10, now=NOW)

    assert deleted == []
    assert os.path.exists(idx_file)


def test_prune_old_cache_files_handles_single_level_naming(tmp_path):
    # single-level fetches have an empty levelist segment -- see
    # EcmwfOpenDataSource._fetch_grib's target naming.
    cache_dir = str(tmp_path)
    old = _touch_grib(cache_dir, "2026-08-01_18z_sp_.grib2")
    deleted = prune_old_cache_files(cache_dir, max_age_days=10, now=NOW)
    assert deleted == [old]


def test_prune_old_cache_files_handles_missing_cache_dir(tmp_path):
    missing = os.path.join(str(tmp_path), "does_not_exist")
    assert prune_old_cache_files(missing, max_age_days=10, now=NOW) == []


def test_prune_old_output_stores_recognizes_step_suffixed_names(tmp_path):
    # 2026-08-21: store names now carry a "_f<NNN>" step suffix (multi-step
    # forecast product) -- this must be recognized as a valid, prunable name
    # just like the old step-less shape, not skipped as "unrecognized".
    root = str(tmp_path)
    old = _touch_store(root, "best_loss", "2026-08-01T00Z_f006.zarr")  # 20 days old
    new = _touch_store(root, "best_loss", "2026-08-20T18Z_f240.zarr")  # 1 day old

    deleted = prune_old_output_stores(root, max_age_days=10, now=NOW)

    assert deleted == [old]
    assert not os.path.exists(old)
    assert os.path.exists(new)


def test_prune_old_output_stores_still_recognizes_pre_multi_step_names(tmp_path):
    # legacy stores written before 2026-08-21 have no step suffix at all --
    # must still be prunable, not orphaned forever as "unrecognized".
    root = str(tmp_path)
    old = _touch_store(root, "best_loss", "2026-08-01T00Z.zarr")

    deleted = prune_old_output_stores(root, max_age_days=10, now=NOW)

    assert deleted == [old]


def test_prune_old_cache_files_handles_step_suffixed_naming(tmp_path):
    cache_dir = str(tmp_path)
    old = _touch_grib(cache_dir, "2026-08-01_18z_f006_t_1000.grib2")
    new = _touch_grib(cache_dir, "2026-08-20_18z_f240_t_1000.grib2")

    deleted = prune_old_cache_files(cache_dir, max_age_days=10, now=NOW)

    assert deleted == [old]
    assert os.path.exists(new)
