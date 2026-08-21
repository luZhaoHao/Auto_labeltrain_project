"""Tests for the immutable dataset snapshot domain contract (Studio S1.2)."""

import hashlib
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from auto_tune.modules.dataset_snapshot import (
    DatasetSnapshot,
    SnapshotSample,
    SnapshotError,
    SnapshotValidationError,
    SnapshotConflictError,
    SnapshotInsufficientSpaceError,
    SnapshotIOError,
)
from auto_tune.modules.dataset_snapshot.service import (
    SourceSample,
    build_snapshot_plan,
    create_dataset_snapshot,
    discover_samples,
    sha256_file,
    validate_dataset_snapshot,
)


def write_sample(root: Path, relative_image: str,
                 label_text: str | None = "0 0.5 0.5 0.2 0.2\n"):
    """Create an image (and optional sibling .txt label) under ``root``."""
    image = root / relative_image
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes((relative_image + "-image").encode())
    label = image.with_suffix(".txt")
    if label_text is not None:
        label.write_text(label_text, encoding="utf-8")
    return image, label if label_text is not None else None


def test_snapshot_error_codes_are_stable():
    cases = [
        (SnapshotValidationError("bad"), "SNAPSHOT_VALIDATION_FAILED", 400),
        (SnapshotConflictError("changed"), "SNAPSHOT_CONFLICT", 409),
        (SnapshotInsufficientSpaceError("full"), "SNAPSHOT_INSUFFICIENT_SPACE", 507),
        (SnapshotIOError("write"), "SNAPSHOT_IO_FAILED", 500),
    ]
    for error, code, status in cases:
        assert isinstance(error, SnapshotError)
        assert (error.error_code, error.status_code) == (code, status)


def test_dataset_snapshot_is_immutable(tmp_path):
    sample = SnapshotSample(
        source_image="a.jpg", source_label=None,
        snapshot_image="images/train/x_a.jpg", snapshot_label=None,
        split="train", is_background=True, image_size=3, label_size=None,
        image_sha256="abc", label_sha256=None,
    )
    snapshot = DatasetSnapshot(
        schema_version="1.0", snapshot_id="sid",
        snapshot_path=tmp_path / "sid", manifest_path=tmp_path / "sid" / "manifest.json",
        data_yaml_path=tmp_path / "sid" / "data.yaml", source_root=tmp_path / "source",
        source_layout="flat", seed=42, val_ratio=0.2,
        train_count=1, val_count=1, background_count=1, total_bytes=3,
        manifest_digest="digest", reused=False, samples=(sample,),
    )
    with pytest.raises(FrozenInstanceError):
        snapshot.snapshot_id = "changed"


# ── Task 2: safe discovery & validation ──

def test_sha256_file_is_chunked_and_stable(tmp_path):
    payload = b"hello world"
    p = tmp_path / "f.bin"
    p.write_bytes(payload)
    assert sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_discover_flat_sorted_by_relative_image(tmp_path):
    write_sample(tmp_path, "sub/b.jpg")
    write_sample(tmp_path, "a.jpg")
    write_sample(tmp_path, "sub/deep/c.jpg")
    layout, samples = discover_samples(tmp_path, {0: "defect"})
    assert layout == "flat"
    rels = [s.relative_image for s in samples]
    assert rels == sorted(rels)
    assert rels == ["a.jpg", "sub/b.jpg", "sub/deep/c.jpg"]
    assert all(isinstance(s, SourceSample) for s in samples)


def test_discover_empty_and_missing_labels_are_background(tmp_path):
    write_sample(tmp_path, "a.jpg", label_text="")
    write_sample(tmp_path, "b.jpg", label_text=None)
    write_sample(tmp_path, "c.jpg", label_text="0 0.5 0.5 0.2 0.2\n")
    layout, samples = discover_samples(tmp_path, {0: "defect"})
    by_name = {s.source_image.name: s for s in samples}
    assert layout == "flat"
    assert by_name["a.jpg"].is_background is True
    assert by_name["b.jpg"].is_background is True
    assert by_name["c.jpg"].is_background is False
    assert by_name["a.jpg"].relative_label is not None
    assert by_name["a.jpg"].label_sha256 is not None
    assert by_name["b.jpg"].relative_label is None
    assert by_name["b.jpg"].label_sha256 is None
    assert by_name["c.jpg"].relative_label is not None


def test_discover_presplit_preserves_assignment(tmp_path):
    write_sample(tmp_path, "images/train/t1.jpg", label_text=None)
    write_sample(tmp_path, "images/train/t2.jpg", label_text=None)
    write_sample(tmp_path, "images/val/v1.jpg", label_text=None)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "t1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (tmp_path / "labels" / "train" / "t2.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (tmp_path / "labels" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val" / "v1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    layout, samples = discover_samples(tmp_path, {0: "defect"})
    assert layout == "presplit"
    splits = {s.source_image.name: s.split for s in samples}
    assert splits == {"t1.jpg": "train", "t2.jpg": "train", "v1.jpg": "val"}


def test_discover_rejects_less_than_two_images(tmp_path):
    write_sample(tmp_path, "only.jpg")
    with pytest.raises(SnapshotValidationError):
        discover_samples(tmp_path, {0: "defect"})


@pytest.mark.parametrize("bad_label", [
    "0 0.5 0.5 0.2\n",                      # 4 columns
    "0 0.5 0.5 0.2 0.2 extra\n",            # 6 columns
    "-1 0.5 0.5 0.2 0.2\n",                 # negative class id
    "9 0.5 0.5 0.2 0.2\n",                  # unknown class id
    "0 nan 0.5 0.2 0.2\n",                  # NaN coordinate
    "0 0.5 inf 0.2 0.2\n",                  # Infinity coordinate
    "0 1.5 0.5 0.2 0.2\n",                  # x_center out of [0,1]
    "0 0.5 0.5 0 0.2\n",                    # zero width
    "0 0.5 0.5 0.2 -0.1\n",                 # negative height
])
def test_discover_rejects_invalid_labels(tmp_path, bad_label):
    write_sample(tmp_path, "a.jpg", label_text="0 0.5 0.5 0.2 0.2\n")
    write_sample(tmp_path, "b.jpg", label_text=bad_label)
    with pytest.raises(SnapshotValidationError):
        discover_samples(tmp_path, {0: "defect"})


def test_discover_rejects_reparse_point(tmp_path):
    write_sample(tmp_path, "a.jpg")
    write_sample(tmp_path, "b.jpg")
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    (outside / "c.jpg").write_bytes(b"fake")
    link = tmp_path / "linked.jpg"
    try:
        os.symlink(outside / "c.jpg", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")
    with pytest.raises(SnapshotValidationError):
        discover_samples(tmp_path, {0: "defect"})


# ── Task 3: deterministic split, identity & manifest plan ──

def make_four_samples(root):
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        write_sample(root, name)


def test_plan_is_deterministic_without_global_random_side_effect(tmp_path):
    import random

    make_four_samples(tmp_path)
    random.seed(999)
    before = random.getstate()
    first = build_snapshot_plan(tmp_path, 0.25, 42, {0: "defect"})
    after = random.getstate()
    second = build_snapshot_plan(tmp_path, 0.25, 42, {0: "defect"})
    assert before == after
    assert first.snapshot_id == second.snapshot_id
    assert first.manifest_digest == second.manifest_digest
    assert first.train_count == 3 and first.val_count == 1
    assert [(x.source_image, x.split) for x in first.samples] == [
        (x.source_image, x.split) for x in second.samples
    ]


def test_plan_flat_seed_or_ratio_changes_identity(tmp_path):
    make_four_samples(tmp_path)
    base = build_snapshot_plan(tmp_path, 0.25, 42, {0: "defect"})
    other_seed = build_snapshot_plan(tmp_path, 0.25, 43, {0: "defect"})
    other_ratio = build_snapshot_plan(tmp_path, 0.5, 42, {0: "defect"})
    assert base.snapshot_id != other_seed.snapshot_id
    assert base.snapshot_id != other_ratio.snapshot_id


def test_plan_presplit_identity_ignores_request_seed(tmp_path):
    write_sample(tmp_path, "images/train/a.jpg", label_text=None)
    write_sample(tmp_path, "images/val/b.jpg", label_text=None)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (tmp_path / "labels" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val" / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    plan1 = build_snapshot_plan(tmp_path, 0.25, 42, {0: "defect"})
    plan2 = build_snapshot_plan(tmp_path, 0.25, 99, {0: "defect"})
    assert plan1.source_layout == "presplit"
    assert plan1.snapshot_id == plan2.snapshot_id


def test_snapshot_filenames_unique_for_same_name_in_different_dirs(tmp_path):
    write_sample(tmp_path, "a/x.jpg")
    write_sample(tmp_path, "b/x.jpg")
    plan = build_snapshot_plan(tmp_path, 0.5, 42, {0: "defect"})
    names = [s.snapshot_image.split("/")[-1] for s in plan.samples]
    assert len(names) == len(set(names))
    # Same request reproduces the same filenames.
    again = build_snapshot_plan(tmp_path, 0.5, 42, {0: "defect"})
    assert [s.snapshot_image for s in plan.samples] == [
        s.snapshot_image for s in again.samples
    ]


@pytest.mark.parametrize("seed", [True, 42.0, "42"])
def test_plan_rejects_non_int_seed(tmp_path, seed):
    make_four_samples(tmp_path)
    with pytest.raises(SnapshotValidationError):
        build_snapshot_plan(tmp_path, 0.25, seed, {0: "defect"})


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_plan_rejects_boundary_ratio(tmp_path, ratio):
    make_four_samples(tmp_path)
    with pytest.raises(SnapshotValidationError):
        build_snapshot_plan(tmp_path, ratio, 42, {0: "defect"})


def test_plan_background_and_total_bytes(tmp_path):
    write_sample(tmp_path, "a.jpg", label_text="")
    write_sample(tmp_path, "b.jpg", label_text=None)
    write_sample(tmp_path, "c.jpg")
    plan = build_snapshot_plan(tmp_path, 0.5, 42, {0: "defect"})
    assert plan.background_count == 2
    expected = 0
    for s in plan.samples:
        expected += s.image_size
        if s.label_size is not None:
            expected += s.label_size
    assert plan.total_bytes == expected


# ── Task 4: materialization, reuse, lock, atomic publish ──

def fingerprint(root):
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_create_snapshot_copies_without_mutating_source(tmp_path):
    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    before = fingerprint(source)
    result = create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    assert fingerprint(source) == before
    assert result.snapshot_path.is_dir()
    assert result.data_yaml_path.is_file()
    assert result.manifest_path.is_file()
    assert (result.train_count, result.val_count, result.reused) == (3, 1, False)
    assert validate_dataset_snapshot(result.snapshot_path).snapshot_id == result.snapshot_id


def test_create_reuses_existing_valid_snapshot(tmp_path):
    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    first = create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    snapshot_path = first.snapshot_path
    before = fingerprint(snapshot_path)
    second = create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    assert second.snapshot_id == first.snapshot_id
    assert second.reused is True
    assert fingerprint(snapshot_path) == before


def test_create_snapshot_insufficient_space_leaves_nothing(tmp_path, monkeypatch):
    from collections import namedtuple
    import shutil as _sh
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(_sh, "disk_usage", lambda p: Usage(0, 0, 0))
    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    with pytest.raises(SnapshotInsufficientSpaceError):
        create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    if snapshots.exists():
        assert list(snapshots.iterdir()) == []


def test_create_snapshot_copy_failure_leaves_no_final(tmp_path, monkeypatch):
    def boom(src, dst, *a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(
        "auto_tune.modules.dataset_snapshot.service.shutil.copy2", boom
    )
    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    with pytest.raises(SnapshotIOError):
        create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    published = [p for p in snapshots.iterdir() if p.name != ".locks"]
    assert all(not (p / "manifest.json").exists() for p in published)


def test_create_snapshot_source_changed_during_copy_conflicts(tmp_path, monkeypatch):
    import shutil as _sh
    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    real_copy2 = _sh.copy2
    calls = {"n": 0}

    def mutating_copy2(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            source / "a.jpg"
            # mutate the first source image being copied
            (source / "a.jpg").write_bytes(b"changed-on-disk")
        return real_copy2(src, dst, *a, **k)

    monkeypatch.setattr(_sh, "copy2", mutating_copy2)
    with pytest.raises(SnapshotConflictError):
        create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})


def test_validate_rejects_tampered_snapshot(tmp_path):
    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    result = create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    img = next((result.snapshot_path / "images" / "train").glob("*.jpg"))
    img.write_bytes(b"tampered")
    with pytest.raises(SnapshotError):
        validate_dataset_snapshot(result.snapshot_path)


def test_create_rejects_tampered_existing_snapshot_without_overwrite(tmp_path):
    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    result = create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    snapshot_dir = result.snapshot_path
    img = next((snapshot_dir / "images" / "train").glob("*.jpg"))
    img.write_bytes(b"tampered")
    before = fingerprint(snapshot_dir)
    with pytest.raises(SnapshotError):
        create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    assert fingerprint(snapshot_dir) == before


def test_create_rejects_nested_snapshot_root_in_source(tmp_path):
    source = tmp_path / "source"
    make_four_samples(source)
    snapshots = source / "snapshots"
    with pytest.raises(SnapshotValidationError):
        create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})


def test_create_rejects_nested_source_in_snapshot_root(tmp_path):
    snapshots = tmp_path / "snapshots"
    source = snapshots / "source"
    make_four_samples(source)
    with pytest.raises(SnapshotValidationError):
        create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})


def test_create_snapshot_lock_timeout_conflicts(tmp_path, monkeypatch):
    import auto_tune.modules.dataset_snapshot.service as svc
    monkeypatch.setattr(svc, "LOCK_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(svc, "LOCK_POLL_INTERVAL", 0.05)
    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    plan = build_snapshot_plan(source, 0.25, 42, {0: "defect"})
    lock_dir = snapshots / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / f"{plan.snapshot_id}.lock").write_text(
        "other-process", encoding="utf-8"
    )
    with pytest.raises(SnapshotConflictError):
        create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})


# ── P0: data.yaml must resolve to this snapshot from any cwd ──

def test_data_yaml_uses_absolute_snapshot_path(tmp_path):
    import yaml
    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    result = create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    data_cfg = yaml.safe_load(result.data_yaml_path.read_text(encoding="utf-8"))
    assert data_cfg["path"] == str(result.snapshot_path.resolve())
    assert data_cfg["train"] == "images/train"
    assert data_cfg["val"] == "images/val"


def test_data_yaml_resolves_from_other_cwd(tmp_path, monkeypatch):
    """Ultralytics must resolve train/val inside the snapshot from a foreign cwd."""
    from ultralytics.data.utils import check_det_dataset

    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    result = create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    other = tmp_path / "other_cwd"
    other.mkdir(exist_ok=True)
    monkeypatch.chdir(other)
    info = check_det_dataset(str(result.data_yaml_path))
    assert info["train"] == str((result.snapshot_path / "images" / "train").resolve())
    assert info["val"] == str((result.snapshot_path / "images" / "val").resolve())


# ── P1: manifest internal paths must be strictly normalized ──

@pytest.mark.parametrize("malicious", [
    "",
    "/absolute/escape.jpg",
    "C:/absolute/escape.jpg",
    "..\\..\\escape.jpg",
    "images/train/../../escape.jpg",
    "images\\train\\escape.jpg",
])
def test_validate_rejects_malicious_manifest_paths(tmp_path, malicious):
    import json
    from auto_tune.modules.dataset_snapshot.service import _compute_manifest_digest

    source, snapshots = tmp_path / "source", tmp_path / "snapshots"
    make_four_samples(source)
    result = create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    manifest_path = result.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"][0]["snapshot_image"] = malicious
    samples = tuple(SnapshotSample(**m) for m in manifest["samples"])
    manifest["manifest_digest"] = _compute_manifest_digest(
        manifest["snapshot_id"], manifest["source_layout"],
        int(manifest["seed"]), float(manifest["val_ratio"]), samples,
        int(manifest["train_count"]), int(manifest["val_count"]),
        int(manifest["background_count"]), int(manifest["total_bytes"]),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SnapshotError):
        validate_dataset_snapshot(result.snapshot_path)
