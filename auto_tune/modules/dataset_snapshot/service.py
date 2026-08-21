"""Dataset snapshot discovery, planning, materialization, and validation (Studio S1.2)."""

import datetime
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath

import yaml

from .models import (
    DatasetSnapshot,
    SnapshotConflictError,
    SnapshotError,
    SnapshotInsufficientSpaceError,
    SnapshotIOError,
    SnapshotSample,
    SnapshotValidationError,
)

# Supported image extensions, case-insensitive.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SCHEMA_VERSION = "1.0"

SAFETY_MIN_BYTES = 64 * 1024 * 1024
SAFETY_RATIO = 0.05

LOCK_TIMEOUT_SECONDS = 30.0
LOCK_POLL_INTERVAL = 0.2


def required_snapshot_bytes(source_bytes: int) -> int:
    """Estimate required disk bytes: source bytes plus a safety margin."""
    return source_bytes + max(SAFETY_MIN_BYTES, math.ceil(source_bytes * SAFETY_RATIO))


@dataclass(frozen=True)
class SourceSample:
    """A discovered image (and optional label) from the source directory."""

    source_image: Path
    relative_image: str
    relative_label: str | None
    split: str | None
    is_background: bool
    image_size: int
    label_size: int | None
    image_sha256: str
    label_sha256: str | None


def sha256_file(path: Path, chunk_size: int = 1048576) -> str:
    """Return the SHA-256 hex digest of ``path``, hashing in bounded chunks.

    The file size and mtime are compared before and after reading; if either
    changes the file was modified mid-read and a conflict is raised.
    """
    size_before = os.path.getsize(path)
    mtime_before = os.stat(path).st_mtime_ns
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    size_after = os.path.getsize(path)
    mtime_after = os.stat(path).st_mtime_ns
    if (size_before, mtime_before) != (size_after, mtime_after):
        raise SnapshotConflictError(f"file changed while hashing: {path}")
    return hasher.hexdigest()


def _is_reparse_point(path: Path) -> bool:
    """Return True when ``path`` is a symlink, junction, or other reparse point."""
    try:
        st = path.lstat()
    except OSError:
        return False
    if os.name == "nt":
        return bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return stat.S_ISLNK(st.st_mode)


def _reject_reparse_up_to(path: Path, root: Path) -> None:
    """Reject when any component from ``root`` down to ``path`` is a reparse point."""
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise SnapshotValidationError(f"path escapes source root: {path}") from exc
    current = root
    for part in rel.parts:
        current = current / part
        if _is_reparse_point(current):
            raise SnapshotValidationError(f"reparse point not allowed: {current}")


def _find_label(image_path: Path, root: Path) -> tuple[Path | None, str | None]:
    """Locate the label file for an image.

    Candidates are the same-stem ``.txt`` next to the image and, when the
    image lives under an ``images/`` directory, the mirrored ``labels/`` path.
    Multiple candidates are rejected as ambiguous.
    """
    rel = image_path.relative_to(root)
    parts = rel.parts
    candidates: list[Path] = []
    same_dir = image_path.with_suffix(".txt")
    if same_dir.exists():
        candidates.append(same_dir)
    if len(parts) >= 2 and parts[0] == "images":
        mirrored = root / "labels" / Path(*parts[1:]).with_suffix(".txt")
        if mirrored.exists():
            candidates.append(mirrored)
    if len(candidates) > 1:
        raise SnapshotValidationError(f"ambiguous label candidates for {image_path}")
    if not candidates:
        return None, None
    label_path = candidates[0]
    return label_path, label_path.relative_to(root).as_posix()


def _validate_label_content(content: str, class_names: dict[int, str], label_path: str) -> None:
    """Validate every non-empty line of a label file against the Detect contract."""
    for lineno, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SnapshotValidationError(f"label must have 5 columns ({label_path}:{lineno})")
        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise SnapshotValidationError(f"invalid class id ({label_path}:{lineno})") from exc
        if class_id < 0 or class_id not in class_names:
            raise SnapshotValidationError(f"class id not in class names ({label_path}:{lineno})")
        try:
            x_center, y_center, width, height = (float(v) for v in parts[1:])
        except ValueError as exc:
            raise SnapshotValidationError(f"invalid coordinate ({label_path}:{lineno})") from exc
        if not all(math.isfinite(v) for v in (x_center, y_center, width, height)):
            raise SnapshotValidationError(f"non-finite coordinate ({label_path}:{lineno})")
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            raise SnapshotValidationError(f"center out of [0,1] ({label_path}:{lineno})")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise SnapshotValidationError(f"width/height must be in (0,1] ({label_path}:{lineno})")


def _walk_images(base: Path, root: Path, split: str | None) -> list[tuple[Path, str | None]]:
    """Yield (image_path, split) pairs for supported images under ``base``.

    Directory reparse points are rejected; symlinked directories are not
    followed.
    """
    found: list[tuple[Path, str | None]] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dir_path = Path(dirpath)
        if _is_reparse_point(dir_path):
            raise SnapshotValidationError(f"reparse point not allowed: {dir_path}")
        for name in filenames:
            candidate = dir_path / name
            if candidate.suffix.lower() in IMAGE_EXTENSIONS:
                found.append((candidate, split))
    found.sort(key=lambda item: item[0].relative_to(root).as_posix())
    return found


def discover_samples(source_dir: Path, class_names: dict[int, str]) -> tuple[str, tuple[SourceSample, ...]]:
    """Discover and validate Detect samples under ``source_dir``.

    Returns ``(source_layout, samples)`` where ``source_layout`` is ``"flat"``
    or ``"presplit"`` and samples are sorted by their canonical relative path.
    """
    if not source_dir.exists():
        raise SnapshotValidationError(f"source directory does not exist: {source_dir}")
    if _is_reparse_point(source_dir):
        raise SnapshotValidationError(f"source directory is a reparse point: {source_dir}")
    source_root = source_dir.resolve(strict=True)
    if not source_root.is_dir():
        raise SnapshotValidationError(f"source is not a directory: {source_dir}")

    has_train = (source_root / "images" / "train").is_dir()
    has_val = (source_root / "images" / "val").is_dir()
    layout = "presplit" if (has_train and has_val) else "flat"

    images: list[tuple[Path, str | None]] = []
    if layout == "presplit":
        for split in ("train", "val"):
            images.extend(_walk_images(source_root / "images" / split, source_root, split))
    else:
        images = _walk_images(source_root, source_root, None)

    if len(images) < 2:
        raise SnapshotValidationError("snapshot requires at least two images")

    samples: list[SourceSample] = []
    for image_path, split in images:
        _reject_reparse_up_to(image_path, source_root)
        label_path, relative_label = _find_label(image_path, source_root)
        if label_path is not None:
            _reject_reparse_up_to(label_path, source_root)

        image_size = os.path.getsize(image_path)
        image_sha256 = sha256_file(image_path)
        label_size = None
        label_sha256 = None
        is_background = True
        if label_path is not None:
            label_size = os.path.getsize(label_path)
            label_sha256 = sha256_file(label_path)
            content = label_path.read_text(encoding="utf-8", errors="replace")
            is_background = content.strip() == ""
            if not is_background:
                _validate_label_content(content, class_names, str(label_path))

        samples.append(SourceSample(
            source_image=image_path,
            relative_image=image_path.relative_to(source_root).as_posix(),
            relative_label=relative_label,
            split=split,
            is_background=is_background,
            image_size=image_size,
            label_size=label_size,
            image_sha256=image_sha256,
            label_sha256=label_sha256,
        ))

    samples.sort(key=lambda s: s.relative_image)
    return layout, tuple(samples)


@dataclass(frozen=True)
class SnapshotPlan:
    """Deterministic plan: split assignment, identity, and manifest digest."""

    source_root: Path
    source_layout: str
    seed: int
    val_ratio: float
    train_count: int
    val_count: int
    background_count: int
    total_bytes: int
    samples: tuple[SnapshotSample, ...]
    snapshot_id: str
    manifest_digest: str


def _ratio_str(val_ratio: float) -> str:
    """Stable decimal string for a ratio (no float noise)."""
    return format(Decimal(str(val_ratio)), "f")


def _ratio_float(val_ratio: float) -> float:
    """Normalized float whose JSON serialization matches ``_ratio_str``."""
    return float(_ratio_str(val_ratio))


def _canonical_json_bytes(payload) -> bytes:
    """UTF-8, key-sorted, fixed-separator JSON bytes for stable digests."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sample_manifest_dict(sample: SnapshotSample) -> dict:
    return {
        "source_image": sample.source_image,
        "source_label": sample.source_label,
        "snapshot_image": sample.snapshot_image,
        "snapshot_label": sample.snapshot_label,
        "split": sample.split,
        "is_background": sample.is_background,
        "image_size": sample.image_size,
        "label_size": sample.label_size,
        "image_sha256": sample.image_sha256,
        "label_sha256": sample.label_sha256,
    }


def _assign_splits(
    samples: tuple[SourceSample, ...],
    layout: str,
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[SourceSample, str]], int, int]:
    """Assign deterministic splits.

    Presplit sources keep their existing train/val membership. Flat sources
    use a local ``random.Random(seed)`` shuffle; the global random state is
    never touched. Both groups must be non-empty.
    """
    if layout == "presplit":
        train = [s for s in samples if s.split == "train"]
        val = [s for s in samples if s.split == "val"]
        if not train or not val:
            raise SnapshotValidationError("presplit source must contain both train and val images")
        return [(s, s.split) for s in samples], len(train), len(val)

    rng = random.Random(seed)
    ordered = list(samples)
    rng.shuffle(ordered)
    total = len(ordered)
    val_count = min(total - 1, max(1, round(total * val_ratio)))
    train_count = total - val_count
    assigned = [(s, "train" if idx < train_count else "val") for idx, s in enumerate(ordered)]
    return assigned, train_count, val_count


def _snapshot_filename_pair(relative_image: str) -> tuple[str, str]:
    """Return (image_filename, label_filename) sharing a content-prefix.

    The prefix is the first 12 hex chars of the canonical source relative path
    so files with the same basename in different subdirectories never collide.
    """
    prefix = hashlib.sha256(relative_image.encode("utf-8")).hexdigest()[:12]
    image_name = PurePosixPath(relative_image).name
    stem = PurePosixPath(image_name).stem
    return f"{prefix}_{image_name}", f"{prefix}_{stem}.txt"


def _compute_snapshot_id(
    layout: str,
    seed: int,
    val_ratio: float,
    plan_samples: tuple[SnapshotSample, ...],
    class_names: dict[int, str],
) -> str:
    """Deterministic snapshot identity.

    Absolute source paths, mtimes, and created times are excluded. Flat
    sources include seed/ratio because they drive the split; presplit identity
    is based on the actual membership only.
    """
    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "source_layout": layout,
        "class_names": {str(k): v for k, v in sorted(class_names.items())},
        "samples": [
            {
                "source_image": s.source_image,
                "source_label": s.source_label,
                "split": s.split,
                "is_background": s.is_background,
                "image_sha256": s.image_sha256,
                "label_sha256": s.label_sha256,
            }
            for s in plan_samples
        ],
    }
    if layout == "flat":
        payload["seed"] = seed
        payload["val_ratio"] = _ratio_str(val_ratio)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _compute_manifest_digest(
    snapshot_id: str,
    layout: str,
    seed: int,
    val_ratio: float,
    plan_samples: tuple[SnapshotSample, ...],
    train_count: int,
    val_count: int,
    background_count: int,
    total_bytes: int,
) -> str:
    """Canonical digest over the manifest payload that excludes volatile fields.

    ``created_at``, ``source_root``, and ``manifest_digest`` itself never enter
    the digest, keeping it stable across local paths and creation times.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "source_layout": layout,
        "seed": seed,
        "val_ratio": _ratio_float(val_ratio),
        "train_count": train_count,
        "val_count": val_count,
        "background_count": background_count,
        "total_bytes": total_bytes,
        "samples": [_sample_manifest_dict(s) for s in plan_samples],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def build_snapshot_plan(
    source_dir: Path,
    val_ratio: float,
    seed: int,
    class_names: dict[int, str],
) -> SnapshotPlan:
    """Build a deterministic snapshot plan for ``source_dir``."""
    if type(seed) is not int:
        raise SnapshotValidationError("seed must be an integer")
    if not isinstance(val_ratio, (int, float)) or not (0.0 < val_ratio < 1.0):
        raise SnapshotValidationError("val_ratio must be a number in (0, 1)")

    source_root = source_dir.resolve(strict=True)
    layout, source_samples = discover_samples(source_dir, class_names)
    assigned, train_count, val_count = _assign_splits(source_samples, layout, val_ratio, seed)

    plan_samples: list[SnapshotSample] = []
    total_bytes = 0
    background_count = 0
    for source_sample, split in assigned:
        image_filename, label_filename = _snapshot_filename_pair(source_sample.relative_image)
        snapshot_image = f"images/{split}/{image_filename}"
        snapshot_label = (
            f"labels/{split}/{label_filename}" if source_sample.relative_label is not None else None
        )
        if source_sample.is_background:
            background_count += 1
        total_bytes += source_sample.image_size
        if source_sample.label_size is not None:
            total_bytes += source_sample.label_size
        plan_samples.append(SnapshotSample(
            source_image=source_sample.relative_image,
            source_label=source_sample.relative_label,
            snapshot_image=snapshot_image,
            snapshot_label=snapshot_label,
            split=split,
            is_background=source_sample.is_background,
            image_size=source_sample.image_size,
            label_size=source_sample.label_size,
            image_sha256=source_sample.image_sha256,
            label_sha256=source_sample.label_sha256,
        ))

    plan_samples_tuple = tuple(plan_samples)
    snapshot_id = _compute_snapshot_id(layout, seed, val_ratio, plan_samples_tuple, class_names)
    manifest_digest = _compute_manifest_digest(
        snapshot_id, layout, seed, val_ratio, plan_samples_tuple,
        train_count, val_count, background_count, total_bytes,
    )
    return SnapshotPlan(
        source_root=source_root,
        source_layout=layout,
        seed=seed,
        val_ratio=val_ratio,
        train_count=train_count,
        val_count=val_count,
        background_count=background_count,
        total_bytes=total_bytes,
        samples=plan_samples_tuple,
        snapshot_id=snapshot_id,
        manifest_digest=manifest_digest,
    )


# ── Task 4: materialization, validation, reuse, lock, atomic publish ──


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_nested(source_root: Path, snapshot_root: Path) -> None:
    """Reject when either directory contains the other."""
    source = source_root.resolve()
    snapshot = snapshot_root.resolve()
    try:
        source.relative_to(snapshot)
        raise SnapshotValidationError("snapshot root must not contain the source directory")
    except ValueError:
        pass
    try:
        snapshot.relative_to(source)
        raise SnapshotValidationError("source directory must not contain the snapshot root")
    except ValueError:
        pass


def _write_file_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` atomically (temp file in same dir + fsync + os.replace).

    On failure the temp file is intentionally left in place to match the
    snapshot failure policy (no silent cleanup of half-written output).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except OSError as exc:
        raise SnapshotIOError(f"atomic write failed: {path}") from exc


def _copy_verified(source_path: Path, dest_path: Path, expected_digest: str) -> None:
    """Copy a file and verify the copy's digest matches the source plan.

    The source is re-hashed before copying to detect mid-creation changes; the
    destination is re-hashed after copying to detect a corrupt copy.
    """
    actual = sha256_file(source_path)
    if actual != expected_digest:
        raise SnapshotConflictError(f"source changed during snapshot creation: {source_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    copied = sha256_file(dest_path)
    if copied != expected_digest:
        raise SnapshotConflictError(f"copied file digest mismatch: {dest_path}")


def _materialize_snapshot(plan: SnapshotPlan, temp_dir: Path) -> None:
    """Copy every image and label into the temp snapshot directory."""
    for sample in plan.samples:
        source_image = plan.source_root / sample.source_image
        dest_image = temp_dir / sample.snapshot_image
        _copy_verified(source_image, dest_image, sample.image_sha256)
        if sample.snapshot_label is not None:
            source_label = plan.source_root / sample.source_label
            dest_label = temp_dir / sample.snapshot_label
            _copy_verified(source_label, dest_label, sample.label_sha256)


def _write_data_yaml(data_yaml_path: Path, class_names: dict[int, str], snapshot_root: Path) -> None:
    """Write data.yaml that resolves to this snapshot from any working directory.

    ``path`` is the absolute snapshot root so Ultralytics resolves
    ``images/train`` and ``images/val`` inside the snapshot regardless of the
    directory the training process is launched from.
    """
    sorted_names = {k: v for k, v in sorted(class_names.items())}
    payload = {
        "path": str(snapshot_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(sorted_names),
        "names": sorted_names,
    }
    text = yaml.safe_dump(payload, default_flow_style=False, allow_unicode=True, sort_keys=False)
    _write_file_atomic(data_yaml_path, text.encode("utf-8"))


def _write_manifest(manifest_path: Path, plan: SnapshotPlan) -> None:
    """Write the snapshot manifest with a stable canonical digest."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": plan.snapshot_id,
        "created_at": _utc_now_iso(),
        "source_root": str(plan.source_root),
        "source_layout": plan.source_layout,
        "seed": plan.seed,
        "val_ratio": _ratio_float(plan.val_ratio),
        "train_count": plan.train_count,
        "val_count": plan.val_count,
        "background_count": plan.background_count,
        "total_bytes": plan.total_bytes,
        "samples": [_sample_manifest_dict(s) for s in plan.samples],
        "manifest_digest": plan.manifest_digest,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    _write_file_atomic(manifest_path, text.encode("utf-8"))


def _validate_data_yaml(data_yaml_path: Path, snapshot_root: Path, require_dirs: bool = False) -> None:
    """Verify data.yaml resolves strictly inside the snapshot root."""
    try:
        with open(data_yaml_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SnapshotValidationError(f"data.yaml unreadable: {data_yaml_path}") from exc
    if not isinstance(data, dict):
        raise SnapshotValidationError("data.yaml must be a mapping")
    root = snapshot_root.resolve()
    path_val = data.get("path")
    if not isinstance(path_val, str) or not path_val:
        raise SnapshotValidationError("data.yaml path must be a non-empty string")
    path_resolved = Path(path_val)
    if not path_resolved.is_absolute():
        raise SnapshotValidationError("data.yaml path must be absolute")
    if path_resolved.resolve() != root:
        raise SnapshotValidationError("data.yaml path must point at the snapshot root")
    if data.get("train") != "images/train":
        raise SnapshotValidationError("data.yaml train must be images/train")
    if data.get("val") != "images/val":
        raise SnapshotValidationError("data.yaml val must be images/val")
    for key in ("train", "val"):
        resolved = (root / data[key]).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise SnapshotValidationError(f"data.yaml {key} escapes the snapshot root") from exc
        if require_dirs and not resolved.is_dir():
            raise SnapshotValidationError(f"data.yaml {key} directory missing: {resolved}")
    names = data.get("names") or {}
    if not isinstance(names, dict) or not names:
        raise SnapshotValidationError("data.yaml names must be a non-empty mapping")
    try:
        int_ids = [int(k) for k in names]
    except (TypeError, ValueError) as exc:
        raise SnapshotValidationError("data.yaml names keys must be integer ids") from exc
    if sorted(int_ids) != list(range(len(int_ids))):
        raise SnapshotValidationError("data.yaml names ids must be contiguous from 0")
    if data.get("nc") != len(int_ids):
        raise SnapshotValidationError("data.yaml nc must match the number of names")


def _result_from_plan(plan: SnapshotPlan, final_dir: Path, reused: bool) -> DatasetSnapshot:
    return DatasetSnapshot(
        schema_version=SCHEMA_VERSION,
        snapshot_id=plan.snapshot_id,
        snapshot_path=final_dir,
        manifest_path=final_dir / "manifest.json",
        data_yaml_path=final_dir / "data.yaml",
        source_root=plan.source_root,
        source_layout=plan.source_layout,
        seed=plan.seed,
        val_ratio=plan.val_ratio,
        train_count=plan.train_count,
        val_count=plan.val_count,
        background_count=plan.background_count,
        total_bytes=plan.total_bytes,
        manifest_digest=plan.manifest_digest,
        reused=reused,
        samples=plan.samples,
    )


class _SnapshotAlreadyPublished(Exception):
    """Internal signal: a valid identical snapshot was published while waiting."""


def _acquire_snapshot_lock(lock_path: Path, snapshot_id: str, final_dir: Path) -> str:
    """Acquire an exclusive per-snapshot lock; reuse the final snapshot if published."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = os.urandom(8).hex()
    payload = {
        "snapshot_id": snapshot_id,
        "pid": os.getpid(),
        "created_at": _utc_now_iso(),
        "token": token,
    }
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if final_dir.exists():
                existing = validate_dataset_snapshot(final_dir)
                if existing.snapshot_id == snapshot_id:
                    raise _SnapshotAlreadyPublished()
                raise SnapshotValidationError(
                    f"existing snapshot {snapshot_id} failed validation"
                )
            if time.monotonic() >= deadline:
                raise SnapshotConflictError(
                    f"snapshot creation conflict for {snapshot_id}: lock timeout"
                )
            time.sleep(LOCK_POLL_INTERVAL)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return token


def _release_snapshot_lock(lock_path: Path, token: str) -> None:
    """Remove the lock only when it is still owned by this caller's token."""
    try:
        with open(lock_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("token") == token:
            lock_path.unlink()
    except (OSError, ValueError):
        pass


def create_dataset_snapshot(
    source_dir: Path,
    snapshot_root: Path,
    val_ratio: float,
    seed: int,
    class_names: dict[int, str],
) -> DatasetSnapshot:
    """Materialize an immutable, validated snapshot of ``source_dir``.

    The source directory is never modified. Identical requests reuse an
    already-published snapshot without re-copying. A temp directory under the
    snapshot root is atomically renamed only after full verification.
    """
    source_path = Path(source_dir)
    snap_root = Path(snapshot_root)
    _reject_nested(source_path, snap_root)

    plan = build_snapshot_plan(source_path, val_ratio, seed, class_names)
    final_dir = snap_root / plan.snapshot_id

    # Reuse a valid identical snapshot; never overwrite a corrupted one.
    if final_dir.exists():
        existing = validate_dataset_snapshot(final_dir)
        if existing.snapshot_id == plan.snapshot_id:
            return _result_from_plan(plan, final_dir, reused=True)
        raise SnapshotValidationError(f"existing snapshot {plan.snapshot_id} failed validation")

    # Disk-space precheck before creating any temp directory.
    disk_path = snap_root if snap_root.exists() else snap_root.parent
    required = required_snapshot_bytes(plan.total_bytes)
    free_bytes = shutil.disk_usage(disk_path).free
    if free_bytes < required:
        raise SnapshotInsufficientSpaceError(
            f"insufficient disk space: need {required} bytes, only {free_bytes} free"
        )

    lock_path = snap_root / ".locks" / f"{plan.snapshot_id}.lock"
    token: str | None = None
    try:
        token = _acquire_snapshot_lock(lock_path, plan.snapshot_id, final_dir)
    except _SnapshotAlreadyPublished:
        return _result_from_plan(plan, final_dir, reused=True)

    try:
        # Re-check after acquiring the lock (a concurrent creator may have finished).
        if final_dir.exists():
            existing = validate_dataset_snapshot(final_dir)
            if existing.snapshot_id == plan.snapshot_id:
                return _result_from_plan(plan, final_dir, reused=True)
            raise SnapshotValidationError(
                f"existing snapshot {plan.snapshot_id} failed validation"
            )

        temp_dir = snap_root / f"{plan.snapshot_id}.tmp-{os.urandom(8).hex()}"
        try:
            temp_dir.mkdir(parents=False)
            _materialize_snapshot(plan, temp_dir)
            data_yaml_path = temp_dir / "data.yaml"
            _write_data_yaml(data_yaml_path, class_names, snapshot_root=final_dir)
            manifest_path = temp_dir / "manifest.json"
            _write_manifest(manifest_path, plan)
            # Self-check inside the temp dir; data.yaml points at the final root.
            validate_dataset_snapshot(temp_dir, expected_root=final_dir, require_data_dirs=False)
            os.replace(temp_dir, final_dir)
        except (SnapshotError, OSError) as exc:
            # The temp dir is intentionally left in place (snapshot failure policy).
            if isinstance(exc, SnapshotError):
                raise
            raise SnapshotIOError(f"snapshot materialization failed: {exc}") from exc
    finally:
        if token is not None:
            _release_snapshot_lock(lock_path, token)

    validate_dataset_snapshot(final_dir)
    return _result_from_plan(plan, final_dir, reused=False)


def _validate_manifest_relpath(rel: str, snap_root: Path, what: str) -> None:
    """Reject malicious manifest paths: absolute, ``..``, backslashes, escapes."""
    if not isinstance(rel, str) or not rel:
        raise SnapshotValidationError(f"manifest {what} must be a non-empty relative path")
    if "\\" in rel:
        raise SnapshotValidationError(f"manifest {what} must use forward slashes: {rel!r}")
    p = Path(rel)
    if p.is_absolute():
        raise SnapshotValidationError(f"manifest {what} must be relative: {rel!r}")
    parts = p.parts
    if any(part in (".", "..") for part in parts):
        raise SnapshotValidationError(f"manifest {what} must be normalized: {rel!r}")
    resolved = (snap_root / p).resolve(strict=False)
    try:
        resolved.relative_to(snap_root.resolve(strict=False))
    except ValueError as exc:
        raise SnapshotValidationError(f"manifest {what} escapes the snapshot root: {rel!r}") from exc


def validate_dataset_snapshot(
    snapshot_dir: Path,
    expected_root: Path | None = None,
    require_data_dirs: bool = True,
) -> DatasetSnapshot:
    """Validate a published snapshot: manifest, digests, files, and data.yaml.

    ``expected_root`` lets a still-unpublished temp snapshot declare the final
    root its data.yaml points at; ``require_data_dirs`` skips existence checks
    before the atomic publish has happened.
    """
    snap_path = Path(snapshot_dir)
    data_root = Path(expected_root) if expected_root is not None else snap_path
    manifest_path = snap_path / "manifest.json"
    data_yaml_path = snap_path / "data.yaml"

    if _is_reparse_point(snap_path):
        raise SnapshotValidationError(f"snapshot directory is a reparse point: {snap_path}")
    if not manifest_path.is_file():
        raise SnapshotValidationError(f"manifest missing: {manifest_path}")
    if not data_yaml_path.is_file():
        raise SnapshotValidationError(f"data.yaml missing: {data_yaml_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SnapshotValidationError(f"manifest unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise SnapshotValidationError("manifest must be a JSON object")

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotValidationError("unsupported manifest schema_version")
    snapshot_id = manifest.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot_id):
        raise SnapshotValidationError("manifest has an invalid snapshot_id")

    digest_payload = {
        k: v for k, v in manifest.items()
        if k not in ("created_at", "source_root", "manifest_digest")
    }
    actual_digest = hashlib.sha256(_canonical_json_bytes(digest_payload)).hexdigest()
    if actual_digest != manifest.get("manifest_digest"):
        raise SnapshotValidationError("manifest digest mismatch")

    samples_raw = manifest.get("samples")
    if not isinstance(samples_raw, list):
        raise SnapshotValidationError("manifest samples must be a list")

    for sample in samples_raw:
        if not isinstance(sample, dict):
            raise SnapshotValidationError("manifest sample must be an object")
        image_rel = sample.get("snapshot_image")
        if not isinstance(image_rel, str):
            raise SnapshotValidationError("manifest sample missing snapshot_image")
        _validate_manifest_relpath(image_rel, snap_path, "snapshot_image")
        image_path = snap_path / image_rel
        _reject_reparse_up_to(image_path, snap_path)
        if not image_path.is_file():
            raise SnapshotValidationError(f"snapshot image missing: {image_rel}")
        if os.path.getsize(image_path) != sample.get("image_size"):
            raise SnapshotValidationError(f"snapshot image size mismatch: {image_rel}")
        if sha256_file(image_path) != sample.get("image_sha256"):
            raise SnapshotValidationError(f"snapshot image digest mismatch: {image_rel}")

        label_rel = sample.get("snapshot_label")
        if label_rel is not None:
            if not isinstance(label_rel, str):
                raise SnapshotValidationError("manifest sample label must be a string")
            _validate_manifest_relpath(label_rel, snap_path, "snapshot_label")
            label_path = snap_path / label_rel
            _reject_reparse_up_to(label_path, snap_path)
            if not label_path.is_file():
                raise SnapshotValidationError(f"snapshot label missing: {label_rel}")
            if os.path.getsize(label_path) != sample.get("label_size"):
                raise SnapshotValidationError(f"snapshot label size mismatch: {label_rel}")
            if sha256_file(label_path) != sample.get("label_sha256"):
                raise SnapshotValidationError(f"snapshot label digest mismatch: {label_rel}")

    train_count = manifest.get("train_count", 0)
    val_count = manifest.get("val_count", 0)
    if len(samples_raw) != train_count + val_count:
        raise SnapshotValidationError("manifest sample count does not match train/val counts")

    _validate_data_yaml(data_yaml_path, data_root, require_dirs=require_data_dirs)

    samples: list[SnapshotSample] = []
    for sample in samples_raw:
        try:
            samples.append(SnapshotSample(**sample))
        except TypeError as exc:
            raise SnapshotValidationError("manifest sample has unexpected fields") from exc

    return DatasetSnapshot(
        schema_version=SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        snapshot_path=snap_path,
        manifest_path=manifest_path,
        data_yaml_path=data_yaml_path,
        source_root=Path(manifest.get("source_root") or "."),
        source_layout=manifest.get("source_layout", "flat"),
        seed=manifest.get("seed", 0),
        val_ratio=manifest.get("val_ratio", 0.2),
        train_count=train_count,
        val_count=val_count,
        background_count=manifest.get("background_count", 0),
        total_bytes=manifest.get("total_bytes", 0),
        manifest_digest=manifest["manifest_digest"],
        reused=False,
        samples=tuple(samples),
    )
