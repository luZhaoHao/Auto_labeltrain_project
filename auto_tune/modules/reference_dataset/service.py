"""Reference dataset resolution service (Bugfix P2).

Deterministically resolves the dataset snapshot a reference training run actually
trained on and freezes it as a ``ReferenceDatasetResolution``. Resolution source
priority:

1. SQLite experiment association (``experiments.run_name -> dataset_id -> dataset``)
   verified through the strict S1.2 snapshot validation;
2. the reference run's ``args.yaml`` ``data`` pointing at a controlled
   ``log/dataset_snapshots/<snapshot_id>/data.yaml``.

The service never reads the global ``latest_dataset`` as a candidate or fallback,
never starts training, never mutates shared config, and only performs an
idempotent dataset backfill on the index after a validated args fallback. Any
unresolved, ambiguous, out-of-bounds, corrupt, or identity-conflicting state is
rejected before the tuning loop is created.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .models import (
    LocalIndexUnavailableError,
    ReferenceDatasetAmbiguousError,
    ReferenceDatasetError,
    ReferenceDatasetResolution,
    ReferenceDatasetUnresolvedError,
    ReferenceRunInvalidError,
    ReferenceSnapshotInvalidError,
)
from auto_tune.modules.dataset_snapshot import SnapshotError, validate_dataset_snapshot
from auto_tune.modules.dataset_snapshot.service import _reject_reparse_up_to
from auto_tune.modules.local_index import LocalIndexError

_SNAPSHOT_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROLLED_DATA_YAML_NAME = "data.yaml"


def _dataset_id_for_snapshot(snapshot_id: str) -> str:
    """Dataset id shared with the local-index projection for a snapshot identity."""
    return hashlib.sha256(f"snapshot:{snapshot_id}".encode("utf-8")).hexdigest()


def _validate_run_name(reference_run: str) -> None:
    if not isinstance(reference_run, str) or not reference_run:
        raise ReferenceRunInvalidError("参考运行名称不能为空")
    if reference_run in (".", ".."):
        raise ReferenceRunInvalidError("参考运行名称无效")
    if re.search(r"[/\\:\[\]\x00]", reference_run):
        raise ReferenceRunInvalidError("参考运行名称无效")
    if Path(reference_run).name != reference_run:
        raise ReferenceRunInvalidError("参考运行名称无效")


def _read_reference_args_yaml(reference_run: str, detect_dir: Path) -> dict:
    """Read the reference run's args.yaml as the authoritative fact file."""
    ref_dir = os.path.join(str(detect_dir), reference_run)
    if not os.path.isdir(ref_dir):
        raise ReferenceRunInvalidError("参考运行目录不存在")
    args_path = os.path.join(ref_dir, "args.yaml")
    if not os.path.isfile(args_path):
        raise ReferenceRunInvalidError("参考运行 args.yaml 缺失")
    try:
        with open(args_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise ReferenceRunInvalidError("参考运行 args.yaml 无法解析") from exc
    if not isinstance(data, dict):
        raise ReferenceRunInvalidError("参考运行 args.yaml 无效")
    return data


def _resolve_args_snapshot(args_yaml: dict, snapshots_root: Path) -> tuple[Path, str]:
    """Return ``(snapshot_dir, snapshot_id)`` for the args.yaml ``data`` path.

    Accepts only a controlled ``snapshots_root/<64-hex>/data.yaml`` path. The
    snapshot directory name itself carries the identity; ``..`` escapes and
    external paths are rejected.
    """
    raw = args_yaml.get("data")
    if not isinstance(raw, str) or not raw.strip():
        raise ReferenceDatasetUnresolvedError("参考运行未记录可验证的数据集")
    data_abs = os.path.abspath(str(raw))
    norm_data = os.path.normcase(os.path.normpath(data_abs))
    norm_root = os.path.normcase(os.path.normpath(os.path.abspath(str(snapshots_root))))
    if not (norm_data == norm_root or norm_data.startswith(norm_root + os.sep)):
        raise ReferenceSnapshotInvalidError("参考数据集不在受控快照目录内")
    data_yaml = Path(norm_data)
    snapshot_dir = data_yaml.parent
    if data_yaml.name != _CONTROLLED_DATA_YAML_NAME:
        raise ReferenceSnapshotInvalidError("参考数据集不是受控快照的 data.yaml")
    if not _SNAPSHOT_ID_RE.match(snapshot_dir.name):
        raise ReferenceSnapshotInvalidError("参考快照目录名无效")
    return snapshot_dir, snapshot_dir.name


def _validate_snapshot_dir(snapshot_dir: Path, snapshots_root: Path):
    """Validate a snapshot directory with the strict S1.2 algorithm.

    ``snapshot_dir`` must be the snapshot id-named child of ``snapshots_root``,
    its manifest must pass ``validate_dataset_snapshot``, the manifest identity
    must match the directory name, and no component below ``snapshots_root`` may
    be a reparse point (symlink/junction/reparse-point escape rejected).
    """
    try:
        rel = snapshot_dir.relative_to(snapshots_root.resolve(strict=False))
    except ValueError as exc:
        raise ReferenceSnapshotInvalidError("参考快照不在受控快照目录内") from exc
    if rel.parts and rel.parts[0] == "..":
        raise ReferenceSnapshotInvalidError("参考快照路径越界")
    data_yaml = snapshot_dir / _CONTROLLED_DATA_YAML_NAME
    try:
        _reject_reparse_up_to(data_yaml, snapshots_root.resolve(strict=False))
    except SnapshotError as exc:
        raise ReferenceSnapshotInvalidError("参考快照路径包含符号链接") from exc
    try:
        snapshot = validate_dataset_snapshot(snapshot_dir)
    except SnapshotError as exc:
        raise ReferenceSnapshotInvalidError("参考快照校验失败") from exc
    if snapshot.snapshot_id != snapshot_dir.name:
        raise ReferenceSnapshotInvalidError("参考快照身份不一致")
    return snapshot


def _args_snapshot_id_if_controlled(args_yaml: dict, snapshots_root: Path) -> str | None:
    """Return the args snapshot id when args.yaml points at a valid controlled
    snapshot; ``None`` when it does not (no cross-check possible)."""
    try:
        snapshot_dir, _ = _resolve_args_snapshot(args_yaml, snapshots_root)
        snapshot = _validate_snapshot_dir(snapshot_dir, snapshots_root)
        return snapshot.snapshot_id
    except ReferenceDatasetError:
        return None


def _resolve_sqlite_association(
    service,
    reference_run: str,
    snapshots_root: Path,
    args_yaml: dict,
) -> tuple[str, str, str, str] | None:
    """Resolve via the SQLite index, or return ``None`` to fall back to args.

    Raises ``ReferenceDatasetAmbiguousError`` on multiple dataset ids and
    ``ReferenceSnapshotInvalidError`` on a corrupt/inconsistent snapshot. Storage
    failures propagate as ``LocalIndexError``; the caller decides the fallback.
    """
    experiments = service.find_reference_experiments(reference_run)
    non_empty = [e for e in experiments if e.get("dataset_id")]
    ids = {e["dataset_id"] for e in non_empty}
    if len(ids) > 1:
        raise ReferenceDatasetAmbiguousError("参考运行关联多个不同数据集")
    if not ids:
        return None
    dataset_id = next(iter(ids))
    ds = service.get_dataset(dataset_id)
    if ds is None:
        return None
    snapshot_id = ds.get("snapshot_id")
    data_yaml_raw = ds.get("data_yaml_path")
    if not snapshot_id or not data_yaml_raw:
        return None
    if not _SNAPSHOT_ID_RE.match(snapshot_id):
        raise ReferenceSnapshotInvalidError("数据集快照 ID 无效")
    snapshot_dir = Path(os.path.abspath(str(data_yaml_raw))).parent
    if snapshot_dir.name != snapshot_id:
        raise ReferenceSnapshotInvalidError("数据集快照路径不一致")
    snapshot = _validate_snapshot_dir(snapshot_dir, snapshots_root)
    if snapshot.snapshot_id != snapshot_id:
        raise ReferenceSnapshotInvalidError("数据集快照身份不一致")

    args_snapshot_id = _args_snapshot_id_if_controlled(args_yaml, snapshots_root)
    if args_snapshot_id is not None and args_snapshot_id != snapshot_id:
        raise ReferenceDatasetAmbiguousError("参考运行记录与索引关联的数据集不一致")

    display_name = ds.get("display_name")
    if not isinstance(display_name, str) or not display_name:
        display_name = os.path.basename(str(snapshot.source_root).rstrip("/\\"))
    return dataset_id, snapshot_id, str(snapshot.data_yaml_path), display_name


def _backfill_dataset(service, snapshot, dataset_id: str) -> None:
    """Idempotently register a validated snapshot in the index."""
    payload = {
        "source_dataset_path": str(snapshot.source_root),
        "data_yaml_path": str(snapshot.data_yaml_path),
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_valid": True,
        "snapshot_manifest_digest": snapshot.manifest_digest,
        "display_name": os.path.basename(str(snapshot.source_root).rstrip("/\\")),
        "dataset_id": dataset_id,
    }
    service.index_dataset(payload)


def _resolve_from_args(
    reference_run: str,
    args_yaml: dict,
    snapshots_root: Path,
    local_index_service,
    sqlite_failure: LocalIndexError | None,
) -> ReferenceDatasetResolution:
    """Safe fallback: reference args.yaml points at a validated controlled snapshot."""
    snapshot_dir, _ = _resolve_args_snapshot(args_yaml, snapshots_root)
    snapshot = _validate_snapshot_dir(snapshot_dir, snapshots_root)
    dataset_id = _dataset_id_for_snapshot(snapshot.snapshot_id)
    display_name = os.path.basename(str(snapshot.source_root).rstrip("/\\"))

    index_warning = None
    if sqlite_failure is not None:
        index_warning = "本地索引不可用，已使用参考运行快照解析"
    if local_index_service is not None and index_warning is None:
        try:
            _backfill_dataset(local_index_service, snapshot, dataset_id)
        except LocalIndexError as exc:
            index_warning = "数据集已解析，但本地索引更新失败"
        except Exception:
            index_warning = "数据集已解析，但本地索引更新失败"

    return ReferenceDatasetResolution(
        reference_run=reference_run,
        dataset_id=dataset_id,
        snapshot_id=snapshot.snapshot_id,
        data_yaml_path=snapshot.data_yaml_path,
        resolution_source="reference_args",
        dataset_display_name=display_name,
        index_warning=index_warning,
    )


def _resolution_from_sqlite(
    reference_run: str,
    sqlite_result: tuple[str, str, str, str],
) -> ReferenceDatasetResolution:
    dataset_id, snapshot_id, data_yaml_path, display_name = sqlite_result
    return ReferenceDatasetResolution(
        reference_run=reference_run,
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        data_yaml_path=Path(data_yaml_path),
        resolution_source="sqlite",
        dataset_display_name=display_name or None,
    )


def resolve_reference_dataset(
    reference_run: str,
    detect_dir: Path,
    log_dir: Path,
    local_index_service=None,
) -> ReferenceDatasetResolution:
    """Resolve and freeze the dataset identity for ``reference_run``.

    Deterministic: validates the run name and fact files, tries the SQLite
    association first, then the reference args.yaml controlled-snapshot fallback.
    Never consults ``latest_dataset``; any unverifiable or conflicting state
    raises a stable ``ReferenceDatasetError``.
    """
    _validate_run_name(reference_run)
    args_yaml = _read_reference_args_yaml(reference_run, detect_dir)
    # Build through os.path.abspath so tests that redirect os.path.join("log", ...)
    # get an isolated snapshot root without touching the real project log/.
    snapshots_root = Path(os.path.abspath(os.path.join(str(log_dir), "dataset_snapshots")))

    sqlite_failure: LocalIndexError | None = None
    if local_index_service is not None:
        try:
            sqlite_result = _resolve_sqlite_association(
                local_index_service, reference_run, snapshots_root, args_yaml
            )
        except ReferenceDatasetError:
            raise
        except LocalIndexError as exc:
            sqlite_failure = exc
            sqlite_result = None
        if sqlite_result is not None:
            return _resolution_from_sqlite(reference_run, sqlite_result)

    try:
        return _resolve_from_args(
            reference_run, args_yaml, snapshots_root, local_index_service, sqlite_failure
        )
    except ReferenceDatasetError as exc:
        if sqlite_failure is not None:
            raise LocalIndexUnavailableError("本地索引不可用且参考数据集无法解析") from exc
        raise
