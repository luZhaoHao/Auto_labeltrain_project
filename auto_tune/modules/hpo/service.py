"""H1.1 HpoService — study 创建、候选生成、结果登记与恢复幂等。

服务无 LLM/config.yaml/训练启动依赖。持久化由 :mod:`storage` 负责，采样由
:mod:`sampler` 负责；本服务在锁内组合两者并执行绑定/版本/预算/幂等规则。
"""

import hashlib
import os
import stat
import uuid
from pathlib import Path

from pydantic import ValidationError

from ..dataset_snapshot.models import SnapshotError
from ..dataset_snapshot.service import validate_dataset_snapshot
from .models import (
    ALLOWED_REASON_BY_STATE,
    EnvironmentSnapshot,
    HpoError,
    ModelBinding,
    REQUEST_ID_RE,
    ResultInput,
    ResultPayload,
    SnapshotBinding,
    StudyConfig,
    StudyRecord,
    TrialRecord,
    utc_now_iso,
)
from .sampler import sample_next
from .search_space import validate_candidate
from .storage import StudyStore, _is_reparse_point, reject_link_chain
from .validation import check_success_evidence


def _current_environment() -> dict:
    import platform
    from importlib import metadata

    def _version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return ""

    return {
        "python_version": platform.python_version(),
        "optuna_version": _version("optuna"),
        "numpy_version": _version("numpy"),
        "ultralytics_version": _version("ultralytics"),
    }


def _sha256_file(path: Path, chunk_size: int = 1048576) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class HpoService:
    """组合存储、采样与搜索空间的 HPO 服务。"""

    def __init__(self, storage_root: Path):
        self._root = Path(storage_root)
        self._store = StudyStore(self._root)

    @property
    def storage_root(self) -> Path:
        """Read-only study storage root (used by the listing route)."""
        return self._root

    # ── 创建 ────────────────────────────────────────────────────────

    def create_study(self, config, *, snapshot_dir: Path,
                     model_path: Path) -> StudyRecord:
        try:
            data = config.model_dump(mode='python', warnings=False) if isinstance(config, StudyConfig) else config
            study_config = StudyConfig.model_validate(data)
        except (ValidationError, ValueError, TypeError) as exc:
            raise HpoError('HPO_INVALID_CONFIG', 'invalid study config') from exc

        reject_link_chain(self._root, code='HPO_INVALID_CONFIG')

        snapshot = self._validate_snapshot(snapshot_dir)
        model_abs, model_bytes, model_sha, model_mtime_ns = self._validate_model_file(
            model_path)

        snapshot_binding = SnapshotBinding(
            snapshot_id=snapshot.snapshot_id,
            manifest_digest=snapshot.manifest_digest,
            snapshot_path=str(snapshot.snapshot_path),
            data_yaml_path=str(snapshot.data_yaml_path),
        )
        model_binding = ModelBinding(model_path=model_abs,
                                     model_bytes=model_bytes,
                                     model_sha256=model_sha,
                                     model_mtime_ns=model_mtime_ns)
        environment = EnvironmentSnapshot(**_current_environment())
        now = utc_now_iso()
        study_id = "hpo_" + uuid.uuid4().hex
        record = StudyRecord(
            study_id=study_id,
            config=study_config,
            created_at=now,
            updated_at=now,
            revision=0,
            snapshot_binding=snapshot_binding,
            model_binding=model_binding,
            environment=environment,
            trials=[],
        )

        study_dir = self._root / study_id
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            study_dir.mkdir(exist_ok=False)
        except OSError as exc:
            raise HpoError('HPO_PERSISTENCE_ERROR', 'cannot create study directory') from exc
        with self._store.locked(study_id):
            self._store.write(record)
        return record.model_copy(deep=True)

    def _validate_snapshot(self, snapshot_dir: Path):
        try:
            return validate_dataset_snapshot(Path(snapshot_dir))
        except SnapshotError as exc:
            raise HpoError("HPO_INVALID_CONFIG",
                           f"snapshot is not a valid published dataset: {exc}") from exc

    def _validate_model_file(self, model_path: Path):
        path = Path(model_path)
        reject_link_chain(path, code='HPO_INVALID_CONFIG')
        if path.suffix.lower() != ".pt":
            raise HpoError("HPO_INVALID_CONFIG",
                           "model_path must end with .pt")
        if _is_reparse_point(path):
            raise HpoError("HPO_INVALID_CONFIG",
                           "model_path must not be a reparse point")
        try:
            st = path.stat()
        except OSError as exc:
            raise HpoError("HPO_INVALID_CONFIG",
                           "model_path is not an accessible local file") from exc
        if not stat.S_ISREG(st.st_mode):
            raise HpoError("HPO_INVALID_CONFIG",
                           "model_path must be a regular local file")
        model_abs = os.path.abspath(os.path.normpath(str(path)))
        # 大小与 mtime 来自同一次 stat()；SHA-256 随后按内容计算
        return (model_abs, int(st.st_size), _sha256_file(path),
                int(st.st_mtime_ns))

    # ── 读取 ────────────────────────────────────────────────────────

    def load_study(self, study_id: str) -> StudyRecord:
        with self._store.locked(study_id):
            record = self._store.read(study_id)
        return record.model_copy(deep=True)

    def validate_binding(self, study_id: str) -> None:
        """公开只读绑定/环境复验，供执行器在幂等 PENDING 启动前逐文件检查。

        与 ask 内部校验一致：环境漂移抛 HPO_VERSION_MISMATCH，数据集快照或模型
        内容变化抛 HPO_BINDING_MISMATCH。不采样、不写盘。
        """
        with self._store.locked(study_id):
            record = self._store.read(study_id)
            self._check_environment(record)
            self._check_binding(record)

    # ── 候选生成 ────────────────────────────────────────────────────

    def ask(self, study_id: str, *, request_id: str) -> TrialRecord:
        if not isinstance(request_id, str) or not REQUEST_ID_RE.fullmatch(request_id):
            raise HpoError("HPO_INVALID_CONFIG",
                           "request_id must be 32 lowercase hex chars")
        with self._store.locked(study_id):
            record = self._store.read(study_id)
            for trial in record.trials:
                if trial.request_id == request_id:
                    return trial.model_copy(deep=True)
            if any(t.state == "PENDING" for t in record.trials):
                raise HpoError("HPO_PENDING_TRIAL",
                               "study already has an outstanding candidate")
            if len(record.trials) >= record.config.budget:
                raise HpoError("HPO_BUDGET_EXHAUSTED",
                               f"study budget {record.config.budget} exhausted")
            self._check_environment(record)
            self._check_binding(record)

            sampled, candidate, distributions = sample_next(
                record.config, record.trials)
            try:
                validate_candidate(candidate, epochs=record.config.epochs)
            except HpoError as exc:
                raise HpoError("HPO_CORRUPT_STUDY",
                               "sampled candidate failed strict validation") from exc

            number = len(record.trials)
            trial = TrialRecord(
                number=number,
                trial_id=f"{study_id}_t{number:04d}",
                request_id=request_id,
                state="PENDING",
                sampled_params=sampled,
                distributions=distributions,
                candidate_params=candidate,
                created_at=utc_now_iso(),
            )
            new = record.model_copy(deep=True)
            new.updated_at = utc_now_iso()
            new.revision = record.revision + 1
            new.trials.append(trial)
            self._store.write(new)
            return trial.model_copy(deep=True)

    # ── 结果登记 ────────────────────────────────────────────────────

    def tell(self, study_id: str, trial_number: int,
             result) -> TrialRecord:
        try:
            data = result.model_dump(mode='python', warnings=False) if isinstance(result, ResultInput) else result
            result_input = ResultInput.model_validate(data)
        except (ValidationError, ValueError, TypeError) as exc:
            raise HpoError('HPO_INVALID_RESULT', 'invalid result input') from exc
        if isinstance(trial_number, bool) or not isinstance(trial_number, int):
            raise HpoError("HPO_INVALID_RESULT",
                           "trial_number must be an int")

        with self._store.locked(study_id):
            record = self._store.read(study_id)
            if not (0 <= trial_number < len(record.trials)):
                raise HpoError("HPO_NOT_FOUND",
                               f"trial {trial_number} not found in study {study_id}")
            trial = record.trials[trial_number]
            if trial.state != "PENDING":
                if self._same_terminal_result(trial, result_input):
                    return trial.model_copy(deep=True)
                raise HpoError("HPO_RESULT_CONFLICT",
                               "trial already has a different terminal result")
            payload = self._validated_payload(record, result_input)
            new = record.model_copy(deep=True)
            new.updated_at = utc_now_iso()
            new.revision = record.revision + 1
            updated = new.trials[trial_number]
            updated.state = result_input.state
            updated.finished_at = utc_now_iso()
            updated.result = payload
            self._store.write(new)
            return updated.model_copy(deep=True)

    # ── 校验辅助 ────────────────────────────────────────────────────

    def _check_environment(self, record: StudyRecord) -> None:
        current = _current_environment()
        stored = record.environment
        if (current["optuna_version"] != stored.optuna_version
                or current["numpy_version"] != stored.numpy_version):
            raise HpoError("HPO_VERSION_MISMATCH",
                           "Optuna/NumPy environment changed; refusing new candidates")

    def _check_binding(self, record: StudyRecord) -> None:
        binding = record.snapshot_binding
        try:
            validated = validate_dataset_snapshot(Path(binding.snapshot_path))
        except SnapshotError as exc:
            raise HpoError("HPO_BINDING_MISMATCH",
                           "dataset snapshot binding changed") from exc
        if (validated.snapshot_id != binding.snapshot_id
                or validated.manifest_digest != binding.manifest_digest):
            raise HpoError("HPO_BINDING_MISMATCH",
                           "dataset snapshot binding changed")

        model = record.model_binding
        mpath = Path(model.model_path)
        reject_link_chain(mpath, code='HPO_BINDING_MISMATCH')
        if _is_reparse_point(mpath) or not mpath.is_file():
            raise HpoError("HPO_BINDING_MISMATCH",
                           "model binding no longer a regular local file")
        try:
            stat_result = mpath.stat()
        except OSError as exc:
            raise HpoError("HPO_BINDING_MISMATCH",
                           "model binding inaccessible") from exc
        if (stat_result.st_size != model.model_bytes
                or _sha256_file(mpath) != model.model_sha256):
            raise HpoError("HPO_BINDING_MISMATCH",
                           "model binding content changed")
        # 新研究同时冻结 mtime；旧记录没有该事实（None），继续按路径/大小/
        # SHA-256 复核，不因缺少 mtime 被判损坏
        if (model.model_mtime_ns is not None
                and int(stat_result.st_mtime_ns) != model.model_mtime_ns):
            raise HpoError("HPO_BINDING_MISMATCH",
                           "model binding modification time changed")

    def _same_terminal_result(self, trial: TrialRecord,
                              result: ResultInput) -> bool:
        if trial.state != result.state or trial.result is None:
            return False
        existing = trial.result
        evidence_ok = (existing.evidence is None and result.evidence is None) or (
            existing.evidence is not None and result.evidence is not None
            and existing.evidence.model_dump() == result.evidence.model_dump())
        return (existing.value == result.value
                and existing.reason_code == result.reason_code
                and evidence_ok)

    def _validated_payload(self, record: StudyRecord,
                           result: ResultInput) -> ResultPayload:
        state = result.state
        if state == "SUCCESS":
            if result.value is None:
                raise HpoError("HPO_INVALID_RESULT",
                               "SUCCESS requires a finite 0..1 value")
            if result.evidence is None:
                raise HpoError("HPO_INVALID_RESULT",
                               "SUCCESS requires full evidence")
            if result.reason_code is not None:
                raise HpoError("HPO_INVALID_RESULT",
                               "SUCCESS must not carry a reason_code")
            if result.evidence.epoch > record.config.epochs:
                raise HpoError("HPO_INVALID_RESULT",
                               f"evidence epoch exceeds config.epochs "
                               f"({record.config.epochs})")
            if result.evidence.evaluation_mode is not None \
                    and result.evidence.evaluation_mode != record.config.evaluation_mode:
                raise HpoError("HPO_INVALID_RESULT",
                               "evidence evaluation mode differs from the study")
            try:
                check_success_evidence(result.evidence, result.value)
            except ValueError as exc:
                raise HpoError("HPO_INVALID_RESULT",
                               "composite score does not match its components") from exc
            return ResultPayload(value=result.value,
                                 evidence=result.evidence,
                                 reason_code=None)
        if result.value is not None or result.evidence is not None:
            raise HpoError("HPO_INVALID_RESULT",
                           f"{state} must not carry value/evidence")
        if result.reason_code is None:
            raise HpoError("HPO_INVALID_RESULT",
                           f"{state} requires a reason_code")
        allowed = ALLOWED_REASON_BY_STATE[state]
        if result.reason_code not in allowed:
            raise HpoError("HPO_INVALID_RESULT",
                           f"{state} does not allow reason_code "
                           f"{result.reason_code}")
        return ResultPayload(value=None, evidence=None,
                             reason_code=result.reason_code)
