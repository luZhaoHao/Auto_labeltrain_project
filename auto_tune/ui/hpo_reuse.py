"""H1.3 best-config reuse: rebuild a fixed-config verification from HPO facts.

The client only ever sends ``source_hpo = {study_id, trial_id}``; every actual
parameter is rebuilt server-side from the authoritative study/execution records
(re-validated rank-1 SUCCESS trial). The client candidate dictionary is never
trusted, the original HPO study budget/ranking is never modified, and the
verification training starts from the study's **bound initial weights** — never
the winner's ``best.pt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from auto_tune.modules.hpo import HpoError, rank_trials
from auto_tune.modules.hpo.execution_adapter import FIXED_PARAMS
from auto_tune.modules.hpo.search_space import validate_candidate

SEARCH_KEYS = ("optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs")


@dataclass(frozen=True)
class VerifiedHpoConfig:
    """Server-verified effective configuration for a fixed-config verification."""

    study_id: str
    trial_id: str
    trial_number: int
    value: float
    epoch: int
    effective: dict = field(default_factory=dict)
    source_public: dict = field(default_factory=dict)


def resolve_hpo_verification(service, runner, study_id, trial_id) -> VerifiedHpoConfig:
    """Re-validate the rank-1 SUCCESS trial and rebuild the full effective config.

    Raises ``HpoError`` with a stable code when the source is not usable:
    ``HPO_NOT_FOUND`` / ``HPO_NO_SUCCESS`` / ``HPO_SOURCE_INVALID`` /
    ``HPO_EXECUTION_CONFLICT`` / ``HPO_BINDING_MISMATCH`` / ``HPO_VERSION_MISMATCH``
    / ``HPO_CORRUPT_STUDY``.
    """
    if not isinstance(trial_id, str) or not trial_id:
        raise HpoError("HPO_SOURCE_INVALID", "缺少来源试验 ID")

    study = service.load_study(study_id)  # HPO_NOT_FOUND for an unknown study

    ranked = rank_trials(study)
    if not ranked:
        raise HpoError("HPO_NO_SUCCESS", "该研究没有可用的成功试验结果")
    top = ranked[0]
    if top.trial_id != trial_id:
        raise HpoError("HPO_SOURCE_INVALID", "来源不是当前排名第一的成功试验")

    # Reject a binding/environment drift before anything is built or started.
    service.validate_binding(study_id)

    try:
        validate_candidate(top.candidate_params, epochs=study.config.epochs)
    except HpoError as exc:
        raise HpoError("HPO_CORRUPT_STUDY",
                       "候选参数未通过严格校验") from exc

    try:
        execution = runner.status(study_id)
    except HpoError as exc:
        if exc.code == "HPO_NOT_FOUND":
            raise HpoError("HPO_EXECUTION_CONFLICT",
                           "该研究尚未绑定执行配置") from exc
        raise

    effective = dict(FIXED_PARAMS)
    effective["model"] = study.model_binding.model_path
    effective["data"] = study.snapshot_binding.data_yaml_path
    effective["epochs"] = study.config.epochs
    effective["seed"] = study.config.seed
    effective["batch"] = execution.config.batch
    effective["imgsz"] = execution.config.imgsz
    effective["device"] = execution.config.device
    effective.update(top.candidate_params)

    source_public = {
        "study_id": study_id,
        "trial_id": top.trial_id,
        "trial_number": top.number,
        "value": top.result.value,
        "epoch": top.result.evidence.epoch,
    }
    return VerifiedHpoConfig(
        study_id=study_id,
        trial_id=top.trial_id,
        trial_number=top.number,
        value=top.result.value,
        epoch=top.result.evidence.epoch,
        effective=effective,
        source_public=source_public,
    )
