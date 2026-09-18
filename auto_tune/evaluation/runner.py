"""受控回放运行器：对每个离线案例跑一遍**真实**决策链路并记录全部中间产物。

链路与生产完全一致：``build_perception`` → ``build_tuning_fact_package`` →
``decide_hyperparameters``（含一次受控纠错）→ ``sanitize_and_merge_tuning_params``。

刻意复用生产的 ``_read_reference_before_metrics``：before 指标必须与
``baseline.params`` 来自同一个参考运行，评估口径不能与生产口径分叉。

不变量：本模块只读生产代码，绝不修改 fact package / decision / 任何生产文件；
唯一写盘位置是调用方指定的工作区与产物目录。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from auto_tune.modules.agent_engine import decision_agent
from auto_tune.modules.agent_engine.decision_agent import decide_hyperparameters
from auto_tune.modules.agent_engine.decision_facts import (
    FactPackageError,
    build_tuning_fact_package,
)
from auto_tune.modules.agent_engine.executor import read_args_yaml
from auto_tune.modules.agent_engine.loop import (
    _read_reference_before_metrics,
    sanitize_and_merge_tuning_params,
)
from auto_tune.modules.agent_engine.perception import (
    build_perception,
    perception_blocking_error,
)

from .cases import EvalCase
from .workspace import CaseWorkspace, materialize

# 与生产 loop 一致的终态归类，便于逐项对账失败类别。
OUTCOME_VALID = "valid"
OUTCOME_CONTRACT_FAILED = "contract_failed"
OUTCOME_SEMANTIC_FAILED = "semantic_failed"
OUTCOME_PROVIDER_ERROR = "provider_error"
OUTCOME_FACT_PACKAGE_INVALID = "fact_package_invalid"
OUTCOME_PERCEPTION_BLOCKED = "perception_blocked"


@dataclass
class PreparedCase:
    """不含任何 LLM 调用的确定性准备结果。"""

    case: EvalCase
    workspace: CaseWorkspace
    perception: dict
    base_args: dict
    before_metrics: dict
    metrics_source: dict
    fact_package: dict | None
    fact_error: str | None = None
    blocking_code: str | None = None


@dataclass
class CaseRunRecord:
    case_id: str
    run_index: int
    timestamp: str
    reference_run: str
    fact_package_id: str | None
    fact_ids: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    before_metrics: dict = field(default_factory=dict)
    base_args: dict = field(default_factory=dict)
    llm_calls: list[dict] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    decision: dict | None = None
    guardrails: dict | None = None
    outcome: str = OUTCOME_CONTRACT_FAILED
    error_code: str | None = None
    would_start_training: bool = False

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "run_index": self.run_index,
            "timestamp": self.timestamp,
            "reference_run": self.reference_run,
            "fact_package_id": self.fact_package_id,
            "fact_ids": list(self.fact_ids),
            "facts": dict(self.facts),
            "before_metrics": dict(self.before_metrics),
            "base_args": dict(self.base_args),
            "llm_calls": list(self.llm_calls),
            "attempts": list(self.attempts),
            "decision": self.decision,
            "guardrails": self.guardrails,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "would_start_training": self.would_start_training,
        }


def prepare_case(case: EvalCase, root: str) -> PreparedCase:
    """物化案例并冻结事实包。只读生产代码，不发起任何 LLM 调用。"""
    workspace = materialize(case, root)
    perception = build_perception(
        log_dir=workspace.log_dir, reference_run=case.reference_run,
    )
    run_dir = os.path.join(workspace.detect_dir, case.reference_run)
    base_args = read_args_yaml(run_dir)
    before_metrics, metrics_source = _read_reference_before_metrics(
        case.reference_run, workspace.detect_dir,
    )

    blocking_code, _ = perception_blocking_error(perception)
    if blocking_code:
        return PreparedCase(
            case=case, workspace=workspace, perception=perception,
            base_args=base_args, before_metrics=before_metrics,
            metrics_source=metrics_source, fact_package=None,
            blocking_code=blocking_code,
        )

    try:
        fact_package = build_tuning_fact_package(
            perception, case.reference_run, base_args, before_metrics, metrics_source,
        )
    except FactPackageError as exc:
        return PreparedCase(
            case=case, workspace=workspace, perception=perception,
            base_args=base_args, before_metrics=before_metrics,
            metrics_source=metrics_source, fact_package=None,
            fact_error=exc.detail,
        )

    return PreparedCase(
        case=case, workspace=workspace, perception=perception,
        base_args=base_args, before_metrics=before_metrics,
        metrics_source=metrics_source, fact_package=fact_package,
    )


class _RecordingLLM:
    """临时包裹 ``call_decision_llm``，记录每次调用的提示词与原始响应。

    只做记录，不改变调用语义、不注入内容、不吞异常；退出时恢复原函数。
    """

    def __init__(self):
        self.calls: list[dict] = []
        self._original = None

    def __enter__(self):
        self._original = decision_agent.call_decision_llm

        def _wrapped(prompt, config, **kwargs):
            response = self._original(prompt, config, **kwargs)
            self.calls.append({"call": len(self.calls) + 1, "prompt": prompt,
                               "response": response})
            return response

        decision_agent.call_decision_llm = _wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        decision_agent.call_decision_llm = self._original
        return False


def _classify_failure(decision: dict) -> str:
    error = decision.get("error")
    if not error:
        return OUTCOME_VALID
    if "API error" in str(error) or "credential" in str(error):
        return OUTCOME_PROVIDER_ERROR
    semantic = decision.get("semantic_validation")
    if semantic is not None:
        return OUTCOME_SEMANTIC_FAILED
    return OUTCOME_CONTRACT_FAILED


def run_case(case: EvalCase, config: dict, run_index: int, root: str) -> dict:
    """跑一次完整的案例回放，返回可 JSON 序列化的记录。"""
    prepared = prepare_case(case, root)
    record = CaseRunRecord(
        case_id=case.case_id,
        run_index=run_index,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        reference_run=case.reference_run,
        fact_package_id=(prepared.fact_package or {}).get("fact_package_id"),
        fact_ids=[f["fact_id"] for f in (prepared.fact_package or {}).get("facts", [])],
        facts={f["fact_id"]: f["value"]
               for f in (prepared.fact_package or {}).get("facts", [])},
        before_metrics=dict(prepared.before_metrics),
        base_args={k: v for k, v in prepared.base_args.items()
                   if not str(k).startswith("_")},
    )

    if prepared.blocking_code:
        record.outcome = OUTCOME_PERCEPTION_BLOCKED
        record.error_code = prepared.blocking_code
        return record.to_dict()
    if prepared.fact_package is None:
        record.outcome = OUTCOME_FACT_PACKAGE_INVALID
        record.error_code = prepared.fact_error
        return record.to_dict()

    with _RecordingLLM() as recorder:
        decision = decide_hyperparameters(
            prepared.fact_package, config, on_attempt=record.attempts.append,
        )
    record.llm_calls = recorder.calls
    record.decision = {
        "diagnosis": decision.get("diagnosis"),
        "action": decision.get("action"),
        "hyperparameter_changes": decision.get("hyperparameter_changes", {}),
        "training_overrides": decision.get("training_overrides", {}),
        "evidence_ids": decision.get("evidence_ids", {}),
        "retried": decision.get("retried", False),
    }
    record.outcome = _classify_failure(decision)
    record.error_code = decision.get("error")

    if record.outcome == OUTCOME_VALID:
        _, guard = sanitize_and_merge_tuning_params(
            prepared.base_args,
            decision.get("hyperparameter_changes", {}),
            decision.get("training_overrides", {}),
            prepared.perception.get("dataset", {}),
        )
        record.guardrails = {
            "valid": guard.valid,
            "errors": list(guard.errors),
            "warnings": list(guard.warnings),
            "clamped": dict(getattr(guard, "clamped", {})),
        }
        record.would_start_training = bool(guard.valid)
    return record.to_dict()


def write_record(record: dict, out_dir: str, run_index: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{record['case_id']}__run{run_index}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path
