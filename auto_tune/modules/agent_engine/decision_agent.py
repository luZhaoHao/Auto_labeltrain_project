"""Decision Agent — LLM-driven hyperparameter suggestion engine.

Builds a structured prompt from perception data + expert mapping rules,
calls DeepSeek, and extracts a JSON hyperparameter change plan.
"""

import json
import re
import requests
from typing import Any, Callable

from auto_tune.modules.security.credentials import resolve_credential
from auto_tune.modules.security.endpoint_policy import (
    DEFAULT_DEEPSEEK_ENDPOINT,
    EndpointPolicyError,
    validate_endpoint,
)
from auto_tune.modules.security.redaction import safe_provider_error

from .parameter_registry import get_tunable_parameter_names
from .decision_contract import (
    DecisionContractError,
    parse_tuning_decision_response,
    validate_decision_evidence,
)
from .decision_semantics import validate_decision_semantics
from .semantic_rules import build_parameter_rule_summary, build_semantic_rule_summary


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from LLM response text."""
    # Try to find ```json ... ``` block first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare {...} spanning multiple lines
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_decision_response(text: str) -> dict:
    """Parse and structurally validate one LLM tuning decision."""
    parsed = _extract_json(text)
    if not isinstance(parsed, dict):
        return {"error": "Failed to parse JSON from LLM response"}

    diagnosis = parsed.get("diagnosis")
    action = parsed.get("action")
    changes = parsed.get("hyperparameter_changes", {})
    overrides = parsed.get("training_overrides", {})
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        return {"error": "diagnosis must be a non-empty string"}
    if not isinstance(action, str) or not action.strip():
        return {"error": "action must be a non-empty string"}
    if not isinstance(changes, dict) or not isinstance(overrides, dict):
        return {"error": "parameter sections must be JSON objects"}

    combined = {**changes, **overrides}
    unknown = sorted(set(combined) - get_tunable_parameter_names())
    if unknown:
        return {"error": f"Unknown parameter(s): {', '.join(unknown)}"}
    if not combined and action != "keep_params":
        return {"error": "empty changes require action=keep_params"}
    if len(combined) > 3:
        return {"error": "正常模式每轮最多修改 3 个参数"}

    return {
        "diagnosis": diagnosis.strip(),
        "action": action.strip(),
        "hyperparameter_changes": changes,
        "training_overrides": overrides,
        "error": None,
    }


def build_decision_prompt(
    perception_summary: str,
    project_info: dict | None = None,
    previous_attempts: list[dict] | None = None,
) -> str:
    """Build the full Decision Agent prompt.

    Args:
        perception_summary: string from summarize_perception().
        project_info: project context dict.
        previous_attempts: list of previous tuning attempts with their results.

    Returns:
        Full prompt string for LLM.
    """

    # Expert mapping rules from the reference doc
    mapping_table = """
## 专家映射规则（感知 -> 调参动作）

请根据以下规则将感知到的现象映射为调参动作：

### 规则1：小目标占比高
- 现象：tiny_bbox_ratio > 0.3 或 avg_relative_area < 0.01
- 动作：imgsz=1280（或更高）, box 升至 10.0-12.0, scale 升至 0.8-0.9

### 规则2：类别极度不均衡（长尾分布）
- 现象：class_balance.is_balanced=false, long_tail_classes 存在
- 动作：cls 升至 1.0-2.0, 开启 copy_paste=0.3, fl_gamma 设为 1.5

### 规则3：工业反光/光照问题
- 现象：overexposure_ratio > 0.3 或 underexposure_ratio > 0.3
- 动作：hsv_h=0.05, hsv_s=0.7, hsv_v=0.6 增强光照鲁棒性

### 规则4：严重过拟合
- 现象：issues 包含 overfitting, 或 val_loss 上升 train_loss 下降
- 动作：weight_decay 升至 0.001-0.005, mosaic=1.0, mixup=0.1-0.15, 减小 epochs 或增大 patience

### 规则5：梯度爆炸 / Loss 震荡
- 现象：issues 包含 unstable_training, 或曲线 trend 含 unstable
- 动作：lr0 降至 0.001 以下, warmup_epochs=5.0, optimizer='AdamW'

### 规则6：背景误检严重
- 现象：低 Precision 或 issues 中相关提示
- 动作：cls 升至 1.0-2.0, 推理时提高 conf 阈值

### 规则7：mAP 停滞（Plateau）
- 现象：issues 包含 plateau, 或 mAP50 曲线 trend='plateau'
- 动作：lr0 降低 50% 使用余弦退火, 数据增强微调

### 规则8：欠拟合（所有指标偏低）
- 现象：mAP50 < 0.3, Recall < 0.4
- 动作：换更大模型, imgsz 提升, lr0 适当提高, 增加 epochs
"""

    prompt = f"""你是YOLOv8超参数优化专家。你的任务是基于数据集分析和训练结果，给出精确的超参数调整建议。

## 输出格式

你必须输出严格的JSON格式，不能包含其他文本：

```json
{{
  "diagnosis": "简要诊断（一句话概括核心问题）",
  "action": "调参策略说明",
  "hyperparameter_changes": {{
    "lr0": 0.005,
    "box": 10.0,
    ...
  }},
  "training_overrides": {{
    "epochs": 200,
    "patience": 30,
    "imgsz": 640
  }}
}}
```

- `hyperparameter_changes`: 只包含需要**修改**的参数（从当前值改为新值）
- `training_overrides`: 训练配置层面的修改（epochs, patience, imgsz, optimizer, model等）
- 只修改必要的参数，不要一次性改太多（正常模式每轮最多修改 3 个参数）
- 如果判断当前无需调整超参数，`action` 必须为 `keep_params`，此时两个参数对象都为空

## 感知数据

{perception_summary}

{mapping_table}
"""

    # Project background
    if project_info:
        proj_name = project_info.get("name") or ""
        proj_desc = project_info.get("description") or ""
        proj_target = project_info.get("detection_target") or ""
        proj_data = project_info.get("data_type") or ""
        prompt += "\n## 项目背景\n"
        if proj_name:
            prompt += f"- 项目名称: {proj_name}\n"
        if proj_desc:
            prompt += f"- 项目描述: {proj_desc}\n"
        if proj_target:
            prompt += f"- 检测目标: {proj_target}\n"
        if proj_data:
            prompt += f"- 数据类型: {proj_data}\n"

    # Previous attempts context
    if previous_attempts:
        prompt += "\n## 历史调参记录\n"
        prompt += "之前的调参尝试及结果如下（请避免重复相同的无效调整）：\n"
        for i, attempt in enumerate(previous_attempts, 1):
            changes = attempt.get("changes", {})
            result = attempt.get("result", "unknown")
            prompt += f"\n尝试 {i}: 修改 {json.dumps(changes, ensure_ascii=False)}"
            prompt += f"\n结果: {result}\n"

    prompt += """
## 注意事项
1. 参数值必须在合理范围内：lr0 [1e-5, 0.1], box [1, 20], cls [0.1, 5], dfl [0.5, 5]
2. 数据增强参数（mosaic, mixup, degrees等）取值范围 [0, 1]
3. 小数据集（<500张）不要用强几何增强
4. 不要同时大幅提高 dropout 和 weight_decay
5. 如果用 AdamW，lr0 不要超过 0.005
6. 每次调整要有针对性，解释你注意到了什么现象才做此调整
"""
    return prompt


def call_decision_llm(prompt: str, config: dict) -> str:
    """Call DeepSeek API for decision using the resolved text credential.

    Args:
        prompt: the decision prompt.
        config: full config dict (uses llm section).

    Returns:
        Raw response text.
    """
    llm_cfg = config.get("llm", {})
    api_key = resolve_credential("text")
    if not api_key:
        raise RuntimeError("DeepSeek API error: credential_missing")
    try:
        endpoint = validate_endpoint(
            llm_cfg.get("endpoint", DEFAULT_DEEPSEEK_ENDPOINT),
            bool(llm_cfg.get("allow_private_endpoint", False)),
        )
    except EndpointPolicyError:
        raise RuntimeError("DeepSeek API error: endpoint_rejected")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": llm_cfg.get("model", "deepseek-v4-flash"),
        "messages": [
            {"role": "system", "content": "你是YOLOv8超参数优化专家。输出严格的JSON格式，不要包含JSON之外的文本。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }

    try:
        resp = requests.post(
            endpoint, headers=headers, json=payload, timeout=(10, 120), allow_redirects=False
        )
    except requests.exceptions.RequestException:
        raise RuntimeError("DeepSeek API error: network_failed")
    if resp.status_code == 200:
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError("DeepSeek API error: incompatible_response")
    raise RuntimeError(
        f"DeepSeek API error ({resp.status_code}): {safe_provider_error(resp.status_code)}"
    )


def build_tuning_decision_prompt(fact_package: dict) -> str:
    """Build the auto-tuning prompt from the frozen FactPackage v1 only.

    The fact package is the single fact source: no free-form perception summary,
    no hard-coded fact values and no history. The model must echo the same
    ``fact_package_id`` and may only reference ``fact_id`` values that exist in
    the package.
    """
    package_json = json.dumps(fact_package, ensure_ascii=False, sort_keys=True)
    allowed = ", ".join(sorted(get_tunable_parameter_names()))
    semantic_summary = build_semantic_rule_summary()
    return f"""你是YOLOv8超参数优化专家。你只能依据下方"事实包"中给出的真实事实做决策，禁止编造事实。

## 事实包（唯一事实来源）

```json
{package_json}
```

## 输出格式（TuningDecision v1）

你必须只输出一个严格 JSON 对象，不要包含其他文本：

```json
{{
  "schema_version": "1.0",
  "fact_package_id": "<必须与上方事实包的 fact_package_id 完全一致>",
  "diagnosis": "简要诊断（一句话）",
  "action": "adjust 或 keep_params",
  "hyperparameter_changes": {{}},
  "training_overrides": {{}},
  "evidence_ids": {{}}
}}
```

## 规则

1. 只允许调整以下参数：{allowed}
2. hyperparameter_changes 与 training_overrides 合并后必须有 1-3 个参数；若判断无需调整，action 必须为 keep_params 且两个参数对象与 evidence_ids 都为空。
3. 每个修改参数必须在 evidence_ids 中给出至少一个 fact_id；evidence_ids 的键必须与本次修改的参数完全一致。
4. 只能引用上方事实包中真实存在的 fact_id，禁止编造事实。
5. 响应中的 fact_package_id 必须与事实包完全一致。
6. 每个修改参数必须符合下方"允许的超参数修改关系"：引用的证据必须能支持该参数，方向与幅度必须匹配；不满足任何关系、方向相反或幅度越界的修改会导致本轮失败。
7. 系统不会为你提供或猜测替代参数值。你必须依据同一事实包及上述语义规则，自行输出完整且合法的 TuningDecision v1。

## 允许的超参数修改关系（唯一事实—参数映射）

{semantic_summary}
"""


def build_tuning_correction_prompt(
    prompt: str,
    fact_package: dict,
    error_code: str,
    error_detail: str,
) -> str:
    """Build a controlled one-time correction prompt for a TuningDecision v1.

    The correction only re-states the stable error code, the exact schema and
    the same fact package; it never asks the model to invent new facts.
    """
    return (
        "你上一次的输出未通过自动调优决策校验，请重新输出严格合法的 TuningDecision v1 JSON。\n"
        f"稳定错误码: {error_code}\n"
        f"错误详情: {error_detail}\n"
        "要求：\n"
        "1. 只输出一个 JSON 对象，不要包含 JSON 之外的文本。\n"
        "2. 只能引用原事实包中真实存在的 fact_id，禁止编造事实。\n"
        "3. fact_package_id 必须与原事实包完全一致。\n"
        "4. evidence_ids 的键必须与本次修改的参数完全一致。\n"
        '5. 输出结构必须严格为：\n'
        '```json\n'
        '{\n'
        '  "schema_version": "1.0",\n'
        '  "fact_package_id": "<原事实包的 fact_package_id>",\n'
        '  "diagnosis": "非空字符串",\n'
        '  "action": "adjust 或 keep_params",\n'
        '  "hyperparameter_changes": {},\n'
        '  "training_overrides": {},\n'
        '  "evidence_ids": {}\n'
        '}\n'
        '```\n'
        f"## 原事实包\n{json.dumps(fact_package, ensure_ascii=False, sort_keys=True)}\n"
        f"## 原始任务\n{prompt}"
    )


def build_semantic_correction_prompt(
    prompt: str,
    fact_package: dict,
    error_code: str,
    parameter: str | None,
    reason_code: str | None,
) -> str:
    """Build a controlled one-time correction prompt for a semantic failure.

    Only carries the stable error code, the failing parameter, the parameter's
    allowed evidence/direction/amplitude from the registry, and the requirement
    to re-emit a full TuningDecision v1. Never guesses or offers substitute
    parameter values.
    """
    param_summary = build_parameter_rule_summary(parameter) if parameter else ""
    return (
        "你上一次的输出未通过自动调优语义校验，请重新输出严格合法的 TuningDecision v1 JSON。\n"
        f"稳定错误码: {error_code}\n"
        f"失败参数: {parameter}\n"
        f"失败原因: {reason_code}\n"
        f"{param_summary}\n"
        "要求：\n"
        "1. 只输出一个 JSON 对象，不要包含 JSON 之外的文本。\n"
        "2. 只能引用原事实包中真实存在的 fact_id，禁止编造事实。\n"
        "3. 每个修改参数必须有一条证据落在上方允许关系中，方向与幅度必须匹配。\n"
        "4. 系统不会提供具体修正值。请依据原事实包、该参数允许的证据关系、方向和幅度，"
        "重新独立输出完整且合法的 TuningDecision v1。\n"
        "5. fact_package_id 必须与原事实包完全一致。\n"
        f"## 原事实包\n{json.dumps(fact_package, ensure_ascii=False, sort_keys=True)}\n"
        f"## 原始任务\n{prompt}"
    )


def _default_tuning_validation(retried: bool) -> dict:
    return {
        "valid": False,
        "error_code": None,
        "error_detail": None,
        "retried": retried,
        "referenced_fact_ids": [],
    }


def _validate_tuning_response(raw: str, fact_package: dict) -> tuple[dict | None, dict]:
    """Parse + evidence-validate (Q1.1) then semantic-validate (Q1.2) a response.

    Returns ``(decision, validation)``. ``decision`` is None only when Q1.1
    contract/evidence validation fails (Q1.2 is not reached). On a semantic
    failure ``decision`` is the parsed decision and ``validation`` carries the
    stable semantic error plus the ``semantic`` sub-result.
    """
    try:
        decision = parse_tuning_decision_response(raw)
        decision = validate_decision_evidence(decision, fact_package)
    except DecisionContractError as exc:
        return None, {
            "valid": False,
            "error_code": exc.error_code,
            "error_detail": exc.detail,
            "retried": False,
            "referenced_fact_ids": [],
        }
    referenced = sorted({fid for ids in decision["evidence_ids"].values() for fid in ids})
    semantic = validate_decision_semantics(decision, fact_package)
    if not semantic["valid"]:
        return decision, {
            "valid": False,
            "error_code": semantic["error_code"],
            "error_detail": "semantic validation failed",
            "retried": False,
            "referenced_fact_ids": referenced,
            "semantic": semantic,
        }
    return decision, {
        "valid": True,
        "error_code": None,
        "error_detail": None,
        "retried": False,
        "referenced_fact_ids": referenced,
        "semantic": semantic,
    }


def _build_attempt_record(
    attempt_index: int, retried: bool, decision: dict | None, validation: dict
) -> dict:
    """Build one per-response audit attempt record.

    The record carries the parsed decision (or None on a Q1.1 contract failure),
    the Q1.1 decision_validation and the Q1.2 semantic_validation. It never
    embeds the raw LLM response, credentials, absolute paths or free-form error
    text — the same redaction boundary as the rest of the audit.
    """
    semantic = validation.get("semantic")
    return {
        "attempt": attempt_index,
        "retried": retried,
        "decision": dict(decision) if decision is not None else None,
        "decision_validation": {
            "valid": validation.get("valid"),
            "error_code": validation.get("error_code"),
            "error_detail": validation.get("error_detail"),
            "referenced_fact_ids": validation.get("referenced_fact_ids", []),
        },
        "semantic_validation": dict(semantic) if semantic is not None else None,
    }


def _persist_attempt(
    on_attempt: Callable | None,
    attempt_index: int,
    retried: bool,
    decision: dict | None,
    validation: dict,
) -> None:
    """Persist one validated model response before deciding to continue.

    ``on_attempt`` is provided by the loop and writes through the audit object;
    a raise here must propagate so the loop stops before any further LLM call,
    Guardrails, command construction or training launch.
    """
    if on_attempt is None:
        return
    on_attempt(_build_attempt_record(attempt_index, retried, decision, validation))


def _run_tuning_decision_with_retry(
    prompt: str, config: dict, fact_package: dict, on_attempt: Callable | None = None
) -> tuple[str | None, dict | None, dict]:
    """Run one TuningDecision v1 decision with exactly one controlled retry.

    Structural/evidence contract errors retry once against the same fact
    package; provider/transport exceptions never retry. A second contract
    failure is terminal.

    Each model response's Q1.1/Q1.2 validation result is handed to
    ``on_attempt`` for audit persistence immediately after validation and before
    any decision to retry or continue; a raise from ``on_attempt`` propagates
    and stops the decision.

    Returns ``(raw_response, normalized_decision | dict | str | None, validation)``.
    A ``str`` result is the safe provider error (never retried). A ``None``
    result means the contract failed on both attempts; ``validation.error_code``
    carries the stable code.
    """
    try:
        raw = call_decision_llm(prompt, config)
    except Exception as e:
        return None, str(e), _default_tuning_validation(False)

    decision, validation = _validate_tuning_response(raw, fact_package)
    _persist_attempt(on_attempt, 1, False, decision, validation)
    if decision is not None and validation["valid"]:
        return raw, decision, validation

    if validation.get("semantic") is not None:
        semantic = validation["semantic"]
        correction = build_semantic_correction_prompt(
            prompt, fact_package, validation["error_code"],
            semantic.get("parameter"), semantic.get("reason_code"))
    else:
        correction = build_tuning_correction_prompt(
            prompt, fact_package, validation["error_code"], validation["error_detail"])
    try:
        raw2 = call_decision_llm(correction, config)
    except Exception as e:
        return raw, str(e), _default_tuning_validation(True)

    decision2, validation2 = _validate_tuning_response(raw2, fact_package)
    validation2["retried"] = True
    _persist_attempt(on_attempt, 2, True, decision2, validation2)
    if decision2 is not None and validation2["valid"]:
        return raw2, decision2, validation2
    return raw2, None, validation2


def build_json_fix_prompt(original_prompt: str, error: str) -> str:
    """Build a controlled "only fix the JSON format" retry prompt.

    The retry never guesses parameters locally; it re-runs the same task and
    asks the model to re-emit strict valid JSON through the same schema.
    """
    return (
        "你上一次的输出未通过结构化校验，请重新输出严格合法的 JSON。\n"
        f"错误信息: {error}\n"
        "要求：\n"
        "1. 仅输出一个 JSON 对象，不要包含 JSON 之外的任何文本。\n"
        "2. 仅包含以下四个字段：\n"
        '```json\n'
        '{\n'
        '  "diagnosis": "非空字符串",\n'
        '  "action": "非空字符串",\n'
        '  "hyperparameter_changes": {},\n'
        '  "training_overrides": {}\n'
        '}\n'
        '```\n'
        "3. 正常模式每轮最多修改 3 个参数（hyperparameter_changes 与 "
        "training_overrides 合并计算）。\n"
        "4. 若无需修改任何超参数，action 必须为 keep_params，两个参数对象都为空。\n"
        "5. 除 keep_params 外，两个参数对象合并后不得为空。\n"
        "6. 参数名必须来自可调参数列表。\n\n"
        f"## 原始任务\n{original_prompt}"
    )


def _run_decision_with_retry(prompt: str, config: dict) -> tuple[str | None, dict | str, bool]:
    """Call the Decision LLM and strict-parse the result, with one controlled retry.

    A single JSON-format-fix retry is allowed only when the first attempt
    produced a response that failed strict validation (never for API/transport
    errors, and never guessed locally). The retry still goes through the same
    strict parser.

    Returns (raw_response, parsed_dict | stable_error_str, retried).
    """
    try:
        raw = call_decision_llm(prompt, config)
    except Exception as e:
        return None, str(e), False

    parsed = parse_decision_response(raw)
    if not parsed.get("error"):
        return raw, parsed, False

    fix_prompt = build_json_fix_prompt(prompt, parsed["error"])
    try:
        raw2 = call_decision_llm(fix_prompt, config)
    except Exception as e:
        return raw, str(e), True

    parsed2 = parse_decision_response(raw2)
    if not parsed2.get("error"):
        return raw2, parsed2, True
    return raw2, parsed2["error"], True


def _semantic_validation_result(validation: dict) -> dict | None:
    """Extract the semantic result plus the retried flag, or None when Q1.2
    was not reached (contract/provider failure)."""
    semantic = validation.get("semantic")
    if semantic is None:
        return None
    out = dict(semantic)
    out["retried"] = validation.get("retried", False)
    return out


def decide_hyperparameters(fact_package: dict, config: dict, on_attempt: Callable | None = None) -> dict:
    """Run the auto-tuning Decision Agent against a frozen FactPackage v1.

    The decision contract is TuningDecision v1: every changed parameter must
    reference a fact in ``fact_package`` and echo its ``fact_package_id``.
    Q1.1 contract/evidence errors and Q1.2 semantic errors each retry once and
    return the stable error code on failure.

    Each model response's validation result is handed to ``on_attempt`` for
    audit persistence immediately after validation; the loop uses this to stop
    on any audit write failure before retrying or continuing.

    Args:
        fact_package: dict from build_tuning_fact_package().
        config: full app config.
        on_attempt: optional callable(attempt_record) invoked after each
            validated response; a raise propagates and stops the decision.

    Returns:
        Dict with diagnosis, action, hyperparameter_changes, training_overrides,
        raw_response, error (None or stable code), retried, schema_version,
        fact_package_id, evidence_ids, validation and semantic_validation.
    """
    prompt = build_tuning_decision_prompt(fact_package)
    raw, result, validation = _run_tuning_decision_with_retry(
        prompt, config, fact_package, on_attempt=on_attempt,
    )
    semantic_validation = _semantic_validation_result(validation)

    if isinstance(result, str):
        return {
            "diagnosis": None,
            "action": None,
            "hyperparameter_changes": {},
            "training_overrides": {},
            "raw_response": raw,
            "error": result,
            "retried": validation["retried"],
            "schema_version": None,
            "fact_package_id": fact_package["fact_package_id"],
            "evidence_ids": {},
            "validation": validation,
            "semantic_validation": None,
        }
    if result is None:
        return {
            "diagnosis": None,
            "action": None,
            "hyperparameter_changes": {},
            "training_overrides": {},
            "raw_response": raw,
            "error": validation["error_code"],
            "retried": validation["retried"],
            "schema_version": None,
            "fact_package_id": fact_package["fact_package_id"],
            "evidence_ids": {},
            "validation": validation,
            "semantic_validation": semantic_validation,
        }
    return {
        "diagnosis": result["diagnosis"],
        "action": result["action"],
        "hyperparameter_changes": result["hyperparameter_changes"],
        "training_overrides": result["training_overrides"],
        "raw_response": raw,
        "error": None,
        "retried": validation["retried"],
        "schema_version": result["schema_version"],
        "fact_package_id": result["fact_package_id"],
        "evidence_ids": result["evidence_ids"],
        "validation": validation,
        "semantic_validation": semantic_validation,
    }


def generate_suggestion(
    summary_text: str,
    project_info: dict | None,
    config: dict,
) -> dict:
    """Unified structured suggestion for one training report.

    Shared by the intelligent-analysis entries (folder/ZIP) so they all use the
    exact same strict parser, the same single JSON-fix retry, and the same
    honest failure shape. Never returns the raw model response.
    """
    prompt = build_decision_prompt(summary_text, project_info)
    _, result_or_err, retried = _run_decision_with_retry(prompt, config)
    if isinstance(result_or_err, str):
        return {
            "diagnosis": None,
            "action": None,
            "hyperparameter_changes": {},
            "training_overrides": {},
            "error": result_or_err,
            "retried": retried,
        }
    return {
        "diagnosis": result_or_err["diagnosis"],
        "action": result_or_err["action"],
        "hyperparameter_changes": result_or_err["hyperparameter_changes"],
        "training_overrides": result_or_err["training_overrides"],
        "error": None,
        "retried": retried,
    }


def summarize_perception_for_decision(perception: dict) -> str:
    """Build a concise decision-focused summary from perception data.

    This is shorter than the full human-readable summary — designed to
    focus the LLM on actionable diagnostics.

    Args:
        perception: dict from build_perception().

    Returns:
        Concise markdown string.
    """
    from .perception import summarize_perception
    return summarize_perception(perception)
