"""Stage 2: Per-run text LLM diagnosis of training results."""

import re
import requests

from auto_tune.modules.security.credentials import resolve_credential
from auto_tune.modules.security.endpoint_policy import (
    DEFAULT_DEEPSEEK_ENDPOINT,
    EndpointPolicyError,
    validate_endpoint,
)
from auto_tune.modules.security.redaction import safe_provider_error


def clean_llm_response(text: str) -> str:
    """Strip markdown prefixes, weak opening phrases, and trailing whitespace."""
    # Remove leading markdown headers like "# " or "## "
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove leading list markers like "- " or "* " at line start
    text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
    # Remove weak opening phrases at the very start of the response
    weak_openings = [
        r'^(好的[，,]\s*)?(根据分析|基于以上数据|综上所述|总而言之|总体来看|从以上数据可以看出|分析如下[：:]|诊断如下[：:]|以下是对|以下为).*?(：|:)',
        r'^(好的[，,]\s*)?(这是|以下是|给你|为你).*?(：|:)',
    ]
    for pat in weak_openings:
        text = re.sub(pat, '', text, count=1)
    # Remove trailing "如果您有任何问题..." or similar
    text = re.sub(r'\n+如果您有任何问题.*$', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def build_run_prompt(run_name: str, run_data: dict, summary: dict,
                     project_info: dict | None = None) -> str:
    """Build prompt for a single training run.

    Args:
        run_name: name of the training run.
        run_data: per-run analysis data.
        summary: cross-run summary dict.
        project_info: optional project context (name, description, detection_target, data_type).
    """
    args = run_data.get("args", {})
    res = run_data.get("results", {})
    final = res.get("final_metrics", {})
    issues = run_data.get("issues", [])
    curve = run_data.get("curve_analysis", {})

    prompt = f"你是YOLO训练结果分析专家。请根据以下单次训练的详细分析数据，给出诊断和改进建议。\n"

    # Project background (optional but strongly recommended)
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
        prompt += "\n"

    prompt += f"## 训练概况\n"

    if final:
        prompt += (f"## 最终指标\n"
                   f"- mAP50: {final.get('metrics/mAP50(B)', '?')}\n"
                   f"- mAP50-95: {final.get('metrics/mAP50-95(B)', '?')}\n"
                   f"- Precision: {final.get('metrics/precision(B)', '?')}\n"
                   f"- Recall: {final.get('metrics/recall(B)', '?')}\n\n")

    # Curve trends
    val_box = curve.get("val_box", {})
    val_cls = curve.get("val_cls", {})
    prompt += "## 损失曲线趋势\n"
    prompt += f"- val_box_loss: {val_box.get('trend', '?')} (slope={val_box.get('slope', '?')})\n"
    prompt += f"- val_cls_loss: {val_cls.get('trend', '?')} (slope={val_cls.get('slope', '?')})\n"

    mAP_curve = curve.get("mAP50", {})
    if mAP_curve:
        prompt += f"- mAP50曲线: {mAP_curve.get('trend', '?')}\n"

    # Early stopping
    es = curve.get("early_stopping", {})
    if es:
        stopped_early = es.get("stopped_early", False)
        prompt += f"- 早停触发: {'是' if stopped_early else '否'}\n"
        if stopped_early:
            prompt += f"  原因: {es.get('reason', '?')}\n"

    prompt += "\n"

    # Issues
    if issues:
        prompt += "## 检测到的问题\n"
        for iss in issues:
            prompt += f"- [{iss['severity']}] {iss['type']}: {iss.get('detail', '')}\n"
    else:
        prompt += "## 检测到的问题\n- 未检测到明显问题\n"

    prompt += f"""

## 要求
请针对这一次训练({run_name})给出具体分析:

1. **质量评估**: 这次训练质量如何？指标是否正常？
2. **关键问题**: 指出本次训练的主要问题及可能原因。
3. **超参建议**: 针对epochs、学习率、数据增强、模型大小等给出建议。

用中文回答。直接输出分析内容，不使用任何标记符号（不要 #、-、* 等），不要以'好的'、'根据分析'、'综上所述'等套话开头，保持专业简洁、条理清晰。"""
    return prompt


def call_deepseek(prompt: str, config: dict) -> str:
    """Call DeepSeek API using the resolved text credential."""
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
            {"role": "system", "content": "你是YOLO训练分析专家。请直接输出分析结果，不使用任何标记符号（#、-、*等），不要用'好的'、'根据分析'等套话开头，保持专业简洁。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": llm_cfg.get("temperature", 0.3),
        "max_tokens": llm_cfg.get("max_tokens", 2000),
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


def analyze_with_llm(stage1_report: dict, config: dict) -> dict:
    """Stage 2: Per-run analysis using text LLM.

    Returns:
        dict keyed by run name, each value has llm_diagnosis, model_used, error fields.
    """
    if not config.get("llm", {}).get("enabled", True):
        return {"_global": {"llm_diagnosis": None, "model_used": None, "error": "LLM analysis disabled"}}

    runs = stage1_report.get("runs", {})
    summary = stage1_report.get("summary", {})
    project_info = stage1_report.get("project", {})
    model_used = config.get("llm", {}).get("model", "deepseek-v4-flash")
    results = {}

    for run_name, run_data in runs.items():
        if run_data.get("results", {}).get("error"):
            results[run_name] = {
                "llm_diagnosis": None,
                "model_used": model_used,
                "error": f"Parse error: {run_data['results']['error']}",
            }
            continue

        prompt = build_run_prompt(run_name, run_data, summary, project_info)
        try:
            response = call_deepseek(prompt, config)
            cleaned = clean_llm_response(response)
            results[run_name] = {
                "llm_diagnosis": cleaned,
                "model_used": model_used,
                "error": None,
            }
        except Exception as e:
            results[run_name] = {
                "llm_diagnosis": None,
                "model_used": model_used,
                "error": str(e),
            }

    return results
