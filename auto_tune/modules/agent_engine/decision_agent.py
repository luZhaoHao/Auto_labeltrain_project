"""Decision Agent — LLM-driven hyperparameter suggestion engine.

Builds a structured prompt from perception data + expert mapping rules,
calls DeepSeek, and extracts a JSON hyperparameter change plan.
"""

import json
import re
import requests
from typing import Any


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
- 只修改必要的参数，不要一次性改太多（最多5-8个）

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
    """Call DeepSeek API for decision.

    Args:
        prompt: the decision prompt.
        config: full config dict (uses llm section).

    Returns:
        Raw response text.
    """
    llm_cfg = config.get("llm", {})
    headers = {
        "Authorization": f"Bearer {llm_cfg.get('api_key', '')}",
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
    endpoint = llm_cfg.get("endpoint", "https://api.deepseek.com/v1/chat/completions")

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    err = resp.json().get("error", {}).get("message", resp.text[:200])
    raise RuntimeError(f"DeepSeek API error ({resp.status_code}): {err}")


def decide_hyperparameters(
    perception: dict,
    config: dict,
    previous_attempts: list[dict] | None = None,
) -> dict:
    """Run the Decision Agent to get hyperparameter suggestions.

    Args:
        perception: dict from build_perception().
        config: full app config.
        previous_attempts: previous tuning loop results for context.

    Returns:
        Dict with diagnosis, action, hyperparameter_changes, training_overrides,
        raw_response, and error (if any).
    """
    summary = summarize_perception_for_decision(perception)
    project_info = perception.get("project", {})
    prompt = build_decision_prompt(summary, project_info, previous_attempts)

    try:
        raw = call_decision_llm(prompt, config)
    except Exception as e:
        return {
            "diagnosis": None,
            "action": None,
            "hyperparameter_changes": {},
            "training_overrides": {},
            "raw_response": None,
            "error": str(e),
        }

    parsed = _extract_json(raw)
    if parsed is None:
        return {
            "diagnosis": None,
            "action": None,
            "hyperparameter_changes": {},
            "training_overrides": {},
            "raw_response": raw,
            "error": "Failed to parse JSON from LLM response",
        }

    return {
        "diagnosis": parsed.get("diagnosis"),
        "action": parsed.get("action"),
        "hyperparameter_changes": parsed.get("hyperparameter_changes", {}),
        "training_overrides": parsed.get("training_overrides", {}),
        "raw_response": raw,
        "error": None,
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
