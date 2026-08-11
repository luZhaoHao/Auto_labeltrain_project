"""Stage 3: Multimodal vision consultation for training results."""

import os
import re
import base64
import requests


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


def _encode_image(image_path: str) -> str:
    """Read and base64-encode an image."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_confusion_matrix(run_dir: str, config: dict,
                             project_info: dict | None = None) -> dict:
    """Send confusion matrix image to vision LLM for analysis.

    Args:
        run_dir: training run directory.
        config: full config dict (uses vision section).
        project_info: optional project context for more specific analysis.

    Returns:
        dict with analysis, model_used, or error.
    """
    cm_path = os.path.join(run_dir, "confusion_matrix_normalized.png")
    if not os.path.exists(cm_path):
        return {"error": f"confusion_matrix_normalized.png not found in {run_dir}"}

    # Optional project context prefix
    ctx = ""
    if project_info:
        parts = [v for v in [
            project_info.get("description"),
            project_info.get("detection_target"),
            project_info.get("data_type"),
        ] if v]
        if parts:
            ctx = f"（项目背景: {' / '.join(parts)}）"

    return _call_vision_api(
        image_path=cm_path,
        config=config,
        prompt=(
            f"分析这张YOLO混淆矩阵图（归一化），给出：\n"
            f"1. 包含哪些类别？{ctx}\n"
            f"2. 哪些类别间易混淆？程度如何？\n"
            f"3. 背景(background)误检率如何？\n"
            f"4. 整体分类性能和改进建议？"
        ),
    )


def analyze_error_crops(run_dir: str, config: dict,
                        project_info: dict | None = None) -> dict:
    """Generate error crops and send to vision LLM for analysis.

    Args:
        run_dir: training run directory.
        config: full config dict.
        project_info: optional project context for more specific analysis.

    Returns:
        dict with analysis or error.
    """
    try:
        from .crop_utils import generate_error_crops
    except ImportError as e:
        return {"error": f"Error crops not available (opencv-python required): {e}"}

    crop_path = generate_error_crops(run_dir)
    if crop_path is None:
        return {"error": "No error crops generated (val_batch images not found or no differences)."}

    # Optional project context prefix
    ctx = ""
    if project_info:
        parts = [v for v in [
            project_info.get("description"),
            project_info.get("detection_target"),
        ] if v]
        if parts:
            ctx = f"（{' / '.join(parts)}）"

    return _call_vision_api(
        image_path=crop_path,
        config=config,
        prompt=(
            f"这些是YOLO验证集预测的错检区域（标签与预测不一致处）{ctx}。请分析：\n"
            f"1. 是假阳性（误检）还是假阴性（漏检）？\n"
            f"2. 错误类型（定位偏差、大小不匹配、完全漏检、背景误检）？\n"
            f"3. 有无共同特征（小目标、遮挡、边界目标）？\n"
            f"4. 改进建议？"
        ),
    )


def _call_vision_api(image_path: str, config: dict, prompt: str) -> dict:
    """Internal: call Qwen-VL API with an image."""
    vision_cfg = config.get("vision", {})
    api_key = vision_cfg.get("api_key", "")
    model = vision_cfg.get("model", "qwen3-vl-flash")
    endpoint = vision_cfg.get("endpoint", "")
    temperature = vision_cfg.get("temperature", 0.3)
    max_tokens = vision_cfg.get("max_tokens", 500)

    if not endpoint:
        return {"error": "Vision API endpoint not configured."}

    try:
        img_b64 = _encode_image(image_path)
    except Exception as e:
        return {"error": f"Failed to read image: {e}"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是YOLO训练视觉分析专家。请直接输出分析结果，不使用任何标记符号（#、-、*等），不要用'好的'、'根据分析'等套话开头，保持专业简洁。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            cleaned = clean_llm_response(content)
            return {
                "model_used": model,
                "image_analyzed": os.path.basename(image_path),
                "analysis": cleaned,
                "error": None,
            }
        err = resp.json().get("error", {}).get("message", resp.text[:200])
        return {"error": f"Vision API error ({resp.status_code}): {err}"}
    except Exception as e:
        return {"error": f"Vision API exception: {e}"}


def multimodal_consult(run_dir: str, config: dict,
                       project_info: dict | None = None) -> dict:
    """Run full vision consultation on a training run.

    Analyzes confusion matrix and optionally error crops.

    Args:
        run_dir: training run directory.
        config: full config dict.
        project_info: optional project context dict (flows from config.yaml project section).

    Returns:
        dict with confusion_matrix_analysis and optional error_crop_analysis.
    """
    run_name = os.path.basename(run_dir)
    result = {"run_name": run_name}

    if not config.get("vision", {}).get("enabled", True):
        return {**result, "error": "Vision analysis disabled"}

    result["confusion_matrix_analysis"] = analyze_confusion_matrix(run_dir, config, project_info)

    if config.get("vision", {}).get("error_crops_enabled", True):
        result["error_crop_analysis"] = analyze_error_crops(run_dir, config, project_info)

    return result
