"""FastAPI web application for Auto-Tune dashboard.

Provides three major pages:
- Dashboard: overview of all modules
- Dataset Analysis: Module A results
- Training Analysis & Tuning: Module B results + Module C auto-tuning
"""

import json
import os
import time
import secrets
import asyncio
import threading
import zipfile
import tempfile
import shutil
import datetime
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlsplit
from fastapi import FastAPI, Request, Query, Cookie, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import requests
import yaml

from auto_tune.modules.security.credentials import (
    CredentialError,
    UnsupportedPlatformError,
    clear_last_test_result,
    delete_credential,
    get_credential_status,
    invalidate_credential_cache,
    resolve_credential,
    set_last_test_result,
    store_credential,
)
from auto_tune.modules.security.endpoint_policy import (
    DEFAULT_DEEPSEEK_ENDPOINT,
    DEFAULT_QWEN_ENDPOINT,
    EndpointPolicyError,
    validate_endpoint,
)
from auto_tune.modules.security.redaction import safe_provider_error


def _safe_extract_zip(zf: zipfile.ZipFile, destination: str | Path) -> None:
    """Extract an archive only when every member stays inside destination."""
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    for member in zf.infolist():
        member_path = (target / member.filename).resolve()
        try:
            member_path.relative_to(target)
        except ValueError as exc:
            raise ValueError(f"unsafe ZIP member path: {member.filename}") from exc
    zf.extractall(target)


from auto_tune.modules.train_analyzer.training_finalizer import finalize_training_run
from auto_tune.modules.agent_engine.training_log import (
    append_training_log,
    build_training_sse_payload,
    classify_training_line,
)
from auto_tune.modules.dataset_snapshot import (
    SnapshotConflictError,
    SnapshotError,
    SnapshotInsufficientSpaceError,
    SnapshotIOError,
    SnapshotValidationError,
)
from auto_tune.modules.dataset_snapshot.service import (
    create_dataset_snapshot,
    validate_dataset_snapshot,
)


def _finalize_and_build_event(
    returncode: int,
    train_dir: str,
    train_name: str,
    config: dict,
    log_dir: str,
    started_at: str | None,
) -> dict:
    """Finalize a finished training and build the unified SSE completion event."""
    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    training_status = "completed" if returncode == 0 else "failed"
    training_error = None
    if returncode != 0:
        if _running_training.get("status") == "aborted":
            error_type, message = "user_cancelled", "训练被用户取消"
        else:
            error_type, message = "training_process_failed", f"训练进程退出码 {returncode}"
        training_error = {
            "stage": "training",
            "error_type": error_type,
            "message": message,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    result = finalize_training_run(
        train_dir,
        train_name,
        "manual",
        config,
        log_dir=log_dir,
        training_status=training_status,
        started_at=started_at,
        finished_at=finished_at,
        training_error=training_error,
    )
    if result["status"] == "completed" and result["analysis_status"] == "failed":
        return {
            "status": "done",
            "level": "warning",
            "message": "训练完成，结果分析失败",
            "result": result,
        }
    if result["status"] == "completed":
        return {
            "status": "done",
            "level": "success",
            "message": f"训练完成: {train_name}",
            "result": result,
        }
    return {
        "status": "error",
        "level": "error",
        "message": f"训练失败 (exit code {returncode})",
        "result": result,
    }


def _remove_status_file() -> None:
    """Delete the training_running.json status file if present (best-effort)."""
    _sf = os.path.join("log", "training_running.json")
    if os.path.exists(_sf):
        try:
            os.remove(_sf)
        except OSError:
            pass


def _sse_chunk(payloads) -> str:
    """Serialize payload dicts into one SSE chunk (multiple ``data:`` events)."""
    return "".join("data: " + json.dumps(p, ensure_ascii=False) + "\n\n" for p in payloads)


def _process_training_output_line(line, train_name, log_path, warned_once):
    """Sanitize/persist/classify one stdout line into an SSE payload dict.

    Returns ``None`` when a persistence warning was already emitted for
    ``log_path`` (warn-once per run). Never raises on a broken log file.
    """
    event = classify_training_line(line)
    try:
        append_training_log(log_path, event.raw)
    except OSError:
        if log_path in warned_once:
            return None
        warned_once.add(log_path)
        return {
            "status": "running",
            "event": "log_persistence_error",
            "level": "warning",
            "message": "日志保存失败，训练继续运行",
        }
    return build_training_sse_payload(event, train_name)


def _enqueue_bounded(msg_queue, payload) -> None:
    """Put a transient SSE payload on a bounded queue, dropping on overflow.

    Terminal events are never routed through here, so a slow client can only
    lose transient progress (constraints 9/11c), never the final status.
    """
    import queue as _queue_module

    try:
        msg_queue.put_nowait(json.dumps(payload, ensure_ascii=False))
    except _queue_module.Full:
        pass


class _SseBatch:
    """Bounded SSE batch buffer: flush on max size or a short time window.

    Keeps the event rate bounded without accumulating the full log in memory.
    """

    def __init__(self, max_batch=20, flush_interval=0.1):
        self.max_batch = max_batch
        self.flush_interval = flush_interval
        self._payloads: list[dict] = []
        self._last_flush = time.monotonic()

    def add(self, payload) -> None:
        self._payloads.append(payload)

    @property
    def pending(self) -> int:
        return len(self._payloads)

    def should_flush(self) -> bool:
        return (
            len(self._payloads) >= self.max_batch
            or (time.monotonic() - self._last_flush) > self.flush_interval
        )

    def take(self) -> str:
        chunk = _sse_chunk(self._payloads)
        self._payloads = []
        self._last_flush = time.monotonic()
        return chunk


from .components.dataset_panel import get_dataset_report, format_dataset_summary
from .components.train_panel import get_training_report
from .components.tuning_panel import get_tuning_history, get_tuning_status
from .components.experiment_panel import get_experiment_history
from .i18n import make_translator, translate

app = FastAPI(title="Auto-Tune Dashboard")

# ── Direct Jinja2 (avoid Starlette TemplateResponse compatibility issue) ──
import jinja2
_templates_dir = str(Path(__file__).parent / "templates")
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_templates_dir),
    enable_async=False,
    auto_reload=True,
)

# Config — plaintext api_key is never loaded into the public APP_CONFIG.
config_path = Path(__file__).parent.parent / "config.yaml"


def _load_public_config() -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for _section in ("llm", "vision"):
        _sec = cfg.get(_section)
        if isinstance(_sec, dict):
            _sec.pop("api_key", None)
    return cfg


APP_CONFIG = _load_public_config()

# Track running training process (single-worker only)
_running_training: dict = {}

# Cancel signal for tuning loop (shared across threads)
_tuning_cancel_event = threading.Event()

# Reference to current TrainingProcess for stop endpoint
_current_tuning_train_proc = None


def _update_config(section: str, data: dict) -> tuple[bool, str]:
    """Update a section of config.yaml and reload APP_CONFIG.

    Returns (success, message).
    """
    try:
        # Reload current config from disk to avoid overwriting concurrent edits
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg.setdefault(section, {}).update(data)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        # Update in-memory config
        APP_CONFIG.setdefault(section, {}).update(data)
        return True, "ok"
    except Exception as e:
        return False, str(e)


# ── S1.3 AI settings / credential security ──
_CSRF_TOKEN = secrets.token_urlsafe(32)
_PLACEHOLDER_KEYS = {"YOUR_DEEPSEEK_API_KEY", "YOUR_QWEN_API_KEY"}
_AI_SETTINGS_WHITELIST = {
    "enabled",
    "provider",
    "model",
    "endpoint",
    "allow_private_endpoint",
}
_MAX_KEY_LENGTH = 512
_MAX_REQUEST_BODY = 64 * 1024


class _SecurityRejected(Exception):
    pass


class _BodyError(Exception):
    pass


def _section_for(purpose: str) -> str:
    return "llm" if purpose == "text" else "vision"


def _default_endpoint_for(purpose: str) -> str:
    return DEFAULT_DEEPSEEK_ENDPOINT if purpose == "text" else DEFAULT_QWEN_ENDPOINT


def _default_model_for(purpose: str) -> str:
    return "deepseek-chat" if purpose == "text" else "qwen-vl-plus"


def _read_config_file() -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _atomic_write_yaml(path: object, data: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            yaml.safe_dump(
                data, handle, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _legacy_key_in_config(purpose: str) -> str | None:
    """Detect a non-empty, non-placeholder legacy api_key for a purpose.

    The value is never exposed outside this module; only the boolean state is.
    """
    section = _section_for(purpose)
    cfg = _read_config_file()
    value = (cfg.get(section) or {}).get("api_key")
    if isinstance(value, str) and value.strip() and value not in _PLACEHOLDER_KEYS:
        return value
    return None


def _validate_key_value(key: object) -> None:
    if not isinstance(key, str):
        raise ValueError("key must be a string")
    if not key.strip():
        raise ValueError("key must not be empty")
    if any(ord(c) < 32 for c in key):
        raise ValueError("key must not contain control characters")
    if len(key) > _MAX_KEY_LENGTH:
        raise ValueError("key is too long")
    if key in _PLACEHOLDER_KEYS:
        raise ValueError("key must not be a placeholder")


def _check_same_origin(request: Request) -> bool:
    host = request.headers.get("host", "")
    origin = request.headers.get("origin")
    if not origin:
        return True
    try:
        origin_host = urlsplit(origin).netloc
    except ValueError:
        return False
    return origin_host == host


def _require_security(request: Request) -> None:
    if not _check_same_origin(request):
        raise _SecurityRejected("cross-origin request rejected")
    token = request.headers.get("x-csrf-token", "")
    if not token or not secrets.compare_digest(token, _CSRF_TOKEN):
        raise _SecurityRejected("missing or invalid CSRF token")


async def _safe_json_body(request: Request) -> object:
    raw = await request.body()
    if not raw:
        return {}
    if len(raw) > _MAX_REQUEST_BODY:
        raise _BodyError("request body too large")
    try:
        return json.loads(raw)
    except Exception:
        raise _BodyError("request body must be valid JSON")


def _ai_settings_for(purpose: str) -> dict:
    section = _section_for(purpose)
    cfg_section = (APP_CONFIG.get(section) or {}) if APP_CONFIG else {}
    status = get_credential_status(purpose)
    return {
        "purpose": purpose,
        "enabled": bool(cfg_section.get("enabled", True)),
        "provider": cfg_section.get("provider", "deepseek" if purpose == "text" else "qwen"),
        "model": cfg_section.get("model", _default_model_for(purpose)),
        "endpoint": cfg_section.get("endpoint", ""),
        "allow_private_endpoint": bool(cfg_section.get("allow_private_endpoint", False)),
        "default_endpoint": _default_endpoint_for(purpose),
        "configured": status.configured,
        "source": status.source,
        "writable": status.writable,
        "last_tested_at": status.last_tested_at,
        "last_test_result": status.last_test_result,
        "migration_required": _legacy_key_in_config(purpose) is not None,
    }


def _ai_settings_context() -> dict:
    return {"text": _ai_settings_for("text"), "vision": _ai_settings_for("vision")}


def _ai_config_basic() -> dict:
    """Non-sensitive settings for the template (no credential status reads)."""
    result = {}
    for purpose in ("text", "vision"):
        section = _section_for(purpose)
        cfg_section = (APP_CONFIG.get(section) or {}) if APP_CONFIG else {}
        result[purpose] = {
            "purpose": purpose,
            "enabled": bool(cfg_section.get("enabled", True)),
            "provider": cfg_section.get("provider", "deepseek" if purpose == "text" else "qwen"),
            "model": cfg_section.get("model", _default_model_for(purpose)),
            "endpoint": cfg_section.get("endpoint", ""),
            "allow_private_endpoint": bool(cfg_section.get("allow_private_endpoint", False)),
            "default_endpoint": _default_endpoint_for(purpose),
            "migration_required": _legacy_key_in_config(purpose) is not None,
        }
    return result


def _probe_connection(purpose: str, api_key_override: str | None = None) -> str:
    """Send a minimal probe using the current (or candidate) credential.

    Returns one of the fixed safe categories; never returns provider bodies.
    """
    section = _section_for(purpose)
    cfg_section = (APP_CONFIG.get(section) or {}) if APP_CONFIG else {}
    api_key = api_key_override if api_key_override is not None else resolve_credential(purpose)
    if not api_key:
        return "credential_missing"
    try:
        endpoint = validate_endpoint(
            cfg_section.get("endpoint") or _default_endpoint_for(purpose),
            bool(cfg_section.get("allow_private_endpoint", False)),
        )
    except EndpointPolicyError:
        return "endpoint_rejected"
    payload = {
        "model": cfg_section.get("model", _default_model_for(purpose)),
        "messages": [{"role": "user", "content": "Reply with OK"}],
        "max_tokens": 5,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(
            endpoint, headers=headers, json=payload, timeout=(10, 30), allow_redirects=False
        )
    except requests.exceptions.RequestException:
        return "network_failed"
    if resp.status_code != 200:
        return safe_provider_error(resp.status_code)
    try:
        resp.json()
    except Exception:
        return "incompatible_response"
    return "success"


def _atomic_update_ai_section(purpose: str, data: dict) -> None:
    section = _section_for(purpose)
    cfg = _read_config_file()
    current = dict(cfg.get(section) or {})
    for key in _AI_SETTINGS_WHITELIST:
        if key in data:
            current[key] = data[key]
    cfg[section] = current
    _atomic_write_yaml(config_path, cfg)
    if APP_CONFIG:
        APP_CONFIG[section] = dict(current)


def _remove_legacy_key_from_config(purpose: str) -> None:
    section = _section_for(purpose)
    cfg = _read_config_file()
    sec = cfg.get(section)
    if isinstance(sec, dict):
        sec.pop("api_key", None)
    _atomic_write_yaml(config_path, cfg)
    if APP_CONFIG:
        cur = APP_CONFIG.setdefault(section, {})
        if isinstance(cur, dict):
            cur.pop("api_key", None)


# ── Simple TTL Cache ──
_cache: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 300.0  # seconds — data only changes when dataset uploaded or tuning completes


def _cached(key: str, ttl: float = _CACHE_TTL) -> object:
    """Get cached value by key, or None if missing/expired."""
    entry = _cache.get(key)
    if entry and time.time() - entry[0] < ttl:
        return entry[1]
    return None


def _set_cache(key: str, value: object):
    _cache[key] = (time.time(), value)


def _invalidate_cache(key_prefix: str = ""):
    """Invalidate cache entries starting with prefix (empty = all)."""
    global _cache
    if not key_prefix:
        _cache.clear()
    else:
        _cache = {k: v for k, v in _cache.items() if not k.startswith(key_prefix)}


# ── Language detection ──
def _detect_lang(request: Request) -> str:
    """Detect language: query param ?lang= > cookie > default zh."""
    lang = request.query_params.get("lang")
    if lang in ("zh", "en"):
        return lang
    lang = request.cookies.get("lang")
    if lang in ("zh", "en"):
        return lang
    return "zh"


def _render(template_name: str, request: Request, **context) -> str:
    lang = _detect_lang(request)
    _ = make_translator(lang)
    template = _jinja_env.get_template(template_name)
    return template.render(_=_, current_lang=lang, **context)


# ── Helper: load report for template ──
def _load_data():
    cache_key = "load_data"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    dataset_raw = get_dataset_report()
    dataset = format_dataset_summary(dataset_raw) if dataset_raw else None
    training = get_training_report()
    project = APP_CONFIG.get("project", {}) if APP_CONFIG else {}
    tuning_history = get_tuning_history()
    experiment_history = get_experiment_history()
    dataset_analyzer_config = APP_CONFIG.get("dataset_analyzer", {})

    result = (dataset, training, project, tuning_history, experiment_history, dataset_analyzer_config)
    _set_cache(cache_key, result)
    return result


# ── Helper: assemble suggestion from tuning history or LLM analysis ──
def _get_latest_suggestion(tuning_history, training):
    """Extract the latest hyperparameter suggestion from tuning history or LLM analysis."""
    if tuning_history:
        latest = tuning_history[-1]
        decision = latest.get("decision", {})
        changes = decision.get("hyperparameter_changes", {})
        if changes:
            return {
                "diagnosis": decision.get("diagnosis", ""),
                "rationale": decision.get("rationale") or decision.get("action", ""),
                "hyperparameter_changes": changes,
                "training_overrides": decision.get("training_overrides", {}),
            }
    # Check training report's own suggestion (from ZIP upload → LLM + Decision Agent)
    if training and training.get("suggestion"):
        sug = training["suggestion"]
        if sug.get("hyperparameter_changes") or sug.get("diagnosis"):
            return {
                "diagnosis": sug.get("diagnosis", ""),
                "rationale": sug.get("action", ""),
                "hyperparameter_changes": sug.get("hyperparameter_changes", {}),
                "training_overrides": sug.get("training_overrides", {}),
            }
    # Fallback: use LLM analysis text as diagnosis (no structured changes)
    if training and training.get("llm_analysis"):
        for rn, diag in training["llm_analysis"].items():
            if diag.get("llm_diagnosis"):
                return {
                    "diagnosis": diag["llm_diagnosis"],
                    "rationale": diag.get("llm_rationale", ""),
                    "hyperparameter_changes": {},
                    "training_overrides": {},
                }
    return None


def _get_current_args(training):
    """Get the current hyperparameter values from the best training run."""
    if training and training.get("runs"):
        best = training.get("summary", {}).get("best_overall_run")
        if best and best in training["runs"]:
            return training["runs"][best].get("args", {})
        for rn, rd in training["runs"].items():
            return rd.get("args", {})
    return None


# ── Routes ──

def _common_context():
    """Load all data needed by the SPA template."""
    dataset, training, project, tuning_history, experiment_history, dataset_analyzer_config = _load_data()
    # Read latest dataset info
    latest_dataset = None
    if LATEST_DATASET_PATH.exists():
        try:
            with open(LATEST_DATASET_PATH, encoding="utf-8") as f:
                ld = json.load(f)
            source_path = ld.get("source_dataset_path") or ld.get("dataset_path", "")
            snapshot_path = ld.get("snapshot_path")
            # A registered snapshot remains usable even if the original source
            # directory was later removed (immutability goal).
            if (source_path and os.path.isdir(source_path)) or (snapshot_path and os.path.isdir(snapshot_path)):
                ld["snapshot_valid"] = _latest_snapshot_valid(ld)
                latest_dataset = ld
        except Exception:
            pass
    return {
        "dataset": dataset,
        "training": training,
        "project": project,
        "tuning_history": tuning_history,
        "experiment_history": experiment_history,
        "latest_suggestion": _get_latest_suggestion(tuning_history, training),
        "current_args": _get_current_args(training),
        "dataset_analyzer_config": dataset_analyzer_config,
        "training_config": APP_CONFIG.get("training", {}),
        "llm_analysis": training.get("llm_analysis") if training else None,
        "vision_analysis": training.get("vision_analysis") if training else None,
        "latest_dataset": latest_dataset,
        "csrf_token": _CSRF_TOKEN,
        "ai_config": _ai_config_basic(),
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    import time as _time
    _t0 = _time.time()
    ctx = _common_context()
    html = _render("single_page.html", request, active_page="dashboard", **ctx)
    _t1 = _time.time()
    with open("log/_startup_timing.txt", "a") as _fh:
        _fh.write(f"[{_time.strftime('%H:%M:%S')}] First request rendered in {_t1 - _t0:.3f}s\n")
    return HTMLResponse(html)


@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request):
    ctx = _common_context()
    html = _render("single_page.html", request, active_page="dataset", **ctx)
    return HTMLResponse(html)


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    """Backward-compatible redirect to agent_suggestion."""
    ctx = _common_context()
    html = _render("single_page.html", request, active_page="agent_suggestion", **ctx)
    return HTMLResponse(html)


@app.get("/agent_suggestion", response_class=HTMLResponse)
async def agent_suggestion_page(request: Request):
    ctx = _common_context()
    html = _render("single_page.html", request, active_page="agent_suggestion", **ctx)
    return HTMLResponse(html)


@app.get("/training_monitor", response_class=HTMLResponse)
async def training_monitor_page(request: Request):
    ctx = _common_context()
    html = _render("single_page.html", request, active_page="training_monitor", **ctx)
    return HTMLResponse(html)


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    ctx = _common_context()
    html = _render("single_page.html", request, active_page="history", **ctx)
    return HTMLResponse(html)


# ── Module-level helpers for supplementing reports ──

def _ensure_report_issues(report):
    """Add top-level issues list from runs."""
    if "issues" in report:
        return
    all_issues = []
    for rn, rd in report.get("runs", {}).items():
        for iss in rd.get("issues", []):
            item = dict(iss) if isinstance(iss, dict) else {"issue": str(iss)}
            item.setdefault("run", rn)
            all_issues.append(item)
    report["issues"] = all_issues


def _ensure_report_llm(report):
    """Populate missing LLM analysis (Stage 2) in a report."""
    if "llm_analysis" in report or not APP_CONFIG.get("llm", {}).get("enabled", False):
        return
    try:
        from auto_tune.modules.train_analyzer.llm_analyzer import analyze_with_llm
        llm_result = analyze_with_llm(report, APP_CONFIG)
        first_key = next(iter(llm_result), None)
        if first_key and isinstance(llm_result[first_key], dict):
            report["llm_analysis"] = {
                "diagnosis": llm_result[first_key].get("llm_diagnosis"),
                "model_used": llm_result[first_key].get("model_used"),
            }
        else:
            report["llm_analysis"] = {"diagnosis": None}
    except Exception as llm_err:
        report["llm_analysis"] = {"error": str(llm_err)}


def _ensure_report_vision(report, detect_dir):
    """Populate missing Vision analysis (Stage 3) in a report."""
    if "vision_analysis" in report or not APP_CONFIG.get("vision", {}).get("enabled", False):
        return
    if not detect_dir or not os.path.isdir(detect_dir):
        return
    try:
        from auto_tune.modules.train_analyzer.vision_analyzer import multimodal_consult
        vision_result = multimodal_consult(detect_dir, APP_CONFIG, APP_CONFIG.get("project", {}))
        if isinstance(vision_result, dict) and "error" not in vision_result:
            report["vision_analysis"] = vision_result
        elif isinstance(vision_result, dict):
            report["vision_analysis"] = {"error": vision_result["error"]}
    except Exception as vis_err:
        report["vision_analysis"] = {"error": str(vis_err)}


# ── API Routes ──

@app.get("/api/dataset")
async def api_dataset():
    dataset_raw = get_dataset_report()
    if dataset_raw:
        return JSONResponse(dataset_raw)
    return JSONResponse({"error": "No dataset report found"}, status_code=404)


@app.get("/api/training")
async def api_training():
    training = get_training_report()
    if training:
        return JSONResponse(training)
    return JSONResponse({"error": "No training report found"}, status_code=404)


@app.get("/api/training/report-by-name")
async def api_training_report_by_name(name: str = Query("")):
    """Return Module B report JSON for a specific training run.
    If an existing report is found, loads it and supplements missing LLM/Vision analysis.
    If no existing report is found, runs full Module B analysis on the fly."""
    if not name:
        return JSONResponse({"error": "Missing 'name' query parameter"}, status_code=400)
    import glob as _glob
    import datetime as _dt

    # ── Step 1: Locate existing report or training directory ──
    report_path = os.path.join("log", f"{name}_report.json")
    report_found = os.path.exists(report_path)
    if not report_found:
        matches = _glob.glob(os.path.join("log", f"*{name}*_report.json"))
        if matches:
            report_path = matches[0]
            report_found = True

    detect_dir = "detect"
    run_dir = os.path.join(detect_dir, name)
    run_dir_exists = os.path.isdir(run_dir)

    if not report_found and not run_dir_exists:
        return JSONResponse({"error": f"No report or training directory found for {name}"}, status_code=404)

    # ── Case A: Existing report found — load and supplement ──
    if report_found:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        _ensure_report_issues(report)
        _ensure_report_llm(report)
        _ensure_report_vision(report, run_dir if run_dir_exists else report.get("detect_dir", ""))
        return JSONResponse(report)

    # ── Case B: No existing report — run full Module B on-the-fly ──
    try:
        from auto_tune.modules.train_analyzer.results_parser import load_training_run
        from auto_tune.modules.train_analyzer.curve_analysis import (
            analyze_loss_curves, analyze_metric_curves, detect_early_stopping,
        )
        from auto_tune.modules.train_analyzer.issue_detector import detect_issues
        from auto_tune.modules.train_analyzer.run_comparator import compare_runs, summarize_runs

        run_data = load_training_run(run_dir)
        run_data["name"] = os.path.basename(run_dir)

        ta_config = APP_CONFIG.get("train_analyzer", {})
        curve_analysis = analyze_loss_curves(run_data["results"], ta_config)
        metric_analysis = analyze_metric_curves(run_data["results"], ta_config)
        early_stop = detect_early_stopping(run_data, ta_config)
        curve_analysis["early_stopping"] = early_stop
        issues = detect_issues(run_data, ta_config)
        run_data["curve_analysis"] = curve_analysis
        run_data["metric_analysis"] = metric_analysis
        run_data["issues"] = issues

        report = {
            "module": "train_analyzer",
            "version": "1.0",
            "analysis_timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "detect_dir": run_dir,
            "project": APP_CONFIG.get("project", {}),
            "total_runs": 1,
            "runs": {
                run_data["name"]: {
                    "name": run_data["name"],
                    "args": run_data.get("args", {}),
                    "results": run_data.get("results", {}),
                    "curve_analysis": curve_analysis,
                    "metric_analysis": metric_analysis,
                    "issues": issues,
                }
            },
            "comparison": compare_runs([run_data], ta_config),
            "summary": summarize_runs([run_data], ta_config),
        }
        _ensure_report_issues(report)
        _ensure_report_llm(report)
        _ensure_report_vision(report, run_dir)
        return JSONResponse(report)
    except Exception as e:
        return JSONResponse({"error": f"Module B analysis failed: {e}"}, status_code=500)


@app.get("/api/audit/{filename}")
async def api_audit_record(filename: str):
    """Return a stored tuning audit record by basename (traversal-safe)."""
    if filename != os.path.basename(filename) or not filename.startswith("tuning_audit_"):
        return JSONResponse({"error": "Invalid audit file name"}, status_code=400)
    path = os.path.join("log", filename)
    if not os.path.isfile(path):
        return JSONResponse({"error": "Audit file not found"}, status_code=404)
    try:
        with open(path, encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read audit file: {exc}"}, status_code=500)


@app.post("/api/training/save-report-text")
async def api_training_save_report_text(request: Request):
    """Save the module B analysis report as a TXT file in the training directory."""
    body = await request.json()
    train_name = body.get("train_name", "").strip()
    if not train_name:
        return JSONResponse({"error": "Missing train_name"}, status_code=400)

    # Re-fetch the report (same logic as report-by-name endpoint)
    import glob as _glob
    report_path = os.path.join("log", f"{train_name}_report.json")
    report_found = os.path.exists(report_path)
    if not report_found:
        matches = _glob.glob(os.path.join("log", f"*{train_name}*_report.json"))
        if matches:
            report_path = matches[0]
            report_found = True

    run_dir = os.path.join("detect", train_name)
    if not report_found and not os.path.isdir(run_dir):
        return JSONResponse({"error": f"No report found for {train_name}"}, status_code=404)

    # Load the report
    report = {}
    if report_found:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

    # Supplement missing analyses (same as report-by-name endpoint)
    _ensure_report_issues(report)
    _ensure_report_llm(report)
    _ensure_report_vision(report, run_dir if os.path.isdir(run_dir) else "")

    # Build text content
    lines = []
    lines.append("=" * 60)
    lines.append(f"训练分析报告 - {train_name}")
    if report.get("analysis_timestamp"):
        lines.append(f"分析时间: {report['analysis_timestamp']}")
    lines.append("=" * 60)
    lines.append("")

    # ── Summary at top level ──
    summary = report.get("summary", {}) or {}
    best_run_name = summary.get("best_overall_run") or report.get("comparison", {}).get("best_run") or train_name
    best_mAP = summary.get("best_mAP50")
    avg_mAP = summary.get("average_mAP50")
    total_analyzed = summary.get("total_runs_analyzed")
    if total_analyzed is not None:
        lines.append(f"分析总运行数: {total_analyzed}")
    if best_mAP is not None:
        lines.append(f"最佳 mAP50: {best_mAP}")
    if avg_mAP is not None:
        lines.append(f"平均 mAP50: {avg_mAP}")
    if summary.get("common_issues"):
        lines.append(f"常见问题: {', '.join(summary['common_issues'])}")
    lines.append("")

    # ── Issues: nested inside runs[best_run_name].issues ──
    runs = report.get("runs", {}) or {}
    run_data = runs.get(best_run_name, {}) if isinstance(runs, dict) else {}
    issues = run_data.get("issues", []) if isinstance(run_data, dict) else []
    if issues:
        lines.append("【关键问题】")
        for i, iss in enumerate(issues, 1):
            text = iss.get("issue", str(iss)) if isinstance(iss, dict) else str(iss)
            lines.append(f"  {i}. {text}")
        lines.append("")

    # ── Curve analysis ──
    curve = run_data.get("curve_analysis", {}) or {}
    metric = run_data.get("metric_analysis", {}) or {}
    if metric:
        lines.append("【指标分析】")
        for m_name in ("mAP50", "mAP50-95", "precision", "recall"):
            m = metric.get(m_name, {}) or {}
            if m.get("trend"):
                lines.append(f"  {m_name}: {m['trend']} (best={m.get('best', '-')})")
        lines.append("")
    if curve.get("overfitting_detected"):
        lines.append(f"  过拟合检测: {'是' if curve['overfitting_detected'] else '否'}")
        lines.append("")

    # ── Metric range from comparison ──
    comparison = report.get("comparison", {}) or {}
    metric_range = comparison.get("metric_range", {}) or summary.get("metric_range", {}) or {}
    if metric_range and isinstance(metric_range, dict):
        lines.append("【指标范围】")
        for k, v in metric_range.items():
            if isinstance(v, list) and len(v) == 2:
                lines.append(f"  {k}: {v[0]} ~ {v[1]}")
        lines.append("")

    # ── LLM analysis at top level ──
    llm = report.get("llm_analysis", {}) or {}

    # ── Vision analysis: may be flat dict or keyed by run name ──
    vision = report.get("vision_analysis", {}) or {}
    if isinstance(vision, dict):
        # Detect if vision_analysis is keyed by run name or flat
        run_keys = list(report.get("runs", {}).keys())
        matching_run_keys = [k for k in vision if k in run_keys]
        if matching_run_keys:
            # Keyed by run name format
            target = best_run_name if best_run_name in vision else matching_run_keys[0]
            run_vision = vision.get(target, {})
        else:
            # Flat format (direct from _ensure_report_vision)
            run_vision = vision
    else:
        run_vision = vision
    if not isinstance(run_vision, dict):
        run_vision = {}
    cm = run_vision.get("confusion_matrix_analysis", {}) or {} if not run_vision.get("error") else {}
    ec = run_vision.get("error_crop_analysis", {}) or {} if not run_vision.get("error") else {}

    # Collect the three special sections to put at the end
    special_sections = []

    if llm.get("error"):
        special_sections.append(("【大模型分析报告】", f"Error: {llm['error']}"))
    elif llm.get("diagnosis"):
        special_sections.append(("【大模型分析报告】", llm["diagnosis"]))

    if run_vision.get("error"):
        # Append vision error as a single section
        special_sections.append(("【视觉分析错误】", run_vision["error"]))
    else:
        if cm.get("analysis"):
            special_sections.append(("【混淆矩阵分析】", cm["analysis"]))
        if ec.get("analysis"):
            special_sections.append(("【错误裁剪分析】", ec["analysis"]))

    # Append special sections at the end
    for title, content in special_sections:
        lines.append(title)
        lines.append(content)
        lines.append("")

    text_content = "\n".join(lines)

    # Write to training directory
    os.makedirs(run_dir, exist_ok=True)
    txt_path = os.path.join(run_dir, "analysis_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    return JSONResponse({"success": True, "path": txt_path})


@app.get("/api/tuning/history")
async def api_tuning_history():
    history = get_tuning_history()
    return JSONResponse(history)


@app.get("/api/experiments/history")
async def api_experiments_history():
    """Unified experiment history (manual + tuning, incl. legacy) for the UI."""
    return JSONResponse(get_experiment_history())


@app.get("/api/tuning/status")
async def api_tuning_status():
    return JSONResponse({"status": get_tuning_status()})


# ── S1.3 AI settings / credentials / migration API ──


@app.get("/api/ai-settings")
async def get_ai_settings():
    return JSONResponse(_ai_settings_context())


@app.put("/api/ai-settings/{purpose}")
async def update_ai_settings(purpose: str, request: Request):
    if purpose not in ("text", "vision"):
        return JSONResponse({"error": "invalid purpose"}, status_code=404)
    try:
        _require_security(request)
    except _SecurityRejected as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    try:
        body = await _safe_json_body(request)
    except _BodyError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
    unknown = set(body) - _AI_SETTINGS_WHITELIST
    if unknown:
        return JSONResponse(
            {"error": f"unsupported field(s): {', '.join(sorted(unknown))}"}, status_code=400
        )
    if "enabled" in body and not isinstance(body["enabled"], bool):
        return JSONResponse({"error": "enabled must be a boolean"}, status_code=400)
    if "allow_private_endpoint" in body and not isinstance(body["allow_private_endpoint"], bool):
        return JSONResponse({"error": "allow_private_endpoint must be a boolean"}, status_code=400)
    for field in ("provider", "model", "endpoint"):
        if field in body:
            value = body[field]
            if not isinstance(value, str) or not value.strip():
                return JSONResponse(
                    {"error": f"{field} must be a non-empty string"}, status_code=400
                )
    if "endpoint" in body:
        try:
            validate_endpoint(
                body["endpoint"], bool(body.get("allow_private_endpoint", False))
            )
        except EndpointPolicyError:
            return JSONResponse({"error": "endpoint rejected by policy"}, status_code=400)
    try:
        _atomic_update_ai_section(purpose, body)
    except Exception:
        return JSONResponse({"error": "failed to save settings"}, status_code=500)
    invalidate_credential_cache(purpose)
    return JSONResponse(_ai_settings_for(purpose))


@app.put("/api/credentials/{purpose}")
async def put_credential(purpose: str, request: Request):
    if purpose not in ("text", "vision"):
        return JSONResponse({"error": "invalid purpose"}, status_code=404)
    try:
        _require_security(request)
    except _SecurityRejected as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    try:
        body = await _safe_json_body(request)
    except _BodyError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
    key = body.get("key")
    test_before_replace = bool(body.get("test_before_replace", False))
    try:
        _validate_key_value(key)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    env_var = "AUTO_TUNE_TEXT_API_KEY" if purpose == "text" else "AUTO_TUNE_VISION_API_KEY"
    if os.environ.get(env_var):
        return JSONResponse(
            {"error": "credential is managed by the environment and cannot be modified"},
            status_code=409,
        )
    if test_before_replace:
        category = _probe_connection(purpose, api_key_override=key)
        if category != "success":
            return JSONResponse({"error": f"credential test failed: {category}"}, status_code=400)
    try:
        store_credential(purpose, key)
    except (CredentialError, UnsupportedPlatformError) as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    set_last_test_result(purpose, "success" if test_before_replace else "untested")
    return JSONResponse({"status": "stored", "purpose": purpose})


@app.delete("/api/credentials/{purpose}")
async def delete_credential_route(purpose: str, request: Request):
    if purpose not in ("text", "vision"):
        return JSONResponse({"error": "invalid purpose"}, status_code=404)
    try:
        _require_security(request)
    except _SecurityRejected as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    try:
        body = await _safe_json_body(request)
    except _BodyError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not (isinstance(body, dict) and body.get("confirm") is True):
        return JSONResponse({"error": "confirmation required"}, status_code=400)
    env_var = "AUTO_TUNE_TEXT_API_KEY" if purpose == "text" else "AUTO_TUNE_VISION_API_KEY"
    if os.environ.get(env_var):
        return JSONResponse(
            {"error": "credential is managed by the environment and cannot be deleted"},
            status_code=409,
        )
    try:
        delete_credential(purpose)
    except CredentialError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    invalidate_credential_cache(purpose)
    return JSONResponse({"status": "deleted", "purpose": purpose})


@app.post("/api/credentials/{purpose}/test")
async def test_credential_route(purpose: str, request: Request):
    if purpose not in ("text", "vision"):
        return JSONResponse({"error": "invalid purpose"}, status_code=404)
    try:
        _require_security(request)
    except _SecurityRejected as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    category = _probe_connection(purpose)
    if category == "credential_missing":
        return JSONResponse({"error": "no credential configured"}, status_code=400)
    set_last_test_result(purpose, category)
    return JSONResponse({"result": category})


@app.post("/api/credentials/{purpose}/migrate")
async def migrate_credential(purpose: str, request: Request):
    if purpose not in ("text", "vision"):
        return JSONResponse({"error": "invalid purpose"}, status_code=404)
    try:
        _require_security(request)
    except _SecurityRejected as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    legacy = _legacy_key_in_config(purpose)
    if not legacy:
        return JSONResponse({"error": "no legacy credential to migrate"}, status_code=400)
    env_var = "AUTO_TUNE_TEXT_API_KEY" if purpose == "text" else "AUTO_TUNE_VISION_API_KEY"
    if os.environ.get(env_var):
        return JSONResponse(
            {"error": "credential is managed by the environment"}, status_code=409
        )
    # Conservative migration: never overwrite an existing secure credential.
    # Leave both the YAML legacy key and the secure store untouched on conflict.
    invalidate_credential_cache(purpose)
    if resolve_credential(purpose):
        return JSONResponse(
            {
                "error": "a secure credential already exists for this service; "
                "delete it before migrating the legacy key"
            },
            status_code=409,
        )
    try:
        store_credential(purpose, legacy)
        invalidate_credential_cache(purpose)
        verify = resolve_credential(purpose)
        if not verify or verify != legacy:
            raise CredentialError("credential read-back verification failed")
        _remove_legacy_key_from_config(purpose)
        invalidate_credential_cache(purpose)
        clear_last_test_result(purpose)
    except CredentialError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except OSError:
        # Secure store succeeded but the YAML rewrite failed: keep the secure
        # credential and the migration_required state for a safe retry.
        return JSONResponse(
            {"error": "credential stored but config update failed; migration incomplete"},
            status_code=500,
        )
    return JSONResponse({"status": "migrated", "purpose": purpose})


@app.post("/tuning/start")
async def start_tuning(request: Request):
    """Start the auto-tuning loop with real-time SSE progress streaming."""
    body = await request.json()
    reference_run = body.get("reference_run") or None
    max_retries = body.get("max_retries", 3)
    mode = body.get("mode", "dry_run")
    skip_execute = mode == "dry_run"
    keep_params = mode == "keep_params"
    auto_analyze = body.get("auto_analyze", False)
    auto_loop = body.get("auto_loop", False)
    # When auto_analyze is enabled and multiple iterations configured,
    # auto_loop must be on so all iterations complete before best is selected
    if auto_analyze and max_retries > 1 and not auto_loop:
        auto_loop = True
    eval_mode = body.get("eval_mode", "comprehensive")

    # Snapshot gate: auto-tuning must never bypass a registered snapshot.
    latest = _read_latest_dataset()
    snapshot_data_yaml = None
    if latest and latest.get("snapshot_id"):
        try:
            snapshot_data_yaml = _resolve_validated_snapshot_data_yaml(latest)
        except SnapshotError as exc:
            return _snapshot_error_response(exc)
    elif not skip_execute:
        return JSONResponse(
            {"error": "没有已校验的不可变数据集快照，无法启动自动调优",
             "error_code": "SNAPSHOT_VALIDATION_FAILED"},
            status_code=400,
        )

    global _tuning_cancel_event, _current_tuning_train_proc
    _tuning_cancel_event.clear()
    _current_tuning_train_proc = None

    # Write tuning running status file so /api/tuning/status returns "running"
    _tuning_status_file = os.path.join("log", "tuning_running.json")
    try:
        with open(_tuning_status_file, "w", encoding="utf-8") as _tsf:
            json.dump({"status": "running"}, _tsf)
    except Exception:
        pass

    async def event_stream():
        import queue as _queue
        import asyncio as _asyncio

        # Use a thread-safe queue so on_progress (running in thread pool)
        # can deliver messages to the async generator in real time.
        # Bounded (constraint 9/11c): under slow-client backpressure transient
        # progress is dropped; terminal events are emitted by the generator
        # itself, never through on_progress, so they can never be dropped.
        msg_queue: _queue.Queue[str] = _queue.Queue(maxsize=500)
        last_iteration = 0

        def _enqueue(payload: dict) -> None:
            _enqueue_bounded(msg_queue, payload)

        def on_progress(iteration, message, step=None, params=None, **kwargs):
            nonlocal last_iteration
            if iteration != last_iteration:
                last_iteration = iteration
                _enqueue({
                    "status": "running",
                    "message": f"--- Iter {iteration} ---",
                    "level": "info",
                    "iteration": iteration,
                    "step": "iteration_start",
                })
            data = {
                "status": "running",
                "message": message,
                "level": kwargs.get("level", "info"),
                "iteration": iteration,
            }
            if step:
                data["step"] = step
            if params:
                data["params"] = params
            for _k in ("event", "log_kind", "detail", "epoch", "total_epochs", "train_name"):
                if _k in kwargs and kwargs[_k] is not None:
                    data[_k] = kwargs[_k]
            _enqueue(data)

        from auto_tune.modules.agent_engine.loop import run_tuning_loop

        def run():
            # Inject the validated snapshot data.yaml into a config copy; never
            # mutate the shared APP_CONFIG.
            cfg = APP_CONFIG
            if snapshot_data_yaml is not None:
                cfg = dict(APP_CONFIG)
                training = dict(APP_CONFIG.get("training", {}) or {})
                training["data_yaml"] = str(snapshot_data_yaml)
                cfg["training"] = training
            return run_tuning_loop(
                config=cfg,
                reference_run=reference_run,
                max_retries=max_retries,
                log_dir="log",
                skip_execute=skip_execute,
                auto_analyze=auto_analyze,
                auto_loop=auto_loop,
                on_progress=on_progress,
                cancel_event=_tuning_cancel_event,
                keep_params=keep_params,
                eval_mode=eval_mode,
            )

        loop = _asyncio.get_event_loop()
        future = loop.run_in_executor(None, run)

        result = None
        error = None

        # Immediate feedback so the user sees "Starting..." right away
        yield f"data: {json.dumps({'status': 'running', 'message': 'Starting tuning...', 'level': 'info', 'iteration': 0})}\n\n"

        # Yield messages in real time while the tuning loop runs in a thread.
        # NOTE: We use get_nowait() + await sleep() instead of get(timeout=0.5)
        # because blocking the event loop with get(timeout=...) prevents it from
        # processing the call_soon_threadsafe callback that resolves future.done().
        while True:
            # Non-blocking drain of all available messages
            try:
                while True:
                    msg = msg_queue.get_nowait()
                    yield f"data: {msg}\n\n"
            except _queue.Empty:
                pass

            if future.done():
                try:
                    result = future.result()
                except Exception as e:
                    error = str(e)
                break

            # Check for cancellation signal
            if _tuning_cancel_event.is_set():
                # Give the thread a moment to clean up, then break
                if future.done():
                    try:
                        result = future.result()
                    except Exception as e:
                        error = str(e)
                else:
                    error = "用户取消"
                break

            # Yield control to event loop so it can process callbacks
            # (critical: without this, future.done() never becomes True
            #  because the event loop is blocked by the async generator)
            await _asyncio.sleep(0.15)

        try:
            _invalidate_cache("load_data")

            if error:
                if "用户取消" in str(error):
                    msg = json.dumps({"status": "cancelled", "message": "训练已取消", "level": "warn"})
                else:
                    msg = json.dumps({"status": "error", "message": f"Tuning failed: {error}", "level": "error"})
                yield f"data: {msg}\n\n"
            elif result is None:
                msg = json.dumps({"status": "error", "message": "Tuning returned no result", "level": "error"})
                yield f"data: {msg}\n\n"
            else:
                final = result.get("final_result", {}) or {}
                msg = json.dumps({
                    "status": "done",
                    "message": "Tuning complete",
                    "level": "success",
                    "result": {
                        "train_name": final.get("train_name"),
                        "changes": final.get("changes"),
                        "best_iteration": result.get("best_iteration"),
                        "best_train_name": result.get("best_train_name"),
                        "best_metrics": result.get("best_metrics"),
                        "eval_mode": result.get("eval_mode", eval_mode),
                    },
                })
                yield f"data: {msg}\n\n"

            # Drain any remaining messages that arrived between future completion and yield
            while True:
                try:
                    msg = msg_queue.get_nowait()
                    yield f"data: {msg}\n\n"
                except _queue.Empty:
                    break
        except Exception as _e:
            # Ensure the client always gets a terminal event, even on error
            try:
                yield f"data: {json.dumps({'status': 'error', 'message': f'Stream error: {_e}', 'level': 'error'})}\n\n"
            except Exception:
                pass
        finally:
            # Clean up tuning running status file when stream ends
            try:
                if os.path.exists(_tuning_status_file):
                    os.remove(_tuning_status_file)
            except Exception:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/tuning/stop")
async def stop_tuning():
    """Stop the currently running tuning loop."""
    global _tuning_cancel_event
    _tuning_cancel_event.set()
    # Clean up tuning running status file
    sf = os.path.join("log", "tuning_running.json")
    if os.path.exists(sf):
        try:
            os.remove(sf)
        except Exception:
            pass
    return JSONResponse({"status": "stopped"})


@app.post("/api/training/stop")
async def stop_first_training():
    """Stop the currently running first-time training."""
    if _running_training.get("proc"):
        proc = _running_training["proc"]
        try:
            if proc.returncode is None:
                proc.terminate()
        except Exception:
            pass
        _running_training["status"] = "aborted"
        _running_training["proc"] = None
    # Clean up status file
    sf = os.path.join("log", "training_running.json")
    if os.path.exists(sf):
        try:
            os.remove(sf)
        except Exception:
            pass
    return JSONResponse({"status": "stopped"})


# ── Dataset Upload & Analysis ──

UPLOAD_DIR = Path("log") / "uploads"

# ── Immutable dataset snapshot (Studio S1.2) ──
LATEST_DATASET_PATH = Path("log") / "latest_dataset.json"
DATASET_SNAPSHOT_ROOT = Path("log") / "dataset_snapshots"


def _read_latest_dataset() -> dict | None:
    """Read latest_dataset.json without rewriting it."""
    if not LATEST_DATASET_PATH.exists():
        return None
    try:
        with open(LATEST_DATASET_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON atomically: same-dir temp, flush, fsync, os.replace.

    On failure the temp file is intentionally left in place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_name, path)


def _parse_split_params(body: dict) -> tuple[float, int]:
    """Strictly parse split request params (reject coercions of ambiguous types)."""
    val_ratio_raw = body.get("val_ratio", 0.2)
    seed_raw = body.get("seed", 42)
    if isinstance(val_ratio_raw, bool) or not isinstance(val_ratio_raw, (int, float)):
        raise SnapshotValidationError("val_ratio 必须是 (0,1) 内的数字")
    val_ratio = float(val_ratio_raw)
    if not (0.0 < val_ratio < 1.0):
        raise SnapshotValidationError("val_ratio 必须是 (0,1) 内的数字")
    if type(seed_raw) is not int:
        raise SnapshotValidationError("seed 必须是整数")
    return val_ratio, seed_raw


def _load_class_names(source_path: Path, ds_info: dict) -> dict[int, str] | None:
    """Derive the real class-name mapping from a registered data.yaml (no inventing)."""
    candidates: list[Path] = []
    registered = ds_info.get("data_yaml_path")
    if registered:
        candidates.append(Path(registered))
    if source_path.is_dir():
        candidates.extend(source_path.rglob("data.yaml"))
        candidates.extend(source_path.rglob("data.yml"))
    for yaml_path in candidates:
        if not yaml_path.is_file():
            continue
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            names = data.get("names") or {}
            if names:
                return {int(k): v for k, v in names.items()}
        except (OSError, ValueError, yaml.YAMLError, TypeError):
            continue
    return None


def _snapshot_error_response(exc: SnapshotError) -> JSONResponse:
    """Map a snapshot domain error to a structured HTTP response."""
    return JSONResponse({"error": str(exc), "error_code": exc.error_code}, status_code=exc.status_code)


def _snapshot_created_at(snapshot) -> str | None:
    try:
        with open(snapshot.manifest_path, encoding="utf-8") as f:
            return json.load(f).get("created_at")
    except (OSError, ValueError):
        return None


def _snapshot_to_latest_info(existing: dict, snapshot) -> dict:
    """Merge snapshot facts into the latest_dataset.json payload."""
    info = dict(existing)
    info.update({
        "source_dataset_path": str(snapshot.source_root),
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_path": str(snapshot.snapshot_path),
        "manifest_path": str(snapshot.manifest_path),
        "data_yaml_path": str(snapshot.data_yaml_path),
        "snapshot_schema_version": snapshot.schema_version,
        "snapshot_manifest_digest": snapshot.manifest_digest,
        "snapshot_created_at": _snapshot_created_at(snapshot),
        "train_count": snapshot.train_count,
        "val_count": snapshot.val_count,
        "background_count": snapshot.background_count,
        "split": True,
    })
    return info


def _latest_snapshot_valid(ld: dict) -> bool:
    """Return True only when the registered snapshot validates end-to-end."""
    snapshot_id = ld.get("snapshot_id")
    snapshot_path = ld.get("snapshot_path")
    if not snapshot_id or not snapshot_path or not os.path.isdir(snapshot_path):
        return False
    try:
        validated = validate_dataset_snapshot(Path(snapshot_path))
        return validated.snapshot_id == snapshot_id
    except SnapshotError:
        return False


def _resolve_validated_snapshot_data_yaml(latest_info: dict) -> Path:
    """Return the absolute snapshot data.yaml path, or raise when invalid."""
    snapshot_id = latest_info.get("snapshot_id")
    snapshot_path = latest_info.get("snapshot_path")
    data_yaml_path = latest_info.get("data_yaml_path")
    manifest_digest = latest_info.get("snapshot_manifest_digest")
    if not snapshot_id or not snapshot_path or not data_yaml_path:
        raise SnapshotValidationError("数据集没有已注册的不可变快照")
    snap_dir = Path(snapshot_path)
    if not snap_dir.is_dir():
        raise SnapshotValidationError(f"快照目录不存在: {snapshot_path}")
    validated = validate_dataset_snapshot(snap_dir)
    if validated.snapshot_id != snapshot_id:
        raise SnapshotValidationError("快照身份不一致")
    if manifest_digest and validated.manifest_digest != manifest_digest:
        raise SnapshotValidationError("快照 manifest 摘要不一致")
    data_yaml = Path(data_yaml_path)
    try:
        data_yaml.relative_to(snap_dir)
    except ValueError as exc:
        raise SnapshotValidationError("data.yaml 必须位于快照目录内") from exc
    if not data_yaml.is_file():
        raise SnapshotValidationError(f"快照 data.yaml 不存在: {data_yaml_path}")
    return data_yaml.resolve()


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a ZIP dataset, run analysis, return results."""
    _invalidate_cache("load_data")

    if not file.filename or not file.filename.endswith(".zip"):
        return JSONResponse({"error": "Only ZIP files are supported"}, status_code=400)

    # Save uploaded file
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = f"ds_{int(time.time())}"
    extract_path = UPLOAD_DIR / upload_id
    extract_path.mkdir(parents=True)

    zip_path = extract_path / file.filename
    try:
        content = await file.read()
        zip_path.write_bytes(content)

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, extract_path)
        zip_path.unlink()  # remove zip after extraction

        # Find dataset root: look for data.yaml or images/ directory
        dataset_dir = str(extract_path)
        data_yaml = None

        # Try to find data.yaml
        yaml_paths = list(extract_path.rglob("data.yaml")) + list(extract_path.rglob("data.yml"))
        if yaml_paths:
            import yaml
            with open(yaml_paths[0], encoding="utf-8") as f:
                data_yaml = yaml.safe_load(f)
            dataset_dir = str(yaml_paths[0].parent)
        else:
            # Check if images/train exists
            train_img = extract_path / "images" / "train"
            if train_img.exists():
                dataset_dir = str(extract_path)
                data_yaml = {"names": {0: "object"}}
            else:
                # Check flat structure
                jpg_files = list(extract_path.glob("*.jpg")) + list(extract_path.glob("*.png"))
                if jpg_files:
                    dataset_dir = str(extract_path)
                    data_yaml = {"names": {0: "object"}}
                else:
                    return JSONResponse({
                        "error": "Cannot find data.yaml or images in the ZIP. "
                                 "Please ensure your ZIP contains a YOLO-format dataset."
                    }, status_code=400)

        # Run analysis
        from auto_tune.modules.dataset_analyzer.analyzer import analyze_dataset

        ds_config = APP_CONFIG.get("dataset_analyzer", {})
        result = analyze_dataset(dataset_dir, data_yaml, ds_config)

        # Add dataset path if not present
        result.setdefault("dataset_path", dataset_dir)

        # Save report to log directory
        report_path = Path("log") / f"dataset_report_{upload_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Track latest dataset in a status file (do not delete extracted files)
        latest_info = {
            "upload_id": upload_id,
            "dataset_path": dataset_dir,
            "has_data_yaml": bool(yaml_paths),
            "upload_time": time.time(),
            "split": False,
            "data_yaml_path": str(yaml_paths[0]) if yaml_paths else None,
        }
        latest_ds_path = Path("log") / "latest_dataset.json"
        latest_ds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latest_ds_path, "w", encoding="utf-8") as f:
            json.dump(latest_info, f, ensure_ascii=False, indent=2)

        return JSONResponse({
            "status": "success",
            "dataset_path": dataset_dir,
            "data_yaml_path": str(yaml_paths[0]) if yaml_paths else None,
            "report_path": str(report_path),
            "summary": format_dataset_summary(result),
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            err_log = Path("log") / "upload_errors.log"
            err_log.parent.mkdir(parents=True, exist_ok=True)
            with open(err_log, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Upload error:\n{tb}\n")
        except Exception:
            pass
        print(f"[ERROR] Upload failed: {e}\n{tb}", flush=True)
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)


# ── Folder Browse API ──


@app.post("/api/browse-folder")
async def browse_folder(request: Request):
    """List subdirectories of a given path for server-side folder browsing."""
    try:
        data = await request.json()
        path = data.get("path", "").strip()

        if not path or not os.path.isdir(path):
            # Return available drives on Windows
            if os.name == "nt":
                import string
                drives = []
                for d in string.ascii_uppercase:
                    dp = f"{d}:\\"
                    if os.path.exists(dp):
                        drives.append({"name": f"({d}:)", "path": dp, "is_dir": True})
                return JSONResponse({"path": path, "entries": drives})
            return JSONResponse({"path": path, "entries": []})

        entries = []
        try:
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                try:
                    if os.path.isdir(item_path):
                        entries.append({"name": item, "path": item_path, "is_dir": True})
                except OSError:
                    pass
        except PermissionError:
            pass

        # On Windows, drive roots (C:\, D:\) have no parent directory;
        # use empty string to signal "go up to drives list".
        if os.name == "nt" and len(path) == 3 and path[1:] == ":\\":
            parent = ""
        else:
            parent = os.path.dirname(path)
            parent = parent if parent != path else None
        return JSONResponse({
            "path": path,
            "parent": parent,
            "entries": entries,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Dataset Folder Analyze API ──


@app.post("/api/dataset/analyze-folder")
async def analyze_dataset_folder(request: Request):
    """Analyze a dataset folder directly from a server path (no ZIP upload)."""
    _invalidate_cache("load_data")

    try:
        data = await request.json()
        folder_path = data.get("path", "").strip()

        if not folder_path or not os.path.isdir(folder_path):
            return JSONResponse({"error": f"文件夹路径无效或不存在: {folder_path}"}, status_code=400)

        # Find data.yaml or detect dataset structure
        dataset_dir = folder_path
        data_yaml = None

        folder = Path(folder_path)
        yaml_paths = list(folder.rglob("data.yaml")) + list(folder.rglob("data.yml"))
        if yaml_paths:
            import yaml as _yaml
            with open(yaml_paths[0], encoding="utf-8") as f:
                data_yaml = _yaml.safe_load(f)
            dataset_dir = str(yaml_paths[0].parent)
        else:
            train_img = folder / "images" / "train"
            if train_img.exists():
                data_yaml = {"names": {0: "object"}}
            else:
                jpg_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                if jpg_files:
                    data_yaml = {"names": {0: "object"}}
                else:
                    return JSONResponse({
                        "error": "找不到 data.yaml 或 images/ 目录。请确保路径指向有效的 YOLO 格式数据集。"
                    }, status_code=400)

        # Run Module A analysis
        from auto_tune.modules.dataset_analyzer.analyzer import analyze_dataset

        ds_config = APP_CONFIG.get("dataset_analyzer", {})
        result = analyze_dataset(dataset_dir, data_yaml, ds_config)
        result.setdefault("dataset_path", dataset_dir)

        # Save report
        upload_id = f"ds_{int(time.time())}"
        report_path = Path("log") / f"dataset_report_{upload_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Track latest dataset
        latest_info = {
            "upload_id": upload_id,
            "dataset_path": dataset_dir,
            "has_data_yaml": bool(yaml_paths),
            "upload_time": time.time(),
            "split": False,
            "data_yaml_path": str(yaml_paths[0]) if yaml_paths else None,
        }
        latest_ds_path = Path("log") / "latest_dataset.json"
        latest_ds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latest_ds_path, "w", encoding="utf-8") as f:
            json.dump(latest_info, f, ensure_ascii=False, indent=2)

        return JSONResponse({
            "status": "success",
            "dataset_path": dataset_dir,
            "data_yaml_path": str(yaml_paths[0]) if yaml_paths else None,
            "report_path": str(report_path),
            "summary": format_dataset_summary(result),
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] Dataset folder analyze failed: {e}\n{tb}", flush=True)
        return JSONResponse({"error": f"分析失败: {str(e)}"}, status_code=500)


# ── Language API ──

@app.get("/api/lang")
async def set_lang(lang: str = Query("zh"), redirect: str = Query("/")):
    """Set language preference (redirects back to referring page)."""
    if lang not in ("zh", "en"):
        lang = "zh"
    from fastapi.responses import RedirectResponse
    response = RedirectResponse(url=redirect)
    response.set_cookie(key="lang", value=lang, max_age=86400 * 365)
    return response


# ── Project API ──

@app.put("/api/project")
async def update_project(request: Request):
    """Update project info in config.yaml."""
    body = await request.json()
    allowed = {"name", "description", "detection_target", "data_type", "data_yaml", "model"}
    to_save = {k: v for k, v in body.items() if k in allowed and isinstance(v, str)}
    if not to_save:
        return JSONResponse({"error": "No valid fields provided"}, status_code=400)
    success, msg = _update_config("project", to_save)
    if success:
        _invalidate_cache("load_data")
        return JSONResponse({"status": "success"})
    return JSONResponse({"error": msg}, status_code=500)


# ── Dataset Config API ──

@app.put("/api/dataset/config")
async def update_dataset_config(request: Request):
    """Update dataset_analyzer thresholds in config.yaml."""
    body = await request.json()
    # Accept any key-value pairs where value is numeric
    to_save = {k: v for k, v in body.items() if isinstance(v, (int, float))}
    if not to_save:
        return JSONResponse({"error": "No valid numeric fields provided"}, status_code=400)
    success, msg = _update_config("dataset_analyzer", to_save)
    if success:
        return JSONResponse({"status": "success"})
    return JSONResponse({"error": msg}, status_code=500)


# ── Dataset Split API ──

@app.get("/api/dataset/latest")
async def get_latest_dataset():
    """Get info about the most recently uploaded dataset."""
    if not LATEST_DATASET_PATH.exists():
        return JSONResponse({"dataset": None})
    with open(LATEST_DATASET_PATH, encoding="utf-8") as f:
        info = json.load(f)
    # Check if dataset path still exists
    ds_path = info.get("source_dataset_path") or info.get("dataset_path", "")
    info["path_exists"] = os.path.isdir(ds_path) if ds_path else False
    info["snapshot_valid"] = _latest_snapshot_valid(info)
    return JSONResponse({"dataset": info})


@app.post("/api/dataset/split")
async def split_dataset(request: Request):
    """Create an immutable dataset snapshot (Studio S1.2).

    The original dataset directory is never moved, renamed, or rewritten. The
    snapshot is materialized, verified, and atomically published by the
    dataset_snapshot service; only after success is ``latest_dataset.json``
    atomically updated.

    JSON body:
      - val_ratio: float in (0, 1), default 0.2
      - seed: int, default 42
    """
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    try:
        val_ratio, seed = _parse_split_params(body)
    except SnapshotValidationError as exc:
        return _snapshot_error_response(exc)

    ds_info = _read_latest_dataset()
    if not ds_info:
        return JSONResponse(
            {"error": "没有已上传的数据集，请先上传数据集", "error_code": "SNAPSHOT_VALIDATION_FAILED"},
            status_code=400,
        )
    source_path = Path(ds_info.get("source_dataset_path") or ds_info.get("dataset_path") or "")
    if not source_path or not source_path.is_dir():
        return JSONResponse(
            {"error": f"数据集目录不存在: {source_path}", "error_code": "SNAPSHOT_VALIDATION_FAILED"},
            status_code=400,
        )

    class_names = _load_class_names(source_path, ds_info)
    if not class_names:
        return JSONResponse(
            {"error": "无法从数据集确定类别映射 names，请确保 data.yaml 提供 names 字段",
             "error_code": "SNAPSHOT_VALIDATION_FAILED"},
            status_code=400,
        )

    try:
        snapshot = await asyncio.to_thread(
            create_dataset_snapshot,
            source_path,
            DATASET_SNAPSHOT_ROOT,
            val_ratio,
            seed,
            class_names,
        )
    except SnapshotValidationError as exc:
        return _snapshot_error_response(exc)
    except SnapshotConflictError as exc:
        return _snapshot_error_response(exc)
    except SnapshotInsufficientSpaceError as exc:
        return _snapshot_error_response(exc)
    except SnapshotIOError as exc:
        return _snapshot_error_response(exc)

    # Register latest only after the snapshot is fully published and verified.
    try:
        _write_json_atomic(LATEST_DATASET_PATH, _snapshot_to_latest_info(ds_info, snapshot))
    except OSError as exc:
        return JSONResponse(
            {"error": f"数据集快照已创建但登记失败: {exc}", "error_code": "SNAPSHOT_IO_FAILED"},
            status_code=500,
        )

    _invalidate_cache("load_data")
    return JSONResponse({
        "status": "success",
        "reused": snapshot.reused,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_path": str(snapshot.snapshot_path),
        "manifest_path": str(snapshot.manifest_path),
        "data_yaml_path": str(snapshot.data_yaml_path),
        "train_count": snapshot.train_count,
        "val_count": snapshot.val_count,
        "background_count": snapshot.background_count,
        "total_bytes": snapshot.total_bytes,
    })


# ── Training Upload & Analyze API ──

import zipfile
import tempfile
import shutil
import datetime


@app.post("/api/training/analyze")
async def upload_training(file: UploadFile = File(...)):
    """Upload a training report JSON or a ZIP of YOLO train directory for analysis."""
    if not file.filename:
        return JSONResponse({"error": "No file provided"}, status_code=400)

    if file.filename.endswith(".zip"):
        return await _analyze_train_zip(file)
    if file.filename.endswith(".json"):
        return await _analyze_train_json(file)
    return JSONResponse({"error": "Only JSON and ZIP files are supported"}, status_code=400)


async def _analyze_train_json(file: UploadFile) -> JSONResponse:
    """Handle JSON training report upload (existing behavior)."""
    try:
        content = await file.read()
        report = json.loads(content.decode("utf-8"))

        upload_id = f"train_{int(time.time())}"
        report_path = Path("log") / f"{upload_id}_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        _invalidate_cache("load_data")

        summary = {
            "total_runs": len(report.get("runs", {})),
            "best_mAP50": report.get("summary", {}).get("best_mAP50"),
            "avg_mAP50": report.get("summary", {}).get("avg_mAP50"),
            "runs_with_issues": report.get("summary", {}).get("runs_with_issues"),
            "common_issues": report.get("summary", {}).get("common_issues", []),
        }
        return JSONResponse({"status": "success", "summary": summary})

    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON file"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)


def _build_decision_summary(report: dict) -> str:
    """Build a concise training summary for the decision agent from a single-run report."""
    runs = report.get("runs", {})
    if not runs:
        return "No training run data available."

    lines = ["## Training Summary", ""]
    for run_name, run_data in runs.items():
        res = run_data.get("results", {})
        final = res.get("final_metrics", {})
        curve = run_data.get("curve_analysis", {})
        issues = run_data.get("issues", [])
        args = run_data.get("args", {})

        lines.append(f"### Run: {run_name}")
        lines.append(f"- Total epochs: {res.get('total_epochs', '?')}")
        lines.append(f"- Best epoch: {res.get('best_epoch', '?')}")
        if final:
            lines.append(f"- mAP50: {final.get('metrics/mAP50(B)', '?')}")
            lines.append(f"- mAP50-95: {final.get('metrics/mAP50-95(B)', '?')}")
            lines.append(f"- Precision: {final.get('metrics/precision(B)', '?')}")
            lines.append(f"- Recall: {final.get('metrics/recall(B)', '?')}")
        val_box = curve.get("val_box", {})
        if val_box:
            lines.append(f"- val_box_loss trend: {val_box.get('trend', '?')} (slope={val_box.get('slope', '?')})")
        es = curve.get("early_stopping", {})
        if es:
            lines.append(f"- Early stopping: {'triggered' if es.get('stopped_early') else 'not triggered'}")
        if issues:
            lines.append("- Detected issues:")
            for iss in issues:
                lines.append(f"  * [{iss.get('severity', '?')}] {iss.get('type', '?')}: {iss.get('detail', '')}")
        if args:
            lines.append(f"- Training args: epochs={args.get('epochs', '?')}, batch={args.get('batch', '?')}, lr0={args.get('lr0', '?')}, imgsz={args.get('imgsz', '?')}")
        lines.append("")
    return "\n".join(lines)


async def _analyze_train_zip(file: UploadFile) -> JSONResponse:
    """Handle ZIP upload of a YOLO train directory — parse results.csv + args.yaml, run full analysis."""
    import os

    tmp_dir = None
    try:
        content = await file.read()

        # Extract to temp directory
        tmp_dir = tempfile.mkdtemp(prefix="train_zip_")
        zip_path = os.path.join(tmp_dir, file.filename)
        with open(zip_path, "wb") as f:
            f.write(content)

        extract_dir = os.path.join(tmp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, extract_dir)

        # Find results.csv and args.yaml
        run_dir = None
        for root, dirs, files in os.walk(extract_dir):
            if "results.csv" in files and "args.yaml" in files:
                run_dir = root
                break

        if not run_dir:
            return JSONResponse({
                "error": "ZIP must contain results.csv and args.yaml (standard YOLO train output)"
            }, status_code=400)

        # Parse training run — override name with ZIP filename
        from auto_tune.modules.train_analyzer.results_parser import load_training_run
        from auto_tune.modules.train_analyzer.curve_analysis import (
            analyze_loss_curves, analyze_metric_curves, detect_early_stopping
        )
        from auto_tune.modules.train_analyzer.issue_detector import detect_issues
        from auto_tune.modules.train_analyzer.run_comparator import compare_runs, summarize_runs

        run_data = load_training_run(run_dir)
        # Use ZIP filename without extension as run name
        zip_name = os.path.splitext(os.path.basename(file.filename))[0]
        run_data["name"] = zip_name

        # Run full analysis (Stage 1 — Python-based, no token cost)
        ta_config = APP_CONFIG.get("train_analyzer", {})
        curve_analysis = analyze_loss_curves(run_data["results"], ta_config)
        metric_analysis = analyze_metric_curves(run_data["results"], ta_config)
        early_stop = detect_early_stopping(run_data, ta_config)
        curve_analysis["early_stopping"] = early_stop
        issues = detect_issues(run_data, ta_config)

        run_data["curve_analysis"] = curve_analysis
        run_data["metric_analysis"] = metric_analysis
        run_data["issues"] = issues

        # Build report
        report = {
            "module": "train_analyzer",
            "version": "1.0",
            "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "detect_dir": run_dir,
            "project": APP_CONFIG.get("project", {}),
            "total_runs": 1,
            "runs": {run_data["name"]: run_data},
            "comparison": compare_runs([run_data], ta_config),
            "summary": summarize_runs([run_data], ta_config),
        }

        # Stage 2: Text LLM diagnosis (if enabled)
        if APP_CONFIG.get("llm", {}).get("enabled", False):
            try:
                from auto_tune.modules.train_analyzer.llm_analyzer import analyze_with_llm
                llm_analysis = analyze_with_llm(report, APP_CONFIG)
                report["llm_analysis"] = llm_analysis
            except Exception as llm_err:
                report["llm_analysis"] = {"error": str(llm_err)}

        # Generate structured hyperparameter suggestions via Decision Agent
        if report.get("llm_analysis") and isinstance(report.get("llm_analysis"), dict) and not report["llm_analysis"].get("error"):
            try:
                from auto_tune.modules.agent_engine.decision_agent import (
                    build_decision_prompt, call_decision_llm, _extract_json
                )
                summary_text = _build_decision_summary(report)
                decision_prompt = build_decision_prompt(summary_text, APP_CONFIG.get("project", {}))
                raw_response = call_decision_llm(decision_prompt, APP_CONFIG)
                parsed = _extract_json(raw_response)
                if parsed:
                    report["suggestion"] = {
                        "diagnosis": parsed.get("diagnosis"),
                        "action": parsed.get("action"),
                        "hyperparameter_changes": parsed.get("hyperparameter_changes", {}),
                        "training_overrides": parsed.get("training_overrides", {}),
                    }
                else:
                    report["suggestion"] = {"error": "Failed to parse JSON from LLM response"}
            except Exception as sug_err:
                report["suggestion"] = {"error": str(sug_err)}

        # Stage 3: Vision consultation (if enabled — requires confusion matrix PNGs in train dir)
        if APP_CONFIG.get("vision", {}).get("enabled", False):
            try:
                from auto_tune.modules.train_analyzer.vision_analyzer import multimodal_consult
            except ImportError:
                multimodal_consult = None

            vision_results = {}
            for name, rd in report["runs"].items():
                if multimodal_consult is None:
                    vision_results[name] = {"run_name": name, "error": "Vision analysis module not available (missing dependencies)"}
                else:
                    try:
                        vision_results[name] = multimodal_consult(run_dir, APP_CONFIG, report.get("project", {}))
                    except Exception as vis_err:
                        vision_results[name] = {"run_name": name, "error": str(vis_err)}
            report["vision_analysis"] = vision_results

        # Save report
        upload_id = f"train_{int(time.time())}"
        report_path = Path("log") / f"{upload_id}_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        _invalidate_cache("load_data")

        s = report["summary"]
        summary = {
            "total_runs": 1,
            "best_mAP50": s.get("best_mAP50"),
            "avg_mAP50": s.get("average_mAP50"),
            "runs_with_issues": s.get("runs_with_issues"),
            "common_issues": s.get("common_issues", []),
            "run_name": run_data["name"],
            "epochs": run_data["results"].get("total_epochs"),
            "best_epoch": run_data["results"].get("best_epoch"),
        }
        return JSONResponse({"status": "success", "summary": summary})

    except Exception as e:
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Training Folder Analyze API ──


@app.post("/api/training/analyze-folder")
async def analyze_training_folder(request: Request):
    """Analyze a YOLO train directory directly from a server path (no ZIP upload)."""
    try:
        data = await request.json()
        folder_path = data.get("path", "").strip()

        if not folder_path or not os.path.isdir(folder_path):
            return JSONResponse({"error": f"文件夹路径无效或不存在: {folder_path}"}, status_code=400)

        # Check for results.csv and args.yaml
        run_dir = folder_path
        has_csv = os.path.exists(os.path.join(run_dir, "results.csv"))
        has_args = os.path.exists(os.path.join(run_dir, "args.yaml"))
        if not has_csv or not has_args:
            return JSONResponse({
                "error": "训练目录必须包含 results.csv 和 args.yaml（标准 YOLO 训练输出）"
            }, status_code=400)

        # Run full analysis (same logic as _analyze_train_zip)
        from auto_tune.modules.train_analyzer.results_parser import load_training_run
        from auto_tune.modules.train_analyzer.curve_analysis import (
            analyze_loss_curves, analyze_metric_curves, detect_early_stopping,
        )
        from auto_tune.modules.train_analyzer.issue_detector import detect_issues
        from auto_tune.modules.train_analyzer.run_comparator import compare_runs, summarize_runs

        run_data = load_training_run(run_dir)
        run_data["name"] = os.path.basename(run_dir)

        ta_config = APP_CONFIG.get("train_analyzer", {})
        curve_analysis = analyze_loss_curves(run_data["results"], ta_config)
        metric_analysis = analyze_metric_curves(run_data["results"], ta_config)
        early_stop = detect_early_stopping(run_data, ta_config)
        curve_analysis["early_stopping"] = early_stop
        issues = detect_issues(run_data, ta_config)

        run_data["curve_analysis"] = curve_analysis
        run_data["metric_analysis"] = metric_analysis
        run_data["issues"] = issues

        report = {
            "module": "train_analyzer",
            "version": "1.0",
            "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "detect_dir": run_dir,
            "project": APP_CONFIG.get("project", {}),
            "total_runs": 1,
            "runs": {run_data["name"]: run_data},
            "comparison": compare_runs([run_data], ta_config),
            "summary": summarize_runs([run_data], ta_config),
        }

        # Stage 2: LLM analysis (if enabled)
        if APP_CONFIG.get("llm", {}).get("enabled", False):
            try:
                from auto_tune.modules.train_analyzer.llm_analyzer import analyze_with_llm
                llm_analysis = analyze_with_llm(report, APP_CONFIG)
                report["llm_analysis"] = llm_analysis
            except Exception as llm_err:
                report["llm_analysis"] = {"error": str(llm_err)}

        # Decision Agent suggestions
        if report.get("llm_analysis") and isinstance(report.get("llm_analysis"), dict) and not report["llm_analysis"].get("error"):
            try:
                from auto_tune.modules.agent_engine.decision_agent import (
                    build_decision_prompt, call_decision_llm, _extract_json,
                )
                summary_text = _build_decision_summary(report)
                decision_prompt = build_decision_prompt(summary_text, APP_CONFIG.get("project", {}))
                raw_response = call_decision_llm(decision_prompt, APP_CONFIG)
                parsed = _extract_json(raw_response)
                if parsed:
                    report["suggestion"] = {
                        "diagnosis": parsed.get("diagnosis"),
                        "action": parsed.get("action"),
                        "hyperparameter_changes": parsed.get("hyperparameter_changes", {}),
                        "training_overrides": parsed.get("training_overrides", {}),
                    }
                else:
                    report["suggestion"] = {"error": "Failed to parse JSON from LLM response"}
            except Exception as sug_err:
                report["suggestion"] = {"error": str(sug_err)}

        # Stage 3: Vision consultation (if enabled)
        if APP_CONFIG.get("vision", {}).get("enabled", False):
            try:
                from auto_tune.modules.train_analyzer.vision_analyzer import multimodal_consult
            except ImportError:
                multimodal_consult = None
            vision_results = {}
            for name, rd in report["runs"].items():
                if multimodal_consult is None:
                    vision_results[name] = {"run_name": name, "error": "Vision analysis module not available"}
                else:
                    try:
                        vision_results[name] = multimodal_consult(run_dir, APP_CONFIG, report.get("project", {}))
                    except Exception as vis_err:
                        vision_results[name] = {"run_name": name, "error": str(vis_err)}
            report["vision_analysis"] = vision_results

        # Save report
        upload_id = f"train_{int(time.time())}"
        report_path = Path("log") / f"{upload_id}_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        _invalidate_cache("load_data")

        s = report["summary"]
        summary = {
            "total_runs": 1,
            "best_mAP50": s.get("best_mAP50"),
            "avg_mAP50": s.get("average_mAP50"),
            "runs_with_issues": s.get("runs_with_issues"),
            "common_issues": s.get("common_issues", []),
            "run_name": run_data["name"],
            "epochs": run_data["results"].get("total_epochs"),
            "best_epoch": run_data["results"].get("best_epoch"),
        }
        return JSONResponse({"status": "success", "summary": summary})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] Training folder analyze failed: {e}\n{tb}", flush=True)
        return JSONResponse({"error": f"分析失败: {str(e)}"}, status_code=500)


# ── First-time Training API ──

@app.post("/api/training/start")
async def start_first_training(request: Request):
    """Start a first-time YOLO training with SSE progress streaming."""
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    import os as _os

    # Read params from request body or config.yaml
    training_cfg = APP_CONFIG.get("training", {})
    project_cfg = APP_CONFIG.get("project", {})
    data_yaml = body.get("data_yaml") or ""
    if not data_yaml:
        latest = _read_latest_dataset()
        if latest and latest.get("snapshot_id"):
            try:
                data_yaml = str(_resolve_validated_snapshot_data_yaml(latest))
            except SnapshotError as exc:
                return _snapshot_error_response(exc)
        else:
            data_yaml = project_cfg.get("data_yaml") or training_cfg.get("data_yaml", "")
    model = body.get("model") or project_cfg.get("model") or training_cfg.get("model", "yolov8n.pt")
    epochs = int(body.get("epochs", training_cfg.get("default_epochs", 100)))
    imgsz = int(body.get("imgsz", training_cfg.get("imgsz", 640)))
    batch = int(body.get("batch", training_cfg.get("batch", 16)))
    workers = int(body.get("workers", training_cfg.get("workers", 8)))
    patience = int(training_cfg.get("patience", 20))

    if not data_yaml:
        return JSONResponse(
            {"error": "数据集路径 (data.yaml) 未配置，请在项目设置中填写"},
            status_code=400,
        )

    async def event_stream():
        nonlocal data_yaml, model, epochs, imgsz, batch, workers, patience

        # Import executor helpers
        from auto_tune.modules.agent_engine.executor import find_detect_dir

        cfg = {}
        try:
            import yaml as _yaml
            import json as _json
            import os as _os2

            # Determine directories
            detect_dir = find_detect_dir()
            # Find next train name (regular train dirs)
            _os2.makedirs(detect_dir, exist_ok=True)
            max_n = 0
            for _d in _os2.listdir(detect_dir):
                if _os2.path.isdir(_os2.path.join(detect_dir, _d)):
                    import re as _re
                    _m = _re.match(r"^train(\d+)$", _d)
                    if _m:
                        max_n = max(max_n, int(_m.group(1)))
            train_name = f"train{max_n + 1}"
            train_dir = _os2.path.join(detect_dir, train_name)
            _os2.makedirs(train_dir, exist_ok=True)

            # Build params
            params = {
                "model": model,
                "data": _os2.path.abspath(data_yaml),
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "workers": workers,
                "patience": patience,
                "name": train_name,
                "project": _os2.path.abspath(detect_dir),
                "exist_ok": "True",
                "plots": True,
                "save": True,
                "device": "0",
            }

            # Write args.yaml for reference
            with open(_os2.path.join(train_dir, "args.yaml"), "w", encoding="utf-8") as _f:
                _yaml.dump(params, _f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            # Build yolo command
            from auto_tune.modules.agent_engine.executor import resolve_yolo_executable
            cmd = [resolve_yolo_executable(), "train"]
            for _k, _v in params.items():
                cmd.append(f"{_k}={_v}")

            yield f"data: {_json.dumps({'status': 'running', 'message': f'启动训练: {train_name}', 'level': 'info', 'train_name': train_name})}\n\n"
            yield f"data: {_json.dumps({'status': 'running', 'message': f'数据集: {data_yaml}', 'level': 'info'})}\n\n"
            yield f"data: {_json.dumps({'status': 'running', 'message': f'模型: {model}  |  轮次: {epochs}  |  batch: {batch}  |  imgsz: {imgsz}', 'level': 'info'})}\n\n"

            # Capture started_at before launch, then launch subprocess
            start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=1024 * 128,
            )

            # Track globally
            _running_training.clear()
            _running_training.update({
                "train_name": train_name,
                "train_dir": train_dir,
                "proc": proc,
                "start_time": time.time(),
                "start_iso": start_iso,
                "status": "running",
            })
            # Also write a status file so it survives reload
            status_f = _os2.path.join("log", "training_running.json")
            with open(status_f, "w", encoding="utf-8") as _sf:
                _json.dump({
                    "train_name": train_name,
                    "status": "running",
                    "start_time": time.time(),
                }, _sf)

            # Open the run's full-log file (new run dir => empty file is safe)
            _log_path = os.path.join(train_dir, "training.log")
            with open(_log_path, "w", encoding="utf-8") as _lf:
                _lf.write("")

            # Single consumer: this coroutine reads the merged stdout/stderr,
            # persists every line to training.log and emits bounded SSE batches.
            _warned_once = set()
            _epoch_keys = set()
            _batcher = _SseBatch()
            while True:
                _line_b = await proc.stdout.readline()
                if not _line_b:
                    break
                _line = _line_b.decode("utf-8", errors="replace").rstrip()
                if not _line:
                    continue
                _payload = _process_training_output_line(_line, train_name, _log_path, _warned_once)
                if _payload is None:
                    continue
                # Dedup default summary for the same epoch/validation row: the
                # detail still reaches the log and full panel, only message drops.
                if _payload.get("event") == "training_log" and _payload.get("log_kind") in ("epoch", "validation"):
                    _key = (_payload.get("log_kind"), _payload.get("epoch"), _payload.get("message"))
                    if _key in _epoch_keys:
                        _payload["message"] = None
                    else:
                        _epoch_keys.add(_key)
                _batcher.add(_payload)
                if _batcher.should_flush():
                    yield _batcher.take()
            if _batcher.pending:
                yield _batcher.take()

            await proc.wait()

            # Preserve an aborted state set by /api/training/stop; otherwise
            # record the real terminal state of the subprocess.
            if _running_training.get("status") != "aborted":
                _running_training["status"] = "completed" if proc.returncode == 0 else "failed"
            # The status file must not stay "running" in any terminal state.
            _remove_status_file()
            event = _finalize_and_build_event(
                proc.returncode, train_dir, train_name, APP_CONFIG, "log", _running_training.get("start_iso")
            )
            yield f"data: {_json.dumps(event, ensure_ascii=False, default=str)}\n\n"

        except Exception as _exc:
            yield f"data: {_json.dumps({'status': 'error', 'message': f'异常: {_exc}', 'level': 'error'})}\n\n"
        finally:
            _invalidate_cache("load_data")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/training/running")
async def training_running_status():
    """Check if a first-time training is currently running.

    Reconciles the in-memory process first: if it finished, the terminal
    status (completed/failed) is recorded and the status file is cleaned up so
    it can never stay "running". Terminal in-memory states take precedence over
    the status file; the file only reports running=True to survive a server
    restart that happened mid-training (in-memory state was lost).
    """
    if _running_training.get("proc") and _running_training["status"] == "running":
        _proc = _running_training["proc"]
        _ret = _proc.poll()
        if _ret is None:
            return JSONResponse({
                "running": True,
                "train_name": _running_training.get("train_name"),
                "status": "running",
                "elapsed": time.time() - _running_training.get("start_time", time.time()),
            })
        _running_training["status"] = "completed" if _ret == 0 else "failed"
        _remove_status_file()

    _status = _running_training.get("status")
    if _status in ("completed", "failed", "aborted"):
        return JSONResponse({"running": False, "status": _status})

    # Status file fallback (survives server restart).
    _sf = os.path.join("log", "training_running.json")
    if os.path.exists(_sf):
        try:
            with open(_sf, encoding="utf-8") as _f:
                data = json.load(_f)
            if data.get("status") == "running":
                return JSONResponse({"running": True, **data})
        except Exception:
            pass

    return JSONResponse({"running": False, "status": "idle"})


# ── Run ──
def start_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    import time
    # Preload cache so the first request is fast
    t0 = time.time()
    _log = f"[{time.strftime('%H:%M:%S')}] Preloading data cache ...\n"
    _ = _common_context()
    t1 = time.time()
    _log += f"[{time.strftime('%H:%M:%S')}] Cache warmed in {t1 - t0:.1f}s\n"
    # Write to a marker file that we can read later
    with open("log/_startup_timing.txt", "w") as _fh:
        _fh.write(_log)
    print(_log.strip(), flush=True)
    print(f"[Auto-Tune] Dashboard at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info",
                timeout_keep_alive=30)


if __name__ == "__main__":
    start_server()
