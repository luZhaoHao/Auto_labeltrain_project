"""FastAPI web application for Auto-Tune dashboard.

Provides three major pages:
- Dashboard: overview of all modules
- Dataset Analysis: Module A results
- Training Analysis & Tuning: Module B results + Module C auto-tuning
"""

import json
import os
import re
import time
import secrets
import asyncio
import threading
import zipfile
import tempfile
import shutil
import datetime
from contextlib import asynccontextmanager
from dataclasses import replace
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
from auto_tune.modules.agent_engine.decision_agent import generate_suggestion
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
from auto_tune.modules.input_safety import (
    InputSafetyError,
    InputSafetyPolicy,
    list_safe_subdirectories,
    load_input_safety_policy,
    scan_directory_bounded,
    validate_directory_path,
)
from auto_tune.modules.local_index import (
    ExperimentQuery,
    LocalIndexError,
    LocalIndexService,
    load_local_index_config,
)
from auto_tune.modules.reference_dataset import (
    ReferenceDatasetError,
    resolve_reference_dataset,
)
from auto_tune.modules.agent_engine.perception import find_module_b_report
from auto_tune.modules.agent_engine.executor import find_detect_dir
from auto_tune.modules.run_state.models import RunStatePersistenceError
from auto_tune.modules.run_state.service import (
    new_run_state,
    project_public_state,
    read_run_state,
    with_terminal,
    write_run_state,
)
from auto_tune.modules.run_state.process_identity import (
    capture_process_identity,
    reconcile_persisted_state,
)
from auto_tune.modules.run_state.gate import (
    evaluate_persisted_run_state,
    hpo_storage_has_live_process,
)
from auto_tune.modules.run_state.events import EventBroker
from auto_tune.modules.run_state.manager import (
    _RUN_MANAGER,
    TrainingBusyError,
)
from auto_tune.modules.run_state.manual_controller import ManualRunController
from auto_tune.modules.run_state.tuning_controller import TuningRunController
from auto_tune.modules.hpo import HpoError, HpoRunner, HpoService
from auto_tune.modules.model_store import ModelStore, ModelStoreError
from .hpo_api import create_hpo_router, safe_hpo_error_code
from .hpo_reuse import resolve_hpo_verification
from .hpo_training import TrainingSubmitDeps, create_hpo_training_router
from .model_store_api import create_model_store_router


def _finalize_and_build_event(
    returncode: int,
    train_dir: str,
    train_name: str,
    config: dict,
    log_dir: str,
    started_at: str | None,
    cancelled: bool = False,
    runtime_run_id: str | None = None,
    local_index_service=None,
) -> dict:
    """Finalize a finished training and build the unified SSE completion event."""
    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    training_status = "completed" if returncode == 0 else "failed"
    training_error = None
    if returncode != 0:
        if cancelled or _running_training.get("status") == "aborted":
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
        runtime_run_id=runtime_run_id,
        local_index_service=local_index_service,
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
    """Legacy no-op.

    S1.5 expresses terminal states by persisting them, never by deleting the
    status file. Kept as a real no-op so any stale call site cannot resurrect
    the "delete file to stop" behavior.
    """
    pass


def _persist_run_state(state_file, state) -> None:
    """Atomically persist a run state; propagates RunStatePersistenceError."""
    write_run_state(state_file, state)


def _hpo_storage_root() -> str:
    """Controlled HPO study storage root (frozen below the existing log root)."""
    return os.path.join("log", "hpo", "studies")


def _busy_training_response() -> JSONResponse:
    return JSONResponse(
        {"error": "已有训练在运行（普通训练/大模型调参/HPO），请先停止或等待完成",
         "error_code": "RUN_ALREADY_ACTIVE"},
        status_code=409,
    )


def _reconcile_failed_response() -> JSONResponse:
    """Controlled 503 for a failed terminal write-back (no exception detail).

    Deliberately carries no path, PID, command or original message: the caller
    only needs to know that the slot could not be proven free.
    """
    return JSONResponse(
        {"error": "无法确认上一训练已安全收敛，本次未启动训练",
         "error_code": "RUN_STATE_RECONCILE_FAILED",
         "next_action": "请检查日志目录写入权限、磁盘空间与运行状态记录后重试；"
                        "在确认上一训练已结束后系统才会允许启动新训练。"},
        status_code=503,
    )


def _assert_training_slot_free() -> JSONResponse | None:
    """Real-training mutual exclusion across manual/tuning/hpo + restart facts.

    Returns a 409 response when the single real-training slot is occupied by a
    live in-memory controller, by an already-taken reservation, OR by a leftover
    persisted ``starting``/``running`` record whose process may still be alive
    (restart). A record that is provably gone (MISSING/MISMATCH) is reconciled to
    a terminal ``interrupted`` state — and that write-back must *succeed* before
    a new run is allowed. A failed write-back returns 503
    (``RUN_STATE_RECONCILE_FAILED``) instead of silently freeing the slot: the
    old run may still be alive and the record would still claim it is running.

    The reservation check is not redundant with the controller check: starting a
    run takes the reservation *before* it creates/registers its controller, so a
    gate that only looked at active controllers would let a second caller slip
    into that atomic gap.
    """
    if _RUN_MANAGER.reservation_owner() is not None:
        return _busy_training_response()
    if _RUN_MANAGER.active_train() is not None:
        return _busy_training_response()
    for run_kind, filename in (("manual", "training_running.json"),
                               ("tuning", "tuning_running.json")):
        state_file = os.path.join("log", filename)
        state = read_run_state(state_file, run_kind=run_kind)
        decision = evaluate_persisted_run_state(state)
        if decision.persist is not None:
            try:
                _persist_run_state(state_file, decision.persist)
            except RunStatePersistenceError:
                # Conservatively block: the orphaned record still says the run
                # is alive, and nothing proves the process is gone.
                return _reconcile_failed_response()
        if decision.blocked:
            return _busy_training_response()
    if hpo_storage_has_live_process(_hpo_storage_root()):
        # A leftover RUNNING HPO subprocess may still be alive.
        return _busy_training_response()
    return None


def _reserve_or_busy(run_kind: str, run_id: str) -> tuple[str | None, JSONResponse | None]:
    """Try to reserve the training slot; returns (token, busy_response)."""
    try:
        token = _RUN_MANAGER.reserve(run_kind, run_id)
        return token, None
    except TrainingBusyError:
        return None, _busy_training_response()


def _release_reservation(token) -> None:
    if token is None:
        return
    try:
        _RUN_MANAGER.release(token)
    except Exception:
        pass


_TRAIN_DIR_RE = re.compile(r"^train(\d+)$")
_MAX_TRAIN_NAME_PROBES = 1000


def _next_train_index(detect_dir: str) -> int:
    """Highest existing ``trainN`` index under ``detect_dir`` (read-only)."""
    try:
        names = os.listdir(detect_dir)
    except OSError:
        return 0
    max_n = 0
    for name in names:
        m = _TRAIN_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(detect_dir, name)):
            max_n = max(max_n, int(m.group(1)))
    return max_n


def _pick_train_name(detect_dir: str) -> str:
    """Next free ``trainN`` name; creates nothing and never overwrites.

    A name is skipped while *any* file-system entry already uses it, so a
    leftover file named ``train7`` can never be silently clobbered by a run.
    """
    n = _next_train_index(detect_dir)
    for _ in range(_MAX_TRAIN_NAME_PROBES):
        candidate = f"train{n + 1}"
        if not os.path.exists(os.path.join(detect_dir, candidate)):
            return candidate
        n += 1
    raise OSError("no free training directory name available")


def _create_train_dirs(detect_dir: str, train_dir: str) -> None:
    """Create the run directory (only ever called with the slot reserved)."""
    os.makedirs(detect_dir, exist_ok=True)
    os.mkdir(train_dir)


def _train_dir_error_response() -> JSONResponse:
    return JSONResponse(
        {"error": "训练目录创建失败，未启动训练",
         "error_code": "TRAIN_DIR_CREATE_FAILED",
         "next_action": "请检查 Detect 目录权限、磁盘空间与同名占用后重试。"},
        status_code=500,
    )


def _run_dir_write_error_response(error_code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": message, "error_code": error_code,
         "next_action": "系统未启动训练，请检查运行目录写入权限后重试。"},
        status_code=500,
    )


_RUN_STATE_DETAIL_THROTTLE = 10


async def _run_sse(broker, controller, after_seq):
    """Subscribe to a run's event bus and stream events to the client.

    The controller owns the run; this generator only drains the broker and
    unsubscribes when the connection closes. It never cancels the controller.
    """
    import queue as _queue

    q, replay = broker.subscribe(after_seq)
    try:
        if broker.replay_truncated(after_seq):
            # The ring buffer already evicted some events after after_seq.
            # This is a transport/control warning, NOT a terminal event: it must
            # never look like the run finished, so it carries no terminal status
            # and no "terminal" phase, and it is exempt from the per-run_id
            # event_seq sequence (no event_seq — the frontend skips seq tracking
            # for it and handles it before any terminal check).
            yield _sse_chunk([{
                "status": "running", "level": "warning",
                "message": "部分历史日志不可重放，当前运行仍在继续",
                "run_id": broker.run_id, "phase": "replaying",
                "event": "replay_truncated", "replay_truncated": True,
            }])
        for ev in replay:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        while True:
            try:
                while True:
                    ev = q.get_nowait()
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            except _queue.Empty:
                pass
            if controller is not None and controller.is_done():
                # Terminal events are already published to the buffer; drain any
                # leftovers so a reconnecting client always sees them.
                try:
                    while True:
                        ev = q.get_nowait()
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except _queue.Empty:
                    pass
                break
            await asyncio.sleep(0.15)
    finally:
        broker.unsubscribe(q)


def _manual_finalize_cb(controller, returncode):
    """Build the unified completion event for a finished manual training run."""
    event = _finalize_and_build_event(
        returncode,
        controller.train_dir,
        controller.train_name,
        APP_CONFIG,
        "log",
        controller.started_iso,
        cancelled=controller._stop_applied,
        runtime_run_id=controller.run_id,
        local_index_service=_local_index_service(),
    )
    event["run_id"] = controller.run_id
    event["phase"] = "terminal"
    return event


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
from .components.tuning_panel import get_tuning_history
from .components.experiment_panel import get_experiment_history, get_experiment_history_view
from .i18n import make_translator, translate
from auto_tune.modules.presentation import build_experiment_labels
from auto_tune.modules.local_index.projection import project_outward
from auto_tune.modules.presentation.experiment_views import ExperimentViewError


def _local_index_startup_backfill() -> None:
    """Bounded, idempotent, non-fatal startup catch-up of recent JSON facts.

    The local index is a rebuildable projection; after a restart (or an index
    outage) the newest facts are backfilled on startup. The operation is bounded
    to the most recent records, never guesses terminal states, persists a
    bounded maintenance summary (readable via diagnostics), and a broken SQLite
    index or corrupt fact file never blocks the app or training from starting.
    """
    try:
        service = _local_index_service()
        if service is None:
            return
        service.backfill_startup(max_records=50)
    except LocalIndexError:
        pass
    except Exception:
        pass


@asynccontextmanager
async def _lifespan(_app):
    _local_index_startup_backfill()
    yield


app = FastAPI(title="Auto-Tune Dashboard", lifespan=_lifespan)

# ── Direct Jinja2 (avoid Starlette TemplateResponse compatibility issue) ──
import jinja2
_templates_dir = str(Path(__file__).parent / "templates")
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_templates_dir),
    enable_async=False,
    auto_reload=True,
)


def _jinja_basename(value):
    """Template filter: reduce a path-like value to its basename for display."""
    if isinstance(value, str):
        return os.path.basename(value)
    return value


def _merge_suggestion_changes(sections) -> list:
    """Merge the two LLM suggestion sections into one deduped display list.

    ``sections`` is ``[hyperparameter_changes, training_overrides]``. Matches
    execution-side priority (``sanitize_tuning_parameters`` applies overrides
    last): on a same-named key the training override wins. Returns an ordered
    list of ``[key, value, kind]`` where kind is ``hyperparameter`` or
    ``training``, keeping hyperparameter changes first.
    """
    if not isinstance(sections, (list, tuple)) or len(sections) != 2:
        return []
    result: list[list] = []
    index: dict[str, int] = {}
    for kind, section in (("hyperparameter", sections[0]), ("training", sections[1])):
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if key in ("diagnosis", "action"):
                # Never display pseudo-keys as parameters.
                continue
            if key in index:
                result[index[key]] = [key, value, kind]
            else:
                index[key] = len(result)
                result.append([key, value, kind])
    return result


_jinja_env.filters["basename"] = _jinja_basename
_jinja_env.filters["merge_suggestion_changes"] = _merge_suggestion_changes

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

# Whether the auto-tuning loop is still owned by this worker's memory (used to
# decide controller_owned when reconciling the persisted tuning run state).
_tuning_loop_active = False


def _local_index_service() -> LocalIndexService | None:
    """Build a fresh local-index service; defaults apply when config is absent.

    Paths are resolved through ``os.path.join`` so tests that redirect
    ``os.path.join("log", ...)`` get an isolated database automatically. A
    malformed ``local_index`` section degrades to the documented defaults so a
    bad config can never take the whole dashboard down.
    """
    section = (APP_CONFIG.get("local_index") or {}) if APP_CONFIG else {}
    section = dict(section)
    section.setdefault("database_path", os.path.join("log", "auto_tune.db"))
    section.setdefault("backup_dir", os.path.join("log", "db_backups"))
    try:
        cfg = load_local_index_config({"local_index": section})
    except LocalIndexError:
        cfg = load_local_index_config({"local_index": {
            "database_path": os.path.join("log", "auto_tune.db"),
            "backup_dir": os.path.join("log", "db_backups"),
        }})
    return LocalIndexService(cfg)


def _local_index_error_response(exc: LocalIndexError) -> JSONResponse:
    """Map a local-index domain error to a stable 5xx response.

    Never leaks raw SQL, tracebacks, or the full database path.
    """
    return JSONResponse(
        {"error": exc.error_code, "error_code": exc.error_code},
        status_code=exc.status_code,
    )


def _experiment_view_error_response(exc: ExperimentViewError) -> JSONResponse:
    """Map a P5 report/audit view projection error to a stable HTTP response.

    The stable ``error_code`` drives the UI message; the body never carries a
    traceback, a raw SQL string, or an absolute artifact path.
    """
    return JSONResponse(
        {"error": exc.error_code, "error_code": exc.error_code},
        status_code=exc.status_code,
    )


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
    return "deepseek-flash" if purpose == "text" else "qwen-vl-plus"


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
    # Bugfix P4: inject the shared display vocabulary so the template renders
    # fields/enums through the same module future P5 report generators reuse.
    context.setdefault("experiment_labels", build_experiment_labels(_))
    template = _jinja_env.get_template(template_name)
    return template.render(_=_, current_lang=lang, **context)


# ── Helper: load report for template ──
def _list_dataset_index(service) -> list:
    """Best-effort dataset index for the dataset page; never raises."""
    if service is None:
        return []
    try:
        return service.list_datasets()
    except LocalIndexError:
        return []


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
    service = _local_index_service()
    view = get_experiment_history_view(service=service)
    dataset_index = _list_dataset_index(service)
    dataset_analyzer_config = APP_CONFIG.get("dataset_analyzer", {})

    result = {
        "dataset": dataset,
        "training": training,
        "project": project,
        "tuning_history": tuning_history,
        "experiment_history": view["experiments"],
        "experiment_history_source": view["source"],
        "experiment_index_warning": view["index_warning"],
        "dataset_index": dataset_index,
        "dataset_analyzer_config": dataset_analyzer_config,
    }
    _set_cache(cache_key, result)
    return result


# ── Helper: assemble suggestion from tuning history or LLM analysis ──
def _project_decision(decision):
    """Project a stored decision/suggestion object into the UI suggestion shape.

    Rationale is always sourced from ``action`` (never a non-existent
    ``llm_rationale``). Structured failures surface a stable error instead of a
    fake "no suggestions". Returns ``None`` when the object carries neither a
    structured failure nor an actionable decision.
    """
    if not isinstance(decision, dict):
        return None
    if decision.get("error"):
        return {
            "diagnosis": "",
            "rationale": "",
            "action": "",
            "hyperparameter_changes": {},
            "training_overrides": {},
            "error": decision["error"],
        }
    changes = decision.get("hyperparameter_changes", {})
    if changes or decision.get("action"):
        return {
            "diagnosis": decision.get("diagnosis", ""),
            "rationale": decision.get("action", ""),
            "action": decision.get("action", ""),
            "hyperparameter_changes": changes,
            "training_overrides": decision.get("training_overrides", {}),
            "error": None,
        }
    return None


def _get_latest_suggestion(tuning_history, training):
    """Extract the structured suggestion that belongs to the analysed page.

    Source priority (F1.1): the current training report's own ``suggestion``
    always wins — success, ``keep_params`` and structured failure alike. Tuning
    history is only a fallback, and only entries whose ``train_name`` matches a
    run in the current report may be used; unrelated runs never leak into the
    page. Without a training report the legacy "latest history entry" behaviour
    is preserved. A plain LLM diagnosis without a structured decision is never
    presented as an executable suggestion.
    """
    if training:
        projected = _project_decision(training.get("suggestion"))
        if projected is not None:
            return projected
        # No report-level suggestion: only a history entry bound to the same
        # run may fill in. Never guess by timestamp or filename similarity.
        runs = training.get("runs")
        run_names = {name for name in runs if name} if isinstance(runs, dict) else set()
        for entry in reversed(tuning_history or []):
            if not isinstance(entry, dict):
                continue
            if entry.get("train_name") not in run_names:
                continue
            projected = _project_decision(entry.get("decision"))
            if projected is not None:
                return projected
        return None
    if tuning_history and isinstance(tuning_history[-1], dict):
        projected = _project_decision(tuning_history[-1].get("decision"))
        if projected is not None:
            return projected
    return None


def _llm_analysis_bound_run(training):
    """Run that owns the current report's structured suggestion, or ``None``.

    Single-run reports bind to that run; multi-run reports only bind when
    ``summary.best_overall_run`` names a run that truly exists. Anything else is
    ambiguous and must not be guessed.
    """
    runs = training.get("runs")
    if not isinstance(runs, dict) or not runs:
        return None
    names = [name for name in runs if name]
    if len(names) == 1:
        return names[0]
    best = (training.get("summary") or {}).get("best_overall_run")
    if best and best in runs:
        return best
    return None


def _get_llm_analysis_display(training):
    """Read-only projection of the LLM analysis belonging to the current report.

    Entries with nothing renderable are dropped so the template never shows a
    title-only card. An empty plain diagnosis is filled from the same report's
    structured ``suggestion.diagnosis`` only when the run identity is certain; a
    stored error is kept and never masked by that fallback. The source report is
    never mutated and no value is ever taken from another run's analysis.
    """
    if not training:
        return None
    analysis = training.get("llm_analysis")
    if not isinstance(analysis, dict) or not analysis:
        return None
    suggestion = training.get("suggestion")
    suggestion = suggestion if isinstance(suggestion, dict) else {}
    fallback = suggestion.get("diagnosis")
    fallback = fallback.strip() if isinstance(fallback, str) else ""
    bound_run = _llm_analysis_bound_run(training) if fallback else None

    display = {}
    for run_name, diag in analysis.items():
        if not isinstance(diag, dict):
            continue
        raw = diag.get("llm_diagnosis")
        text = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
        entry = {
            "llm_diagnosis": text,
            "model_used": diag.get("model_used"),
            "error": diag.get("error"),
        }
        if not text.strip() and not entry["error"] and run_name == bound_run:
            entry["llm_diagnosis"] = fallback
        if not entry["llm_diagnosis"].strip() and not entry["error"]:
            continue
        display[run_name] = entry
    return display or None


def _monitor_latest_projection(training, experiment_history):
    """确定性投影“最近有效训练”，供训练监控首屏渲染（P1 返修）。

    顺序复用服务端既有规则：``experiment_history`` 已是统一实验索引的确定性顺序
    （SQLite 按 ``finished_at`` 降序，见 ``ExperimentQuery``）。这里取其中最新一条
    已完成、有指标、且在当前训练报告 ``training.runs`` 里有对应 run 的记录，用报告
    里的 epochs 与最终指标（同一 run 的权威事实）。找不到时返回 ``None``，页面诚实
    显示“—”；绝不依赖 Jinja for 循环里的局部变量逸出，也不按字典遍历顺序猜测。
    """
    runs = (training or {}).get("runs") or {}
    for record in experiment_history or ():
        if not isinstance(record, dict):
            continue
        run_name = record.get("run_name")
        if record.get("status") != "completed" or not run_name or run_name not in runs:
            continue
        entry = runs.get(run_name) or {}
        metrics = (entry.get("results") or {}).get("final_metrics") or {}
        if not metrics:
            continue
        args = entry.get("args") or {}
        mapping = (
            ("mAP50", "metrics/mAP50(B)"),
            ("mAP50_95", "metrics/mAP50-95(B)"),
            ("precision", "metrics/precision(B)"),
            ("recall", "metrics/recall(B)"),
        )
        projection = {"run_name": run_name,
                      "epochs": args.get("epochs")}
        for key, source_key in mapping:
            raw = metrics.get(source_key)
            if raw is None or raw == "":
                projection[key] = None
                continue
            try:
                projection[key] = float(raw)
            except (TypeError, ValueError):
                projection[key] = None
        return projection
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
    data = _load_data()
    dataset = data["dataset"]
    training = data["training"]
    project = data["project"]
    tuning_history = data["tuning_history"]
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
        # 训练监控首屏的“最近有效训练”由服务端确定性选出（不靠模板循环变量）
        "monitor_latest": _monitor_latest_projection(
            training, data["experiment_history"]),
        "project": project,
        "tuning_history": tuning_history,
        "experiment_history": data["experiment_history"],
        "experiment_history_source": data["experiment_history_source"],
        "experiment_index_warning": data["experiment_index_warning"],
        "dataset_index": data["dataset_index"],
        "latest_suggestion": _get_latest_suggestion(tuning_history, training),
        "current_args": _get_current_args(training),
        "dataset_analyzer_config": data["dataset_analyzer_config"],
        "training_config": APP_CONFIG.get("training", {}),
        "llm_analysis": _get_llm_analysis_display(training),
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


@app.get("/static/hpo.js")
async def hpo_static_script():
    """Serve the HPO front-end script from the packaged ui/static directory."""
    from fastapi.responses import Response

    script_path = Path(__file__).parent / "static" / "hpo.js"
    try:
        body = script_path.read_text(encoding="utf-8")
    except OSError:
        return Response(status_code=404, content="hpo.js not found")
    return Response(content=body, media_type="application/javascript")


@app.get("/static/monitor.js")
async def monitor_static_script():
    """Serve the shared training-monitor script (same packaged ui/static dir).

    The module is a separate file so both the ordinary training stream and the
    reconnect stream share one result projection, and so the real front-end
    logic can be executed directly by tests/js/minidom.js.
    """
    from fastapi.responses import Response

    script_path = Path(__file__).parent / "static" / "monitor.js"
    try:
        body = script_path.read_text(encoding="utf-8")
    except OSError:
        return Response(status_code=404, content="monitor.js not found")
    return Response(content=body, media_type="application/javascript")


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


# ── Studio S2 Core: local index query / import API ──


@app.get("/api/local-index/status")
async def local_index_status():
    """Report local index availability, schema and counts."""
    service = _local_index_service()
    try:
        return JSONResponse(service.status())
    except LocalIndexError as exc:
        return _local_index_error_response(exc)


@app.get("/api/datasets")
async def api_datasets(limit: int = Query(100)):
    """List indexed datasets (stable query projection)."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 500):
        return JSONResponse(
            {"error": "limit must be an integer in [1,500]", "error_code": "INVALID_QUERY"},
            status_code=400,
        )
    service = _local_index_service()
    try:
        datasets = service.list_datasets()[:limit]
        return JSONResponse({
            "datasets": [project_outward(d) for d in datasets],
            "count": len(datasets),
        })
    except LocalIndexError as exc:
        return _local_index_error_response(exc)


@app.get("/api/datasets/{dataset_id}")
async def api_dataset_detail(dataset_id: str):
    """Detail for one indexed dataset."""
    service = _local_index_service()
    try:
        dataset = service.get_dataset(dataset_id)
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    if dataset is None:
        return JSONResponse(
            {"error": "dataset not found", "error_code": "NOT_FOUND"}, status_code=404
        )
    return JSONResponse(project_outward(dataset))


@app.get("/api/datasets/{dataset_id}/experiments")
async def api_dataset_experiments(dataset_id: str, limit: int = Query(10)):
    """Dataset association summary: count, best fact and recent experiments."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 100):
        return JSONResponse(
            {"error": "limit must be an integer in [1,100]", "error_code": "INVALID_QUERY"},
            status_code=400,
        )
    service = _local_index_service()
    try:
        summary = service.get_dataset_experiments(dataset_id, limit=limit)
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    if summary is None:
        return JSONResponse(
            {"error": "dataset not found", "error_code": "NOT_FOUND"}, status_code=404
        )
    return JSONResponse(project_outward(summary))


@app.get("/api/training/recent-runs")
async def api_training_recent_runs(limit: str = Query("4")):
    """Recent completed detect training runs for the analysis shortcut (Bugfix P3).

    The narrow response carries public facts plus the verified full ``run_dir``
    (the one business field needed to fill the local input); it never echoes
    params, commands, dataset/audit/weights paths or internal errors. Every
    run_dir is live-validated server-side against the S1.4 input_safety policy.
    """
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = -1
    if isinstance(value, bool) or not (1 <= value <= 4):
        return JSONResponse(
            {"error": "limit must be an integer in [1,4]", "error_code": "INVALID_QUERY"},
            status_code=400,
        )
    service = _local_index_service()
    try:
        policy = _load_input_policy()
        items = service.recent_training_runs(limit=value, policy=policy)
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    except InputSafetyError as exc:
        return _input_safety_error_response(exc)
    return JSONResponse({"items": items, "source": "sqlite"})


@app.get("/api/experiments")
async def api_experiments(
    dataset_id: str | None = Query(None),
    source: str | None = Query(None),
    status: str | None = Query(None),
    search: str | None = Query(None),
    sort: str = Query("finished_at"),
    order: str = Query("desc"),
    limit: int = Query(25),
    offset: int = Query(0),
):
    """Query indexed experiments with stable pagination, search and sorting.

    ``limit`` is bounded to 1-100 (default 25); sorting is backed by a server
    whitelist so a client can never inject a raw SQL fragment.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 100):
        return JSONResponse(
            {"error": "limit must be an integer in [1,100]", "error_code": "INVALID_QUERY"},
            status_code=400,
        )
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return JSONResponse(
            {"error": "offset must be a non-negative integer", "error_code": "INVALID_QUERY"},
            status_code=400,
        )
    try:
        query = ExperimentQuery(
            dataset_id=dataset_id or None,
            source=source or None,
            status=status or None,
            search=search or None,
            sort=sort,
            order=order,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        return JSONResponse(
            {"error": str(exc), "error_code": "INVALID_QUERY"}, status_code=400
        )
    service = _local_index_service()
    try:
        page = service.query_experiments(query)
        page["items"] = [project_outward(e) for e in page["items"]]
        return JSONResponse(page)
    except LocalIndexError as exc:
        return _local_index_error_response(exc)


@app.get("/api/experiments/{run_id}")
async def api_experiment_detail(run_id: str):
    """Extended detail for one indexed experiment (URL-encoded run_id).

    Returns sanitized facts, the dataset association and a controlled artifact
    manifest; it never echoes full business paths and never reads arbitrary
    local paths (the run_id is the only input).
    """
    service = _local_index_service()
    try:
        experiment = service.get_experiment_detail(run_id)
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    if experiment is None:
        return JSONResponse(
            {"error": "experiment not found", "error_code": "NOT_FOUND"}, status_code=404
        )
    return JSONResponse(project_outward(experiment))


@app.get("/api/experiments/{run_id}/report-view")
async def api_experiment_report_view(run_id: str):
    """Bounded, read-only training-report view for one SQLite run_id (Bugfix P5).

    The report is a registered artifact of the exact experiment; its content is
    projected to a minimal display model. No LLM/vision call, no glob, no metric
    recomputation, no full paths and no report file writes.
    """
    service = _local_index_service()
    try:
        view = service.get_report_view(run_id)
    except ExperimentViewError as exc:
        return _experiment_view_error_response(exc)
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    return JSONResponse(view)


@app.get("/api/experiments/{run_id}/audit-view")
async def api_experiment_audit_view(run_id: str):
    """Bounded, read-only tuning-audit view for one SQLite run_id (Bugfix P5).

    The audit is a registered artifact of the exact experiment and its session
    must reconcile with the artifact identity; iterations are projected from the
    stored facts only.
    """
    service = _local_index_service()
    try:
        view = service.get_audit_view(run_id)
    except ExperimentViewError as exc:
        return _experiment_view_error_response(exc)
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    return JSONResponse(view)


@app.post("/api/local-index/import-legacy")
async def local_index_import_legacy(request: Request):
    """Re-run the read-only legacy JSON import over the fixed log files.

    Client-supplied paths are never accepted; the CSRF/origin checks apply.
    """
    try:
        _require_security(request)
    except _SecurityRejected as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    service = _local_index_service()
    paths = [
        os.path.join("log", "experiment_history.json"),
        os.path.join("log", "tuning_history.json"),
    ]
    try:
        summary = service.import_legacy_files(paths)
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    _invalidate_cache("load_data")
    return JSONResponse({
        "source_files": summary.source_files,
        "imported": summary.imported,
        "skipped": summary.skipped,
        "failed": summary.failed,
        "failures": [
            {"error_code": f.error_code, "message": f.message} for f in summary.failures
        ],
    })


@app.get("/api/local-index/audit")
async def local_index_audit():
    """Read-only reconciliation audit of the projection against the facts.

    The audit never initializes, migrates, creates or writes the database; it
    returns a stable dict (with a stable ``error_code`` in the body on failure)
    and never leaks a traceback, SQL, or an absolute path.
    """
    service = _local_index_service()
    try:
        return JSONResponse(project_outward(service.audit()))
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    except Exception:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": "LOCAL_INDEX_AUDIT_FAILED", "error_code": "LOCAL_INDEX_AUDIT_FAILED"},
            status_code=500,
        )


@app.post("/api/local-index/audit/record")
async def local_index_audit_record(request: Request):
    """Explicitly persist a bounded audit summary (CSRF/origin-gated).

    Persisting an audit summary is a write, so it is an explicit POST, never
    part of the read-only GET audit route.
    """
    try:
        _require_security(request)
    except _SecurityRejected as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    service = _local_index_service()
    try:
        return JSONResponse(project_outward(service.persist_audit()))
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    except Exception:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": "LOCAL_INDEX_AUDIT_FAILED", "error_code": "LOCAL_INDEX_AUDIT_FAILED"},
            status_code=500,
        )


@app.post("/api/local-index/rebuild")
async def local_index_rebuild(request: Request):
    """Atomically rebuild the index (backup -> temp -> publish)."""
    try:
        _require_security(request)
    except _SecurityRejected as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    service = _local_index_service()
    try:
        result = service.rebuild()
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    except Exception:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": "LOCAL_INDEX_REBUILD_FAILED", "error_code": "LOCAL_INDEX_REBUILD_FAILED"},
            status_code=500,
        )
    _invalidate_cache("load_data")
    return JSONResponse(project_outward(result))


@app.get("/api/local-index/diagnostics")
async def local_index_diagnostics():
    """Report index health: schema/counts/quick_check/backups/recent events."""
    service = _local_index_service()
    try:
        return JSONResponse(project_outward(service.diagnostics()))
    except LocalIndexError as exc:
        return _local_index_error_response(exc)


@app.post("/api/local-index/checkpoint")
async def local_index_checkpoint(request: Request):
    """Run a WAL checkpoint on the index database."""
    try:
        _require_security(request)
    except _SecurityRejected as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    service = _local_index_service()
    try:
        return JSONResponse(service.checkpoint())
    except LocalIndexError as exc:
        return _local_index_error_response(exc)


@app.post("/api/local-index/backup")
async def local_index_backup(request: Request):
    """Create a manual backup of the index database."""
    try:
        _require_security(request)
    except _SecurityRejected as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    service = _local_index_service()
    try:
        return JSONResponse(service.backup())
    except LocalIndexError as exc:
        return _local_index_error_response(exc)


@app.post("/api/experiments/compare")
async def api_experiments_compare(request: Request):
    """Compare 2-5 experiments against one baseline (read-only, CSRF-gated)."""
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
    service = _local_index_service()
    try:
        result = service.compare_experiments(
            body.get("run_ids"), body.get("baseline_run_id")
        )
    except LocalIndexError as exc:
        return _local_index_error_response(exc)
    return JSONResponse(project_outward(result))


@app.get("/api/tuning/status")
async def api_tuning_status():
    """Reconcile auto-tuning state into the unified public projection.

    A live in-memory tuning controller keeps ``running``; a persisted running
    record without one is conservatively downgraded and never reported as
    running.
    """
    _sf = os.path.join("log", "tuning_running.json")

    controller = _RUN_MANAGER.active_tuning()
    if controller is not None:
        return JSONResponse(project_public_state(controller.run_state))

    _state = read_run_state(_sf, run_kind="tuning")
    _reconciled = reconcile_persisted_state(_state, controller_owned=False)
    if _reconciled is not None and _reconciled != _state:
        try:
            _persist_run_state(_sf, _reconciled)
        except RunStatePersistenceError:
            pass
    return JSONResponse(project_public_state(_reconciled))


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
    # Only the historical legal values may reach this endpoint; ``train`` is a
    # legacy real-tuning value still used by existing callers/tests. Any unknown
    # value (including ``hpo``) must never fall through to the LLM tuning branch;
    # HPO has its own route and controller.
    if mode not in ("dry_run", "keep_params", "full", "train"):
        return JSONResponse(
            {"error": f"未知训练模式: {mode}", "error_code": "INVALID_MODE"},
            status_code=422,
        )
    skip_execute = mode == "dry_run"
    keep_params = mode == "keep_params"
    auto_analyze = body.get("auto_analyze", False)
    auto_loop = body.get("auto_loop", False)
    # When auto_analyze is enabled and multiple iterations configured,
    # auto_loop must be on so all iterations complete before best is selected
    if auto_analyze and max_retries > 1 and not auto_loop:
        auto_loop = True
    eval_mode = body.get("eval_mode", "comprehensive")

    # 大模型调优必须继承参考运行 args.yaml 中已经使用的权重：不接受任何客户端
    # 权重覆盖（哪怕会被静默忽略），也绝不因此用新权重训练却沿用旧基线。
    # 该判断先于参考运行解析，避免把一个被拒绝的请求先暴露成参考运行错误。
    if "model_id" in body or "model" in body:
        return JSONResponse(
            {"error_code": "LLM_MODEL_OVERRIDE_FORBIDDEN",
             "error": "大模型调优必须继承参考运行的初始权重。"},
            status_code=422,
        )

    # Resolve the reference run's bound dataset snapshot BEFORE any controller
    # or YOLO subprocess is created. The global latest_dataset is never a
    # candidate or fallback for auto-tuning (Bugfix P2).
    if not reference_run:
        report = find_module_b_report(log_dir="log")
        if report and report.get("runs"):
            reference_run = list(report["runs"].keys())[0]

    detect_dir = find_detect_dir() if reference_run else None
    reference_dataset = None
    reference_dataset_error = None
    if reference_run:
        try:
            reference_dataset = resolve_reference_dataset(
                reference_run,
                Path(detect_dir),
                Path("log"),
                local_index_service=_local_index_service(),
            )
        except ReferenceDatasetError as exc:
            if not skip_execute:
                return _reference_dataset_error_response(exc)
            reference_dataset_error = {
                "error_code": exc.error_code,
                "error": exc.message,
            }
    elif not skip_execute:
        return JSONResponse(
            {"error": "无法确定唯一的参考训练，无法启动自动调优",
             "error_code": "REFERENCE_RUN_INVALID"},
            status_code=400,
        )
    else:
        reference_dataset_error = {
            "error_code": "REFERENCE_DATASET_UNRESOLVED",
            "error": "无参考训练，dry-run 计划不可执行",
        }

    # A second active tuning run is rejected for every mode (legacy behavior).
    if _RUN_MANAGER.active_tuning() is not None:
        return JSONResponse(
            {"error": "已有活动调优运行，请先停止或等待完成",
             "error_code": "RUN_ALREADY_ACTIVE"},
            status_code=409,
        )
    # Real strategies (keep_params/full) share the unified single training slot
    # with ordinary training and HPO. dry_run is a preview and never reserves.
    if not skip_execute:
        _gate = _assert_training_slot_free()
        if _gate is not None:
            return _gate

    global _tuning_cancel_event, _current_tuning_train_proc
    _tuning_cancel_event.clear()
    _current_tuning_train_proc = None

    # Create the unified tuning run identity and persist it before anything
    # starts; the first write must succeed or the loop never launches.
    run_state = new_run_state("tuning")
    reservation_token = None
    if not skip_execute:
        reservation_token, _busy = _reserve_or_busy("tuning", run_state.run_id)
        if _busy is not None:
            return _busy
    _tuning_status_file = os.path.join("log", "tuning_running.json")
    try:
        _persist_run_state(_tuning_status_file, run_state)
    except RunStatePersistenceError as exc:
        _release_reservation(reservation_token)
        return JSONResponse(
            {"error": f"运行状态写入失败，未启动调优: {exc}",
             "error_code": "RUN_STATE_PERSIST_FAILED"},
            status_code=500,
        )

    from auto_tune.modules.agent_engine.loop import run_tuning_loop

    def loop_runner(on_progress, on_state, cancel_event):
        return run_tuning_loop(
            config=APP_CONFIG,
            reference_run=reference_run,
            max_retries=max_retries,
            log_dir="log",
            skip_execute=skip_execute,
            auto_analyze=auto_analyze,
            auto_loop=auto_loop,
            on_progress=on_progress,
            cancel_event=cancel_event,
            keep_params=keep_params,
            eval_mode=eval_mode,
            on_state=on_state,
            runtime_run_id=run_state.run_id,
            local_index_service=_local_index_service(),
            reference_dataset=reference_dataset,
        )

    # Create + register + start the background controller. SSE is only a
    # subscriber; the controller persists the terminal state by itself.
    broker = EventBroker(run_state.run_id)
    controller = TuningRunController(
        run_state=run_state,
        state_file=_tuning_status_file,
        broker=broker,
        manager=_RUN_MANAGER,
        loop_runner=loop_runner,
        reservation_token=reservation_token,
    )
    _RUN_MANAGER.register(controller)

    # Publish the frozen resolution confirmation before the loop starts so the
    # SSE stream always carries the display-safe dataset/snapshot identity.
    if reference_dataset is not None:
        broker.publish({
            "status": "preparing",
            "event": "reference_dataset",
            "message": "参考数据集已解析",
            "run_id": run_state.run_id,
            "phase": "preparing",
            "executable": True,
            "reference_dataset": _reference_dataset_public(reference_dataset),
        })
    elif reference_dataset_error is not None:
        broker.publish({
            "status": "preparing",
            "event": "reference_dataset",
            "message": "参考数据集未解析，计划不可执行",
            "run_id": run_state.run_id,
            "phase": "preparing",
            "executable": False,
            "error_code": reference_dataset_error["error_code"],
            "error": reference_dataset_error["error"],
        })

    try:
        controller.start()
    except Exception:
        _release_reservation(reservation_token)
        raise
    _invalidate_cache("load_data")

    return StreamingResponse(
        _run_sse(broker, controller, 0),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/tuning/stop")
async def stop_tuning():
    """Stop the active auto-tuning run.

    Only ``starting``/``running`` runs can be stopped; a terminal state is
    never rewritten to cancelled. The controller marks ``stopping``, cancels
    the loop thread, waits for it to finish, then the controller persists
    ``cancelled/terminal``. A failed wait returns an error, never a fake
    success.
    """
    sf = os.path.join("log", "tuning_running.json")
    controller = _RUN_MANAGER.active_tuning()
    if controller is not None:
        ok = controller.request_stop()
        if not ok:
            return JSONResponse(
                {"error": "调优任务无法停止", "error_code": "STOP_FAILED"},
                status_code=500,
            )
        done = await asyncio.to_thread(controller.wait_done, 30)
        if not done:
            return JSONResponse(
                {"error": "等待调优任务退出超时", "error_code": "STOP_TIMEOUT"},
                status_code=500,
            )
        state = read_run_state(sf, run_kind="tuning")
        return JSONResponse({
            "status": "stopped",
            "running": False,
            "status_message": state.status if state else "cancelled",
        })

    # No live controller: never rewrite an existing terminal state.
    _state = read_run_state(sf, run_kind="tuning")
    if _state is None:
        return JSONResponse({"error": "没有活动调优运行", "error_code": "NO_ACTIVE_RUN"}, status_code=404)
    if _state.status in ("starting", "running"):
        reconciled = reconcile_persisted_state(_state, controller_owned=False)
        if reconciled is not None and reconciled != _state:
            try:
                _persist_run_state(sf, reconciled)
            except RunStatePersistenceError:
                pass
        return JSONResponse(
            {"error": "运行控制已中断，无法停止原进程", "error_code": "CONTROLLER_LOST"},
            status_code=409,
        )
    return JSONResponse({"status": "stopped", "running": False, "status_message": _state.status})


@app.post("/api/training/stop")
async def stop_first_training():
    """Stop the active ordinary training run.

    Only ``starting``/``running`` runs can be stopped. The controller marks
    ``stopping``, terminates the subprocess, waits for it to exit, then the
    controller persists ``cancelled/terminal``. A failed termination or
    timeout returns an error, never a fake success.
    """
    sf = os.path.join("log", "training_running.json")
    controller = _RUN_MANAGER.active_manual()
    if controller is not None:
        ok = controller.request_stop()
        if not ok:
            return JSONResponse(
                {"error": "终止训练进程失败", "error_code": "STOP_FAILED"},
                status_code=500,
            )
        try:
            await controller.wait_done(timeout=20)
        except asyncio.TimeoutError:
            return JSONResponse(
                {"error": "等待训练进程退出超时", "error_code": "STOP_TIMEOUT"},
                status_code=500,
            )
        state = read_run_state(sf, run_kind="manual")
        return JSONResponse({
            "status": "stopped",
            "running": False,
            "status_message": state.status if state else "cancelled",
        })

    # No live controller: never rewrite an existing terminal state.
    _state = read_run_state(sf, run_kind="manual")
    if _state is None:
        return JSONResponse({"error": "没有活动训练运行", "error_code": "NO_ACTIVE_RUN"}, status_code=404)
    if _state.status in ("starting", "running"):
        reconciled = reconcile_persisted_state(_state, controller_owned=False)
        if reconciled is not None and reconciled != _state:
            try:
                _persist_run_state(sf, reconciled)
            except RunStatePersistenceError:
                pass
        return JSONResponse(
            {"error": "运行控制已中断，无法停止原进程", "error_code": "CONTROLLER_LOST"},
            status_code=409,
        )
    return JSONResponse({"status": "stopped", "running": False, "status_message": _state.status})


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


def _reference_dataset_public(res) -> dict:
    """Display-safe projection of a resolution (no absolute paths, short IDs).

    ``resolution_source`` is the stable code (``sqlite`` / ``reference_args``);
    the UI translates it via the existing i18n map.
    """
    snapshot_id = res.snapshot_id or ""
    return {
        "reference_run": res.reference_run,
        "dataset_display_name": res.dataset_display_name,
        "snapshot_short_id": snapshot_id[:8] if snapshot_id else None,
        "resolution_source": res.resolution_source,
    }


def _reference_dataset_error_response(exc: ReferenceDatasetError) -> JSONResponse:
    """Map a resolution error to a stable 400/409/503 without leaking paths."""
    return JSONResponse(
        {"error": exc.message, "error_code": exc.error_code},
        status_code=exc.status_code,
    )


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
    """(Legacy) ZIP dataset upload disabled — use folder selection (Studio S1.4).

    The request body is never read; the handler responds 410 unconditionally.
    """
    return JSONResponse(
        {"error": "ZIP 上传已停用，请使用目录选择", "error_code": "LEGACY_UPLOAD_DISABLED"},
        status_code=410,
    )


# ── Folder Browse API ──


def _load_input_policy() -> InputSafetyPolicy:
    """Load the current input_safety policy from the public config."""
    return load_input_safety_policy(APP_CONFIG)


def _input_safety_error_response(exc: InputSafetyError) -> JSONResponse:
    """Map an input-safety domain error to a stable structured HTTP response."""
    return JSONResponse({"error": str(exc), "error_code": exc.error_code}, status_code=exc.status_code)


def _browse_roots(policy: InputSafetyPolicy) -> JSONResponse:
    """Root listing for an empty browse path.

    With ``allowed_roots`` configured, only those roots are offered; otherwise
    available Windows drives are listed, matching the pre-S1.4 behavior.
    """
    if policy.allowed_roots:
        entries = [
            {"name": p.name or str(p), "path": str(p), "is_dir": True}
            for p in (Path(root).resolve(strict=False) for root in policy.allowed_roots)
        ]
        return JSONResponse({"path": "", "parent": None, "entries": entries})
    if os.name == "nt":
        import string
        drives = []
        for d in string.ascii_uppercase:
            dp = f"{d}:\\"
            if os.path.exists(dp):
                drives.append({"name": f"({d}:)", "path": dp, "is_dir": True})
        return JSONResponse({"path": "", "parent": None, "entries": drives})
    return JSONResponse({"path": "", "parent": None, "entries": []})


def _browse_parent(root: Path, policy: InputSafetyPolicy) -> str | None:
    """Parent for navigation, or ``None`` when moving up would be unsafe/out of bounds."""
    parent = root.parent
    if parent == root:
        return None
    try:
        validate_directory_path(parent, policy)
    except InputSafetyError:
        return None
    return str(parent)


@app.post("/api/browse-folder")
async def browse_folder(request: Request):
    """List safe subdirectories of a given server path.

    Every entry validates the path and enumerates direct subdirectories through
    the input-safety module; out-of-bounds, link, and permission errors return
    stable error responses instead of an empty listing.
    """
    try:
        data = await request.json()
    except Exception:
        data = {}
    path = (data.get("path") or "").strip()
    try:
        policy = _load_input_policy()
        if not path:
            return _browse_roots(policy)
        root = validate_directory_path(path, policy)
        subs = list_safe_subdirectories(root, policy)
        entries = [{"name": p.name, "path": str(p), "is_dir": True} for p in subs]
        return JSONResponse({
            "path": str(root),
            "parent": _browse_parent(root, policy),
            "entries": entries,
        })
    except InputSafetyError as exc:
        return _input_safety_error_response(exc)
    except Exception:
        return JSONResponse(
            {"error": "浏览目录失败", "error_code": "INPUT_BROWSE_FAILED"}, status_code=500
        )


# ── Dataset Folder Analyze API ──


@app.post("/api/dataset/analyze-folder")
async def analyze_dataset_folder(request: Request):
    """Analyze a dataset folder directly from a server path (no ZIP upload).

    The input is validated and bounded-scanned before any ``rglob``, file read,
    Module A call, or state write; failures never create a report or rewrite
    ``latest_dataset.json``.
    """
    _invalidate_cache("load_data")

    try:
        data = await request.json()
        folder_path = data.get("path", "").strip()

        try:
            policy = _load_input_policy()
            root = validate_directory_path(folder_path, policy)
            scan = scan_directory_bounded(root, policy)
        except InputSafetyError as exc:
            return _input_safety_error_response(exc)

        # Find data.yaml or detect dataset structure
        dataset_dir = str(root)
        data_yaml = None

        folder = root
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
            "input_scan": {
                "member_count": scan.member_count,
                "total_bytes": scan.total_bytes,
            },
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
    latest_info = _snapshot_to_latest_info(ds_info, snapshot)
    try:
        _write_json_atomic(LATEST_DATASET_PATH, latest_info)
    except OSError as exc:
        return JSONResponse(
            {"error": f"数据集快照已创建但登记失败: {exc}", "error_code": "SNAPSHOT_IO_FAILED"},
            status_code=500,
        )

    _invalidate_cache("load_data")
    response = {
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
    }
    # Index the dataset best-effort: an index failure must never change the
    # snapshot success fact, only surface a recoverable warning.
    try:
        service = _local_index_service()
        if service is not None:
            service.index_dataset(latest_info)
    except LocalIndexError:
        response["index_warning"] = {
            "error_code": "LOCAL_INDEX_PERSIST_FAILED",
            "message": "数据集已创建，但本地索引更新失败",
        }
    return JSONResponse(response)


# ── Training Upload & Analyze API ──

import zipfile
import tempfile
import shutil
import datetime


@app.post("/api/training/analyze")
async def upload_training(file: UploadFile = File(...)):
    """(Legacy) training ZIP/JSON upload disabled — use folder selection (Studio S1.4).

    The request body is never read; the handler responds 410 unconditionally.
    """
    return JSONResponse(
        {"error": "ZIP/JSON 上传已停用，请使用目录选择", "error_code": "LEGACY_UPLOAD_DISABLED"},
        status_code=410,
    )


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

        # Structured hyperparameter suggestion via the unified Decision Agent
        if report.get("llm_analysis") and isinstance(report.get("llm_analysis"), dict) and not report["llm_analysis"].get("error"):
            try:
                report["suggestion"] = generate_suggestion(
                    _build_decision_summary(report),
                    APP_CONFIG.get("project", {}),
                    APP_CONFIG,
                )
            except Exception:
                # Never lose the plain diagnosis because the suggestion step
                # itself blew up; persist a stable, safe structured error.
                report["suggestion"] = {"error": "Suggestion generation failed"}

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
    """Analyze a YOLO train directory directly from a server path (no ZIP upload).

    The input is validated and bounded-scanned before any file read, Module B
    call, report, or history write; failures never create/overwrite anything.
    """
    try:
        data = await request.json()
        folder_path = data.get("path", "").strip()

        try:
            policy = _load_input_policy()
            root = validate_directory_path(folder_path, policy)
            scan = scan_directory_bounded(root, policy)
        except InputSafetyError as exc:
            return _input_safety_error_response(exc)

        # Check for results.csv and args.yaml
        run_dir = str(root)
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

        # Structured hyperparameter suggestion via the unified Decision Agent
        if report.get("llm_analysis") and isinstance(report.get("llm_analysis"), dict) and not report["llm_analysis"].get("error"):
            try:
                report["suggestion"] = generate_suggestion(
                    _build_decision_summary(report),
                    APP_CONFIG.get("project", {}),
                    APP_CONFIG,
                )
            except Exception:
                # Never lose the plain diagnosis because the suggestion step
                # itself blew up; persist a stable, safe structured error.
                report["suggestion"] = {"error": "Suggestion generation failed"}

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
        return JSONResponse({
            "status": "success",
            "summary": summary,
            "input_scan": {
                "member_count": scan.member_count,
                "total_bytes": scan.total_bytes,
            },
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] Training folder analyze failed: {e}\n{tb}", flush=True)
        return JSONResponse({"error": f"分析失败: {str(e)}"}, status_code=500)


# ── First-time Training API ──

_HPO_REUSE_STATUS = {
    "HPO_NO_SUCCESS": 409,
    "HPO_SOURCE_INVALID": 409,
    "HPO_NOT_FOUND": 409,
    "HPO_EXECUTION_CONFLICT": 409,
    "HPO_BINDING_MISMATCH": 409,
    "HPO_VERSION_MISMATCH": 409,
    "HPO_CORRUPT_STUDY": 500,
    "HPO_CORRUPT_EXECUTION": 500,
}


def _hpo_reuse_error_response(exc: HpoError) -> JSONResponse:
    """Stable, redacted error for a best-config verification start.

    Only the stable code and fixed text are returned: the underlying
    ``exc.message`` can carry file-system detail from the storage layer and is
    never assumed display-safe.
    """
    code = safe_hpo_error_code(exc.code)
    status = _HPO_REUSE_STATUS.get(code, 500)
    message = {
        409: "该最佳配置当前不可用于验证。",
        500: "研究记录暂不可用。",
    }[status]
    next_action = {
        409: "请刷新研究结果后重试；系统未启动训练。",
        500: "系统未启动训练，请稍后重试。",
    }[status]
    return JSONResponse(
        {"error_code": code, "error": message, "next_action": next_action},
        status_code=status,
    )


async def _start_hpo_verification_training(body: dict):
    """Start one fixed-config verification run from an HPO source study/trial.

    Only ``source_hpo`` may be present; every effective parameter (six search
    params + fixed conditions) is rebuilt server-side from the authoritative
    HPO records. The bound **initial** weights and snapshot are used, a new run
    directory is created and the HPO study/execution/ranking are never touched.
    """
    import os as _os
    import yaml as _yaml

    from auto_tune.modules.agent_engine.executor import (
        find_detect_dir,
        resolve_yolo_executable,
    )

    source = body.get("source_hpo")
    extra = [key for key in body if key != "source_hpo"]
    if extra:
        return JSONResponse(
            {"error": "固定配置验证不能同时提交自由训练参数",
             "error_code": "MIXED_TRAINING_PARAMS",
             "next_action": "请只提交来源研究/试验，其余参数由系统重建。"},
            status_code=422,
        )
    if not isinstance(source, dict) or set(source.keys()) - {"study_id", "trial_id"}:
        return JSONResponse(
            {"error": "来源格式不合法", "error_code": "INVALID_SOURCE_HPO",
             "next_action": "请重新载入最佳配置后再验证。"},
            status_code=422,
        )
    try:
        verified = resolve_hpo_verification(
            _hpo_service, _hpo_runner, source.get("study_id"), source.get("trial_id"))
    except HpoError as exc:
        return _hpo_reuse_error_response(exc)

    effective = verified.effective
    data_yaml = effective["data"]
    model = effective["model"]
    epochs = int(effective["epochs"])

    # Read-only candidate name: an invalid/forged source or a busy slot must
    # leave the Detect tree exactly as it was (no trainN, no args.yaml, no
    # hpo_source.json, no training process).
    detect_dir = find_detect_dir()
    train_name = _pick_train_name(detect_dir)

    run_state = new_run_state("manual", run_name=train_name)
    reservation_token, _busy = _reserve_or_busy("manual", run_state.run_id)
    if _busy is not None:
        return _busy
    _state_file = os.path.join("log", "training_running.json")

    # Same ordering contract as ordinary training: the run directory and its
    # files are written while the slot is held, before the run-state record, so
    # no failure can leave a persisted `starting` record behind.
    train_name = _pick_train_name(detect_dir)  # re-check under the held slot
    train_dir = _os.path.join(detect_dir, train_name)
    try:
        _create_train_dirs(detect_dir, train_dir)
    except OSError:
        _release_reservation(reservation_token)
        return _train_dir_error_response()
    if train_name != run_state.run_name:
        run_state = replace(run_state, run_name=train_name)

    params = dict(effective)
    params["data"] = _os.path.abspath(data_yaml)
    params["model"] = model
    params["name"] = train_name
    params["project"] = _os.path.abspath(detect_dir)
    params["exist_ok"] = "True"
    try:
        with open(_os.path.join(train_dir, "args.yaml"), "w", encoding="utf-8") as _f:
            _yaml.dump(params, _f, default_flow_style=False, allow_unicode=True,
                       sort_keys=False)
    except OSError:
        _release_reservation(reservation_token)
        return _run_dir_write_error_response(
            "ARGS_PERSIST_FAILED", "训练参数文件写入失败，未启动验证训练")

    # Source metadata must be traceable BEFORE the run starts.
    try:
        with open(_os.path.join(train_dir, "hpo_source.json"), "w",
                  encoding="utf-8") as _sf:
            json.dump(verified.source_public, _sf, ensure_ascii=False, indent=2)
    except OSError:
        _release_reservation(reservation_token)
        return JSONResponse(
            {"error": "来源元数据写入失败，未启动验证训练",
             "error_code": "SOURCE_METADATA_PERSIST_FAILED"},
            status_code=500,
        )

    try:
        _persist_run_state(_state_file, run_state)
    except RunStatePersistenceError as exc:
        _release_reservation(reservation_token)
        return JSONResponse(
            {"error": f"运行状态写入失败，未启动验证训练: {exc}",
             "error_code": "RUN_STATE_PERSIST_FAILED"},
            status_code=500,
        )

    cmd = [resolve_yolo_executable(), "train"]
    for _k, _v in params.items():
        cmd.append(f"{_k}={_v}")

    broker = EventBroker(run_state.run_id)
    controller = ManualRunController(
        run_state=run_state,
        state_file=_state_file,
        cmd=cmd,
        params=params,
        train_name=train_name,
        train_dir=train_dir,
        data_yaml=data_yaml,
        model=model,
        epochs=epochs,
        log_path=os.path.join(train_dir, "training.log"),
        finalize_cb=_manual_finalize_cb,
        broker=broker,
        manager=_RUN_MANAGER,
        reservation_token=reservation_token,
    )
    controller.source_hpo = verified.source_public
    _RUN_MANAGER.register(controller)
    try:
        controller.start()
    except Exception:
        _release_reservation(reservation_token)
        raise
    _invalidate_cache("load_data")
    return StreamingResponse(
        _run_sse(broker, controller, 0),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/training/start")
async def start_first_training(request: Request):
    """Start a first-time YOLO training.

    The subprocess, stdout consumption, log writes, state updates, finalizer,
    and terminal-state persistence all run in a background ManualRunController.
    This endpoint only creates and starts the controller, then returns an SSE
    subscription. Disconnecting the SSE client never cancels the controller.
    """
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    import os as _os

    from auto_tune.modules.agent_engine.executor import find_detect_dir

    # Unified gate: ordinary training, LLM tuning and HPO share one real
    # training slot; leftover restart records also occupy it (H1.3).
    _gate = _assert_training_slot_free()
    if _gate is not None:
        return _gate

    # Best-config verification: parameters are rebuilt from the HPO source.
    if "source_hpo" in body:
        return await _start_hpo_verification_training(body)

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
    model = _resolve_training_model(body)
    if isinstance(model, JSONResponse):
        # 解析失败不创建训练目录、不写运行状态、不启动任何进程
        return model
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

    # The candidate run name is computed read-only: nothing is created on disk
    # until this request actually owns the single training slot, so a losing
    # racer leaves no empty directory, no args file and no training process
    # behind (and cannot shift the historical trainN numbering).
    detect_dir = find_detect_dir()
    train_name = _pick_train_name(detect_dir)

    run_state = new_run_state("manual", run_name=train_name)
    reservation_token, _busy = _reserve_or_busy("manual", run_state.run_id)
    if _busy is not None:
        return _busy
    _state_file = os.path.join("log", "training_running.json")

    # Everything below runs only while this request owns the slot. The run
    # directory (and its files) are created before the first run-state write so
    # a failed write can never leave a persisted `starting` record that the
    # restart gate would have to treat as an unresolvable live process.
    train_name = _pick_train_name(detect_dir)  # re-check under the held slot
    train_dir = _os.path.join(detect_dir, train_name)
    try:
        _create_train_dirs(detect_dir, train_dir)
    except OSError:
        _release_reservation(reservation_token)
        return _train_dir_error_response()
    if train_name != run_state.run_name:
        run_state = replace(run_state, run_name=train_name)

    # Build params + args.yaml + the exact command once.
    import yaml as _yaml
    params = {
        "model": model,
        "data": _os.path.abspath(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "patience": patience,
        "name": train_name,
        "project": _os.path.abspath(detect_dir),
        "exist_ok": "True",
        "plots": True,
        "save": True,
        "device": "0",
    }
    try:
        with open(_os.path.join(train_dir, "args.yaml"), "w", encoding="utf-8") as _f:
            _yaml.dump(params, _f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except OSError:
        _release_reservation(reservation_token)
        return _run_dir_write_error_response(
            "ARGS_PERSIST_FAILED", "训练参数文件写入失败，未启动训练")

    # The first state write must succeed or training never starts. The
    # reservation token is bound to this controller and released only when the
    # controller truly finishes (never on an SSE disconnect or stop request).
    try:
        _persist_run_state(_state_file, run_state)
    except RunStatePersistenceError as exc:
        _release_reservation(reservation_token)
        return JSONResponse(
            {"error": f"运行状态写入失败，未启动训练: {exc}",
             "error_code": "RUN_STATE_PERSIST_FAILED"},
            status_code=500,
        )

    from auto_tune.modules.agent_engine.executor import resolve_yolo_executable
    cmd = [resolve_yolo_executable(), "train"]
    for _k, _v in params.items():
        cmd.append(f"{_k}={_v}")

    # Create + register + start the background controller. SSE is only a
    # subscriber: disconnecting it never cancels the controller.
    broker = EventBroker(run_state.run_id)
    controller = ManualRunController(
        run_state=run_state,
        state_file=_state_file,
        cmd=cmd,
        params=params,
        train_name=train_name,
        train_dir=train_dir,
        data_yaml=data_yaml,
        model=model,
        epochs=epochs,
        log_path=os.path.join(train_dir, "training.log"),
        finalize_cb=_manual_finalize_cb,
        broker=broker,
        manager=_RUN_MANAGER,
        reservation_token=reservation_token,
    )
    _RUN_MANAGER.register(controller)
    try:
        controller.start()
    except Exception:
        _release_reservation(reservation_token)
        raise
    _invalidate_cache("load_data")

    return StreamingResponse(
        _run_sse(broker, controller, 0),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/training/running")
async def training_running_status():
    """Reconcile ordinary-training state into the unified public projection.

    A live in-memory controller (which owns the subprocess and the event bus)
    is authoritative. A persisted running record without a live controller is
    conservatively downgraded (interrupted/unknown) and never reported as
    still running.
    """
    _sf = os.path.join("log", "training_running.json")

    controller = _RUN_MANAGER.active_manual()
    if controller is not None:
        return JSONResponse(project_public_state(controller.run_state))

    # Persisted record reconciliation (covers restart, legacy, corrupt files).
    _state = read_run_state(_sf, run_kind="manual")
    _reconciled = reconcile_persisted_state(_state, controller_owned=False)
    if _reconciled is not None and _reconciled != _state:
        try:
            _persist_run_state(_sf, _reconciled)
        except RunStatePersistenceError:
            pass
    return JSONResponse(project_public_state(_reconciled))


@app.get("/api/runs/{run_id}/stream")
async def run_stream(run_id: str, after_seq: int = Query(0)):
    """Resubscribe to an active run's event stream after a page refresh.

    While the controller is alive in this process, the client replays buffered
    events ``> after_seq`` and then receives live events. A controller that
    just finished is briefly retained so a disconnected client can still replay
    the real buffered events (including the terminal and finalizer results).
    Once evicted from retention (or after a server restart, where a running
    record is reconciled to ``interrupted`` and the process is never
    re-adopted), only the persisted terminal is returned, flagged
    ``replay_truncated``. A stopped run cannot reconnect as an active stream.
    """
    controller = _RUN_MANAGER.get(run_id)
    if controller is not None:
        return StreamingResponse(
            _run_sse(controller.broker, controller, after_seq),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    run_kind = None
    if run_id.startswith("manual:"):
        run_kind = "manual"
    elif run_id.startswith("tuning:"):
        run_kind = "tuning"
    if run_kind is None:
        return JSONResponse({"error": "未知运行", "error_code": "UNKNOWN_RUN"}, status_code=404)

    _sf = os.path.join("log", "training_running.json" if run_kind == "manual" else "tuning_running.json")
    state = read_run_state(_sf, run_kind=run_kind)
    if state is None or state.run_id != run_id:
        return JSONResponse({"error": "未知运行", "error_code": "UNKNOWN_RUN"}, status_code=404)

    if state.status in ("starting", "running"):
        # Server restart lost the controller: never re-adopt the process.
        reconciled = reconcile_persisted_state(state, controller_owned=False)
        if reconciled is not None and reconciled != state:
            try:
                _persist_run_state(_sf, reconciled)
            except RunStatePersistenceError:
                pass
        state = reconciled

    async def terminal_stream():
        # No controller/broker is available to replay missed events, so the
        # persisted terminal is the best we can offer; say so honestly.
        seq = (state.last_event.seq + 1) if state.last_event is not None else 1
        ev = {
            "status": state.status,
            "run_id": run_id,
            "phase": state.phase,
            "message": state.terminal_reason or state.status,
            "event_seq": seq,
            "terminal_reason": state.terminal_reason,
            "replay_truncated": True,
        }
        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        terminal_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── H1.3 HPO API mount (prefix /api/hpo) ──────────────────────────

_HPO_SNAPSHOT_ID_RE = re.compile(r"^[0-9a-f]{64}$")


def _valid_count(value) -> bool:
    """A real non-negative integer count (booleans are never counts)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _resolve_hpo_snapshot(snapshot_id: str):
    """Resolve an explicit snapshot_id to its controlled published directory.

    The snapshot id is looked up only under the controlled snapshot root and is
    never replaced by the global ``latest_dataset``. Full content validation is
    performed later by ``HpoService.create_study``; this resolver only locates
    and sanity-checks identity.
    """
    if not isinstance(snapshot_id, str) or not _HPO_SNAPSHOT_ID_RE.fullmatch(snapshot_id):
        raise HpoError("HPO_INVALID_CONFIG", "快照 ID 格式不合法")
    snap_dir = Path(DATASET_SNAPSHOT_ROOT) / snapshot_id
    if not snap_dir.is_dir():
        raise HpoError("HPO_INVALID_CONFIG", "指定快照不存在或已失效")
    try:
        manifest = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("snapshot_id") != snapshot_id:
            raise HpoError("HPO_INVALID_CONFIG", "快照 manifest 身份不一致")
    except (OSError, ValueError) as exc:
        raise HpoError("HPO_INVALID_CONFIG", "快照 manifest 无法读取") from exc
    return snap_dir


def list_published_snapshots(root=None, include_source_root=False) -> list[dict]:
    """Read-only published-snapshot listing for the HPO snapshot selector.

    Only the controlled snapshot root is scanned; a candidate must be a
    directory whose name is a full 64-hex snapshot id and whose manifest carries
    the same identity. An unreadable/corrupt snapshot stays **visible** but is
    never selectable, so the user sees the real state instead of a silently
    shortened list. No absolute path, no global ``latest_dataset`` fallback and
    no data substitution: selecting an older legal snapshot is allowed even when
    nothing is registered as latest.

    ``image_count`` follows the manifest count contract: the legal sample total
    is ``train_count + val_count`` (equal to the manifest ``samples`` length),
    and ``background_count`` is a **subset** of those samples — it is reported
    separately and never added to the total.
    """
    base = Path(root) if root is not None else Path(DATASET_SNAPSHOT_ROOT)
    rows: list[dict] = []
    if not base.is_dir():
        return rows
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError:
        return rows
    for child in children:
        if not _HPO_SNAPSHOT_ID_RE.fullmatch(child.name) or not child.is_dir():
            continue
        row = {
            "snapshot_id": child.name,
            "short_id": child.name[:8],
            "readable": False,
            "selectable": False,
            "reason_code": "SNAPSHOT_MANIFEST_INVALID",
            "dataset_name": None,
            "created_at": None,
            "image_count": None,
            "train_count": None,
            "val_count": None,
            "background_count": None,
        }
        try:
            manifest = json.loads(
                (child / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get("snapshot_id") != child.name:
                raise ValueError("snapshot identity mismatch")
            source_root = manifest.get("source_root")
            train_count = manifest.get("train_count")
            val_count = manifest.get("val_count")
            background = manifest.get("background_count")
            if not _valid_count(train_count) or not _valid_count(val_count) \
                    or not _valid_count(background):
                raise ValueError("snapshot counts are not non-negative integers")
            samples = manifest.get("samples")
            # 合法样本总数以 train+val 为准（manifest 的 samples 长度契约）；
            # background 是 train/val 的子集，绝不与总数相加。
            if not isinstance(samples, list) or len(samples) != train_count + val_count:
                raise ValueError("snapshot sample count does not match train+val")
            if background > train_count + val_count:
                raise ValueError("snapshot background count exceeds train+val")
            row.update({
                "readable": True,
                "selectable": True,
                "reason_code": None,
                "dataset_name": (os.path.basename(os.path.normpath(source_root))
                                 if isinstance(source_root, str) and source_root else None),
                "created_at": manifest.get("created_at"),
                "train_count": train_count,
                "val_count": val_count,
                "background_count": background,
                "image_count": train_count + val_count,
            })
            if include_source_root:
                row["source_root"] = (os.path.abspath(os.path.normpath(source_root))
                                      if isinstance(source_root, str) and source_root
                                      else None)
        except (OSError, ValueError, UnicodeDecodeError):
            pass
        rows.append(row)
    readable = [r for r in rows if r["readable"]]
    unreadable = [r for r in rows if not r["readable"]]
    readable.sort(key=lambda r: (r["created_at"] or "", r["snapshot_id"]), reverse=True)
    return readable + unreadable


def _model_store_root() -> str:
    """Controlled weight root from config, relative to the working directory."""
    cfg = (APP_CONFIG.get("model_store") or {}) if APP_CONFIG else {}
    configured = cfg.get("root") or os.path.join("models", "weights")
    return os.path.abspath(str(configured))


def _build_model_store() -> "ModelStore":
    cfg = (APP_CONFIG.get("model_store") or {}) if APP_CONFIG else {}
    try:
        max_upload = int(cfg.get("max_upload_bytes", 2_147_483_648))
    except (TypeError, ValueError):
        max_upload = 2_147_483_648
    try:
        max_models = int(cfg.get("max_models", 100))
    except (TypeError, ValueError):
        max_models = 100
    return ModelStore(
        Path(_model_store_root()),
        # 项目根作为遗留兼容来源继续可读；绝不扫描 detect/trainN 训练产物
        legacy_roots=[Path(os.getcwd())],
        max_upload_bytes=max_upload,
        max_models=max_models,
    )


_MODEL_STORE = _build_model_store()


def list_local_models() -> list[dict]:
    """Short-term compatibility wrapper over the one controlled weight store.

    It no longer scans anything itself: the safe public projection (model id,
    name, size, SHA-256, origin, availability) is the only listing, so no
    client ever receives a server-side path and no training artifact is
    presented as an initial weight.
    """
    return [row.public_dict() for row in _MODEL_STORE.list_models()]


def _resolve_training_model(body: dict):
    """Resolve the initial weight of a direct training from the controlled library.

    A client-supplied ``model`` (or any path) is rejected outright — it is never
    used and never guessed — and the configured default is resolved **by
    basename** inside the controlled library, so Ultralytics can never
    implicitly download a checkpoint. The resolved value is an absolute path of
    a file whose existence and SHA-256 are re-checked right here, before any run
    directory, state file, reservation or controller exists.
    """
    model_id = body.get("model_id")
    if model_id:
        try:
            return str(_MODEL_STORE.resolve(model_id))
        except ModelStoreError as exc:
            return JSONResponse({"error_code": exc.code, "error": exc.message},
                                status_code=exc.status_code)
    if "model" in body:
        return JSONResponse(
            {"error_code": "MODEL_PATH_FORBIDDEN",
             "error": "请从受控权重库选择初始权重。"},
            status_code=422,
        )
    project_cfg = APP_CONFIG.get("project", {}) if APP_CONFIG else {}
    training_cfg = APP_CONFIG.get("training", {}) if APP_CONFIG else {}
    configured = project_cfg.get("model") or training_cfg.get("model", "yolov8n.pt")
    try:
        return str(_MODEL_STORE.resolve_configured_name(configured))
    except ModelStoreError as exc:
        return JSONResponse({"error_code": exc.code, "error": exc.message},
                            status_code=exc.status_code)


def _require_security_verdict(request: Request):
    """``require_security`` in the gate style used by the HPO routers."""
    try:
        _require_security(request)
    except _SecurityRejected as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    return None


def _is_reparse_like(path: Path) -> bool:
    try:
        if os.path.islink(path):
            return True
    except OSError:
        return False
    try:
        from auto_tune.modules.hpo.storage import _is_reparse_point

        return bool(_is_reparse_point(Path(path)))
    except Exception:
        return False


def _resolve_hpo_model_id(model_id):
    """Map a controlled model id to the frozen absolute initial-weight path.

    The client never sends a path: a model id is re-resolved against the current
    controlled-library facts here, and ``HpoService.create_study`` freezes the
    normalized path, byte count, SHA-256 and nanosecond mtime (``model_mtime_ns``)
    into the study binding. Legacy records without that field stay readable.
    Existing studies keep their own frozen binding and are never rebound.
    """
    return _MODEL_STORE.resolve(model_id)


_hpo_service = HpoService(Path(_hpo_storage_root()))
_hpo_runner = HpoRunner(Path(_hpo_storage_root()), Path(find_detect_dir()), Path("log"))


def _hpo_training_deps():
    """Build the ordinary-training dependencies for a formal HPO run.

    Resolved per request so the current controlled Detect directory, executable
    and app-level callables are always used. The controller lifecycle stays in
    ``ManualRunController``; this only wires the shared pieces.
    """
    from auto_tune.modules.agent_engine.executor import (
        find_detect_dir as _find_detect_dir,
        resolve_yolo_executable as _resolve_yolo_executable,
    )

    return TrainingSubmitDeps(
        run_manager=_RUN_MANAGER,
        detect_dir=_find_detect_dir,
        resolve_executable=_resolve_yolo_executable,
        new_run_state=new_run_state,
        broker_factory=EventBroker,
        manual_controller_cls=ManualRunController,
        finalize_cb=_manual_finalize_cb,
        pick_train_name=_pick_train_name,
        create_train_dirs=_create_train_dirs,
        persist_run_state=_persist_run_state,
        reserve=_reserve_or_busy,
        release=_release_reservation,
        gate=_assert_training_slot_free,
        log_dir="log",
        state_file=os.path.join("log", "training_running.json"),
        invalidate_cache=_invalidate_cache,
        run_state_replace=lambda state, name: replace(state, run_name=name),
    )


def _hpo_detect_dir() -> str:
    from auto_tune.modules.agent_engine.executor import find_detect_dir as _fdd

    return _fdd()


# ── HPO 服务端权威默认绑定（少让用户选；缺失/不合法才要求一次必要选择）──
#
# 默认值只来自**受控事实**：已登记数据集的合法快照、当前项目/训练配置里的合法本地
# 权重、当前普通训练配置。任何“配置了但不合法”的值都原样返回并附稳定 warning，
# 绝不静默替换成别的值；任何“缺失”才回退到既有默认。

_HPO_SEARCH_DEFAULTS = {"sampler": "tpe", "budget": 10, "epochs": 30, "seed": 42,
                        "timeout_seconds": 3600,
                        # 评价模式：UI 默认全面模式（后端契约仍兼容旧单指标语义）
                        "evaluation_mode": "comprehensive"}
_HPO_BATCH_RANGE = (1, 256)
_HPO_IMGSZ_RANGE = (32, 2048)
_HPO_EPOCHS_RANGE = (1, 1000)
_HPO_DEVICE_RE = re.compile(r"^(?:cpu|[0-9]|[1-5][0-9]|6[0-3])$")


def _cuda_available() -> bool:
    """只读探测本机是否有可用的 CUDA 设备；任何失败都视为不可用。"""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _available_gpus() -> list[int]:
    """只读投影已探测到的 GPU 编号；不可用时为空列表（绝不猜测编号）。"""
    if not _cuda_available():
        return []
    try:
        import torch

        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    if count <= 0:
        # is_available() 为真但拿不到计数时，仍允许用户显式选择 GPU 0
        return [0]
    return list(range(min(count, 64)))


def _default_device(raw) -> tuple[str, dict | None]:
    """计算设备默认值：配置优先；缺失时 GPU 可用默认 ``0``，否则 CPU + 提示。

    在 GPU 可用的本机绝不静默回退 CPU；无 GPU 时明确给出可操作提示。
    """
    if raw is not None:
        if isinstance(raw, str) and _HPO_DEVICE_RE.fullmatch(raw.strip()):
            return raw.strip(), None
        return raw, {"code": "CONFIG_VALUE_INVALID", "field": "device"}
    if _cuda_available():
        return "0", None
    return "cpu", {"code": "GPU_NOT_AVAILABLE_CPU_FALLBACK",
                   "message": "未检测到可用 GPU，已使用 CPU；如需 GPU 训练请在“计算设备”中填写显卡编号。"}


def _valid_int_in(value, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _same_local_path(left: str, right: str) -> bool:
    try:
        return (os.path.normcase(os.path.normpath(os.path.abspath(left)))
                == os.path.normcase(os.path.normpath(os.path.abspath(right))))
    except (OSError, ValueError):
        return False


def _default_snapshot_binding() -> tuple[dict | None, str, bool]:
    """Return ``(snapshot_row, reason_code, registered)`` for the current dataset.

    Only two facts may auto-bind: the **registered** snapshot when it validates,
    or a legal snapshot whose original ``source_root`` equals the registered
    dataset path. A snapshot belonging to another original directory is never
    auto-selected, even when it is the globally newest one.
    """
    latest = _read_latest_dataset() or {}
    registered_id = latest.get("snapshot_id")
    source = latest.get("source_dataset_path") or latest.get("dataset_path")
    registered = isinstance(source, str) and bool(source)

    rows = list_published_snapshots(include_source_root=True)
    by_id = {row["snapshot_id"]: row for row in rows}

    if isinstance(registered_id, str) and registered_id and _latest_snapshot_valid(latest):
        row = by_id.get(registered_id)
        if row is not None and row["readable"]:
            return row, "REGISTERED_SNAPSHOT", True

    if registered:
        # rows are already newest-first, so the first match is the newest legal
        # snapshot of the *same* original directory.
        for row in rows:
            if row["readable"] and row.get("source_root") \
                    and _same_local_path(row["source_root"], source):
                return row, "SAME_SOURCE_ROOT", True

    return None, ("NO_MATCHING_SNAPSHOT" if registered else "NO_REGISTERED_DATASET"), registered


def _model_row_by_name(name: str):
    """Controlled-library record for one configured name, or ``None``.

    Delegates to the store's single configuration-name resolution so HPO
    defaults and direct training can never diverge: a path form is rejected by
    the shared rule instead of being truncated to its last segment. A missing
    name is reported as such and never triggers a download.
    """
    try:
        return _MODEL_STORE.resolve_configured_record(name)
    except ModelStoreError:
        return None


def _default_model_binding(project_cfg: dict, training_cfg: dict) -> dict:
    """Default initial weight: configured basename resolved in the library.

    The projection is client-safe: a ``model_id`` and a display name, never a
    server path. A configured-but-illegal value is never silently swapped for a
    different file (that would change the user's experiment); it is reported so
    the user makes the one necessary selection. A search-stage ``best.pt`` is
    never auto-selected and nothing is ever downloaded.
    """
    for source_key, value in (("project.model", project_cfg.get("model")),
                              ("training.model", training_cfg.get("model"))):
        if isinstance(value, str) and value.strip():
            row = _model_row_by_name(value)
            if row is None:
                return {"model_id": None, "name": None, "source": None,
                        "available": False, "reason_code": "MODEL_CONFIG_INVALID"}
            return {"model_id": row.model_id, "name": row.name,
                    "source": source_key, "available": True,
                    "reason_code": "MODEL_BOUND"}

    fallback = _model_row_by_name("yolov8n.pt")
    if fallback is not None:
        return {"model_id": fallback.model_id, "name": fallback.name,
                "source": "fallback:yolov8n.pt", "available": True,
                "reason_code": "MODEL_BOUND"}
    return {"model_id": None, "name": None, "source": None, "available": False,
            "reason_code": "MODEL_MISSING"}


def build_hpo_defaults() -> dict:
    """Read-only server-side defaults for the HPO draft + formal-training form."""
    project_cfg = dict(APP_CONFIG.get("project") or {})
    training_cfg = dict(APP_CONFIG.get("training") or {})
    warnings: list[dict] = []

    def configured(key: str):
        """Raw configured value (None when absent/empty), warning when illegal."""
        if key not in training_cfg or training_cfg[key] in (None, ""):
            return None
        return training_cfg[key]

    def number(key: str, low: int, high: int, fallback: int, *, multiple=None) -> int:
        raw = configured(key)
        if raw is None:
            return fallback
        if _valid_int_in(raw, low, high) and (multiple is None or raw % multiple == 0):
            return raw
        warnings.append({"code": "CONFIG_VALUE_INVALID", "field": key})
        return raw

    batch = number("batch", *_HPO_BATCH_RANGE, fallback=16)
    imgsz = number("imgsz", *_HPO_IMGSZ_RANGE, fallback=640, multiple=32)
    formal_epochs = number("default_epochs", *_HPO_EPOCHS_RANGE, fallback=100)
    dev, device_notice = _default_device(configured("device"))
    if device_notice is not None:
        if device_notice.get("code") == "CONFIG_VALUE_INVALID":
            warnings.append({"code": "CONFIG_VALUE_INVALID", "field": "device"})

    snapshot_row, reason_code, registered = _default_snapshot_binding()
    return {
        "dataset": {
            "registered": registered,
            "display_name": (snapshot_row or {}).get("dataset_name"),
            "snapshot": snapshot_row,
            "reason_code": reason_code,
            "needs_user_confirmation": snapshot_row is None,
        },
        "model": _default_model_binding(project_cfg, training_cfg),
        "search": {
            "sampler": _HPO_SEARCH_DEFAULTS["sampler"],
            "budget": _HPO_SEARCH_DEFAULTS["budget"],
            "epochs": _HPO_SEARCH_DEFAULTS["epochs"],
            "seed": _HPO_SEARCH_DEFAULTS["seed"],
            "timeout_seconds": _HPO_SEARCH_DEFAULTS["timeout_seconds"],
            "evaluation_mode": _HPO_SEARCH_DEFAULTS["evaluation_mode"],
            "batch": batch,
            "imgsz": imgsz,
            "device": dev,
        },
        # 已探测设备：UI 只列这些 GPU 编号与 CPU（明确选择控件，不用自由文本），
        # 权威默认来自配置或 GPU 探测，绝不静默回退 CPU。
        "devices": {"gpus": _available_gpus(), "default": dev},
        # 无 GPU 时的可操作提示（仅提示，不改变用户已选条件）
        "device_notice": (device_notice
                          if device_notice and device_notice.get("message")
                          else None),
        # 正式训练默认取当前**普通训练**配置，绝不沿用验收研究的 epochs1/imgsz64
        "formal": {"epochs": formal_epochs, "batch": batch, "imgsz": imgsz,
                   "device": dev},
        "config_warnings": warnings,
    }


@app.get("/api/hpo/defaults")
async def hpo_defaults():
    """Read-only default bindings; never creates a study, snapshot or training."""
    return JSONResponse(build_hpo_defaults())


app.include_router(
    create_hpo_router(
        service=_hpo_service,
        runner=_hpo_runner,
        manager=_RUN_MANAGER,
        resolve_snapshot=_resolve_hpo_snapshot,
        resolve_model=_resolve_hpo_model_id,
        assert_training_slot_free=_assert_training_slot_free,
        list_snapshots=list_published_snapshots,
        list_models=list_local_models,
    ),
    prefix="/api/hpo",
)
app.include_router(
    create_model_store_router(
        # accessor, not a captured instance: the module global is the truth
        store=lambda: _MODEL_STORE,
        require_security=_require_security_verdict,
    ),
)
app.include_router(
    create_hpo_training_router(
        # accessors, not captured instances: the module globals are the truth
        service=lambda: _hpo_service,
        runner=lambda: _hpo_runner,
        manager=lambda: _RUN_MANAGER,
        detect_dir=_hpo_detect_dir,
        log_dir="log",
        build_deps=_hpo_training_deps,
        # 权威实验索引的只读访问器：把“查看结果”绑定到索引详情身份（不是 JSON 历史 ID）
        index=lambda: _local_index_service(),
    ),
    prefix="/api/hpo",
)


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
