"""H1.2 顺序 HPO 执行器（HpoRunner）— 写意图→执行→写结果→tell→finalize。

一个 trial 从同一绑定的初始模型开始，不接续上一 trial 权重；失败不重试原 trial、
不自动降 batch、不扩预算。启动意图成功落盘后才 launch；结果意图落盘后才 tell。
恢复只以最后一次成功发布的事实为准；BLOCKED 无强制继续接口，缺失启动身份不能
仅因时间流逝解除。停止/超时先持久化 termination_reason 再终止，未确认退出不得
启动下一个 trial。

H1.2 返修后额外保证：活动进程自 launch 起的异常清理边界（R1）；执行记录与 study
的跨文件语义复验、命令形状与启动前 args 校验（R2）；冻结根路径与祖先链接检查
（R3）；RunState 终态一致且写盘失败阻断预算（R5）；环境漂移只在启动新训练前判定、
不阻断已提交结果的幂等收尾（R6）。
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from .execution_adapter import FIXED_PARAMS, ExecutionAdapter, GuardrailRejection
from .execution_models import ExecutionAttempt, ExecutionConfig, ExecutionRecord
from .execution_storage import ExecutionStore
from .models import HpoError, ResultInput, utc_now_iso
from .search_space import validate_candidate
from .service import HpoService, _current_environment
from .storage import reject_link_chain, writer_waits_for_lock
from auto_tune.modules.agent_engine.executor import build_yolo_command
from auto_tune.modules.run_state.process_identity import (
    IdentityMatch,
    ProcessIdentity,
    compare_process_identity,
)
from auto_tune.modules.run_state.service import (
    new_run_state,
    with_status_phase,
    write_run_state,
)

POLL_INTERVAL = 0.2
TERMINATE_WAIT = 10.0
KILL_WAIT = 10.0

# A read-only request holds a study/execution transaction only for the duration
# of one read, so the execution session waits it out instead of being defeated
# by it. The wait stays bounded: a genuinely long holder still surfaces the
# stable busy code, which the controller then retries in its own bounded window.
SESSION_LOCK_WAIT_SECONDS = 2.0

_CONTINUE = "continue"
_PAUSE = "pause"

_RUN_STATE_STATUS = {
    "SUCCESS": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "INTERRUPTED": "interrupted",
}


def execution_request_id(study_id: str, number: int) -> str:
    """固定请求身份：同一 study+number 永远映射到同一 request_id（uuid5）。"""
    return uuid5(NAMESPACE_URL, f"hpo-execution-v1:{study_id}:{number}").hex


def _stop_requested(stop_event) -> bool:
    return stop_event is not None and stop_event.is_set()


def _authoritative_effective(study, config: ExecutionConfig, trial) -> dict:
    """从权威 study/config/绑定输入与已验证 candidate 重建完整 effective。"""
    params = dict(FIXED_PARAMS)
    params["model"] = study.model_binding.model_path
    params["data"] = study.snapshot_binding.data_yaml_path
    params["epochs"] = study.config.epochs
    params["seed"] = study.config.seed
    params["batch"] = config.batch
    params["imgsz"] = config.imgsz
    params["device"] = config.device
    params.update(trial.candidate_params)
    return params


def _authoritative_command(trial_id: str, run_relpath: str,
                           effective: dict, output_root: Path,
                           executable: str | None = None) -> list:
    """用同一 builder 从权威 effective 与受控输出路径重建完整预期命令。

    传入已冻结的 ``executable``（历史 attempt 的 command[0]）时只做纯参数重建、
    不解析当前 YOLO（R6）；``executable=None`` 表示新启动，走当前环境解析。
    """
    run_dir = Path(output_root) / run_relpath
    args_path = run_dir / "args.yaml"
    return build_yolo_command(trial_id, str(args_path), dict(effective),
                              executable=executable)


class HpoRunner:
    """组合 HpoService/ExecutionStore/ExecutionAdapter 的顺序执行器。"""

    def __init__(self, storage_root: Path, output_root: Path, log_root: Path):
        self._storage_root = Path(os.path.abspath(storage_root))
        self._output_root = Path(os.path.abspath(output_root))
        self._log_root = Path(os.path.abspath(log_root))
        self._service = HpoService(self._storage_root)
        self._exec_store = ExecutionStore(self._storage_root)

    # ── prepare ───────────────────────────────────────────────────

    def prepare(self, study_id: str, config) -> ExecutionRecord:
        try:
            data = config.model_dump(mode='python', warnings=False) if isinstance(config, ExecutionConfig) else config
            execution_config = ExecutionConfig.model_validate(data)
        except (ValidationError, ValueError, TypeError) as exc:
            raise HpoError("HPO_INVALID_EXECUTION_CONFIG",
                           "invalid execution config") from exc
        with self._exec_store.runner_locked():
            self._service.load_study(study_id)
            try:
                existing = self._read_exec(study_id)
            except HpoError as exc:
                if exc.code != "HPO_NOT_FOUND":
                    raise
                existing = None
            if existing is not None:
                self._ensure_roots_match(existing)
                if existing.config != execution_config:
                    raise HpoError("HPO_EXECUTION_CONFLICT",
                                   "execution config differs from the bound record")
                return existing.model_copy(deep=True)
            study = self._service.load_study(study_id)
            if any(t.state != "PENDING" for t in study.trials):
                raise HpoError("HPO_EXECUTION_CONFLICT",
                               "study already has terminal trials before binding executor")
            record = ExecutionRecord(
                study_id=study_id,
                revision=0,
                created_at=utc_now_iso(),
                updated_at=utc_now_iso(),
                config=execution_config,
                roots=_make_roots(self._storage_root, self._output_root,
                                  self._log_root),
                environment=_make_environment(_current_execution_environment()),
                status="READY",
                stop_reason=None,
                attempts=[],
            )
            self._commit(record)
            return record.model_copy(deep=True)

    # ── 只读校验装载 ─────────────────────────────────────────────

    def _load_validated(self, study_id: str):
        record = self._read_exec(study_id)
        self._ensure_roots_match(record)
        study = self._service.load_study(study_id)
        self._cross_validate(study, record)
        return study, record

    # ── status ────────────────────────────────────────────────────

    def status(self, study_id: str) -> ExecutionRecord:
        _, record = self._load_validated(study_id)
        return record.model_copy(deep=True)

    # ── run / resume 门控 ─────────────────────────────────────────

    def run(self, study_id: str, *, stop_event=None) -> ExecutionRecord:
        with writer_waits_for_lock(SESSION_LOCK_WAIT_SECONDS), \
                self._exec_store.runner_locked():
            _, record = self._load_validated(study_id)
            if record.status == "COMPLETED":
                return record.model_copy(deep=True)
            if record.status == "READY":
                return self._run_session(study_id, stop_event)
            if record.status in ("PAUSED", "INTERRUPTED"):
                raise HpoError("HPO_EXECUTION_CONFLICT",
                               "execution is paused/interrupted; call resume()")
            raise HpoError("HPO_RECOVERY_REQUIRED",
                           f"execution is {record.status}; call resume() to reconcile")

    def resume(self, study_id: str, *, stop_event=None) -> ExecutionRecord:
        with writer_waits_for_lock(SESSION_LOCK_WAIT_SECONDS), \
                self._exec_store.runner_locked():
            _, record = self._load_validated(study_id)
            if record.status == "COMPLETED":
                return record.model_copy(deep=True)
            return self._run_session(study_id, stop_event)

    # ── 会话主循环 ────────────────────────────────────────────────

    def _run_session(self, study_id: str, stop_event) -> ExecutionRecord:
        _, record = self._load_validated(study_id)
        if record.status not in ("RUNNING",):
            record = record.model_copy(update={
                "status": "RUNNING",
                "stop_reason": None,
                "updated_at": utc_now_iso(),
            })
            self._commit(self._bump(record))
        while True:
            study = self._service.load_study(study_id)
            record = self._read_exec(study_id)
            self._cross_validate(study, record)
            attempt = self._current_attempt(record)
            if attempt is None:
                has_pending = bool(study.trials) and study.trials[-1].state == "PENDING"
                if not has_pending and len(study.trials) >= study.config.budget:
                    return self._finish_session(record, "COMPLETED", None)
                if _stop_requested(stop_event):
                    return self._finish_session(record, "PAUSED", "user_stopped")
                decision = self._claim_next(study, record, stop_event)
                if decision == _PAUSE:
                    record = self._read_exec(study_id)
                    return self._finish_session(record, "PAUSED", "user_stopped")
                continue
            decision = self._drive_attempt(record, study, attempt, stop_event)
            if decision == _PAUSE:
                record = self._read_exec(study_id)
                return self._finish_session(record, "PAUSED", "user_stopped")
            # _CONTINUE：trial 已 FINALIZED（或已登记），继续认领下一个。

    def _finish_session(self, record, status: str, stop_reason) -> ExecutionRecord:
        now = utc_now_iso()
        next_record = record.model_copy(update={
            "status": status,
            "stop_reason": stop_reason,
            "updated_at": now,
        })
        committed = self._commit(self._bump(next_record))
        return committed.model_copy(deep=True)

    # ── 认领下一个 trial ─────────────────────────────────────────

    def _claim_next(self, study, record, stop_event):
        if _stop_requested(stop_event):
            return _PAUSE
        trials = study.trials
        tail_pending = (trials[-1] if trials and trials[-1].state == "PENDING"
                        else None)
        # 尾部 PENDING 而无执行 attempt：沿用该编号（ask 幂等返回），不另取样。
        number = (tail_pending.number if tail_pending is not None
                  else len(trials))
        if number >= study.config.budget:
            return _CONTINUE  # 外层转为 COMPLETED
        # 环境漂移只在新训练（ask 新候选）前判定。
        self._check_execution_environment(record)
        request_id = execution_request_id(study.study_id, number)
        self._service.validate_binding(study.study_id)
        trial = self._service.ask(study.study_id, request_id=request_id)
        if trial.state != "PENDING":
            return _CONTINUE
        adapter = self._make_adapter()
        try:
            prepared = adapter.prepare(study, trial, record.config)
        except GuardrailRejection:
            self._safe_tell(study.study_id, trial.number, ResultInput(
                state="FAILED", reason_code="invalid_params"))
            return _CONTINUE
        if prepared["run_relpath"] != f"{study.study_id}/{trial.trial_id}":
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "prepared run_relpath is not bound to study/trial")
        # 写 run_state + PREPARED attempt；写盘失败不能启动。
        run_state = new_run_state("tuning")
        run_dir = self._output_root / prepared["run_relpath"]
        reject_link_chain(run_dir, code="HPO_CORRUPT_EXECUTION")
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = self._state_path(prepared["run_relpath"])
        write_run_state(state_path, run_state)
        attempt = ExecutionAttempt(
            trial_number=trial.number,
            trial_id=trial.trial_id,
            request_id=trial.request_id,
            run_id=run_state.run_id,
            phase="PREPARED",
            candidate_params=dict(prepared["candidate_params"]),
            effective_params=dict(prepared["effective_params"]),
            command=list(prepared["command"]),
            command_executable=prepared["command"][0],
            run_relpath=prepared["run_relpath"],
            args_sha256=prepared["args_sha256"],
            started_at=utc_now_iso(),
        )
        next_record = record.model_copy(update={
            "attempts": list(record.attempts) + [attempt],
            "updated_at": utc_now_iso(),
        })
        self._commit(self._bump(next_record))
        return _CONTINUE

    # ── 推进当前 attempt ─────────────────────────────────────────

    def _drive_attempt(self, record, study, attempt, stop_event):
        phase = attempt.phase
        if phase == "PREPARED":
            self._check_execution_environment(record)
            self._service.validate_binding(study.study_id)
            return self._launch_new(record, study, attempt, stop_event)
        if phase == "LAUNCH_INTENT":
            return self._blocked(record, attempt, "HPO_RECOVERY_REQUIRED",
                                 "launch outcome is not provable")
        if phase == "RUNNING":
            return self._reconcile_running(record, attempt, stop_event)
        if phase == "EXITED":
            return self._decide_from_exited(record, attempt, stop_event)
        if phase in ("RESULT_READY", "TOLD"):
            return self._replay(record, study, attempt)
        return _CONTINUE  # pragma: no cover

    def _launch_new(self, record, study, attempt, stop_event):
        """PREPARED→LAUNCH_INTENT→(launch)→RUNNING→monitor→FINALIZED。"""
        if _stop_requested(stop_event):
            # 启动前停止：登记 CANCELLED（已取样消耗预算），PAUSED。
            result = ResultInput(state="CANCELLED", reason_code="user_stopped")
            return self._commit_result(record, attempt, result,
                                       call_finalizer=False, stop_event=stop_event)
        adapter = self._make_adapter()
        prepared = {
            "trial_id": attempt.trial_id,
            "run_relpath": attempt.run_relpath,
            "effective_params": attempt.effective_params,
            "command": attempt.command,
            "args_sha256": attempt.args_sha256,
        }
        # R2a-2：完整启动预验（args 摘要/内容/命令/路径）在发布 LAUNCH_INTENT 之前
        # 完成。预验失败属审计/持久化损坏：保留原稳定错误码、阻断取样与启动、保留
        # 证据，绝不降级为普通 FAILED 继续消耗预算。
        try:
            adapter.validate_launch(prepared)
        except HpoError as exc:
            self._blocked(record, attempt, exc.code, exc.message)
        intent = attempt.model_copy(update={"phase": "LAUNCH_INTENT"})
        next_record = self._commit(self._bump(
            self._replace_attempt(record, intent)))
        try:
            proc = adapter.launch(prepared)
        except HpoError as exc:
            # 已发布 LAUNCH_INTENT 后、临近启动复验失败：adapter.launch 在创建进程
            # 前完成校验（同一调用内可确证“已知没有启动”）。保留原错误码阻断。
            self._blocked(next_record, intent, exc.code, exc.message)
        except Exception as exc:
            # 真实进程创建失败（非 HpoError 审计错误）：登记 FAILED，无 finalizer。
            failed = attempt.model_copy(update={
                "phase": "EXITED",
                "returncode": -1,
                "error_code": "HPO_PREFLIGHT_FAILED",
                "error_message": str(exc)[:2048],
                "finished_at": utc_now_iso(),
            })
            next_record = self._commit(self._bump(
                self._replace_attempt(next_record, failed)))
            result = ResultInput(state="FAILED", reason_code="training_failed")
            return self._commit_result(next_record, failed, result,
                                       call_finalizer=False,
                                       stop_event=stop_event)
        try:
            return self._launch_owned(next_record, attempt, proc,
                                      adapter, study, stop_event)
        except Exception:
            # R1：任何写盘/身份/监控失败都清理本次明确持有的进程。
            self._stop_process(proc)
            raise

    def _launch_owned(self, intent_record, attempt, proc, adapter,
                      study, stop_event):
        pid = getattr(proc, "pid", None)
        if pid is None:
            raise HpoError("HPO_PREFLIGHT_FAILED", "launch returned no process id")
        rc = proc.poll()
        identity = _capture_process_identity(pid)
        if identity is None:
            if rc is not None:
                # 短命进程已退出：直接记录 EXITED 并走结果判定（无法捕获身份）。
                exited = attempt.model_copy(update={
                    "phase": "EXITED",
                    "returncode": rc,
                    "finished_at": utc_now_iso(),
                })
                intent_record = self._commit(self._bump(
                    self._replace_attempt(intent_record, exited)))
                return self._decide_from_exited(intent_record, exited, stop_event)
            # 活动进程身份无法捕获：终止并 wait，阻断本轮。
            self._stop_process(proc)
            return self._blocked(intent_record,
                                 attempt.model_copy(update={"phase": "LAUNCH_INTENT"}),
                                 "HPO_PROCESS_STILL_ACTIVE",
                                 "process identity cannot be captured")
        running = attempt.model_copy(update={
            "phase": "RUNNING",
            "pid": pid,
            "process_create_token": identity,
        })
        intent_record = self._commit(self._bump(
            self._replace_attempt(intent_record, running)))
        state = self._ensure_run_state(attempt)
        try:
            write_run_state(
                self._state_path(attempt.run_relpath),
                with_status_phase(state, status="running", phase="training",
                                  pid=pid, process_create_token=identity))
        except Exception:
            raise HpoError("HPO_PERSISTENCE_ERROR",
                           "failed to persist run_state before training")
        return self._monitor(intent_record, study, running, proc, adapter,
                             stop_event)

    def _monitor(self, record, study, attempt, proc, adapter, stop_event):
        deadline = time.monotonic() + record.config.timeout_seconds
        termination = None
        while True:
            rc = proc.poll()
            if rc is not None:
                exited = attempt.model_copy(update={
                    "phase": "EXITED",
                    "returncode": rc,
                    "termination_reason": termination,
                    "finished_at": utc_now_iso(),
                })
                record = self._commit(self._bump(
                    self._replace_attempt(record, exited)))
                return self._decide_from_exited(record, exited, stop_event)
            if termination is None:
                if _stop_requested(stop_event):
                    termination = "user_stopped"
                elif time.monotonic() >= deadline:
                    termination = "timeout"
                if termination is not None:
                    attempt = attempt.model_copy(
                        update={"termination_reason": termination})
                    record = self._commit(self._bump(
                        self._replace_attempt(record, attempt)))
                    state = self._ensure_run_state(attempt)
                    try:
                        write_run_state(
                            self._state_path(attempt.run_relpath),
                            with_status_phase(state, status="running",
                                              phase="stopping"))
                    except Exception:
                        raise HpoError("HPO_PERSISTENCE_ERROR",
                                       "failed to persist stopping run_state")
                    if not self._stop_process(proc):
                        return self._blocked(record, attempt,
                                             "HPO_PROCESS_STILL_ACTIVE",
                                             "stopped process did not confirm exit")
                    continue
            time.sleep(POLL_INTERVAL)

    def _stop_process(self, proc) -> bool:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=TERMINATE_WAIT)
            if proc.poll() is not None:
                return True
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=KILL_WAIT)
        except Exception:
            return False
        return proc.poll() is not None

    def _ensure_run_state(self, attempt):
        from auto_tune.modules.run_state.service import read_run_state
        path = self._state_path(attempt.run_relpath)
        state = read_run_state(path, run_kind="tuning")
        if state is None:
            state = new_run_state("tuning")
            state = replace(state, run_id=attempt.run_id,
                            started_at=attempt.started_at or state.started_at)
            write_run_state(path, state)
        return state

    def _write_terminal_run_state(self, attempt, result):
        """以结果事实投影 RunState 终态；写盘失败抛 HPO_PERSISTENCE_ERROR。"""
        state = self._ensure_run_state(attempt)
        status = _RUN_STATE_STATUS.get(result.state, "failed")
        reason = None if result.state == "SUCCESS" else result.reason_code
        finished = attempt.finished_at or utc_now_iso()
        terminal = replace(
            state,
            status=status,
            phase="terminal",
            terminal_reason=reason,
            finished_at=finished,
            pid=attempt.pid,
            process_create_token=attempt.process_create_token,
            updated_at=utc_now_iso(),
        )
        try:
            write_run_state(self._state_path(attempt.run_relpath), terminal)
        except Exception as exc:
            raise HpoError("HPO_PERSISTENCE_ERROR",
                           f"failed to persist terminal run_state: {exc}") from exc
        return terminal

    def _decide_from_exited(self, record, attempt, stop_event):
        """EXITED 后按 returncode/termination 决定结果并收尾（不再启动）。"""
        if attempt.returncode == 0 and attempt.termination_reason is None:
            outcome = self._make_adapter().collect(
                self._service.load_study(record.study_id), attempt)
            if outcome.result is not None:
                return self._commit_result(
                    record, attempt, outcome.result, actual=outcome,
                    call_finalizer=True, stop_event=stop_event)
            failure = ResultInput(state="FAILED",
                                  reason_code=outcome.reason_code or "training_failed")
            return self._commit_result_with_error(
                record, attempt, failure, outcome.reason_code or "training_failed",
                outcome.error_code, outcome.error_message, actual=outcome,
                call_finalizer=True, stop_event=stop_event)
        reason = "training_failed"
        result_state = "FAILED"
        if attempt.termination_reason == "user_stopped":
            result_state = "CANCELLED"
            reason = "user_stopped"
        elif attempt.termination_reason == "timeout":
            result_state = "FAILED"
            reason = "timeout"
        elif attempt.returncode != 0:
            reason = "oom" if self._make_adapter().detect_oom(
                self._output_root / attempt.run_relpath) else "training_failed"
        result = ResultInput(state=result_state, reason_code=reason)
        return self._commit_result(record, attempt, result,
                                   call_finalizer=True, stop_event=stop_event)

    def _reconcile_running(self, record, attempt, stop_event):
        """恢复遇到 RUNNING：PID+token 核验；MATCH/UNVERIFIABLE 阻断。"""
        match = compare_process_identity(
            ProcessIdentity(attempt.pid, attempt.process_create_token))
        if match is IdentityMatch.MATCH:
            return self._blocked(record, attempt, "HPO_PROCESS_STILL_ACTIVE",
                                 "original process remains alive")
        if match is IdentityMatch.UNVERIFIABLE:
            return self._blocked(record, attempt, "HPO_RECOVERY_REQUIRED",
                                 "process identity is unverifiable")
        # MISSING/MISMATCH：不杀复用 PID，登记 INTERRUPTED 并继续剩余预算。
        result = ResultInput(state="INTERRUPTED",
                             reason_code="process_interrupted")
        return self._commit_result(record, attempt, result,
                                   call_finalizer=True, stop_event=stop_event)

    def _replay(self, record, study, attempt):
        """RESULT_READY/TOLD：幂等 tell / 重放 finalize，不得重训。

        已提交/已退出事实的重放不进行环境漂移判定；漂移只在启动新训练前阻止。
        """
        result = attempt.result
        if result is None:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "commit phase requires a result")
        self._write_terminal_run_state(attempt, result)
        self._safe_tell(study.study_id, attempt.trial_number, result)
        told = attempt.model_copy(update={"phase": "TOLD"})
        record = self._commit(self._bump(
            self._replace_attempt(self._read_exec(record.study_id), told)))
        finalizer_record = None
        if attempt.pid is not None or attempt.returncode is not None:
            # 已启动的 trial：补 finalize（幂等 upsert，不重训）。
            finalizer_record = self._run_finalizer(study, told)
        finalized = told.model_copy(update={
            "phase": "FINALIZED",
            "finalizer_record": finalizer_record,
        })
        self._commit(self._bump(
            self._replace_attempt(self._read_exec(record.study_id), finalized)))
        return _CONTINUE

    # ── 结果提交 / tell / finalize ───────────────────────────────

    def _commit_result(self, record, attempt, result, *, actual=None,
                       call_finalizer=True, stop_event=None):
        return self._commit_result_with_error(
            record, attempt, result, None, None, None, actual=actual,
            call_finalizer=call_finalizer, stop_event=stop_event)

    def _commit_result_with_error(self, record, attempt, result, _reason,
                                  error_code=None, error_message=None, *,
                                  actual=None, call_finalizer=True,
                                  stop_event=None):
        updates = {
            "phase": "RESULT_READY",
            "result": result,
            "error_code": error_code,
            "error_message": error_message,
            "finished_at": utc_now_iso(),
        }
        if actual is not None:
            if actual.actual_args is not None:
                updates["actual_args"] = actual.actual_args
                updates["actual_args_sha256"] = actual.actual_args_sha256
            if actual.diagnostics is not None:
                updates["metric_diagnostics"] = actual.diagnostics
        attempt = attempt.model_copy(update=updates)
        record = self._commit(self._bump(
            self._replace_attempt(record, attempt)))
        # 终态 RunState 写盘失败稳定返回 HPO_PERSISTENCE_ERROR，阻断后续 trial。
        self._write_terminal_run_state(attempt, result)
        self._safe_tell(record.study_id, attempt.trial_number, result)
        told = attempt.model_copy(update={"phase": "TOLD"})
        record = self._commit(self._bump(
            self._replace_attempt(self._read_exec(record.study_id), told)))
        finalizer_record = None
        if call_finalizer:
            study = self._service.load_study(record.study_id)
            finalizer_record = self._run_finalizer(study, told)
        finalized = told.model_copy(update={
            "phase": "FINALIZED",
            "finalizer_record": finalizer_record,
        })
        self._commit(self._bump(
            self._replace_attempt(self._read_exec(record.study_id), finalized)))
        if _stop_requested(stop_event):
            return _PAUSE
        return _CONTINUE

    def _run_finalizer(self, study, told):
        adapter = self._make_adapter()
        finalizer_record = adapter.finalize(study, told)
        if finalizer_record and finalizer_record.get("history_error"):
            raise HpoError("HPO_PERSISTENCE_ERROR",
                           "finalizer history write failed; stopping budget")
        return finalizer_record

    def _safe_tell(self, study_id, trial_number, result):
        try:
            self._service.tell(study_id, trial_number, result)
        except HpoError as exc:
            if exc.code == "HPO_RESULT_CONFLICT":
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "study terminal state conflicts with execution") from exc
            raise

    def _blocked(self, record, attempt, code: str, message: str):
        marked = attempt.model_copy(update={
            "error_code": code,
            "error_message": message[:2048],
        })
        next_record = record.model_copy(update={
            "attempts": self._replace_attempts(record.attempts, marked),
            "status": "BLOCKED",
            "stop_reason": "audit_failure",
            "updated_at": utc_now_iso(),
        })
        self._commit(self._bump(next_record))
        raise HpoError(code, message)

    # ── 跨文件语义校验 ───────────────────────────────────────────

    def _ensure_roots_match(self, record: ExecutionRecord) -> None:
        roots = record.roots
        if (roots.storage_root != str(self._storage_root)
                or roots.output_root != str(self._output_root)
                or roots.log_root != str(self._log_root)):
            raise HpoError("HPO_EXECUTION_CONFLICT",
                           "runner roots differ from the frozen execution roots")

    def _cross_validate(self, study, record: ExecutionRecord) -> None:
        """execution.json 与 study.json 的同一身份/参数/命令/终态语义校验。"""
        for attempt in record.attempts:
            if attempt.trial_number >= len(study.trials):
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "attempt trial is absent from the study")
            trial = study.trials[attempt.trial_number]
            if (attempt.trial_id != trial.trial_id
                    or attempt.request_id != trial.request_id):
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "attempt identity does not match the study trial")
            if attempt.candidate_params != trial.candidate_params:
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "attempt candidate differs from the study trial")
            try:
                validate_candidate(attempt.candidate_params,
                                   epochs=study.config.epochs)
            except HpoError as exc:
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "attempt candidate violates the search space") from exc
            # R2b：effective 必须与“权威输入重建值”完全一致，不能把可编辑的
            # attempt.effective_params 当作权威；覆盖 epochs/seed、batch/imgsz/
            # device、model/data 与全部 FIXED_PARAMS。
            authoritative = _authoritative_effective(study, record.config, trial)
            if attempt.effective_params != authoritative:
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "attempt effective_params deviate from frozen config")
            # R2a：用权威 effective 重建预期命令并与已发布命令全等（含 executable、
            # 全部参数与值、project/name/exist_ok），拒绝重复/额外覆盖参数。
            # R2a-3/R6：以冻结的 command_executable（与 command 分离的依据）为锚点
            # 重建，不解析当前 YOLO；命令首 token 必须与该冻结依据一致，不能拿被
            # 验证的 command[0] 自身当依据。command[0] 单独被篡改会在模型读入层拒绝。
            if not attempt.command:
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "attempt command is empty")
            if attempt.command[0] != attempt.command_executable:
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "attempt executable token deviates from the "
                               "frozen command_executable")
            expected_command = _authoritative_command(
                attempt.trial_id, attempt.run_relpath,
                authoritative, self._output_root,
                executable=attempt.command_executable)
            if attempt.command != expected_command:
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "attempt command deviates from the rebuilt command")
            if attempt.run_relpath != f"{study.study_id}/{attempt.trial_id}":
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "attempt run_relpath is not bound to study/trial")
            if attempt.result is not None:
                evidence = attempt.result.evidence
                if evidence is not None and evidence.run_id != attempt.run_id:
                    raise HpoError("HPO_CORRUPT_EXECUTION",
                                   "attempt evidence run_id is unrelated")
            if attempt.phase == "FINALIZED":
                if trial.state == "PENDING" or trial.result is None:
                    raise HpoError("HPO_CORRUPT_EXECUTION",
                                   "finalized attempt has a non-terminal study trial")
                result = attempt.result
                if result is None or result.state != trial.state:
                    raise HpoError("HPO_CORRUPT_EXECUTION",
                                   "finalized attempt result contradicts the study")
                if (result.value != trial.result.value
                        or result.reason_code != trial.result.reason_code):
                    raise HpoError("HPO_CORRUPT_EXECUTION",
                                   "finalized attempt payload contradicts the study")
        if record.status == "COMPLETED":
            # R2c：COMPLETED 必须证明预算已耗尽；空 study 或未完成预算均拒绝。
            if any(t.state == "PENDING" for t in study.trials):
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "completed execution has a pending trial")
            if any(a.phase != "FINALIZED" for a in record.attempts):
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "completed execution has unfinished attempts")
            if len(study.trials) != study.config.budget:
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "completed execution has not exhausted the budget")

    # ── 内部读写辅助 ─────────────────────────────────────────────

    def _read_exec(self, study_id: str) -> ExecutionRecord:
        with self._exec_store.locked(study_id):
            return self._exec_store.read(study_id)

    def _commit(self, record: ExecutionRecord) -> ExecutionRecord:
        with self._exec_store.locked(record.study_id):
            self._exec_store.write(record)
        return record

    def _bump(self, record: ExecutionRecord) -> ExecutionRecord:
        return record.model_copy(update={
            "revision": record.revision + 1,
            "updated_at": utc_now_iso(),
        })

    def _replace_attempt(self, record, attempt) -> ExecutionRecord:
        return record.model_copy(update={
            "attempts": self._replace_attempts(record.attempts, attempt)})

    @staticmethod
    def _replace_attempts(attempts, attempt):
        return [a if a.trial_number != attempt.trial_number else attempt
                for a in attempts]

    def _current_attempt(self, record: ExecutionRecord):
        for attempt in reversed(record.attempts):
            if attempt.phase != "FINALIZED":
                return attempt
        return None

    def _state_path(self, run_relpath: str):
        return self._output_root / run_relpath / "run_state.json"

    def _check_execution_environment(self, record: ExecutionRecord) -> None:
        current = _current_execution_environment()
        env = record.environment
        for key in ("sys_executable", "python_version", "cuda_version",
                    "optuna_version", "numpy_version", "torch_version",
                    "ultralytics_version"):
            if current.get(key) != getattr(env, key, ""):
                raise HpoError("HPO_VERSION_MISMATCH",
                               f"environment {key} changed since prepare")

    def _make_adapter(self) -> ExecutionAdapter:
        return ExecutionAdapter(self._output_root, self._log_root)


def _make_roots(storage_root, output_root, log_root):
    from .execution_models import ExecutionRoots
    return ExecutionRoots(storage_root=str(storage_root),
                          output_root=str(output_root),
                          log_root=str(log_root))


def _make_environment(data: dict):
    from .execution_models import ExecutionEnvironment
    return ExecutionEnvironment(
        sys_executable=data["sys_executable"],
        python_version=data["python_version"],
        torch_version=data.get("torch_version", ""),
        cuda_version=data.get("cuda_version", ""),
        ultralytics_version=data.get("ultralytics_version", ""),
        optuna_version=data.get("optuna_version", ""),
        numpy_version=data.get("numpy_version", ""),
    )


def _capture_process_identity(pid: int):
    from auto_tune.modules.run_state.process_identity import capture_process_identity
    identity = capture_process_identity(pid)
    if identity is None:
        return None
    return identity.process_create_token


def _read_run_state(path):
    from auto_tune.modules.run_state.service import read_run_state
    return read_run_state(path, run_kind="tuning")


def _current_execution_environment() -> dict:
    import platform
    import sys
    env = dict(_current_environment())
    env["sys_executable"] = str(sys.executable)
    env["python_version"] = platform.python_version()
    env["torch_version"] = _dist_version("torch")
    env["ultralytics_version"] = _dist_version("ultralytics")
    env["cuda_version"] = ""
    try:
        import torch
        env["cuda_version"] = str(torch.version.cuda or "")
    except Exception:
        pass
    return env


def _dist_version(name: str) -> str:
    from importlib import metadata
    try:
        return str(metadata.version(name))
    except metadata.PackageNotFoundError:
        return ""
