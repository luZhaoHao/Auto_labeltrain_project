"""H1.3 追加返修 Task 2/3: 使用最佳参数的正式训练提交与关联结果。

HPO 来源是真实的 HpoService/HpoRunner（tmp_path），普通训练子进程与收尾被替换为
假实现，因此不会执行真实 YOLO、不会调用网络 LLM。所有负面用例都断言“零进程、
零新训练目录、HPO 事实字节不变”。
"""

import asyncio
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import (
    Evidence,
    ExecutionConfig,
    HpoRunner,
    HpoService,
    ResultInput,
    StudyConfig,
)
from auto_tune.ui.hpo_training import RUNTIME_RUN_ID_RE

_SEARCH_KEYS = ("optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs")


def _rid():
    return uuid.uuid4().hex


def _evidence(epoch=1):
    return Evidence(run_id=f"run-{_rid()}", artifact_relpath="results.csv",
                    artifact_sha256="0" * 64, epoch=epoch)


def _make_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for n in range(4):
        Image.new("RGB", (16, 16)).save(source / f"{n}.jpg")
        (source / f"{n}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    snapshot = create_dataset_snapshot(source, tmp_path / "snapshots",
                                       val_ratio=0.5, seed=42,
                                       class_names={0: "part"})
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"hpo-formal-test-not-a-real-model")
    return snapshot, model


class CompletingRunner:
    """Real prepared execution record, viewed as a terminal (COMPLETED) status.

    ``prepare()`` is the real HpoRunner (so ``execution.json`` is a genuine
    frozen audit fact); only the read-back status is pinned, because a real
    COMPLETED record needs a real training run.
    """

    def __init__(self, inner: HpoRunner, status: str = "COMPLETED"):
        self._inner = inner
        self._status = status
        self.run_calls = 0

    def status(self, study_id):
        record = self._inner.status(study_id)
        return record.model_copy(update={"status": self._status})

    def prepare(self, study_id, config):
        return self._inner.prepare(study_id, config)

    def run(self, study_id, *, stop_event=None):
        self.run_calls += 1

    def resume(self, study_id, *, stop_event=None):
        self.run_calls += 1


class Source:
    """Real HPO source study with deterministic SUCCESS trials."""

    def __init__(self, tmp_path, values=(0.5, 0.9), epochs=10, status="COMPLETED"):
        self.snapshot, self.model = _make_inputs(tmp_path)
        self.root = tmp_path / "storage"
        self.service = HpoService(self.root)
        self.study = self.service.create_study(
            StudyConfig(budget=5, epochs=epochs),
            snapshot_dir=self.snapshot.snapshot_path, model_path=self.model)
        self.exec_config = ExecutionConfig(batch=4, imgsz=64, device="cpu",
                                           timeout_seconds=120)
        # real prepare (writes execution.json) before any trial becomes terminal
        inner = HpoRunner(self.root, tmp_path / "out", tmp_path / "log")
        inner.prepare(self.study.study_id, self.exec_config)
        self.inner_runner = inner
        self.runner = CompletingRunner(inner, status)
        for value in values:
            trial = self.service.ask(self.study.study_id, request_id=_rid())
            self.service.tell(self.study.study_id, trial.number,
                              ResultInput(state="SUCCESS", value=value,
                                          evidence=_evidence(1)))
        trials = self.service.load_study(self.study.study_id).trials
        self.top = trials[-1] if trials else None
        # deterministic warmup so the happy paths do not depend on sampling
        if self.top is not None:
            self.set_warmup(0)
            self.top = self.service.load_study(self.study.study_id).trials[-1]

    @property
    def study_id(self):
        return self.study.study_id

    def set_warmup(self, value: int, trial_number: int | None = None) -> None:
        """Rewrite one stored candidate's warmup_epochs (a legal fact fixture).

        The sampled range is ``0..min(5, epochs-1)``; a value that is legal for
        the original epochs but not for a smaller formal epochs is exactly the
        candidate-condition case the rework must reject (never clamp).
        """
        number = self.top.number if trial_number is None else trial_number
        path = self.root / self.study_id / "study.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        # sampled and mapped values must stay consistent (deep re-validation)
        data["trials"][number]["sampled_params"]["warmup_epochs"] = value
        data["trials"][number]["candidate_params"]["warmup_epochs"] = value
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")


class FakeProc:
    def __init__(self):
        class Stdout:
            async def readline(self):
                return b""

        self.stdout = Stdout()
        self.pid = os.getpid()
        self.returncode = 0

    async def wait(self):
        return 0


def _patch_app(monkeypatch, tmp_path, source, detect_dir=None):
    """Patch app-level HPO source + the ordinary training subprocess/finalizer."""
    from auto_tune.ui import app as app_mod

    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    real_join = os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir.joinpath(*parts[1:]))
        return real_join(*parts)

    monkeypatch.setattr(os.path, "join", fake_join)
    monkeypatch.setattr(app_mod, "_hpo_service", source.service)
    monkeypatch.setattr(app_mod, "_hpo_runner", source.runner)
    target_detect = str(detect_dir or (tmp_path / "detect"))
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: target_detect)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
        lambda: "yolo")
    # the route resolves them through the module attribute at call time
    monkeypatch.setattr("auto_tune.ui.app.find_detect_dir", lambda: target_detect,
                        raising=False)

    launched = []

    async def fake_subprocess_exec(*args, **kwargs):
        launched.append(list(args))
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)

    def fake_finalize(run_dir, run_name, source_kind, config, log_dir, training_status,
                      started_at=None, finished_at=None, training_error=None, **kw):
        return {
            "run_id": f"manual:{run_name}", "run_name": run_name,
            "source": "manual", "status": "completed", "analysis_status": "skipped",
            "metrics": {}, "artifacts": {"report_path": None}, "error": None,
            "analysis_error": None, "history_error": None,
        }

    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run", fake_finalize)
    # The authoritative experiment index used for the "查看结果" identity must be
    # the isolated tmp one, never the developer's real log/auto_tune.db.
    monkeypatch.setattr(
        app_mod, "APP_CONFIG",
        dict(app_mod.APP_CONFIG, local_index={
            "database_path": str(log_dir / "auto_tune.db"),
            "backup_dir": str(log_dir / "db_backups"),
        }),
        raising=False,
    )
    app_mod._running_training.clear()
    return app_mod, launched


def _index_experiment(app_mod, tmp_path, train_name, *, runtime_run_id=None,
                      run_name=None, run_dir=None):
    """Register one real experiment in the app's (isolated) local index."""
    service = app_mod._local_index_service()
    record = {
        "run_id": f"manual:{train_name}",
        "run_name": run_name or train_name,
        "source": "manual",
        "status": "completed",
        "analysis_status": "completed",
        "started_at": "2026-09-14T00:00:00Z",
        "finished_at": "2026-09-14T00:10:00Z",
        "params": {"model": "yolov8n.pt"},
        "metrics": {"mAP50": 0.71, "mAP50_95": 0.44},
        "artifacts": {"report_path": None,
                      "run_dir": run_dir or str(tmp_path / "detect" / train_name)},
        "error": None,
    }
    return service.index_experiment(record, runtime_run_id=runtime_run_id)["run_id"]


def _strip_runtime_identity(train_dir: Path) -> None:
    """Rewrite hpo_source.json as a pre-rework record without ``runtime_run_id``."""
    path = train_dir / "hpo_source.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.pop("runtime_run_id", None)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _post(app_mod, study_id, payload, raw=False):
    client = TestClient(app_mod.app)
    if raw:
        return client.post(f"/api/hpo/studies/{study_id}/train-best",
                           content=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
    return client.post(f"/api/hpo/studies/{study_id}/train-best", json=payload)


def _valid_body(source, **training):
    """正式训练只允许改 epochs；其余三个条件必须等于研究的权威执行条件。

    客户端仍携带完整的 ``training_config``（接口兼容），但 batch/imgsz/device
    一律由前端取当前研究的权威 execution 值，服务端再严格比对。
    """
    cfg = {"epochs": 2, "batch": 4, "imgsz": 64, "device": "cpu"}
    cfg.update(training)
    return {"trial_id": source.top.trial_id, "training_config": cfg}


def _models_dir(app_mod):
    return app_mod._RUN_MANAGER


@pytest.fixture
def source(tmp_path):
    return Source(tmp_path)


# ── 严格参数/身份校验：任何拒绝都必须零进程、零新目录 ──────────────


def _assert_zero_side_effects(tmp_path, launched, source, app_mod):
    assert launched == []
    assert not (tmp_path / "detect").exists()
    assert list(tmp_path.rglob("hpo_source.json")) == []
    assert list(tmp_path.rglob("args.yaml")) == []
    assert app_mod._RUN_MANAGER.reservation_owner() is None
    assert app_mod._RUN_MANAGER.active_manual() is None


def test_train_best_unknown_study_zero_side_effects(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    body = _valid_body(source)
    resp = _post(app_mod, "hpo_" + "0" * 32, body)
    assert resp.status_code in (404, 409)
    assert resp.json()["error_code"] in ("HPO_NOT_FOUND", "HPO_SOURCE_INVALID")
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)


def test_train_best_requires_completed_study(tmp_path, monkeypatch):
    source = Source(tmp_path, status="PAUSED")
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    resp = _post(app_mod, source.study_id, _valid_body(source))
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_SOURCE_INVALID"
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)


def test_train_best_rejects_non_rank_first_trial(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    study = source.service.load_study(source.study_id)
    body = {"trial_id": study.trials[0].trial_id,  # the 0.5 trial
            "training_config": {"epochs": 2, "batch": 4, "imgsz": 64,
                                "device": "cpu"}}
    resp = _post(app_mod, source.study_id, body)
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_SOURCE_INVALID"
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)


def test_train_best_rejects_forged_trial_id(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    body = _valid_body(source)
    body["trial_id"] = source.study_id + "_t9999"
    resp = _post(app_mod, source.study_id, body)
    assert resp.status_code == 409
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)


def test_train_best_rejects_client_supplied_search_params(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    body = _valid_body(source)
    body["training_config"]["lr0"] = 0.001  # extra → forbidden
    resp = _post(app_mod, source.study_id, body)
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "INVALID_HPO_FIELD"
    assert resp.json()["field"] == "lr0"
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)


def test_train_best_rejects_extra_body_field(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    body = _valid_body(source)
    body["search_params"] = {"optimizer": "SGD"}
    resp = _post(app_mod, source.study_id, body)
    assert resp.status_code == 422
    assert resp.json()["field"] == "search_params"
    assert resp.json()["reason_code"] == "FIELD_UNKNOWN"
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)


@pytest.mark.parametrize("label,cfg", [
    ("epochs bool", {"epochs": True}),
    ("epochs str", {"epochs": "3"}),
    ("epochs float", {"epochs": 2.5}),
    ("epochs zero", {"epochs": 0}),
    ("epochs over", {"epochs": 5000}),
    ("batch bool", {"batch": True}),
    ("batch str", {"batch": "8"}),
    ("batch zero", {"batch": 0}),
    ("batch over", {"batch": 999}),
    ("imgsz str", {"imgsz": "96"}),
    ("imgsz not multiple", {"imgsz": 100}),
    ("imgsz too small", {"imgsz": 16}),
    ("imgsz too large", {"imgsz": 4096}),
    ("device int", {"device": 0}),
    ("device bogus", {"device": "gpu0"}),
    ("device gpu over", {"device": "64"}),
    ("nan epochs", {"epochs": float("nan")}),
])
def test_train_best_rejects_bad_training_config(tmp_path, monkeypatch, source,
                                                label, cfg):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    body = _valid_body(source)
    body["training_config"].update(cfg)
    resp = _post(app_mod, source.study_id, body, raw=True)
    assert resp.status_code == 422, label
    body_json = resp.json()
    assert body_json["error_code"] == "INVALID_HPO_FIELD", label
    assert body_json["field"] in ("epochs", "batch", "imgsz", "device"), label
    assert body_json["reason_code"], label
    assert "Input should" not in resp.text
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)


def test_train_best_rejects_missing_training_config_field(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    body = {"trial_id": source.top.trial_id,
            "training_config": {"epochs": 2, "batch": 4, "imgsz": 64}}
    resp = _post(app_mod, source.study_id, body)
    assert resp.status_code == 422
    assert resp.json()["field"] == "device"
    assert resp.json()["reason_code"] == "FIELD_REQUIRED"
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)


@pytest.mark.parametrize("label,drifting", [
    ("batch", {"batch": 8}),
    ("imgsz", {"imgsz": 96}),
    ("device", {"device": "1"}),
])
def test_train_best_rejects_conditions_that_differ_from_the_study(
        tmp_path, monkeypatch, source, label, drifting):
    """正式训练除 epochs 外的条件冻结：与 execution 事实不一致 → 零启动拒绝。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    study_bytes = (source.root / source.study_id / "study.json").read_bytes()
    exec_bytes = (source.root / source.study_id / "execution.json").read_bytes()
    resp = _post(app_mod, source.study_id, _valid_body(source, epochs=2, **drifting))
    assert resp.status_code == 422, label
    body = resp.json()
    # 合法但被冻结的取值不是“字段格式错误”，而是配置冲突 → 稳定配置错误
    assert body["error_code"] == "HPO_INVALID_CONFIG", label
    assert body["error"] and body["next_action"]
    assert set(body) == {"error_code", "error", "next_action"}
    assert "已有训练" not in body["error"]
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)
    # HPO 事实完全没有被改写
    assert (source.root / source.study_id / "study.json").read_bytes() == study_bytes
    assert (source.root / source.study_id / "execution.json").read_bytes() == exec_bytes


def test_train_best_binding_drift_zero_start(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    source.model.write_bytes(b"changed-model-content")
    resp = _post(app_mod, source.study_id, _valid_body(source))
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_BINDING_MISMATCH"
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)


def test_train_best_warmup_incompatible_with_new_epochs_zero_start(
        tmp_path, monkeypatch, source):
    """候选条件在新 epochs 下不合法时必须安全拒绝，不得静默裁剪。"""
    source.set_warmup(3)  # legal for epochs=10 (<=min(5,9)); illegal for epochs=2
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    resp = _post(app_mod, source.study_id, _valid_body(source, epochs=2))
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_SOURCE_INVALID"
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)


def test_train_best_blocked_by_active_training_zero_side_effects(
        tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)

    class Stub:
        run_kind = "tuning"
        run_id = "tuning:stub"

        def is_active(self):
            return True

        def is_done(self):
            return False

    stub = Stub()
    app_mod._RUN_MANAGER.register(stub)
    try:
        resp = _post(app_mod, source.study_id, _valid_body(source))
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "RUN_ALREADY_ACTIVE"
        _assert_zero_side_effects(tmp_path, launched, source, app_mod)
    finally:
        app_mod._RUN_MANAGER.unregister(stub.run_id)


def test_train_best_blocked_by_active_hpo_controller(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)

    class HpoStub:
        run_kind = "hpo"
        run_id = source.study_id
        error_code = None

        def is_active(self):
            return True

        def is_done(self):
            return False

    stub = HpoStub()
    app_mod._RUN_MANAGER.register(stub)
    try:
        resp = _post(app_mod, source.study_id, _valid_body(source))
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "HPO_SOURCE_INVALID"
        _assert_zero_side_effects(tmp_path, launched, source, app_mod)
    finally:
        app_mod._RUN_MANAGER.unregister(stub.run_id)


# ── 成功路径：四项条件生效，其余全部来自服务端权威记录 ──────────────


def test_train_best_uses_new_conditions_and_authoritative_params(
        tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    study = source.service.load_study(source.study_id)
    study_bytes = (source.root / source.study_id / "study.json").read_bytes()
    exec_bytes = (source.root / source.study_id / "execution.json").read_bytes()

    with TestClient(app_mod.app) as client:
        resp = client.post(f"/api/hpo/studies/{source.study_id}/train-best",
                           json=_valid_body(source, epochs=2))
        assert resp.status_code == 202, resp.text
        payload = resp.json()
        assert payload["status"] == "accepted"
        assert payload["run_id"].startswith("manual:")
        assert payload["train_name"] == "train1"
        assert payload["source"]["trial_id"] == source.top.trial_id
        assert payload["source"]["value"] == 0.9
        assert payload["training_config"] == {"epochs": 2, "batch": 4,
                                              "imgsz": 64, "device": "cpu"}
        # 正常情况下差异只可能来自 epochs
        assert payload["differences"] == {"epochs": {"original": 10,
                                                     "requested": 2}}
        assert payload["links"]["formal_runs"].endswith("/formal-runs")
        # never a 202 that pretends to be a finished result
        assert payload["status"] != "completed"

        train_dir = tmp_path / "detect" / "train1"
        args = json.loads(json.dumps(_read_yaml(train_dir / "args.yaml")))
        # 正式训练继承 HPO 的全部固定条件，只把 epochs 换成本次的正式轮数
        assert args["epochs"] == 2 and args["batch"] == 4
        assert args["imgsz"] == 64 and args["device"] == "cpu"
        # six search params + seed + bindings from the authoritative record
        for key in _SEARCH_KEYS:
            assert args[key] == source.top.candidate_params[key], key
        assert args["seed"] == study.config.seed
        assert args["model"] == study.model_binding.model_path
        assert args["data"] == os.path.abspath(study.snapshot_binding.data_yaml_path)
        assert "best.pt" not in json.dumps(args)

        meta = json.loads((train_dir / "hpo_source.json").read_text(encoding="utf-8"))
        assert meta["mode"] == "formal"
        assert meta["study_id"] == source.study_id
        assert meta["trial_id"] == source.top.trial_id
        assert meta["training_config"] == {"epochs": 2, "batch": 4, "imgsz": 64,
                                           "device": "cpu"}
        assert meta["original_conditions"] == {"epochs": 10, "batch": 4,
                                              "imgsz": 64, "device": "cpu"}
        assert meta["differences"] == {"epochs": {"original": 10, "requested": 2}}
        assert meta["search_params"]["optimizer"] == source.top.candidate_params["optimizer"]

    # the command mirrors the persisted args exactly (built once)
    assert launched, "formal training must start one subprocess"
    cmd = launched[0]
    assert cmd[0] == "yolo" and cmd[1] == "train"
    for key in ("epochs=2", "batch=4", "imgsz=64", "device=cpu"):
        assert key in cmd, key
    for key in _SEARCH_KEYS:
        assert f"{key}={source.top.candidate_params[key]}" in cmd, key

    # HPO facts are never written back
    assert (source.root / source.study_id / "study.json").read_bytes() == study_bytes
    assert (source.root / source.study_id / "execution.json").read_bytes() == exec_bytes


def _read_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


# 绘图之外的正式训练固定条件；为修复绘图而改动其中任何一项都是回归。
_FROZEN_FORMAL_CONDITIONS = (
    ("val", True), ("save", True), ("amp", False), ("deterministic", True),
    ("workers", 0), ("resume", False), ("patience", 0),
)


def _assert_other_fixed_conditions(params: dict) -> None:
    for key, expected in _FROZEN_FORMAL_CONDITIONS:
        assert params[key] == expected, key
        assert type(params[key]) is type(expected), key


def test_formal_training_enables_plots_and_leaves_search_params_alone(
        tmp_path, monkeypatch, source):
    """正式训练必须启用 YOLO 标准绘图，且不得靠改写共享搜索固定参数实现。"""
    from auto_tune.modules.hpo.execution_adapter import FIXED_PARAMS
    from auto_tune.modules.run_state.manager import RunManager
    from auto_tune.ui.hpo_training import (
        FormalTrainingConfig,
        resolve_hpo_formal_training,
    )

    verified = resolve_hpo_formal_training(
        source.service, source.runner, RunManager(), source.study_id,
        source.top.trial_id,
        FormalTrainingConfig(epochs=2, batch=4, imgsz=64, device="cpu"))

    assert verified.effective["plots"] is True
    # 搜索阶段（共享常量）仍然关闭绘图：覆盖只作用于本次正式训练的参数副本
    assert FIXED_PARAMS["plots"] is False
    _assert_other_fixed_conditions(verified.effective)


def test_formal_training_persists_plots_true_in_args_and_command(
        tmp_path, monkeypatch, source):
    """落盘 args.yaml、实际命令与收尾读取的参数三处都必须是 plots=True。"""
    from auto_tune.modules.train_analyzer.training_finalizer import _load_params

    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    resp = _post(app_mod, source.study_id, _valid_body(source, epochs=2))
    assert resp.status_code == 202, resp.text
    train_dir = tmp_path / "detect" / "train1"

    args = _read_yaml(train_dir / "args.yaml")
    assert args["plots"] is True
    _assert_other_fixed_conditions(args)

    assert launched, "formal training must start one subprocess"
    cmd = launched[0]
    # 只有一个权威取值：不存在冲突或重复的 plots 参数
    assert [arg for arg in cmd if arg.startswith("plots=")] == ["plots=True"]
    # 收尾阶段的统一历史参数直接读这份 args.yaml
    assert _load_params(str(train_dir))["plots"] is True


def test_train_best_records_source_for_verification_reuse_compat(
        tmp_path, monkeypatch, source):
    """The legacy verification entry stays a separate mode and is never mixed in."""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    meta = json.loads((tmp_path / "detect" / "train1" / "hpo_source.json")
                      .read_text(encoding="utf-8"))
    assert meta["mode"] == "formal"

    # the legacy SSE route still works and still starts a verification run
    with TestClient(app_mod.app) as client:
        resp = client.post("/api/training/start", json={"source_hpo": {
            "study_id": source.study_id, "trial_id": source.top.trial_id}})
    assert resp.status_code == 200
    assert len(launched) == 2
    verify_meta = json.loads((tmp_path / "detect" / "train2" / "hpo_source.json")
                             .read_text(encoding="utf-8"))
    assert verify_meta.get("mode") != "formal"


# ── 409 必须区分来源失效/绑定失效/环境漂移，不一律提示“已有训练” ────


@pytest.mark.parametrize("cause,expected", [
    ("source", "HPO_SOURCE_INVALID"),
    ("binding", "HPO_BINDING_MISMATCH"),
    ("version", "HPO_VERSION_MISMATCH"),
])
def test_train_best_409_messages_distinguish_the_cause(
        tmp_path, monkeypatch, source, cause, expected):
    from auto_tune.modules.hpo import HpoError
    from auto_tune.ui.hpo_training import (
        FormalTrainingConfig,
        hpo_training_error_response,
        resolve_hpo_formal_training,
    )

    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    config = FormalTrainingConfig(epochs=2, batch=4, imgsz=64, device="cpu")
    trial_id = source.top.trial_id
    if cause == "source":
        trial_id = source.study_id + "_t9999"

    if cause == "binding":
        source.model.write_bytes(b"drifted-model")
        resp = _post(app_mod, source.study_id, _valid_body(source))
    elif cause == "version":
        def boom(_sid):
            raise HpoError("HPO_VERSION_MISMATCH", "optuna changed")

        monkeypatch.setattr(source.service, "validate_binding", boom)
        resp = _post(app_mod, source.study_id, _valid_body(source))
    else:
        body = _valid_body(source)
        body["trial_id"] = trial_id
        resp = _post(app_mod, source.study_id, body)

    assert resp.status_code == 409, cause
    payload = resp.json()
    assert payload["error_code"] == expected, cause
    assert payload["error"] and payload["next_action"]
    # the training-slot busy phrasing belongs to RUN_ALREADY_ACTIVE only
    assert "已有训练" not in payload["error"]
    assert set(payload) == {"error_code", "error", "next_action"}
    _assert_zero_side_effects(tmp_path, launched, source, app_mod)

    # the same mapper is exercised directly (defence in depth against a
    # polluted code or an unexpected message reaching a client)
    direct = resolve_direct_error(hpo_training_error_response,
                                 resolve_hpo_formal_training, HpoError,
                                 source, trial_id, config, expected)
    assert direct == expected


def resolve_direct_error(error_mapper, resolver, hpo_error_cls, source, trial_id,
                         config, expected):
    """Run the resolver directly and map its error, returning the stable code."""
    from auto_tune.modules.run_state.manager import RunManager

    try:
        resolver(source.service, source.runner, RunManager(), source.study_id,
                 trial_id, config)
    except hpo_error_cls as exc:
        assert exc.code == expected
        mapped = error_mapper(exc)
        assert mapped.status_code == 409
        body = json.loads(bytes(mapped.body).decode("utf-8"))
        assert body["error_code"] == expected
        assert "已有训练" not in body["error"]
        return body["error_code"]
    return None


# ── 初始化写盘失败：必须释放槽位、零进程、稳定错误 ────────────────


def _deps_with_failure(app_mod, monkeypatch, target):
    """Inject one write failure into the shared submit path.

    The dependencies are looked up by ``_hpo_training_deps`` at request time, so
    patching the app-level helpers (or the serializer used by the metadata step)
    is what actually reaches the route.
    """
    if target == "train_dir":
        def boom(_detect, _train):
            raise OSError("cannot create run dir")

        monkeypatch.setattr(app_mod, "_create_train_dirs", boom)
    elif target == "args":
        import yaml as _yaml

        def boom(*_args, **_kwargs):
            raise OSError("cannot write args.yaml")

        monkeypatch.setattr(_yaml, "dump", boom)
    elif target == "metadata":
        def boom(*_args, **_kwargs):
            raise OSError("cannot write source metadata")

        monkeypatch.setattr(json, "dump", boom)
    else:
        raise AssertionError(target)


@pytest.mark.parametrize("target,code", [
    ("train_dir", "TRAIN_DIR_CREATE_FAILED"),
    ("args", "ARGS_PERSIST_FAILED"),
    ("metadata", "SOURCE_METADATA_PERSIST_FAILED"),
])
def test_train_best_write_failure_releases_slot_zero_start(
        tmp_path, monkeypatch, source, target, code):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _deps_with_failure(app_mod, monkeypatch, target)
    resp = _post(app_mod, source.study_id, _valid_body(source))
    assert resp.status_code == 500, target
    body = resp.json()
    assert body["error_code"] == code, target
    assert body["error"] and body["next_action"]
    assert "OSError" not in resp.text and "cannot write" not in resp.text
    assert launched == []
    assert app_mod._RUN_MANAGER.reservation_owner() is None
    assert app_mod._RUN_MANAGER.active_manual() is None


def test_formal_run_survives_a_client_disconnect(tmp_path, monkeypatch, source):
    """A 202 has no SSE subscription; losing the client never cancels the run."""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    with TestClient(app_mod.app) as client:
        resp = client.post(f"/api/hpo/studies/{source.study_id}/train-best",
                           json=_valid_body(source))
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        assert resp.json()["train_name"] == "train1"
    # the run directory and its source metadata are durable facts on disk
    train_dir = tmp_path / "detect" / "train1"
    assert (train_dir / "args.yaml").is_file()
    assert (train_dir / "hpo_source.json").is_file()
    # the controller is owned by the manager, not by the HTTP request
    assert app_mod._RUN_MANAGER.get(run_id) is not None, \
        "the manual controller must outlive the request"
    assert launched, "the training subprocess was started before the client left"
    # a later read never re-starts or cancels it
    listing = TestClient(app_mod.app).get(
        f"/api/hpo/studies/{source.study_id}/formal-runs")
    assert listing.status_code == 200


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_formal_runs_projects_each_terminal_status(tmp_path, monkeypatch, source,
                                                   status):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    metrics = {"mAP50": 0.8, "mAP50_95": 0.5} if status == "completed" else {}
    _write_history(tmp_path / "log", "train1", status=status, metrics=metrics,
                   default_metrics=(status == "completed"))

    run = TestClient(app_mod.app).get(
        f"/api/hpo/studies/{source.study_id}/formal-runs").json()["runs"][0]
    assert run["status"] == status
    assert run["result_available"] is (status == "completed")
    assert run["metrics"] == metrics
    # a non-completed run is never presented as having a final result
    if status != "completed":
        assert run["best_pt_available"] is False


def test_formal_runs_never_call_a_leftover_weight_a_finished_model(
        tmp_path, monkeypatch, source):
    """文件存在 ≠ 正式训练完成：失败的运行即使留下 best.pt 也不能声称最终模型。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    train_dir = tmp_path / "detect" / "train1"
    (train_dir / "weights").mkdir()
    (train_dir / "weights" / "best.pt").write_bytes(b"leftover-weights")
    _write_history(tmp_path / "log", "train1", status="failed", metrics={},
                   default_metrics=False)

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["status"] == "failed"
    assert run["best_pt_available"] is False
    assert run["result_available"] is False
    assert run["metrics"] == {}


# ── 关联结果：按 study 投影普通 run 事实，服务重启后仍可查 ──────────


def _write_history(log_dir: Path, run_name: str, *, status="completed",
                   metrics=None, epochs=None, default_metrics=True):
    from auto_tune.modules.train_analyzer.experiment_history import (
        ExperimentHistoryStore,
        make_run_id,
    )

    record = {
        "run_id": make_run_id("manual", run_name),
        "run_name": run_name,
        "source": "manual",
        "status": status,
        "analysis_status": "completed",
        "started_at": "2026-09-14T00:00:00Z",
        "finished_at": "2026-09-14T00:10:00Z",
        "params": {},
        "metrics": ({"mAP50": 0.71, "mAP50_95": 0.44}
                    if (metrics is None and default_metrics) else (metrics or {})),
        "epochs": epochs or {"configured": 2, "completed": 2, "best": 2},
        "artifacts": {"report_path": None, "run_dir": "detect/" + run_name},
        "audit_path": None, "analysis_error": None, "history_error": None,
        "index_error": None, "error": None,
    }
    store = ExperimentHistoryStore(str(Path(log_dir) / "experiment_history.json"))
    store.upsert(record)


def test_formal_runs_lists_linked_run_with_final_metrics(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    train_dir = tmp_path / "detect" / "train1"
    (train_dir / "weights").mkdir()
    (train_dir / "weights" / "best.pt").write_bytes(b"weights")
    _write_history(tmp_path / "log", "train1")

    resp = TestClient(app_mod.app).get(
        f"/api/hpo/studies/{source.study_id}/formal-runs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["study_id"] == source.study_id
    assert len(body["runs"]) == 1
    run = body["runs"][0]
    assert run["train_name"] == "train1"
    assert run["status"] == "completed"
    assert run["metrics"]["mAP50"] == 0.71
    assert run["metrics"]["mAP50_95"] == 0.44
    assert run["result_available"] is True
    assert run["best_pt_available"] is True
    assert run["source_trial_id"] == source.top.trial_id
    assert run["training_config"]["epochs"] == 2
    assert run["differences"]["epochs"]["requested"] == 2
    # never a filesystem path
    assert str(tmp_path) not in resp.text
    assert "weights" not in resp.text


def test_formal_runs_survives_restart_and_keeps_all_runs(tmp_path, monkeypatch, source):
    """关联事实必须来自持久化 metadata，而不是内存/DOM。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source, epochs=2))
    _post(app_mod, source.study_id, _valid_body(source, epochs=3))
    names = sorted(p.name for p in (tmp_path / "detect").iterdir())
    assert names == ["train1", "train2"]

    # a brand-new manager (simulating a process restart) still finds both runs
    fresh_manager = type(app_mod._RUN_MANAGER)()
    from auto_tune.ui import hpo_training

    rows = hpo_training.list_formal_runs(
        log_dir=tmp_path / "log", detect_dir=tmp_path / "detect",
        study_id=source.study_id, manager=fresh_manager)
    assert {r["train_name"] for r in rows} == {"train1", "train2"}
    assert all(r["source_readable"] for r in rows)
    # status is honestly unknown without a run fact (never guessed as completed)
    assert {r["status"] for r in rows} == {"unknown"}


def test_formal_runs_does_not_leak_other_studies_runs(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    # a run bound to another study
    other = source.study_id[:-1] + "f" if source.study_id[-1] != "f" else source.study_id[:-1] + "0"
    (tmp_path / "detect" / "train9").mkdir(parents=True)
    (tmp_path / "detect" / "train9" / "hpo_source.json").write_text(
        json.dumps({"mode": "formal", "study_id": other, "trial_id": other + "_t0000"}),
        encoding="utf-8")
    body = TestClient(app_mod.app).get(
        f"/api/hpo/studies/{source.study_id}/formal-runs").json()
    assert {r["train_name"] for r in body["runs"]} == {"train1"}


def test_formal_runs_ignores_verification_and_malformed_metadata(
        tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    for name, content in (("train7", json.dumps({
            "mode": "verification", "study_id": source.study_id})),
            ("train8", "{not json"),
            ("train9", json.dumps({"mode": "formal"}))):
        (tmp_path / "detect" / name).mkdir(parents=True)
        (tmp_path / "detect" / name / "hpo_source.json").write_text(
            content, encoding="utf-8")
    body = TestClient(app_mod.app).get(
        f"/api/hpo/studies/{source.study_id}/formal-runs").json()
    assert {r["train_name"] for r in body["runs"]} == {"train1"}


def test_formal_runs_running_status_from_live_controller(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    controller = app_mod._RUN_MANAGER.active_manual()

    class LiveState:
        status = "running"

    if controller is not None:
        controller.run_state = LiveState()
    from auto_tune.ui import hpo_training

    rows = hpo_training.list_formal_runs(
        log_dir=tmp_path / "log", detect_dir=tmp_path / "detect",
        study_id=source.study_id, manager=app_mod._RUN_MANAGER)
    assert rows and rows[0]["status"] in ("running", "completed", "unknown")
    assert rows[0]["status"] != "failed"


def test_formal_runs_unknown_study_id_never_served(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    resp = TestClient(app_mod.app).get(
        "/api/hpo/studies/" + "hpo_" + "0" * 32 + "/formal-runs")
    assert resp.status_code in (404, 409)
    assert resp.json()["error_code"]


# ── B5: runtime 身份（manual:UUID）与 JSON 历史 ID 必须分开且投影一致 ──


def _formal_runs(app_mod, study_id):
    return TestClient(app_mod.app).get(
        f"/api/hpo/studies/{study_id}/formal-runs").json()


def test_formal_source_metadata_records_the_runtime_identity(
        tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    resp = _post(app_mod, source.study_id, _valid_body(source))
    assert resp.status_code == 202
    accepted = resp.json()

    meta = json.loads((tmp_path / "detect" / "train1" / "hpo_source.json")
                      .read_text(encoding="utf-8"))
    # 启动前写盘的身份：runtime run_id 与 train_name 都是持久化事实
    assert meta["train_name"] == "train1" == accepted["train_name"]
    assert meta["runtime_run_id"] == accepted["run_id"]
    assert meta["runtime_run_id"].startswith("manual:")


def test_formal_runs_project_runtime_and_history_identity_separately(
        tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    accepted = _post(app_mod, source.study_id, _valid_body(source)).json()
    # 收尾后统一历史给这条 run 一个 JSON 历史 ID（manual:train1）
    _write_history(tmp_path / "log", "train1")

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    # 活跃中的投影与启动响应一致（同一 runtime 身份）
    assert run["runtime_run_id"] == accepted["run_id"]
    assert run["history_run_id"] == "manual:train1"
    assert run["runtime_run_id"] != run["history_run_id"]
    assert run["train_name"] == "train1"


def test_formal_runs_without_runtime_field_report_it_missing_not_fabricated(
        tmp_path, monkeypatch, source):
    """旧 metadata 缺少 runtime 字段时明确缺失，绝不补造 UUID。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    train = tmp_path / "detect" / "train9"
    train.mkdir(parents=True)
    (train / "hpo_source.json").write_text(json.dumps({
        "mode": "formal", "study_id": source.study_id,
        "trial_id": source.top.trial_id, "trial_number": source.top.number,
        "training_config": {"epochs": 2, "batch": 4, "imgsz": 96, "device": "cpu"},
    }), encoding="utf-8")

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["train_name"] == "train9"
    assert run["runtime_run_id"] is None
    assert run["runtime_identity_missing"] is True
    assert run["history_run_id"] is None   # no history fact yet


def test_formal_runs_reload_projection_uses_persisted_runtime_identity(
        tmp_path, monkeypatch, source):
    """服务重启（新 manager）后仍能从 metadata 得到同一 runtime 身份。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    accepted = _post(app_mod, source.study_id, _valid_body(source)).json()
    from auto_tune.ui import hpo_training

    rows = hpo_training.list_formal_runs(
        log_dir=tmp_path / "log", detect_dir=tmp_path / "detect",
        study_id=source.study_id, manager=type(app_mod._RUN_MANAGER)())
    assert rows[0]["runtime_run_id"] == accepted["run_id"]
    assert rows[0]["runtime_identity_missing"] is False


# ── C4: 坏元数据/扫描截断必须给出安全 warnings，不能谎报“没有正式历史”──


def test_formal_runs_reports_unreadable_metadata_as_warning(
        tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    for name, content in (("train7", "{not json"),
                          ("train8", json.dumps({"mode": "formal"}))):
        (tmp_path / "detect" / name).mkdir(parents=True)
        (tmp_path / "detect" / name / "hpo_source.json").write_text(
            content, encoding="utf-8")

    body = _formal_runs(app_mod, source.study_id)
    # 严格校验：坏记录不进入 runs
    assert {r["train_name"] for r in body["runs"]} == {"train1"}
    codes = {w["code"] for w in body["warnings"]}
    assert "FORMAL_SOURCE_UNREADABLE" in codes
    assert "FORMAL_SOURCE_INVALID" in codes
    assert body["truncated"] is False
    # warnings 只含稳定码与目录名，绝不含路径或内容
    for warning in body["warnings"]:
        assert set(warning) <= {"code", "train_name"}
        assert str(tmp_path) not in json.dumps(warning)


def test_formal_runs_rejects_metadata_whose_train_identity_differs(
        tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    moved = tmp_path / "detect" / "train5"
    moved.mkdir(parents=True)
    (moved / "hpo_source.json").write_text(json.dumps({
        "mode": "formal", "study_id": source.study_id,
        "trial_id": source.top.trial_id, "train_name": "train1",   # wrong directory
    }), encoding="utf-8")

    body = _formal_runs(app_mod, source.study_id)
    assert body["runs"] == []
    assert any(w["code"] == "FORMAL_SOURCE_INVALID" for w in body["warnings"])


def test_formal_runs_reports_scan_truncation(tmp_path, monkeypatch, source):
    from auto_tune.ui import hpo_training

    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    _post(app_mod, source.study_id, _valid_body(source, epochs=3))
    monkeypatch.setattr(hpo_training, "_MAX_TRAIN_DIRS", 1)

    body = _formal_runs(app_mod, source.study_id)
    assert body["truncated"] is True
    assert any(w["code"] == "FORMAL_RUNS_TRUNCATED" for w in body["warnings"])
    # 被截断时明确告知，而不是把剩余记录当作不存在
    assert len(body["runs"]) <= 1


def test_formal_runs_keeps_valid_rows_when_a_sibling_is_corrupt(
        tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    _write_history(tmp_path / "log", "train1")
    broken = tmp_path / "detect" / "train6"
    broken.mkdir(parents=True)
    (broken / "hpo_source.json").write_bytes(b"\xff\xfe\x00bad")

    body = _formal_runs(app_mod, source.study_id)
    assert len(body["runs"]) == 1
    assert body["runs"][0]["train_name"] == "train1"
    assert body["runs"][0]["result_available"] is True
    assert body["warnings"]


# ── “查看结果”的权威实验详情身份 ─────────────────────────────────────
#
# 回归的正是独立初审复现的 404：前端把 JSON 历史 ID（``manual:trainN``）传给以
# 实验索引主键（``manual:<uuid4>``）查询的详情接口。这里用**真实** LocalIndexService
# + 真实 JSON 历史 + 真实 hpo_source.json 建立反例，并通过真实详情 API 验证返回
# 的是对应训练，而不只是断言按钮文案。


def test_result_entry_resolves_the_index_identity_when_runtime_is_missing(
        tmp_path, monkeypatch, source):
    """旧 metadata 缺 runtime：从权威索引解析已有实验身份，且不伪造 UUID。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    train_dir = tmp_path / "detect" / "train1"
    _write_history(tmp_path / "log", "train1")
    _strip_runtime_identity(train_dir)

    runtime_id = _index_experiment(
        app_mod, tmp_path, "train1",
        runtime_run_id="manual:" + str(uuid.uuid4()))

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    # runtime 与 history 仍是两个不同概念，且旧记录绝不补造 UUID
    assert run["runtime_run_id"] is None
    assert run["runtime_identity_missing"] is True
    assert run["history_run_id"] == "manual:train1"
    # 权威详情身份来自实验索引
    assert run["experiment_run_id"] == runtime_id
    assert run["experiment_identity_reason"] is None
    assert runtime_id.startswith("manual:")

    client = TestClient(app_mod.app)
    # 旧的错误做法（把 JSON 历史 ID 传给详情接口）确实 404 —— 这是真实复现
    assert client.get(f"/api/experiments/{run['history_run_id']}").status_code == 404
    # 权威详情身份返回的就是这条正式训练
    detail = client.get(f"/api/experiments/{run['experiment_run_id']}")
    assert detail.status_code == 200
    assert detail.json()["run_name"] == "train1"
    assert detail.json()["source"] == "manual"


def test_result_entry_uses_the_runtime_identity_when_it_is_present(
        tmp_path, monkeypatch, source):
    """metadata 里有 runtime 时直接用它，并且必须与受控 train 目录一致。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    train_dir = tmp_path / "detect" / "train1"
    _write_history(tmp_path / "log", "train1")
    meta = json.loads((train_dir / "hpo_source.json").read_text(encoding="utf-8"))
    runtime_id = meta["runtime_run_id"]
    assert runtime_id.startswith("manual:")
    _index_experiment(app_mod, tmp_path, "train1", runtime_run_id=runtime_id)

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["runtime_run_id"] == runtime_id
    assert run["experiment_run_id"] == runtime_id
    assert run["experiment_identity_reason"] is None


def test_result_entry_refuses_a_lookalike_run_in_another_directory(
        tmp_path, monkeypatch, source):
    """同名但注册在别的训练目录的记录不算匹配（不跨目录串用）。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    _write_history(tmp_path / "log", "train1")
    _strip_runtime_identity(tmp_path / "detect" / "train1")
    other = tmp_path / "detect" / "train9"
    other.mkdir(parents=True)
    _index_experiment(app_mod, tmp_path, "train1",
                      runtime_run_id="manual:" + str(uuid.uuid4()),
                      run_dir=str(other))

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["experiment_run_id"] is None
    assert run["experiment_identity_reason"] == "EXPERIMENT_NOT_INDEXED"


def test_result_entry_accepts_the_legacy_relative_run_dir(
        tmp_path, monkeypatch, source):
    """历史记录里的相对 run_dir：唯一允许的形式是 detect/trainN（或裸 trainN）。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    _write_history(tmp_path / "log", "train1")
    _strip_runtime_identity(tmp_path / "detect" / "train1")
    runtime_id = _index_experiment(
        app_mod, tmp_path, "train1",
        runtime_run_id="manual:" + str(uuid.uuid4()), run_dir="detect/train1")

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["experiment_run_id"] == runtime_id
    assert run["experiment_identity_reason"] is None


def test_result_entry_rejects_a_relative_run_dir_in_another_directory(
        tmp_path, monkeypatch, source):
    """other/trainN 不是受控 detect 目录，不得当作匹配。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    _write_history(tmp_path / "log", "train1")
    _strip_runtime_identity(tmp_path / "detect" / "train1")
    _index_experiment(app_mod, tmp_path, "train1",
                      runtime_run_id="manual:" + str(uuid.uuid4()),
                      run_dir="other/train1")

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["experiment_run_id"] is None
    assert run["experiment_identity_reason"] == "EXPERIMENT_NOT_INDEXED"


def test_result_entry_is_unavailable_when_not_indexed(
        tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    _write_history(tmp_path / "log", "train1")
    _strip_runtime_identity(tmp_path / "detect" / "train1")

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["history_run_id"] == "manual:train1"   # JSON 历史仍在
    assert run["experiment_run_id"] is None           # 但没有权威详情身份
    assert run["experiment_identity_reason"] == "EXPERIMENT_NOT_INDEXED"


def test_result_entry_is_unavailable_when_the_index_is_ambiguous(
        tmp_path, monkeypatch, source):
    """同名多条记录必须明确不可用，绝不随意取一条。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    _write_history(tmp_path / "log", "train1")
    _strip_runtime_identity(tmp_path / "detect" / "train1")
    _index_experiment(app_mod, tmp_path, "train1",
                      runtime_run_id="manual:" + str(uuid.uuid4()))
    _index_experiment(app_mod, tmp_path, "train1",
                      runtime_run_id="manual:" + str(uuid.uuid4()))

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["experiment_run_id"] is None
    assert run["experiment_identity_reason"] == "EXPERIMENT_AMBIGUOUS"


def test_result_entry_reports_an_unavailable_index(tmp_path, monkeypatch, source):
    """索引不可用是明确原因，不是一个必然 404 的按钮。"""
    from auto_tune.ui import hpo_training

    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    _write_history(tmp_path / "log", "train1")
    monkeypatch.setattr(app_mod, "_local_index_service", lambda: None)

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["experiment_run_id"] is None
    assert run["experiment_identity_reason"] == "LOCAL_INDEX_UNAVAILABLE"
    # 投影层直接调用同样诚实（不依赖 HTTP 层）
    rows = hpo_training.list_formal_runs(
        log_dir=tmp_path / "log", detect_dir=tmp_path / "detect",
        study_id=source.study_id, index=None)
    assert rows[0]["experiment_identity_reason"] == "LOCAL_INDEX_UNAVAILABLE"


def test_result_entry_is_read_only_for_old_facts(tmp_path, monkeypatch, source):
    """解析详情身份不得修改旧来源文件、HPO 研究、预算或排名。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    train_dir = tmp_path / "detect" / "train1"
    _write_history(tmp_path / "log", "train1")
    _strip_runtime_identity(train_dir)
    _index_experiment(app_mod, tmp_path, "train1",
                      runtime_run_id="manual:" + str(uuid.uuid4()))

    study_path = source.root / source.study_id / "study.json"
    exec_path = source.root / source.study_id / "execution.json"
    source_path = train_dir / "hpo_source.json"
    before = (study_path.read_bytes(), exec_path.read_bytes(),
              source_path.read_bytes())

    body = _formal_runs(app_mod, source.study_id)
    assert body["runs"][0]["experiment_run_id"] is not None

    assert (study_path.read_bytes(), exec_path.read_bytes(),
            source_path.read_bytes()) == before


# ── 第三轮返修：已声明的 runtime 身份失配时绝不静默改绑 ──────────────
#
# 独立复现的阻断问题：metadata 声明了 ``runtime_run_id``，但该 ID 不在索引里（或它
# 指向别的目录）时，解析继续按 run_name 回退，于是“查看结果”被**静默改绑**到同名的
# 另一条实验。以下反例用真实 LocalIndexService + 真实 hpo_source.json 锁定正确行为：
# 声明存在即身份，失配即不可用；只有字段真正缺失/为空才允许按名称回退。

_DECLARED_RUNTIME = "manual:11111111-1111-4111-8111-111111111111"
_OTHER_RUNTIME = "manual:22222222-2222-4222-8222-222222222222"


def _set_runtime_identity(train_dir: Path, value) -> None:
    """把 hpo_source.json 的 runtime_run_id 改写成指定值（metadata 事实夹具）。"""
    path = train_dir / "hpo_source.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["runtime_run_id"] = value
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _formal_source(train_dir: Path) -> Path:
    return Path(train_dir) / "hpo_source.json"


def test_declared_runtime_missing_from_index_never_rebinds_to_a_sibling(
        tmp_path, monkeypatch, source):
    """反例 1：声明的 runtime ID 不在索引里，另有同名同目录实验 → 必须不可用。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    train_dir = tmp_path / "detect" / "train1"
    _set_runtime_identity(train_dir, _DECLARED_RUNTIME)
    sibling = _index_experiment(app_mod, tmp_path, "train1",
                                runtime_run_id=_OTHER_RUNTIME)
    assert sibling == _OTHER_RUNTIME

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["runtime_run_id"] == _DECLARED_RUNTIME
    assert run["experiment_run_id"] is None
    assert run["experiment_run_id"] != sibling          # 绝不改绑到另一条实验
    assert run["experiment_identity_reason"] == "EXPERIMENT_NOT_INDEXED"

    # 被拒绝改绑的那条实验本身仍然可查：它只是不能冒充本次正式训练
    detail = TestClient(app_mod.app).get(f"/api/experiments/{sibling}")
    assert detail.status_code == 200
    assert detail.json()["run_name"] == "train1"


def test_declared_runtime_in_another_directory_does_not_fall_back(tmp_path, monkeypatch,
                                                                 source):
    """反例 2：声明的 runtime 身份记录在别的目录，正确目录另有同名实验 → 不可用。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    train_dir = tmp_path / "detect" / "train1"
    _set_runtime_identity(train_dir, _DECLARED_RUNTIME)
    other_dir = tmp_path / "detect" / "train9"
    other_dir.mkdir(parents=True)
    # 声明的 runtime 身份确实存在，但它登记在另一个训练目录 → 不是本次运行
    _index_experiment(app_mod, tmp_path, "train1",
                      runtime_run_id=_DECLARED_RUNTIME, run_dir=str(other_dir))
    sibling = _index_experiment(app_mod, tmp_path, "train1",
                                runtime_run_id=_OTHER_RUNTIME)

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["experiment_run_id"] is None
    assert run["experiment_run_id"] != sibling
    assert run["experiment_identity_reason"] == "EXPERIMENT_NOT_INDEXED"

    detail = TestClient(app_mod.app).get(f"/api/experiments/{sibling}")
    assert detail.status_code == 200


@pytest.mark.parametrize("declared", [
    "manual:train1",                            # 非空但格式非法（JSON 历史 ID）
    "11111111-1111-4111-8111-111111111111",     # 缺少 kind 前缀
    "manual:11111111-1111-4111-8111",           # 不是 UUID4 形态
    123,                                        # 非字符串
    {"run_id": _OTHER_RUNTIME},                 # 非字符串（对象）
])
def test_illegal_declared_runtime_never_falls_back_to_a_name_match(
        tmp_path, monkeypatch, source, declared):
    """反例 3：声明了非空但非法的 runtime 值 → 不得按名称回退、不得补造身份。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    train_dir = tmp_path / "detect" / "train1"
    _set_runtime_identity(train_dir, declared)
    unique = _index_experiment(app_mod, tmp_path, "train1",
                               runtime_run_id=_OTHER_RUNTIME)

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["experiment_run_id"] is None
    assert run["experiment_run_id"] != unique
    assert run["experiment_identity_reason"] == "EXPERIMENT_NOT_INDEXED"


def test_declared_runtime_exact_hit_wins_over_lookalike_records(
        tmp_path, monkeypatch, source):
    """反例 4（正例）：声明身份精确命中同名同目录 → 继续返回该 runtime ID。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    accepted = _post(app_mod, source.study_id, _valid_body(source)).json()
    runtime_id = accepted["run_id"]
    exact = _index_experiment(app_mod, tmp_path, "train1", runtime_run_id=runtime_id)
    assert exact == runtime_id
    other_dir = tmp_path / "detect" / "train9"
    other_dir.mkdir(parents=True)
    _index_experiment(app_mod, tmp_path, "train1",
                      runtime_run_id=_OTHER_RUNTIME, run_dir=str(other_dir))

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["runtime_run_id"] == runtime_id
    assert run["experiment_run_id"] == runtime_id
    assert run["experiment_identity_reason"] is None


def test_rejected_rebind_stays_read_only(tmp_path, monkeypatch, source):
    """反例 8：拒绝改绑的整条路径不得写回元数据、研究、执行审计或预算。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    _post(app_mod, source.study_id, _valid_body(source))
    train_dir = tmp_path / "detect" / "train1"
    _set_runtime_identity(train_dir, _DECLARED_RUNTIME)
    _index_experiment(app_mod, tmp_path, "train1", runtime_run_id=_OTHER_RUNTIME)

    study_path = source.root / source.study_id / "study.json"
    exec_path = source.root / source.study_id / "execution.json"
    source_path = _formal_source(train_dir)
    before = (study_path.read_bytes(), exec_path.read_bytes(),
              source_path.read_bytes())
    before_study = source.service.load_study(source.study_id)

    body = _formal_runs(app_mod, source.study_id)
    assert body["runs"][0]["experiment_run_id"] is None
    assert body["runs"][0]["experiment_identity_reason"] == "EXPERIMENT_NOT_INDEXED"

    assert (study_path.read_bytes(), exec_path.read_bytes(),
            source_path.read_bytes()) == before
    # 研究本体（预算/试验/排名/revision）逐字段不变，也没有被重新写盘
    assert source.service.load_study(source.study_id).model_dump() == \
        before_study.model_dump()


# ── 第四轮：监控身份只认严格合法的 runtime ID ────────────────────────
#
# 详情身份与监控身份是两个概念。metadata 里写了 ``manual:train1``（JSON 历史 ID）
# 或其他非法值时：它仍是一份“已声明但非法”的**详情**线索（→ EXPERIMENT_NOT_INDEXED，
# 禁止按名称改绑），但绝不能当作**运行身份**交给 ``/api/runs/{run_id}/stream``。
# 只有严格 ``<kind>:<uuid4>`` 才可投影；“查看监控”回退到活动控制器时同样受限。

class _LiveManualStub:
    """Minimal active manual controller stand-in for the monitor-identity fallback."""

    run_kind = "manual"

    def __init__(self, train_name: str, run_id):
        self.train_name = train_name
        self.run_id = run_id
        self.run_state = SimpleNamespace(run_id=run_id, status="running")

    def is_active(self):
        return True

    def is_done(self):
        return False


def _write_formal_source(train_dir: Path, source, **overrides) -> None:
    """Write a strictly valid formal ``hpo_source.json`` by hand (fact fixture)."""
    train_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "mode": "formal", "study_id": source.study_id,
        "trial_id": source.top.trial_id, "trial_number": source.top.number,
        "training_config": {"epochs": 2, "batch": 4, "imgsz": 96, "device": "cpu"},
    }
    meta.update(overrides)
    (train_dir / "hpo_source.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_controller_live(monkeypatch, app_mod, run_id):
    """Make the real (already finished) controller look in-flight again.

    A finished manual controller is ``retain()``ed out of the active registry, so
    an in-flight run has to be simulated by re-registering it.
    """
    controller = app_mod._RUN_MANAGER.get(run_id)
    assert controller is not None
    monkeypatch.setattr(controller, "is_active", lambda: True)
    app_mod._RUN_MANAGER.register(controller)


def test_illegal_metadata_runtime_is_never_projected_as_a_monitor_identity(
        tmp_path, monkeypatch, source):
    """反例 A：终态 metadata 的 runtime 是 JSON 历史 ID → 监控身份必须为 None。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    accepted = _post(app_mod, source.study_id, _valid_body(source)).json()
    train_dir = tmp_path / "detect" / "train1"
    _set_runtime_identity(train_dir, "manual:train1")   # JSON 历史 ID 被误存进 runtime 字段
    _write_history(tmp_path / "log", "train1")
    app_mod._RUN_MANAGER.unregister(accepted["run_id"])  # 终态/服务重载：无活动控制器

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["runtime_run_id"] is None              # 绝不把 JSON 历史 ID 当运行身份
    assert run["runtime_identity_missing"] is True
    assert run["experiment_run_id"] is None
    assert run["experiment_identity_reason"] == "EXPERIMENT_NOT_INDEXED"
    assert run["history_run_id"] == "manual:train1"   # JSON 历史事实单独保留


def test_illegal_metadata_runtime_falls_back_to_the_active_controller_for_monitoring(
        tmp_path, monkeypatch, source):
    """反例 C：metadata 非法但活动控制器持有合法身份 → 监控用真实控制器 ID。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    accepted = _post(app_mod, source.study_id, _valid_body(source)).json()
    train_dir = tmp_path / "detect" / "train1"
    _set_runtime_identity(train_dir, "manual:train1")
    _write_history(tmp_path / "log", "train1")
    _make_controller_live(monkeypatch, app_mod, accepted["run_id"])
    # 诱饵：索引里有同名同目录实验，绝不是本次正式训练的详情身份
    sibling = _index_experiment(app_mod, tmp_path, "train1",
                                runtime_run_id=_OTHER_RUNTIME)

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["runtime_run_id"] == accepted["run_id"]
    assert run["runtime_run_id"] != "manual:train1"
    assert run["runtime_run_id"] != sibling
    assert RUNTIME_RUN_ID_RE.fullmatch(run["runtime_run_id"])
    assert run["runtime_identity_missing"] is False
    # 控制器回退只解决监控身份：非法声明仍不得按名称改绑
    assert run["experiment_run_id"] is None
    assert run["experiment_identity_reason"] == "EXPERIMENT_NOT_INDEXED"


def test_illegal_live_controller_identity_is_not_projected(tmp_path, monkeypatch, source):
    """反例 D：活动控制器的 run_id 也非法 → 监控身份仍为 None。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    train_dir = tmp_path / "detect" / "train1"
    _write_formal_source(train_dir, source, runtime_run_id="manual:train1",
                         train_name="train1")
    stub = _LiveManualStub("train1", "manual:train1")
    app_mod._RUN_MANAGER.register(stub)
    try:
        run = _formal_runs(app_mod, source.study_id)["runs"][0]
        assert run["runtime_run_id"] is None
        assert run["runtime_identity_missing"] is True
        assert run["experiment_run_id"] is None
        assert run["experiment_identity_reason"] == "EXPERIMENT_NOT_INDEXED"
    finally:
        app_mod._RUN_MANAGER.unregister(stub.run_id)


def test_valid_metadata_runtime_still_drives_monitoring_and_details(
        tmp_path, monkeypatch, source):
    """反例 E：合法 metadata runtime 保持不变（监控与详情都用它）。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    accepted = _post(app_mod, source.study_id, _valid_body(source)).json()
    runtime_id = accepted["run_id"]
    _index_experiment(app_mod, tmp_path, "train1", runtime_run_id=runtime_id)

    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["runtime_run_id"] == runtime_id
    assert run["runtime_identity_missing"] is False
    assert run["experiment_run_id"] == runtime_id
    assert run["experiment_identity_reason"] is None


def test_missing_metadata_runtime_uses_the_active_controller_for_monitoring(
        tmp_path, monkeypatch, source):
    """反例 F：旧 metadata 真正缺 runtime，活动控制器身份仍可用于监控。"""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    accepted = _post(app_mod, source.study_id, _valid_body(source)).json()
    train_dir = tmp_path / "detect" / "train1"
    _strip_runtime_identity(train_dir)
    _write_history(tmp_path / "log", "train1")
    _make_controller_live(monkeypatch, app_mod, accepted["run_id"])

    # 旧记录的详情回退行为不受影响（唯一同名同目录记录仍可解析）
    indexed = _index_experiment(app_mod, tmp_path, "train1",
                                runtime_run_id=_OTHER_RUNTIME)
    run = _formal_runs(app_mod, source.study_id)["runs"][0]
    assert run["runtime_run_id"] == accepted["run_id"]
    assert run["runtime_identity_missing"] is False
    assert run["experiment_run_id"] == indexed
    assert run["experiment_identity_reason"] is None
