"""H1.3 追加返修 Task 1/4: read-only projections, snapshot selection, field errors.

The HPO source is a real HpoService over a tmp_path storage root; the execution
runner is a fake that never starts YOLO and never calls a network LLM. The
front-end assertions are behavioural (which element/state the script drives),
not bare string-existence checks.
"""

import json
import re
import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import (
    Evidence,
    HpoError,
    HpoService,
)
from auto_tune.modules.run_state.manager import RunManager
from auto_tune.ui.hpo_api import create_hpo_router

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"
_SCRIPT = (_UI_DIR / "static" / "hpo.js").read_text(encoding="utf-8")
_TEMPLATE = (_UI_DIR / "templates" / "single_page.html").read_text(encoding="utf-8")


def _rid():
    return uuid.uuid4().hex


def _evidence(epoch=1):
    return Evidence(run_id=f"run-{_rid()}", artifact_relpath="results.csv",
                    artifact_sha256="0" * 64, epoch=epoch)


def _make_source(tmp_path, name):
    source = tmp_path / name
    source.mkdir()
    for n in range(4):
        Image.new("RGB", (16, 16)).save(source / f"{n}.jpg")
        (source / f"{n}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return source


def _make_inputs(tmp_path):
    snapshot = create_dataset_snapshot(_make_source(tmp_path, "source"),
                                       tmp_path / "snapshots",
                                       val_ratio=0.5, seed=42,
                                       class_names={0: "part"})
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"hpo-ui-rework-test-not-a-real-model")
    return snapshot, model


class FakeExec:
    def __init__(self, study_id, config):
        self.study_id = study_id
        self.config = config
        self.status = "READY"
        self.revision = 0
        self.stop_reason = None
        self.attempts = []

    def set(self, status, stop_reason=None):
        self.status = status
        self.stop_reason = stop_reason
        self.revision += 1


class FakeRunner:
    def __init__(self):
        self._execs = {}
        self._lock = threading.Lock()
        self.run_calls = 0

    def _get(self, study_id):
        record = self._execs.get(study_id)
        if record is None:
            raise HpoError("HPO_NOT_FOUND", f"study {study_id} has no execution")
        return record

    def prepare(self, study_id, config):
        with self._lock:
            existing = self._execs.get(study_id)
            if existing is not None:
                return existing
            record = FakeExec(study_id, config)
            self._execs[study_id] = record
            return record

    def status(self, study_id):
        return self._get(study_id)

    def run(self, study_id, *, stop_event=None):
        with self._lock:
            self.run_calls += 1


class Stack:
    def __init__(self, tmp_path, published_root=None, list_models=None):
        self.snapshot, self.model = _make_inputs(tmp_path)
        self.root = tmp_path / "storage"
        self.service = HpoService(self.root)
        self.runner = FakeRunner()
        self.manager = RunManager()
        self.published_root = published_root or (tmp_path / "snapshots")
        self._list_models = list_models
        self.client = self._client()

    def _list_snapshots(self):
        from auto_tune.ui.app import list_published_snapshots

        return list_published_snapshots(self.published_root)

    def _client(self):
        snapshot = self.snapshot
        model = self.model

        def resolve_snapshot(snapshot_id):
            if snapshot_id == snapshot.snapshot_id:
                return Path(snapshot.snapshot_path)
            raise HpoError("HPO_INVALID_CONFIG", "快照不存在或已失效")

        def validate_model(value):
            path = Path(value)
            if path.is_file() and path.suffix == ".pt":
                return str(path.resolve())
            raise HpoError("HPO_INVALID_CONFIG", "模型文件不存在或不受支持")

        router = create_hpo_router(
            service=self.service, runner=self.runner, manager=self.manager,
            resolve_snapshot=resolve_snapshot, validate_model=validate_model,
            assert_training_slot_free=lambda: None,
            list_snapshots=self._list_snapshots,
            list_models=self._list_models)
        app = FastAPI()
        app.include_router(router, prefix="/api/hpo")
        return TestClient(app)

    def create(self, **overrides):
        payload = {
            "snapshot_id": self.snapshot.snapshot_id,
            "model_path": str(self.model),
            "study_config": {},
            "execution_config": {},
        }
        payload.update(overrides)
        # raw content (allow_nan) so strict numeric cases like NaN reach the
        # server instead of being rejected by the client-side JSON encoder
        return self.client.post("/api/hpo/studies",
                                content=json.dumps(payload),
                                headers={"Content-Type": "application/json"})

    def success_trials(self, study_id, values=(0.5, 0.9)):
        for value in values:
            trial = self.service.ask(study_id, request_id=_rid())
            self.service.tell(study_id, trial.number,
                              _evidence_result(value))
        return self.service.load_study(study_id).trials


def _evidence_result(value):
    from auto_tune.modules.hpo import ResultInput

    return ResultInput(state="SUCCESS", value=value, evidence=_evidence(1))


@pytest.fixture
def stack(tmp_path):
    return Stack(tmp_path)


# ── Task 1: controlled snapshot selection (no latest fallback) ─────


def test_snapshot_selector_lists_controlled_root_without_latest_binding(
        stack, tmp_path):
    # a second, older legal snapshot that is NOT bound as latest_dataset
    other = create_dataset_snapshot(_make_source(tmp_path, "other"),
                                    tmp_path / "snapshots",
                                    val_ratio=0.5, seed=7,
                                    class_names={0: "part"})
    assert not (tmp_path / "log").exists()  # no latest_dataset registration

    resp = stack.client.get("/api/hpo/snapshots")
    assert resp.status_code == 200
    body = resp.json()
    by_id = {row["snapshot_id"]: row for row in body["snapshots"]}
    assert {stack.snapshot.snapshot_id, other.snapshot_id} <= set(by_id)

    row = by_id[other.snapshot_id]
    assert row["selectable"] is True and row["readable"] is True
    assert row["short_id"] == other.snapshot_id[:8]
    assert row["dataset_name"] == "other"
    assert row["created_at"]
    # 合法样本总数 = train + val；background 是 train/val 的子集，不能相加
    assert row["image_count"] == other.train_count + other.val_count
    assert row["train_count"] == other.train_count and row["val_count"] == other.val_count
    assert row["background_count"] == other.background_count
    assert row["background_count"] <= row["image_count"]
    # the full id is available for the detail view, and no absolute path leaks
    assert other.snapshot_id in resp.text
    assert str(tmp_path) not in resp.text


def test_snapshot_selector_counts_background_as_subset_not_added(tmp_path):
    """00915b4c 口径回归：184+46=230，背景 99 含在 230 内，不得显示 329。"""
    source = tmp_path / "bg_source"
    source.mkdir()
    total, background = 230, 99
    for n in range(total):
        Image.new("RGB", (16, 16)).save(source / f"img{n}.jpg")
        if n >= background:
            (source / f"img{n}.txt").write_text("0 0.5 0.5 0.2 0.2\n",
                                                 encoding="utf-8")
    snapshot = create_dataset_snapshot(source, tmp_path / "snapshots",
                                       val_ratio=0.2, seed=42,
                                       class_names={0: "part"})
    assert snapshot.train_count + snapshot.val_count == total
    assert snapshot.background_count == background

    stack = Stack(tmp_path, published_root=tmp_path / "snapshots")
    row = {r["snapshot_id"]: r for r in
           stack.client.get("/api/hpo/snapshots").json()["snapshots"]}[
               snapshot.snapshot_id]
    assert row["readable"] is True and row["selectable"] is True
    assert row["image_count"] == total          # 230, never 329
    assert row["train_count"] == snapshot.train_count
    assert row["val_count"] == snapshot.val_count
    assert row["background_count"] == background  # subset, still shown
    assert row["image_count"] != total + background


def test_snapshot_selector_rejects_inconsistent_count_contract(tmp_path):
    """计数契约（samples == train+val，background <= total）不成立时不可选。"""
    bad = tmp_path / "snapshots" / ("9" * 64)
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text(json.dumps({
        "snapshot_id": "9" * 64,
        "source_root": str(tmp_path / "src"),
        "created_at": "2026-09-14T00:00:00Z",
        "train_count": 10, "val_count": 2, "background_count": 1,
        "samples": [{} for _ in range(5)],   # 5 != 12 → broken contract
    }), encoding="utf-8")

    stack = Stack(tmp_path, published_root=tmp_path / "snapshots")
    row = {r["snapshot_id"]: r for r in
           stack.client.get("/api/hpo/snapshots").json()["snapshots"]}[
               "9" * 64]
    # visible but never presented as usable
    assert row["readable"] is False and row["selectable"] is False
    assert row["image_count"] is None


def test_snapshot_selector_marks_corrupt_snapshot_visible_not_selectable(
        stack, tmp_path):
    broken_id = "f" * 64
    broken = tmp_path / "snapshots" / broken_id
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")

    body = stack.client.get("/api/hpo/snapshots").json()
    by_id = {row["snapshot_id"]: row for row in body["snapshots"]}
    assert broken_id in by_id  # never silently dropped
    assert by_id[broken_id]["readable"] is False
    assert by_id[broken_id]["selectable"] is False
    assert by_id[broken_id]["dataset_name"] is None


def test_snapshot_selector_ignores_non_snapshot_entries(stack, tmp_path):
    junk = tmp_path / "snapshots" / "not-a-snapshot"
    junk.mkdir()
    (junk / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "snapshots" / "loose.txt").write_text("x", encoding="utf-8")

    ids = [row["snapshot_id"] for row in
           stack.client.get("/api/hpo/snapshots").json()["snapshots"]]
    assert "not-a-snapshot" not in ids
    assert stack.snapshot.snapshot_id in ids


def test_snapshot_selector_read_only_and_missing_root(stack, tmp_path):
    before = sorted(p.name for p in (tmp_path / "snapshots").iterdir())
    stack.client.get("/api/hpo/snapshots")
    assert sorted(p.name for p in (tmp_path / "snapshots").iterdir()) == before
    # a missing published root is an empty list, never an exception
    (tmp_path / "empty").mkdir()
    empty = Stack(tmp_path / "empty", published_root=tmp_path / "does-not-exist")
    assert empty.client.get("/api/hpo/snapshots").json()["snapshots"] == []


# ── Task 1: safe field-level errors (no raw Pydantic / paths) ─────


_RAW_LEAK_MARKERS = (
    "Input should", "value_error", "Traceback", "pydantic", "Pydantic",
    "validation error", "For further information",
)

_FIELD_CASES = [
    ("budget", {"study_config": {"budget": 0}}, "budget", "FIELD_RANGE"),
    ("budget bool", {"study_config": {"budget": True}}, "budget", "FIELD_TYPE"),
    ("budget str", {"study_config": {"budget": "10"}}, "budget", "FIELD_TYPE"),
    ("epochs", {"study_config": {"epochs": 0}}, "epochs", "FIELD_RANGE"),
    ("epochs float", {"study_config": {"epochs": 1.5}}, "epochs", "FIELD_TYPE"),
    ("seed", {"study_config": {"seed": -1}}, "seed", "FIELD_RANGE"),
    ("seed nan", {"study_config": {"seed": float("nan")}}, "seed", "FIELD_TYPE"),
    ("sampler", {"study_config": {"sampler": "bogus"}}, "sampler", "FIELD_VALUE"),
    ("batch", {"execution_config": {"batch": 300}}, "batch", "FIELD_RANGE"),
    ("batch bool", {"execution_config": {"batch": True}}, "batch", "FIELD_TYPE"),
    ("imgsz", {"execution_config": {"imgsz": 100}}, "imgsz", "FIELD_MULTIPLE"),
    ("imgsz range", {"execution_config": {"imgsz": 16}}, "imgsz", "FIELD_RANGE"),
    ("imgsz str", {"execution_config": {"imgsz": "640"}}, "imgsz", "FIELD_TYPE"),
    ("device", {"execution_config": {"device": "gpu0"}}, "device", "FIELD_VALUE"),
    ("device int", {"execution_config": {"device": 0}}, "device", "FIELD_TYPE"),
    ("timeout", {"execution_config": {"timeout_seconds": 0}}, "timeout_seconds",
     "FIELD_RANGE"),
    ("snapshot", {"snapshot_id": ""}, "snapshot_id", "FIELD_REQUIRED"),
    ("model", {"model_path": ""}, "model_path", "FIELD_REQUIRED"),
    ("unknown", {"unknown_field": 1}, "unknown_field", "FIELD_UNKNOWN"),
    ("unknown nested", {"study_config": {"nope": 1}}, "nope", "FIELD_UNKNOWN"),
]


@pytest.mark.parametrize("label,payload,field,reason", _FIELD_CASES)
def test_create_field_error_is_safe_and_located(stack, label, payload, field, reason):
    resp = stack.create(**payload)
    assert resp.status_code == 422, label
    body = resp.json()
    assert body["error_code"] == "INVALID_HPO_FIELD"
    assert body["field"] == field, label
    assert body["reason_code"] == reason, label
    assert isinstance(body["error"], str) and body["error"]
    assert body["next_action"]
    # the message must be Chinese guidance, never the underlying exception text
    assert re.search(r"[一-鿿]", body["error"]), label
    for marker in _RAW_LEAK_MARKERS:
        assert marker not in resp.text, label
    assert list(stack.root.glob("hpo_*")) == []


@pytest.mark.parametrize("dirty", [
    r"C:\secret\snapshots", "/etc/secret/data.yaml", "sk-secret-token",
])
def test_create_error_never_echoes_the_offending_value(stack, dirty):
    """A dirty value may be shape-valid but unresolvable; either way it is redacted."""
    resp = stack.create(snapshot_id=dirty)
    body = resp.json()
    assert resp.status_code == 422
    assert body["error_code"] in ("INVALID_HPO_FIELD", "HPO_INVALID_CONFIG")
    assert body["error"] and body["next_action"]
    for marker in _RAW_LEAK_MARKERS:
        assert marker not in resp.text
    assert dirty not in resp.text
    assert list(stack.root.glob("hpo_*")) == []
    assert stack.runner.run_calls == 0


@pytest.mark.parametrize("dirty", ["C:" + "x" * 200, "sk-" + "a" * 200])
def test_create_overlong_snapshot_id_is_field_located_and_redacted(stack, dirty):
    resp = stack.create(snapshot_id=dirty)
    body = resp.json()
    assert resp.status_code == 422
    assert body["error_code"] == "INVALID_HPO_FIELD"
    assert body["field"] == "snapshot_id"
    assert body["reason_code"] == "FIELD_LENGTH"
    assert dirty not in resp.text
    assert list(stack.root.glob("hpo_*")) == []


# ── Task 1: history paging + frozen detail summary ─────────────────


def test_history_second_page_returns_only_the_second_page(stack):
    ids = [stack.create().json()["study_id"] for _ in range(3)]
    page1 = stack.client.get("/api/hpo/studies?offset=0&limit=2").json()
    page2 = stack.client.get("/api/hpo/studies?offset=2&limit=2").json()
    assert page1["count"] == 3 and len(page1["studies"]) == 2
    assert len(page2["studies"]) == 1
    first = {s["study_id"] for s in page1["studies"]}
    assert {s["study_id"] for s in page2["studies"]} == set(ids) - first


def test_status_payload_exposes_frozen_authoritative_summary(stack):
    study_id = stack.create(
        study_config={"sampler": "random", "budget": 3, "epochs": 7, "seed": 11},
        execution_config={"batch": 4, "imgsz": 128, "device": "cpu",
                          "timeout_seconds": 60},
    ).json()["study_id"]
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert body["created_at"]
    assert body["sampler"] == "random"
    assert body["budget"] == 3
    assert body["seed"] == 11
    assert body["study_epochs"] == 7
    assert body["batch"] == 4
    assert body["imgsz"] == 128
    assert body["device"] == "cpu"
    assert body["snapshot_id"] == stack.snapshot.snapshot_id
    assert body["snapshot_short_id"] == stack.snapshot.snapshot_id[:8]
    assert body["model_display"] == "fixture.pt"
    # frozen values are not editable from the client
    stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    after = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert after["study_epochs"] == 7 and after["batch"] == 4


def test_status_payload_describes_search_config_without_llm_content(stack):
    study_id = stack.create().json()["study_id"]
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    space = body["search_space"]
    # 新建研究默认全面模式：搜索配置必须随之给出综合口径，不再是单指标
    assert space["objective"] == "comprehensive_composite_best_epoch_v1"
    assert space["objective_label"]
    assert set(space["parameters"]) == {
        "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs"}
    assert space["parameters"]["optimizer"]["kind"] == "choice"
    assert space["parameters"]["optimizer"]["choices"] == ["SGD", "AdamW"]
    assert space["parameters"]["optimizer"]["conditional"] is True  # momentum link
    assert space["parameters"]["lr0"]["kind"] == "float"
    assert space["parameters"]["lr0"]["low"] == 1e-5
    assert space["parameters"]["warmup_epochs"]["kind"] == "int"
    # fixed conditions are clearly separated from the searched ones
    assert set(space["fixed"]) == {"epochs", "batch", "imgsz", "device", "seed",
                                   "snapshot", "model"}
    assert space["fixed"]["epochs"] == 30
    # never an LLM rationale / suggestion / diagnosis
    for token in ("rationale", "diagnosis", "suggestion", "reason", "llm"):
        assert token not in json.dumps(space).lower()


def test_best_config_keeps_legacy_shape_and_adds_display_fields(stack):
    study_id = stack.create(study_config={"epochs": 10},
                            execution_config={"batch": 4, "imgsz": 64}).json()["study_id"]
    stack.success_trials(study_id, values=(0.5, 0.9))
    payload = stack.client.get(f"/api/hpo/studies/{study_id}/best-config").json()
    # legacy fixed-config verification contract untouched
    assert set(payload["source"]) == {"study_id", "trial_id", "trial_number"}
    assert set(payload["search"]) == {
        "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs"}
    assert payload["fixed"] == {"epochs": 10, "batch": 4, "imgsz": 64, "device": "cpu"}
    assert payload["value"] == 0.9
    assert payload["epoch"] == 1
    # display-only additions
    assert payload["source"]["trial_id"].endswith("_t0001")
    assert payload["approved_for_formal_training"] is False  # still READY
    assert payload["trial_artifacts"]["best_pt_available"] is False
    assert payload["trial_artifacts"]["last_pt_available"] is False


# ── A4/A5/A6/A7: 服务端权威默认绑定（少让用户选，功能留后台） ────────


def _latest_for(snapshot):
    return {
        "source_dataset_path": str(snapshot.source_root),
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_path": str(snapshot.snapshot_path),
        "manifest_path": str(snapshot.manifest_path),
        "data_yaml_path": str(snapshot.data_yaml_path),
        "snapshot_manifest_digest": snapshot.manifest_digest,
        "split": True,
    }


def _defaults(monkeypatch, tmp_path, *, latest=None, config=None,
              snapshots_root=None, cwd=None, cuda=False):
    from auto_tune.ui import app as app_mod

    # 设备默认值取决于本机 GPU 探测：测试固定探测结果，绝不依赖开发机硬件
    monkeypatch.setattr(app_mod, "_cuda_available", lambda: cuda)
    monkeypatch.setattr(app_mod, "APP_CONFIG", config or {})
    monkeypatch.setattr(app_mod, "_read_latest_dataset", lambda: latest)
    monkeypatch.setattr(app_mod, "DATASET_SNAPSHOT_ROOT",
                        Path(snapshots_root or (tmp_path / "snapshots")))
    if cwd is not None:
        monkeypatch.chdir(cwd)
    return app_mod.build_hpo_defaults()


def test_defaults_bind_the_registered_valid_snapshot(tmp_path, monkeypatch):
    snapshot = create_dataset_snapshot(_make_source(tmp_path, "source"),
                                       tmp_path / "snapshots", val_ratio=0.5,
                                       seed=42, class_names={0: "part"})
    payload = _defaults(monkeypatch, tmp_path, latest=_latest_for(snapshot))
    data = payload["dataset"]
    assert data["snapshot"]["snapshot_id"] == snapshot.snapshot_id
    assert data["snapshot"]["dataset_name"] == "source"   # 原数据集名称
    assert data["snapshot"]["source_root"] == str(snapshot.source_root)
    assert data["snapshot"]["created_at"]
    # train/val 为总数；background 是子集，不重复相加
    assert data["snapshot"]["train_count"] == snapshot.train_count
    assert data["snapshot"]["val_count"] == snapshot.val_count
    assert data["snapshot"]["image_count"] == snapshot.train_count + snapshot.val_count
    assert data["snapshot"]["background_count"] == snapshot.background_count
    assert data["needs_user_confirmation"] is False


def test_defaults_never_pick_another_source_roots_newest_snapshot(
        tmp_path, monkeypatch):
    """未登记时只能绑同一原目录的合法快照，不得按全局最新时间挑别的目录。"""
    mine = create_dataset_snapshot(_make_source(tmp_path, "mine"),
                                   tmp_path / "snapshots", val_ratio=0.5,
                                   seed=1, class_names={0: "part"})
    other = create_dataset_snapshot(_make_source(tmp_path, "other"),
                                    tmp_path / "snapshots", val_ratio=0.5,
                                    seed=2, class_names={0: "part"})
    latest = {"source_dataset_path": str(mine.source_root), "dataset_path": str(mine.source_root)}
    payload = _defaults(monkeypatch, tmp_path, latest=latest)
    bound = payload["dataset"]["snapshot"]
    # 同一原目录 → 允许绑定；其他原目录的快照绝不被自动选中
    assert bound["snapshot_id"] == mine.snapshot_id
    assert bound["snapshot_id"] != other.snapshot_id
    assert bound["dataset_name"] == "mine"


def test_defaults_require_confirmation_when_no_reliable_binding(
        tmp_path, monkeypatch):
    other = create_dataset_snapshot(_make_source(tmp_path, "unrelated"),
                                    tmp_path / "snapshots", val_ratio=0.5,
                                    seed=3, class_names={0: "part"})
    payload = _defaults(monkeypatch, tmp_path, latest=None)
    assert payload["dataset"]["snapshot"] is None
    assert payload["dataset"]["needs_user_confirmation"] is True
    assert payload["dataset"]["reason_code"]
    # 有“别人的”快照也绝不自动替代数据
    assert other.snapshot_id not in json.dumps(payload)


def test_defaults_pick_the_legal_local_weight_and_never_a_search_best(
        tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "yolov8n.pt").write_bytes(b"weights")
    (project / "detect" / "train1" / "weights").mkdir(parents=True)
    (project / "detect" / "train1" / "weights" / "best.pt").write_bytes(b"best")
    payload = _defaults(monkeypatch, tmp_path,
                        config={"project": {"model": "yolov8n.pt"}}, cwd=project)
    model = payload["model"]
    assert model["available"] is True
    assert model["path"] == str((project / "yolov8n.pt").resolve())
    assert model["source"] == "project.model"
    assert "train1" not in model["path"]      # 绝不自动选搜索产物 best.pt


def test_defaults_fall_back_to_existing_yolov8n_only_when_unconfigured(
        tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "yolov8n.pt").write_bytes(b"weights")
    payload = _defaults(monkeypatch, tmp_path, config={}, cwd=project)
    assert payload["model"]["source"] == "fallback:yolov8n.pt"
    assert payload["model"]["available"] is True

    # 未配置且本地没有 → 明确要求一次必要选择，绝不猜测
    empty = tmp_path / "empty"
    empty.mkdir()
    payload = _defaults(monkeypatch, tmp_path, config={}, cwd=empty)
    assert payload["model"]["available"] is False
    assert payload["model"]["path"] is None
    assert payload["model"]["reason_code"] == "MODEL_MISSING"


def test_defaults_do_not_silently_replace_an_invalid_configured_model(
        tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "yolov8n.pt").write_bytes(b"weights")   # 本地有可用的回退权重
    payload = _defaults(monkeypatch, tmp_path,
                        config={"project": {"model": "does-not-exist.pt"}},
                        cwd=project)
    model = payload["model"]
    # 已配置但不合法 → 不静默换成 yolov8n.pt，要求用户确认
    assert model["available"] is False
    assert model["reason_code"] == "MODEL_CONFIG_INVALID"
    assert model["path"] is None


def test_defaults_search_conditions_prefer_the_current_training_config(
        tmp_path, monkeypatch):
    payload = _defaults(monkeypatch, tmp_path, config={
        "training": {"batch": 8, "imgsz": 512, "default_epochs": 7}})
    search = payload["search"]
    assert search["sampler"] == "tpe"         # 算法默认 TPE
    assert search["budget"] == 10             # 搜索预算默认 10
    assert search["epochs"] == 30             # 每 Trial 默认 30 轮
    assert search["seed"] == 42
    assert search["timeout_seconds"] == 3600
    assert search["batch"] == 8 and search["imgsz"] == 512
    assert search["device"] == "cpu"


def test_defaults_search_fall_back_only_when_absent(tmp_path, monkeypatch):
    payload = _defaults(monkeypatch, tmp_path, config={})
    assert payload["search"]["batch"] == 16
    assert payload["search"]["imgsz"] == 640
    assert payload["search"]["device"] == "cpu"
    assert payload["formal"] == {"epochs": 100, "batch": 16, "imgsz": 640,
                                 "device": "cpu"}


def test_defaults_never_silently_replace_illegal_configured_values(
        tmp_path, monkeypatch):
    payload = _defaults(monkeypatch, tmp_path,
                        config={"training": {"batch": 300, "imgsz": 100,
                                             "default_epochs": 0}})
    assert payload["search"]["batch"] == 300      # 原值透传，不静默变 16
    assert payload["search"]["imgsz"] == 100
    assert payload["formal"]["epochs"] == 0
    codes = {w["code"] for w in payload["config_warnings"]}
    assert codes == {"CONFIG_VALUE_INVALID"}
    fields = {w["field"] for w in payload["config_warnings"]}
    assert fields == {"batch", "imgsz", "default_epochs"}


def test_defaults_formal_conditions_come_from_ordinary_training_config(
        tmp_path, monkeypatch):
    payload = _defaults(monkeypatch, tmp_path, config={
        "training": {"default_epochs": 120, "batch": 32, "imgsz": 128,
                     "device": "0"}})
    formal = payload["formal"]
    assert formal == {"epochs": 120, "batch": 32, "imgsz": 128, "device": "0"}
    # 绝不沿用验收研究的 epochs1/imgsz64
    assert formal["epochs"] != 1 and formal["imgsz"] != 64


def test_defaults_project_the_probed_device_list(tmp_path, monkeypatch):
    """设备列表只给已探测的 GPU 编号与 CPU，权威默认与探测结果一致。"""
    gpu = _defaults(monkeypatch, tmp_path, cuda=True)
    assert gpu["devices"]["gpus"] and gpu["devices"]["gpus"][0] == 0
    assert gpu["devices"]["default"] == "0"

    cpu = _defaults(monkeypatch, tmp_path, cuda=False)
    assert cpu["devices"]["gpus"] == []
    assert cpu["devices"]["default"] == "cpu"
    assert cpu["device_notice"]["code"] == "GPU_NOT_AVAILABLE_CPU_FALLBACK"

    # 配置里显式指定的 device 优先，并且不在列表里也不会被静默替换
    configured = _defaults(monkeypatch, tmp_path, cuda=True,
                           config={"training": {"device": "2"}})
    assert configured["devices"]["default"] == "2"


def test_defaults_route_is_read_only_and_available(stack, tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod
    from auto_tune.ui.app import app

    client = TestClient(app)
    resp = client.get("/api/hpo/defaults")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"dataset", "model", "search", "formal", "config_warnings",
                         "devices"}


# ── B1: 详情必须投影 approved_for_formal_training（与 best-config 同一规则）──


def test_status_payload_approves_formal_training_only_when_completed_with_success(
        stack):
    study_id = stack.create().json()["study_id"]
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    # READY 且无成功试验 → 不可正式训练（前端不再靠猜）
    assert "approved_for_formal_training" in body
    assert body["approved_for_formal_training"] is False

    stack.success_trials(study_id)
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    # COMPLETED 之前仍有成功试验也不批准
    assert body["approved_for_formal_training"] is False

    stack.runner._execs[study_id].set("COMPLETED")
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert body["approved_for_formal_training"] is True
    # 同一规则：best-config 与 status 绝不互相矛盾
    best = stack.client.get(f"/api/hpo/studies/{study_id}/best-config").json()
    assert best["approved_for_formal_training"] is True


def test_status_payload_disapproves_formal_training_while_search_is_active(stack):
    study_id = stack.create().json()["study_id"]
    stack.success_trials(study_id)
    stack.runner._execs[study_id].set("COMPLETED")

    class LiveHpo:
        run_kind = "hpo"
        run_id = study_id
        error_code = None

        def is_active(self):
            return True

        def is_done(self):
            return False

    controller = LiveHpo()
    stack.manager.register(controller)
    try:
        body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
        assert body["approved_for_formal_training"] is False
        best = stack.client.get(f"/api/hpo/studies/{study_id}/best-config").json()
        assert best["approved_for_formal_training"] is False
    finally:
        stack.manager.unregister(study_id)


# ── Task 4: shared layout / selection consistency (behavioural) ────


def _js():
    return _SCRIPT


def test_device_is_an_explicit_selector_without_a_cpu_staged_default():
    """设备必须用明确选择控件，且 defaults 到达前没有可提交的暂存默认值。"""
    region = _region(_TEMPLATE, "hpoCreateDraft")
    opener = _TEMPLATE[_TEMPLATE.rindex("<select", 0,
                                        _TEMPLATE.index('id="hpoDevice"')):
                      _TEMPLATE.index(">", _TEMPLATE.index('id="hpoDevice"'))]
    assert "hpoDevice" in region
    assert 'value="cpu"' not in opener
    # 选项由服务端探测结果动态填充（GPU 编号 + CPU），模板不预置任何设备值
    assert _TEMPLATE.count('id="hpoDevice"') == 1
    assert 'type="text" id="hpoDevice"' not in _TEMPLATE


def test_new_study_evaluation_selector_offers_only_the_two_product_modes():
    """新建研究下拉框只有全面/快速两项；legacy 仅用于读取与展示旧研究。"""
    region = _region(_TEMPLATE, "hpoEvaluationMode")
    values = re.findall(r'<option value="([^"]*)"', region)
    assert values == ["comprehensive", "quick"]
    assert "legacy_map50_95" not in region
    # 旧版标签仍然存在（旧研究详情/历史必须继续显示它），但只能被读取与展示
    assert "legacy_map50_95" in _js()
    assert "旧版 mAP50-95" in _js()


def test_js_has_selection_generation_guard_for_async_replies():
    script = _js()
    # every detail/best render is guarded by the selection it belongs to
    assert "selection" in script and "generation" in script
    assert re.search(r"function\s+isCurrentSelection|_hpoIsCurrentSelection", script)
    # the guard is consulted by both detail and best render paths
    detail = script.split("function renderStatus", 1)[1].split("\n  }", 1)[0]
    assert "Selection" in detail


def test_js_hpo_mode_has_no_llm_suggestion_content():
    html = _TEMPLATE
    # the HPO content regions never carry LLM suggestion/rationale markup
    hpo_only = re.findall(r'class="[^"]*hpo-only[^"]*"', html)
    assert hpo_only, "HPO needs its own layout region"
    for region in _hpo_regions(html):
        assert "rationale" not in region.lower()
        assert "诊断" not in region
        assert "建议" not in region or "无建议" in region


def test_mode_gate_reveals_hpo_regions_and_hides_empty_cards():
    """The always-on HPO regions are revealed by the mode gate; data-gated
    regions must never pop up empty just because HPO mode was selected."""
    html = _TEMPLATE

    def _opener(element_id):
        start = html.index('id="' + element_id + '"')
        return html[html.rindex("<div", 0, start):html.index(">", start)]

    for element_id in ("hpoCreateDraft", "hpoPanel"):
        opener = _opener(element_id)
        assert "hpo-mode" in opener, element_id
        assert "hpo-only" in opener, element_id
    # data-gated cards are NOT mode-gated: they stay hidden until facts arrive
    for element_id in ("hpoSearchConfig", "hpoBestArea", "hpoProgressCard",
                       "hpoHistorySection"):
        assert "hpo-mode" not in _opener(element_id), element_id
        assert "hpo-only" in _opener(element_id), element_id
    script = _js()
    gate = script.split("var modeEls = document.querySelectorAll('.hpo-mode')", 1)[1] \
        .split("\n", 1)[1]
    assert "classList.toggle('hidden', !isHpo)" in gate


def test_hpo_regions_are_separate_from_llm_regions():
    html = _TEMPLATE
    # llm-only and hpo-only are distinct sibling markers toggled by the same handler
    assert "llm-only" in html and "hpo-only" in html
    script = _js()
    assert "hpo-only" in script and "llm-only" in script


# ── 最终返修 Task 1：HPO 模式只显示 HPO 内容 ───────────────────────


def _mode_opener(element_id):
    html = _TEMPLATE
    start = html.index('id="' + element_id + '"')
    return html[html.rindex("<div", 0, start):html.index(">", start)]


def test_non_hpo_only_regions_carry_an_explicit_non_hpo_contract():
    """训练总结/LLM 分析/视觉分析与逐项调优历史都只属于非 HPO 模式。"""
    for element_id in ("trainingSummaryCard", "llmAnalysisCard",
                       "visionAnalysisCard", "tuningHistoryCard"):
        opener = _mode_opener(element_id)
        assert "non-hpo-only" in opener, element_id
        # 只隐藏、不复制：这些区块在模板里各自唯一
        assert _TEMPLATE.count('id="' + element_id + '"') == 1, element_id
    # HPO 主摘要属于 HPO 模式，必须随模式显示（否则设备提示无处可见）
    assert "hpo-mode" in _mode_opener("hpoMainSummary")
    assert "hpo-only" in _mode_opener("hpoMainSummary")


def test_template_keeps_a_single_mode_select_and_primary_action():
    """第三轮：模式栏与唯一主操作上移到页面顶部的共同控制区，必须排在一切
    LLM/HPO 可变内容之前（加载建议/研究详情/最佳结果都不会推动它的位置）。"""
    html = _TEMPLATE
    assert html.count('id="tuningModeSelect"') == 1
    assert html.count('id="startTuningBtn"') == 1
    assert html.count('id="hpoCreateAndStartBtn"') == 1
    common = html.index('id="tuningCommonControls"')
    mode = html.index('id="tuningModeSelect"')
    # 模式选择栏在共同控制区内
    assert common < mode
    # 共同控制区早于 LLM 建议卡片、HPO 搜索配置/最佳结果与 HPO 草稿
    assert mode < html.index('class="suggestion-card llm-only"')
    assert mode < html.index('id="hpoSearchConfig"')
    assert mode < html.index('id="hpoBestArea"')
    assert mode < html.index('id="hpoCreateDraft"')
    # 主操作按钮移出表单后仍用 form= 关联到共用表单，提交行为不变
    submit = html.index('id="startTuningBtn"')
    button_tag = html[html.rindex('<button', 0, submit):html.index('>', submit)]
    assert 'form="tuningForm"' in button_tag
    assert html.count('<form id="tuningForm">') == 1


def test_hpo_draft_and_frozen_detail_are_distinct_regions():
    html = _TEMPLATE
    assert 'id="hpoCreateDraft"' in html
    assert 'id="hpoStudyDetail"' in html
    draft = html.index('id="hpoCreateDraft"')
    detail = html.index('id="hpoStudyDetail"')
    assert draft != detail


def test_new_collapsed_sections_have_chinese_translations():
    """返修新增的折叠区标题必须在 zh 有译文（P4 双语展示一致）。"""
    from auto_tune.ui.i18n import make_translator

    zh = make_translator("zh")
    for key in ("Advanced Technical Options",
                "Search Range & Fixed Conditions (technical)",
                "Frozen Details", "Linked Final Model / Formal Training",
                # 第四轮：评价方式、结果操作、正式训练监控
                "Best Search Parameters (technical)",
                "Open Tuning Result Folder", "Download best.pt",
                "Formal Training Monitor",
                "Comprehensive (four metrics)", "Quick (two metrics)"):
        assert key in _TEMPLATE, key
        assert zh(key) != key, key
        assert re.search(r"[一-鿿]", zh(key)), key


def test_fourth_round_regions_are_direct_or_collapsed_as_designed():
    """第四轮：绑定选择器直接可见可操作；六参数明细与内部身份仍在折叠区。"""
    html = _TEMPLATE
    for element_id in ("hpoBestDetails", "hpoBindingNotice", "hpoDraftDetails"):
        assert 'id="' + element_id + '"' in html, element_id
    # 绑定区不再折叠：既没有折叠容器，也没有可编辑路径
    assert 'id="hpoInputDetails"' not in html
    assert 'id="hpoModelPath"' not in html
    opener = html[html.rindex("<details", 0, html.index('id="hpoBestDetails"')):
                  html.index(">", html.index('id="hpoBestDetails"'))]
    assert " open" not in opener
    # 数据集/权重选择器在折叠区之外（主界面直接可操作）
    draft = html.index('id="hpoCreateDraft"')
    details = html.index("<details", draft)
    assert html.index('id="hpoSnapshotSelect"') < details
    assert html.index('id="hpoModelSelect"') < details
    # 关联结果/最终模型入口与安全 warnings 都在 best 折叠体之外
    formal_runs = html.index('id="hpoFormalRuns"')
    warning = html.index('id="hpoFormalRunsWarning"')
    details_close = html.index("</details>", html.index('id="hpoBestDetails"'))
    assert formal_runs > details_close
    assert warning > details_close
    # 无任务时不显示永远禁用的停止/恢复/启动按钮
    for element_id in ("hpoStopBtn", "hpoResumeBtn", "hpoStartBtn"):
        opener = html[html.rindex("<button", 0, html.index('id="' + element_id + '"')):
                      html.index(">", html.index('id="' + element_id + '"'))]
        assert " hidden" in opener, element_id


def test_hpo_actions_present_for_new_flows():
    html = _TEMPLATE
    for control in ("hpoCreateAndStartBtn", "hpoSnapshotSelect", "hpoSelectBtn",
                    "hpoFormalForm", "hpoFormalBestBtn", "hpoFormalRunsList",
                    "hpoDownloadBest", "hpoOpenResultFolder", "hpoSearchConfig",
                    "hpoFieldError", "hpoStudyPagePrev", "hpoStudyPageNext",
                    "hpoFrozenSummary", "hpoEvaluationMode", "hpoHistoryToggle",
                    "hpoProgressBar", "hpoFormalConditions",
                    "hpoFormalMonitorHost"):
        assert 'id="' + control + '"' in html, control


# ── 第五轮 Task 2：五项技术设置收入“高级技术选项”折叠区 ─────────────

# 只有编辑控件被折叠：默认值、设备探测、校验与服务端提交字段都不变。
_COLLAPSED_DRAFT_CONTROLS = ("hpoSampler", "hpoEvaluationMode", "hpoDevice",
                             "hpoImgsz", "hpoBatch")
# 主区必须继续直接显示：试验次数、每次训练轮数、数据快照、初始权重
_DIRECT_DRAFT_CONTROLS = ("hpoBudget", "hpoEpochs", "hpoSnapshotSelect",
                          "hpoModelSelect")


def _draft_details_bounds(html: str) -> tuple[int, int]:
    start = html.rindex("<details", 0, html.index('id="hpoDraftDetails"'))
    return start, html.index("</details>", start)


def test_the_five_technical_settings_live_inside_the_collapsed_draft_details():
    html = _TEMPLATE
    start, end = _draft_details_bounds(html)
    for control in _COLLAPSED_DRAFT_CONTROLS:
        index = html.index('id="' + control + '"')
        assert start < index < end, control
    # 技术摘要（种子/超时/完整内部身份）继续留在同一折叠区
    for control in ("hpoSeed", "hpoTimeout", "hpoCreateConfirmDetail"):
        index = html.index('id="' + control + '"')
        assert start < index < end, control


def test_budget_epochs_snapshot_and_weights_stay_outside_the_collapsed_area():
    html = _TEMPLATE
    start, end = _draft_details_bounds(html)
    for control in _DIRECT_DRAFT_CONTROLS:
        index = html.index('id="' + control + '"')
        assert not (start < index < end), control
    # 字段错误必须在折叠区之外可见
    error = html.index('id="hpoFieldError"')
    assert not (start < error < end)


def test_draft_details_is_collapsed_by_default_and_controls_stay_unique():
    html = _TEMPLATE
    opener = html[html.rindex("<details", 0, html.index('id="hpoDraftDetails"')):
                  html.index(">", html.index('id="hpoDraftDetails"'))]
    assert " open" not in opener
    for control in _COLLAPSED_DRAFT_CONTROLS + _DIRECT_DRAFT_CONTROLS:
        assert html.count('id="' + control + '"') == 1, control
    # 每一项都仍然只有一个控件：没有为折叠另建第二套重复控件
    for name in ("sampler", "evaluation_mode", "device", "imgsz", "batch"):
        assert html.count('name="' + name + '"') == 1, name


def test_hpo_best_card_has_no_llm_justification():
    script = _js()
    best = script.split("function renderBest", 1)
    assert len(best) == 2, "the best card renderer must exist"
    body = best[1].split("\n}", 1)[0]
    for token in ("rationale", "diagnosis", "建议修改", "提升"):
        assert token not in body


def _hpo_regions(html):
    """Return the markup of every element carrying the hpo-only class."""
    regions = []
    for match in re.finditer(r'<div class="[^"]*hpo-only[^"]*"[^>]*>', html):
        start = match.end()
        depth = 1
        idx = start
        while depth and idx < len(html):
            nxt_open = html.find("<div", idx)
            nxt_close = html.find("</div>", idx)
            if nxt_close == -1:
                break
            if nxt_open != -1 and nxt_open < nxt_close:
                depth += 1
                idx = nxt_open + 4
            else:
                depth -= 1
                idx = nxt_close + 6
        regions.append(html[start:idx])
    return regions


def _region(html, element_id):
    """Return the markup of the element carrying ``element_id``."""
    marker = 'id="' + element_id + '"'
    start = html.index(marker)
    open_at = html.rindex("<div", 0, start)
    depth = 1
    idx = html.index(">", start) + 1
    while depth and idx < len(html):
        nxt_open = html.find("<div", idx)
        nxt_close = html.find("</div>", idx)
        if nxt_close == -1:
            break
        if nxt_open != -1 and nxt_open < nxt_close:
            depth += 1
            idx = nxt_open + 4
        else:
            depth -= 1
            idx = nxt_close + 6
    assert idx > open_at
    return html[open_at:idx]


def test_formal_training_form_only_allows_epochs():
    """第四轮：正式训练只允许调整轮数；batch/imgsz/device 与搜索阶段冻结一致。"""
    region = _region(_TEMPLATE, "hpoFormalForm")
    ids = re.findall(r'<input[^>]*id="([^"]+)"', region)
    assert ids == ["hpoFormalEpochs"]
    for removed in ("hpoFormalBatch", "hpoFormalImgsz", "hpoFormalDevice"):
        assert 'id="' + removed + '"' not in _TEMPLATE
    # the six search parameters can never be edited here
    for key in ("optimizer", "lr0", "lrf", "momentum", "weight_decay",
                "warmup_epochs"):
        assert 'name="' + key + '"' not in region
        assert 'id="' + key + '"' not in region


def test_formal_training_defaults_come_from_the_server_not_the_study():
    """正式训练默认轮数取服务端权威默认，绝不沿用研究的 1epoch。"""
    script = _js()
    defaults = script.split("function applyDefaults", 1)[1].split("\n  }", 1)[0]
    assert "'hpoFormalEpochs'" in defaults
    assert "formal." in defaults
    # 其余三项条件不是输入框，而是当前研究的权威执行条件
    assert "'hpoFormalBatch'" not in defaults
    # 折叠区展示的是服务端默认，而不是把研究冻结值写进输入框
    notes = script.split("function applyFormalNotes", 1)[1].split("\n  }", 1)[0]
    for field in ("epochs", "batch", "imgsz", "device"):
        assert ".value" not in notes, field
    # 默认只应用一次，轮询/切换不覆盖用户后续输入
    assert "state.defaultsApplied" in script


def test_formal_training_uses_the_json_route_and_reports_the_linked_id():
    script = _js()
    formal = script.split("window.hpoStartFormalTraining = function", 1)[1]
    assert "/train-best" in formal
    assert "trial_id" in formal and "training_config" in formal
    # a 202 is accepted-but-unfinished: the linked run id is shown immediately
    assert "202" in formal
    assert "train_name" in formal and "run_id" in formal
    # never parse an SSE body as JSON to await completion
    assert "await" not in formal
    # local pre-validation mirrors the server's strict integer/range rules
    assert "validateFormal" in script


def test_legacy_verification_entry_stays_a_separate_secondary_action():
    script = _js()
    verify = script.split("window.hpoStartVerification = function", 1)[1]
    assert "/api/training/start" in verify
    assert "source_hpo" in verify
    # it must not be the primary best-parameter path
    assert "/train-best" not in verify


def test_js_selection_switch_clears_best_and_pending_operations():
    script = _js()
    select = script.split("function selectStudy", 1)[1].split("\n  }", 1)[0]
    assert "clearBestAndErrors" in select
    assert "stopPolling" in select          # the previous timer is stopped first
    assert "bumpSelection" in select        # new generation for the new study
    cleared = script.split("function clearBestAndErrors", 1)[1].split("\n  }", 1)[0]
    assert "hpoBestParamsBody" in cleared
    assert "hpoResultActions" in cleared
    assert "hpoDownloadBest" in cleared and "hpoOpenResultFolder" in cleared
    assert "stopPending" in cleared


def test_js_selection_generation_drops_stale_replies():
    script = _js()
    assert "state.pollStudyId" in script
    fetch = script.split("function hpoFetchStatus", 1)[1].split("\n  }", 1)[0]
    assert "isCurrentSelection" in fetch
    formal = script.split("function refreshFormalRuns", 1)[1].split("\n  }", 1)[0]
    assert "isCurrentSelection" in formal


def test_js_draft_confirmation_shows_the_actual_bindings():
    script = _js()
    confirm = script.split("function updateDraftReadouts", 1)[1] \
        .split("\n  }", 1)[0]
    assert "draftInputs" in confirm
    assert "完整数据快照 ID" in confirm and "初始权重" in confirm
    # 图片口径：background 是 train/val 子集，不重复相加
    assert "背景" in confirm and "子集" in confirm
    # 主摘要只给名称/短编号，绝不显示物理路径
    assert "inputs.model_path" in confirm
    assert "path/" not in _region(_TEMPLATE, "hpoCreateConfirm")
    # it is wired to the draft controls
    assert "updateDraftReadouts" in script.split("addEventListener('DOMContentLoaded'", 1)[1]
    assert "hpoCreateConfirm" in _TEMPLATE


def test_js_unknown_create_outcome_is_not_resent():
    script = _js()
    create = script.split("function createStudy", 1)[1].split("\n  function", 1)[0]
    # exactly one create POST; an unknown outcome never triggers a retry
    assert create.count("jsonPost('/api/hpo/studies'") == 1
    assert "创建请求结果未知" in create
    start = script.split("window.hpoCreateAndStart = function", 1)[1] \
        .split("\n  };", 1)[0]
    assert "createStudy" in start and "startStudy" in start
    assert ".then" in start


def test_js_stop_pending_is_not_reported_as_stopped():
    script = _js()
    assert "正在停止" in script
    stop = script.split("window.hpoStopStudy = function", 1)[1].split("\n  };", 1)[0]
    assert "202" in stop and "state.stopPending = true" in stop
    assert "尚未停止" in stop


def test_js_history_paging_uses_offset_and_limit():
    script = _js()
    page = script.split("window.hpoHistoryPage = function", 1)[1].split("\n  };", 1)[0]
    assert "historyOffset" in page and "historyLimit" in page
    hist = script.split("function hpoRefreshHistory", 1)[1].split("\n  }", 1)[0]
    assert "offset=" in hist and "limit=" in hist
    # a bad record stays visible instead of being silently dropped
    assert "row.readable" in script
    # a transient read conflict is never dressed up as corruption
    assert "row.transient" in script


def test_js_reload_restores_only_the_selected_id_and_never_starts_training():
    script = _js()
    assert "localStorage" in script
    # only the study id is cached; no parameters, metrics or results
    assert "SELECTION_KEY" in script
    store = script.split("function rememberSelection", 1)[1].split("\n  }", 1)[0]
    assert "setItem(SELECTION_KEY, String(studyId))" in store
    assert "JSON.stringify" not in store
    restore = script.split("function restoreSelection", 1)[1].split("\n  }", 1)[0]
    assert "selectStudy" in restore
    # restoring never starts, resumes or stops anything
    for forbidden in ("/start", "/resume", "/stop", "train-best"):
        assert forbidden not in restore, forbidden


def test_linked_runs_come_from_the_persisted_projection_not_local_state():
    script = _js()
    assert "/formal-runs" in script
    assert "function refreshFormalRuns" in script
    # a formal run is watched through the API projection, never DOM/localStorage
    assert "state.watchFormal" in script
    run_render = script.split("function renderFormalRuns", 1)[1].split("\n  }", 1)[0]
    assert "localStorage" not in run_render
    # per-run status/result entries: monitor, stored results and the final best.pt
    assert "查看监控" in run_render and "查看结果" in run_render
    assert "下载最终 best.pt" in run_render
    # a missing run fact is stated honestly instead of being fabricated
    assert "未找到该运行的终态事实" in run_render


def test_js_keeps_operations_error_separate_from_polling():
    script = _js()
    # the error area is only written by operations, and the network note is
    # separate so a poll can never erase a real operation error
    assert "hpoStudyDetailError" in script
    assert "hpoRunNote" in script
    assert "显示最近一次可信状态" in script


def test_js_reports_ties_without_claiming_an_improvement():
    script = _js()
    ranking = script.split("function renderRanking", 1)[1].split("\n  }", 1)[0]
    assert "同分" in ranking
    assert "不代表精度更高" in ranking
    # best 卡片绝不出现模型生成的理由或“提升”话术
    best = script.split("function renderBest", 1)[1]
    for token in ("rationale", "diagnosis", "建议修改", "提升"):
        assert token not in best


def test_js_unload_and_mode_switch_never_stop_training():
    script = _js()
    assert "window.addEventListener('beforeunload', stopPolling)" in script
    unload = script.split("window.addEventListener('beforeunload'", 1)[1] \
        .split("\n", 1)[0]
    assert "stop" not in unload.lower() or "stopPolling" in unload
    assert "/stop" not in unload
    mode = script.split("window.onTuningModeChange = function", 1)[1] \
        .split("\n  };", 1)[0]
    assert "/stop" not in mode


# ── local .pt selector: controlled roots only ──────────────────────


def test_local_models_lists_only_controlled_roots(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    project = tmp_path / "project"
    detect = project / "detect"
    detect.mkdir(parents=True)
    (project / "yolov8n.pt").write_bytes(b"weights")
    (detect / "train1" / "weights").mkdir(parents=True)
    (detect / "train1" / "weights" / "best.pt").write_bytes(b"weights")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.pt").write_bytes(b"weights")

    monkeypatch.chdir(project)
    monkeypatch.setattr(app_mod, "_hpo_detect_dir", lambda: str(detect))
    rows = app_mod.list_local_models()
    names = {row["name"] for row in rows}
    assert names == {"yolov8n.pt", "best.pt"}
    for row in rows:
        assert row["size_mb"] >= 0
        assert set(row) == {"name", "path", "size_mb", "kind", "origin"}
        assert "secret" not in row["path"]
    by_name = {row["name"]: row for row in rows}
    # 初始权重与训练产物必须可区分，不能全部显示成同一个 best.pt 标签
    assert by_name["yolov8n.pt"]["kind"] == "initial"
    assert by_name["best.pt"]["kind"] == "training_artifact"
    assert "train1" in by_name["best.pt"]["origin"]
    assert by_name["yolov8n.pt"]["origin"] != by_name["best.pt"]["origin"]


def test_local_models_route_is_read_only_and_bounded(tmp_path, monkeypatch, stack):
    from auto_tune.ui import app as app_mod

    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "yolov8n.pt").write_bytes(b"weights")
    monkeypatch.chdir(project)
    monkeypatch.setattr(app_mod, "_hpo_detect_dir", lambda: str(tmp_path / "none"))
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: str(tmp_path / "none"), raising=False)

    stack._list_models = app_mod.list_local_models
    stack.client = stack._client()
    resp = stack.client.get("/api/hpo/local-models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["models"][0]["name"] == "yolov8n.pt"
    assert body["models"][0]["kind"] == "initial"
    # the listing never mutates anything
    assert sorted(p.name for p in project.iterdir()) == ["yolov8n.pt"]
