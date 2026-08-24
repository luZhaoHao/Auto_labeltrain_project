"""API tests for Studio S1.4 directory-input safety (browse-folder + analyze-folder gates)."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.input_safety import (
    InputPermissionDeniedError,
    InputPolicyInvalidError,
    InputSafetyPolicy,
)
from auto_tune.ui import app as app_mod


def _client():
    return TestClient(app_mod.app)


def _policy(**overrides):
    base = {
        "max_directory_members": 200000,
        "max_directory_bytes": 536870912000,
        "allowed_roots": (),
        "allow_unc_paths": False,
    }
    base.update(overrides)
    return InputSafetyPolicy(**base)


def _try_symlink_dir(target, link):
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")


# ── Task 5: /api/browse-folder ──


def test_browse_empty_path_returns_drive_listing_when_no_roots(tmp_path, monkeypatch):
    resp = _client().post("/api/browse-folder", json={"path": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["path"] == ""
    assert isinstance(data["entries"], list)


def test_browse_root_shows_allowed_roots_when_configured(tmp_path, monkeypatch):
    root = tmp_path / "allowed_root"
    root.mkdir()
    monkeypatch.setattr(
        app_mod, "_load_input_policy", lambda: _policy(allowed_roots=(root.resolve(),))
    )
    resp = _client().post("/api/browse-folder", json={"path": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert any(e["path"] == str(root.resolve()) for e in data["entries"])


def test_browse_lists_subdirectories_in_order(tmp_path, monkeypatch):
    d = tmp_path / "data"
    (d / "sub1").mkdir(parents=True)
    (d / "sub2").mkdir()
    (d / "file.txt").write_bytes(b"x")
    resp = _client().post("/api/browse-folder", json={"path": str(d)})
    assert resp.status_code == 200
    data = resp.json()
    names = [e["name"] for e in data["entries"]]
    assert names == ["sub1", "sub2"]
    assert all(e["is_dir"] for e in data["entries"])


def test_browse_outside_allowed_root_403(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(
        app_mod, "_load_input_policy", lambda: _policy(allowed_roots=(root.resolve(),))
    )
    resp = _client().post("/api/browse-folder", json={"path": str(outside)})
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "INPUT_PATH_NOT_ALLOWED"


def test_browse_link_400(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    d = tmp_path / "data"
    d.mkdir()
    link = d / "lnk"
    _try_symlink_dir(str(real), str(link))
    resp = _client().post("/api/browse-folder", json={"path": str(link)})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "INPUT_LINK_NOT_ALLOWED"


def test_browse_permission_403(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise InputPermissionDeniedError("目录访问被拒绝")

    monkeypatch.setattr(app_mod, "list_safe_subdirectories", denied)
    resp = _client().post("/api/browse-folder", json={"path": str(tmp_path)})
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "INPUT_PERMISSION_DENIED"


def test_browse_policy_invalid_500(tmp_path, monkeypatch):
    def bad_policy():
        raise InputPolicyInvalidError("input_safety 配置非法")

    monkeypatch.setattr(app_mod, "_load_input_policy", bad_policy)
    resp = _client().post("/api/browse-folder", json={"path": ""})
    assert resp.status_code == 500
    assert resp.json()["error_code"] == "INPUT_POLICY_INVALID"


def test_browse_error_response_has_no_stacktrace(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(
        app_mod, "_load_input_policy", lambda: _policy(allowed_roots=(root.resolve(),))
    )
    resp = _client().post("/api/browse-folder", json={"path": str(outside)})
    assert resp.status_code == 403
    assert "Traceback" not in resp.text
    assert 'File "' not in resp.text


# ── Task 6: analyze-folder preflight gates ──

SAMPLE_CSV = (
    "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),"
    "metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n"
    "1,1.5,3.0,2.0,0.1,0.2,0.05,0.01,1.6,3.1,2.1\n"
)


def _boom(*args, **kwargs):
    raise AssertionError("analysis must not run")


def _dataset_dir(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    return d


def _train_dir(tmp_path):
    t = tmp_path / "train"
    t.mkdir()
    (t / "results.csv").write_text(SAMPLE_CSV, encoding="utf-8")
    (t / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
    return t


def _use_tmp_log(monkeypatch, tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_mod, "LATEST_DATASET_PATH", log_dir / "latest_dataset.json")
    return log_dir


def _mock_training_analysis(monkeypatch):
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.results_parser.load_training_run",
        lambda d: {"name": "train", "results": {"total_epochs": 2, "best_epoch": 1}, "args": {}},
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.curve_analysis.analyze_loss_curves",
        lambda r, c: {},
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.curve_analysis.analyze_metric_curves",
        lambda r, c: {},
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.curve_analysis.detect_early_stopping",
        lambda r, c: {},
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.issue_detector.detect_issues", lambda r, c: []
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.run_comparator.compare_runs", lambda r, c: {}
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.run_comparator.summarize_runs",
        lambda r, c: {
            "best_mAP50": None,
            "average_mAP50": None,
            "runs_with_issues": 0,
            "common_issues": [],
        },
    )
    monkeypatch.setitem(app_mod.APP_CONFIG.setdefault("llm", {}), "enabled", False)
    monkeypatch.setitem(app_mod.APP_CONFIG.setdefault("vision", {}), "enabled", False)


def test_dataset_analyze_member_limit_blocks_before_analysis(tmp_path, monkeypatch):
    d = _dataset_dir(tmp_path)
    for i in range(4):
        (d / f"img{i}.jpg").write_bytes(b"x")
    monkeypatch.setattr(app_mod, "_load_input_policy", lambda: _policy(max_directory_members=2))
    monkeypatch.setattr("auto_tune.modules.dataset_analyzer.analyzer.analyze_dataset", _boom)
    resp = _client().post("/api/dataset/analyze-folder", json={"path": str(d)})
    assert resp.status_code == 413
    assert resp.json()["error_code"] == "INPUT_MEMBER_LIMIT_EXCEEDED"


def test_dataset_analyze_size_limit_blocks_before_analysis(tmp_path, monkeypatch):
    d = _dataset_dir(tmp_path)
    for i in range(3):
        (d / f"img{i}.jpg").write_bytes(b"xxxx")
    monkeypatch.setattr(app_mod, "_load_input_policy", lambda: _policy(max_directory_bytes=5))
    monkeypatch.setattr("auto_tune.modules.dataset_analyzer.analyzer.analyze_dataset", _boom)
    resp = _client().post("/api/dataset/analyze-folder", json={"path": str(d)})
    assert resp.status_code == 413
    assert resp.json()["error_code"] == "INPUT_SIZE_LIMIT_EXCEEDED"


def test_dataset_analyze_outside_root_blocks_before_analysis(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    d = _dataset_dir(tmp_path)
    (d / "img0.jpg").write_bytes(b"x")
    monkeypatch.setattr(
        app_mod, "_load_input_policy", lambda: _policy(allowed_roots=(root.resolve(),))
    )
    monkeypatch.setattr("auto_tune.modules.dataset_analyzer.analyzer.analyze_dataset", _boom)
    resp = _client().post("/api/dataset/analyze-folder", json={"path": str(d)})
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "INPUT_PATH_NOT_ALLOWED"


def test_dataset_analyze_link_blocks_before_analysis(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    d = _dataset_dir(tmp_path)
    (d / "ok.jpg").write_bytes(b"x")
    _try_symlink_dir(str(real), str(d / "lnk"))
    monkeypatch.setattr("auto_tune.modules.dataset_analyzer.analyzer.analyze_dataset", _boom)
    resp = _client().post("/api/dataset/analyze-folder", json={"path": str(d)})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "INPUT_LINK_NOT_ALLOWED"


def test_dataset_analyze_policy_invalid_blocks_before_analysis(tmp_path, monkeypatch):
    d = _dataset_dir(tmp_path)
    (d / "img0.jpg").write_bytes(b"x")

    def bad_policy():
        raise InputPolicyInvalidError("input_safety 配置非法")

    monkeypatch.setattr(app_mod, "_load_input_policy", bad_policy)
    monkeypatch.setattr("auto_tune.modules.dataset_analyzer.analyzer.analyze_dataset", _boom)
    resp = _client().post("/api/dataset/analyze-folder", json={"path": str(d)})
    assert resp.status_code == 500
    assert resp.json()["error_code"] == "INPUT_POLICY_INVALID"


def test_training_analyze_member_limit_blocks_before_analysis(tmp_path, monkeypatch):
    t = _train_dir(tmp_path)
    monkeypatch.setattr(app_mod, "_load_input_policy", lambda: _policy(max_directory_members=1))
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.results_parser.load_training_run", _boom
    )
    resp = _client().post("/api/training/analyze-folder", json={"path": str(t)})
    assert resp.status_code == 413
    assert resp.json()["error_code"] == "INPUT_MEMBER_LIMIT_EXCEEDED"


def test_training_analyze_size_limit_blocks_before_analysis(tmp_path, monkeypatch):
    t = _train_dir(tmp_path)
    monkeypatch.setattr(app_mod, "_load_input_policy", lambda: _policy(max_directory_bytes=5))
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.results_parser.load_training_run", _boom
    )
    resp = _client().post("/api/training/analyze-folder", json={"path": str(t)})
    assert resp.status_code == 413
    assert resp.json()["error_code"] == "INPUT_SIZE_LIMIT_EXCEEDED"


def test_training_analyze_outside_root_blocks_before_analysis(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    t = _train_dir(tmp_path)
    monkeypatch.setattr(
        app_mod, "_load_input_policy", lambda: _policy(allowed_roots=(root.resolve(),))
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.results_parser.load_training_run", _boom
    )
    resp = _client().post("/api/training/analyze-folder", json={"path": str(t)})
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "INPUT_PATH_NOT_ALLOWED"


def test_training_analyze_link_blocks_before_analysis(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    t = _train_dir(tmp_path)
    _try_symlink_dir(str(real), str(t / "lnk"))
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.results_parser.load_training_run", _boom
    )
    resp = _client().post("/api/training/analyze-folder", json={"path": str(t)})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "INPUT_LINK_NOT_ALLOWED"


def test_training_analyze_policy_invalid_blocks_before_analysis(tmp_path, monkeypatch):
    t = _train_dir(tmp_path)

    def bad_policy():
        raise InputPolicyInvalidError("input_safety 配置非法")

    monkeypatch.setattr(app_mod, "_load_input_policy", bad_policy)
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.results_parser.load_training_run", _boom
    )
    resp = _client().post("/api/training/analyze-folder", json={"path": str(t)})
    assert resp.status_code == 500
    assert resp.json()["error_code"] == "INPUT_POLICY_INVALID"


def test_dataset_analyze_failure_does_not_touch_state(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    d = _dataset_dir(tmp_path)
    for i in range(4):
        (d / f"img{i}.jpg").write_bytes(b"x")
    latest = log_dir / "latest_dataset.json"
    latest.write_text(json.dumps({"dataset_path": str(d), "split": False}), encoding="utf-8")
    before = latest.read_bytes()
    monkeypatch.setattr(app_mod, "_load_input_policy", lambda: _policy(max_directory_members=2))
    resp = _client().post("/api/dataset/analyze-folder", json={"path": str(d)})
    assert resp.status_code == 413
    assert latest.read_bytes() == before
    assert not list(log_dir.glob("dataset_report_*.json"))


def test_training_analyze_failure_does_not_touch_state(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    t = _train_dir(tmp_path)
    monkeypatch.setattr(app_mod, "_load_input_policy", lambda: _policy(max_directory_members=1))
    resp = _client().post("/api/training/analyze-folder", json={"path": str(t)})
    assert resp.status_code == 413
    assert not list(log_dir.glob("train_*_report.json"))
    assert not (log_dir / "experiment_history.json").exists()


def test_dataset_analyze_success_includes_input_scan(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    ds = tmp_path / "ds"
    (ds / "images" / "train").mkdir(parents=True)
    (ds / "images" / "train" / "img0.jpg").write_bytes(b"xxx")
    monkeypatch.setattr(
        "auto_tune.modules.dataset_analyzer.analyzer.analyze_dataset",
        lambda dir_, yaml_, cfg: {"status": "ok", "quality_score": 0.5},
    )
    resp = _client().post("/api/dataset/analyze-folder", json={"path": str(ds)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["input_scan"]["member_count"] >= 1
    assert data["input_scan"]["total_bytes"] >= 0


def test_training_analyze_success_includes_input_scan(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    t = _train_dir(tmp_path)
    _mock_training_analysis(monkeypatch)
    resp = _client().post("/api/training/analyze-folder", json={"path": str(t)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["input_scan"]["member_count"] == 2
    assert data["input_scan"]["total_bytes"] >= 0
