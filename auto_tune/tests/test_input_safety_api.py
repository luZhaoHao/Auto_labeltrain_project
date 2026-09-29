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
from auto_tune.delivery.runtime import INPUT_ALLOWED_ROOTS_ENV
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


# ── Task 5b: the container's controlled browse roots ────────────────────────

CONTAINER_ROOTS = "/data/datasets;/opt/auto-tune/detect"


def _posix(path) -> str:
    """A path as the container spells it, on any host platform."""
    return str(path).replace("\\", "/")


def _offered_container_roots() -> list[str]:
    """The two declared roots as this host can spell them.

    The browser resolves every root before it offers it, so on a Windows
    development host a container-absolute ``/data/datasets`` becomes
    ``<drive>:\\data\\datasets``. What the API must not do is offer anything
    *other* than the two declared roots — or nothing at all. The exact container
    spelling is pinned by the Docker delivery suites.
    """
    return [str(Path(root).resolve(strict=False)) for root in CONTAINER_ROOTS.split(";")]


def _input_safety_section(*roots):
    return {"max_directory_members": 200000,
            "max_directory_bytes": 536870912000,
            "allowed_roots": [str(root) for root in roots],
            "allow_unc_paths": False}


def test_the_environment_roots_are_offered_when_the_configuration_has_none(monkeypatch):
    """This is the container start: the configuration lists no root at all and
    the two declared directories are the ones the browser offers."""
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", _input_safety_section())
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, CONTAINER_ROOTS)

    resp = _client().post("/api/browse-folder", json={"path": ""})

    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert [entry["path"] for entry in entries] == _offered_container_roots()
    assert all(entry["is_dir"] for entry in entries)


def test_the_environment_roots_replace_the_configured_ones(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    configured.mkdir()
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", _input_safety_section(configured))
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, CONTAINER_ROOTS)

    resp = _client().post("/api/browse-folder", json={"path": ""})

    assert resp.status_code == 200
    assert [entry["path"] for entry in resp.json()["entries"]] == _offered_container_roots()


def test_the_configured_roots_stay_in_force_when_the_environment_is_unset(monkeypatch, tmp_path):
    """The desktop default is unchanged: no variable, no override."""
    configured = tmp_path / "configured"
    configured.mkdir()
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", _input_safety_section(configured))
    monkeypatch.delenv(INPUT_ALLOWED_ROOTS_ENV, raising=False)

    resp = _client().post("/api/browse-folder", json={"path": ""})

    assert resp.status_code == 200
    assert [entry["path"] for entry in resp.json()["entries"]] == [str(configured.resolve())]


def test_a_configured_root_is_unreachable_once_the_environment_replaces_it(monkeypatch, tmp_path):
    """The override must bound *browsing*, not only the root listing."""
    configured = tmp_path / "configured"
    configured.mkdir()
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", _input_safety_section(configured))
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, CONTAINER_ROOTS)

    resp = _client().post("/api/browse-folder", json={"path": str(configured)})

    assert resp.status_code == 403
    assert resp.json()["error_code"] == "INPUT_PATH_NOT_ALLOWED"


def test_the_offered_roots_have_no_parent_to_navigate_up_to(monkeypatch):
    """No client request can walk the browser above a declared root."""
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", _input_safety_section())
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, CONTAINER_ROOTS)
    policy = app_mod._load_input_policy()

    assert policy.allowed_roots, "the environment must really bound the policy"
    for root in policy.allowed_roots:
        assert app_mod._browse_parent(root, policy) is None


def test_an_invalid_environment_root_is_a_stable_policy_error(monkeypatch):
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", _input_safety_section())
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, "relative/datasets")

    resp = _client().post("/api/browse-folder", json={"path": ""})

    assert resp.status_code == 500
    assert resp.json()["error_code"] == "INPUT_POLICY_INVALID"
    assert "Traceback" not in resp.text
    assert "relative/datasets" not in resp.text


def test_a_directory_the_delivery_does_not_declare_is_refused_by_the_api(monkeypatch):
    """The picker cannot be pointed at the container's own directories: the value
    is refused before any policy is built, with the value never echoed."""
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", _input_safety_section())
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, "/data/datasets;/opt/auto-tune/runs")

    resp = _client().post("/api/browse-folder", json={"path": ""})

    assert resp.status_code == 500
    assert resp.json()["error_code"] == "INPUT_POLICY_INVALID"
    assert "/opt/auto-tune/runs" not in resp.text


# ── an upgraded install: the authoritative roots win before the policy exists ──


def test_an_old_configuration_cannot_block_the_container_start(monkeypatch):
    """A container is started with the ``config.yaml`` an earlier, Windows
    install wrote. Its ``allowed_roots`` are host paths the container cannot
    use, and the container's own value is authoritative: the environment roots
    must replace them *before* the policy is built, so neither the start nor a
    browse request fails on a root the override is about to discard."""
    monkeypatch.setitem(
        app_mod.APP_CONFIG,
        "input_safety",
        _input_safety_section(r"C:\data\datasets", "relative/datasets"),
    )
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, CONTAINER_ROOTS)

    resp = _client().post("/api/browse-folder", json={"path": ""})

    assert resp.status_code == 200
    assert [entry["path"] for entry in resp.json()["entries"]] == _offered_container_roots()


def test_the_other_limits_still_come_from_the_configuration(monkeypatch):
    """Only the roots are authoritative in the environment; the bounded-scan
    limits stay the operator's configuration."""
    section = _input_safety_section()
    section["max_directory_members"] = 7
    section["max_directory_bytes"] = 4096
    section["allow_unc_paths"] = True
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", section)
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, CONTAINER_ROOTS)

    policy = app_mod._load_input_policy()

    assert policy.max_directory_members == 7
    assert policy.max_directory_bytes == 4096
    assert policy.allow_unc_paths is True
    assert [_posix(root) for root in policy.allowed_roots] == CONTAINER_ROOTS.split(";")


def test_a_still_invalid_limit_is_still_a_policy_error(monkeypatch):
    """The override is scoped to the roots: a genuinely invalid limit must keep
    being refused instead of being silently replaced along with them."""
    section = _input_safety_section()
    section["max_directory_members"] = 0
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", section)
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, CONTAINER_ROOTS)

    resp = _client().post("/api/browse-folder", json={"path": ""})

    assert resp.status_code == 500
    assert resp.json()["error_code"] == "INPUT_POLICY_INVALID"


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
