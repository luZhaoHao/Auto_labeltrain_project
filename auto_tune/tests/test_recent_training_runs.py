"""Bugfix P3: recent training runs shortcut (service + API).

The intelligent-analysis page gains a "recent training" list (max 4) that only
fills the local training-directory input after server-side verification. This
suite pins the service query (bounded completed/detect candidates, run_dir
artifact lookup, live input_safety validation) and the narrow API contract
(limit 1-4, stable 400, minimal public fields, no path/credential leaks).
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.input_safety import InputSafetyPolicy
from auto_tune.modules.local_index import LocalIndexError
from auto_tune.ui import app as app_mod


def _use_tmp_log(monkeypatch, tmp_path):
    import os as real_os

    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    real_join = real_os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir / parts[1])
        return real_join(*parts)

    monkeypatch.setattr(real_os.path, "join", fake_join)
    return log_dir


def _make_run_dir(base, name, model="yolov8n.pt"):
    d = Path(base) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "args.yaml").write_text(f"model: {model}\n", encoding="utf-8")
    (d / "results.csv").write_text("epoch,metrics/mAP50(B)\n0,0.5\n", encoding="utf-8")
    return d


def _seed_experiment(svc, run_id, run_name, source, *, status="completed",
                     task="detect", finished_at=None, run_dir=None, model="yolov8n.pt",
                     map50=0.8, started_at=None):
    record = {
        "run_id": run_id,
        "run_name": run_name,
        "source": source,
        "status": status,
        "finished_at": finished_at,
        "started_at": started_at,
        "params": {"model": model, "task": task},
        "metrics": {"mAP50": map50},
    }
    if run_dir is not None:
        record["artifacts"] = {"run_dir": str(run_dir)}
    svc.index_experiment(record)
    return record


def _fresh_service(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    return svc


def _client():
    return TestClient(app_mod.app)


# ── service: candidate selection ──


def test_returns_max_four_completed_detect_runs(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    for i in range(6):
        _seed_experiment(
            svc, f"manual:{i}", f"train{i}", "manual",
            finished_at=f"2026-08-01T00:{i:02d}:00Z",
            run_dir=_make_run_dir(tmp_path, f"train{i}"),
        )
    items = svc.recent_training_runs(limit=4)
    assert len(items) == 4


def test_default_limit_is_four(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    for i in range(5):
        _seed_experiment(
            svc, f"manual:{i}", f"train{i}", "manual",
            finished_at=f"2026-08-01T00:{i:02d}:00Z",
            run_dir=_make_run_dir(tmp_path, f"train{i}"),
        )
    assert len(svc.recent_training_runs()) == 4


def test_finished_at_descending_order(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    for i in range(4):
        _seed_experiment(
            svc, f"manual:{i}", f"train{i}", "manual",
            finished_at=f"2026-08-0{i + 1}T00:00:00Z",
            run_dir=_make_run_dir(tmp_path, f"train{i}"),
        )
    names = [i["run_name"] for i in svc.recent_training_runs(limit=4)]
    assert names == ["train3", "train2", "train1", "train0"]


def test_same_finished_at_uses_stable_second_sort_key(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    stamp = "2026-08-01T00:00:00Z"
    _seed_experiment(svc, "manual:a", "train_a", "manual", finished_at=stamp,
                     run_dir=_make_run_dir(tmp_path, "train_a"))
    _seed_experiment(svc, "manual:b", "train_b", "manual", finished_at=stamp,
                     run_dir=_make_run_dir(tmp_path, "train_b"))
    items = svc.recent_training_runs(limit=4)
    assert [i["run_id"] for i in items] == ["manual:b", "manual:a"]


def test_manual_and_tuning_sources_returned(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    _seed_experiment(svc, "manual:m", "train_m", "manual",
                     finished_at="2026-08-01T00:02:00Z",
                     run_dir=_make_run_dir(tmp_path, "train_m"))
    _seed_experiment(svc, "tuning:t", "autotune_t", "tuning",
                     finished_at="2026-08-01T00:01:00Z",
                     run_dir=_make_run_dir(tmp_path, "autotune_t"))
    items = svc.recent_training_runs(limit=4)
    assert {i["source"] for i in items} == {"manual", "tuning"}


def test_non_completed_statuses_excluded(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    for idx, status in enumerate(["running", "failed", "cancelled", "interrupted", "unknown"]):
        _seed_experiment(
            svc, f"manual:{status}", f"train_{status}", "manual", status=status,
            finished_at=f"2026-08-01T00:{idx + 1:02d}:00Z",
            run_dir=_make_run_dir(tmp_path, f"train_{status}"),
        )
    _seed_experiment(svc, "manual:ok", "train_ok", "manual",
                     finished_at="2026-08-01T00:00:00Z",
                     run_dir=_make_run_dir(tmp_path, "train_ok"))
    items = svc.recent_training_runs(limit=4)
    assert [i["run_name"] for i in items] == ["train_ok"]


def test_non_detect_task_excluded(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    _seed_experiment(svc, "manual:cls", "cls_run", "manual", task="classify",
                     finished_at="2026-08-01T00:02:00Z",
                     run_dir=_make_run_dir(tmp_path, "cls_run"))
    _seed_experiment(svc, "manual:det", "det_run", "manual",
                     finished_at="2026-08-01T00:01:00Z",
                     run_dir=_make_run_dir(tmp_path, "det_run"))
    items = svc.recent_training_runs(limit=4)
    assert [i["run_name"] for i in items] == ["det_run"]


# ── service: run_dir verification ──


def test_missing_run_dir_artifact_skipped(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    _seed_experiment(svc, "manual:norun", "train_norun", "manual",
                     finished_at="2026-08-01T00:01:00Z")
    _seed_experiment(svc, "manual:ok", "train_ok", "manual",
                     finished_at="2026-08-01T00:00:00Z",
                     run_dir=_make_run_dir(tmp_path, "train_ok"))
    items = svc.recent_training_runs(limit=4)
    assert [i["run_name"] for i in items] == ["train_ok"]


def test_run_dir_no_longer_exists_skipped(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    gone = tmp_path / "train_gone"
    # Never created on disk; artifact still references the path.
    _seed_experiment(svc, "manual:gone", "train_gone", "manual",
                     finished_at="2026-08-01T00:02:00Z", run_dir=gone)
    _seed_experiment(svc, "manual:ok", "train_ok", "manual",
                     finished_at="2026-08-01T00:01:00Z",
                     run_dir=_make_run_dir(tmp_path, "train_ok"))
    items = svc.recent_training_runs(limit=4)
    assert [i["run_name"] for i in items] == ["train_ok"]


def test_missing_args_yaml_skipped(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    no_args = tmp_path / "train_no_args"
    no_args.mkdir(parents=True, exist_ok=True)
    (no_args / "results.csv").write_text("epoch,metrics/mAP50(B)\n0,0.5\n", encoding="utf-8")
    _seed_experiment(svc, "manual:noargs", "train_no_args", "manual",
                     finished_at="2026-08-01T00:02:00Z", run_dir=no_args)
    _seed_experiment(svc, "manual:ok", "train_ok", "manual",
                     finished_at="2026-08-01T00:01:00Z",
                     run_dir=_make_run_dir(tmp_path, "train_ok"))
    items = svc.recent_training_runs(limit=4)
    assert [i["run_name"] for i in items] == ["train_ok"]


def test_missing_results_csv_skipped(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    no_csv = tmp_path / "train_no_csv"
    no_csv.mkdir(parents=True, exist_ok=True)
    (no_csv / "args.yaml").write_text("model: yolov8n.pt\n", encoding="utf-8")
    _seed_experiment(svc, "manual:nocsv", "train_no_csv", "manual",
                     finished_at="2026-08-01T00:02:00Z", run_dir=no_csv)
    _seed_experiment(svc, "manual:ok", "train_ok", "manual",
                     finished_at="2026-08-01T00:01:00Z",
                     run_dir=_make_run_dir(tmp_path, "train_ok"))
    items = svc.recent_training_runs(limit=4)
    assert [i["run_name"] for i in items] == ["train_ok"]


def test_outside_allowed_roots_skipped(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    allowed = tmp_path / "allowed"
    allowed.mkdir(exist_ok=True)
    _seed_experiment(svc, "manual:in", "train_in", "manual",
                     finished_at="2026-08-01T00:02:00Z",
                     run_dir=_make_run_dir(allowed, "train_in"))
    _seed_experiment(svc, "manual:out", "train_out", "manual",
                     finished_at="2026-08-01T00:03:00Z",
                     run_dir=_make_run_dir(tmp_path, "train_out"))
    policy = InputSafetyPolicy(allowed_roots=(allowed,))
    items = svc.recent_training_runs(limit=4, policy=policy)
    assert [i["run_name"] for i in items] == ["train_in"]


def test_symlink_run_dir_skipped(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    real = _make_run_dir(tmp_path, "real_target")
    link = tmp_path / "train_link"
    try:
        os.symlink(str(real), str(link), target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")
    _seed_experiment(svc, "manual:link", "train_link", "manual",
                     finished_at="2026-08-01T00:02:00Z", run_dir=link)
    _seed_experiment(svc, "manual:ok", "train_ok", "manual",
                     finished_at="2026-08-01T00:01:00Z",
                     run_dir=_make_run_dir(tmp_path, "train_ok"))
    items = svc.recent_training_runs(limit=4)
    assert [i["run_name"] for i in items] == ["train_ok"]


def test_skips_invalid_candidates_until_four_collected(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    # Two newer-but-invalid candidates must be skipped so four valid older ones
    # still fill the result (bounded candidate scan).
    for i in range(2):
        bad = tmp_path / f"train_bad{i}"
        bad.mkdir(parents=True, exist_ok=True)
        _seed_experiment(
            svc, f"manual:bad{i}", f"train_bad{i}", "manual",
            finished_at=f"2026-08-02T00:{i:02d}:00Z", run_dir=bad,
        )
    for i in range(4):
        _seed_experiment(
            svc, f"manual:ok{i}", f"train_ok{i}", "manual",
            finished_at=f"2026-08-01T00:{i:02d}:00Z",
            run_dir=_make_run_dir(tmp_path, f"train_ok{i}"),
        )
    items = svc.recent_training_runs(limit=4)
    assert [i["run_name"] for i in items] == ["train_ok3", "train_ok2", "train_ok1", "train_ok0"]


def test_map50_missing_yields_none(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    record = {
        "run_id": "manual:nomap", "run_name": "train_nomap", "source": "manual",
        "status": "completed", "finished_at": "2026-08-01T00:00:00Z",
        "params": {"model": "yolov8n.pt", "task": "detect"}, "metrics": {},
        "artifacts": {"run_dir": str(_make_run_dir(tmp_path, "train_nomap"))},
    }
    svc.index_experiment(record)
    items = svc.recent_training_runs(limit=4)
    assert items[0]["run_name"] == "train_nomap"
    assert items[0]["map50"] is None


# ── P3 返修: 多轮自动调优 run_name / run_dir 选择 ──


def _seed_multi_iteration(svc, base, *, run_name="autotune_x_iter02",
                          iter_names=("autotune_x_iter01", "autotune_x_iter02")):
    """Register one tuning runtime run_id with several run_dir artifacts (one
    per iteration), each an existing dir containing args.yaml + results.csv."""
    dirs = {}
    for name in iter_names:
        d = _make_run_dir(base, name)
        dirs[name] = d
        svc.index_experiment({
            "run_id": "tuning:sess:1", "run_name": run_name, "source": "tuning",
            "status": "completed", "finished_at": "2026-08-28T10:20:30Z",
            "params": {"model": "yolov8n.pt", "task": "detect"},
            "metrics": {"mAP50": 0.78207},
            "artifacts": {"run_dir": str(d)},
        })
    return dirs


def test_multi_iteration_run_dir_matches_run_name(tmp_path, monkeypatch):
    """回归修复: 同一 run_id 积累 iter01 + iter02 时，必须返回 basename 与
    run_name 匹配的 iter02，而不是存储/排序顺序上的第一个 iter01。"""
    svc = _fresh_service(tmp_path, monkeypatch)
    _seed_multi_iteration(svc, tmp_path)
    items = svc.recent_training_runs(limit=4)
    assert len(items) == 1
    assert items[0]["run_name"] == "autotune_x_iter02"
    assert os.path.basename(items[0]["run_dir"]) == "autotune_x_iter02"


def test_multi_iteration_match_is_not_first_artifact_in_sort_order(tmp_path, monkeypatch):
    """精确匹配目录不是 kind,path 排序的第一项时仍被选中。"""
    svc = _fresh_service(tmp_path, monkeypatch)
    _seed_multi_iteration(
        svc, tmp_path,
        iter_names=("autotune_x_iter00", "autotune_x_iter01", "autotune_x_iter02"),
    )
    items = svc.recent_training_runs(limit=4)
    assert len(items) == 1
    assert os.path.basename(items[0]["run_dir"]) == "autotune_x_iter02"


def test_multi_valid_dirs_without_name_match_skipped(tmp_path, monkeypatch):
    """多个安全有效目录且无精确名称匹配时，跳过实验，不得猜测。"""
    svc = _fresh_service(tmp_path, monkeypatch)
    _seed_multi_iteration(svc, tmp_path, run_name="autotune_y_iter99")
    items = svc.recent_training_runs(limit=4)
    assert items == []


def test_single_valid_dir_without_name_match_kept(tmp_path, monkeypatch):
    """只有一个安全有效目录时保留旧记录兼容（即便 basename != run_name）。"""
    svc = _fresh_service(tmp_path, monkeypatch)
    d = _make_run_dir(tmp_path, "train_legacy")
    svc.index_experiment({
        "run_id": "manual:legacy", "run_name": "some_other_name", "source": "manual",
        "status": "completed", "finished_at": "2026-08-01T00:00:00Z",
        "params": {"model": "yolov8n.pt", "task": "detect"},
        "metrics": {"mAP50": 0.5},
        "artifacts": {"run_dir": str(d)},
    })
    items = svc.recent_training_runs(limit=4)
    assert len(items) == 1
    assert items[0]["run_dir"] == str(d)


def test_unsafe_name_matching_dir_not_returned(tmp_path, monkeypatch):
    """名称匹配不能绕过安全验证：精确匹配目录是符号链接时不得返回。"""
    svc = _fresh_service(tmp_path, monkeypatch)
    real = _make_run_dir(tmp_path, "autotune_x_iter02_real")
    link = tmp_path / "autotune_x_iter02"
    try:
        os.symlink(str(real), str(link), target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")
    svc.index_experiment({
        "run_id": "tuning:sess:1", "run_name": "autotune_x_iter02", "source": "tuning",
        "status": "completed", "finished_at": "2026-08-28T10:20:30Z",
        "params": {"model": "yolov8n.pt", "task": "detect"},
        "metrics": {"mAP50": 0.5},
        "artifacts": {"run_dir": str(link)},
    })
    items = svc.recent_training_runs(limit=4)
    assert items == []


def test_api_run_name_matches_basename_run_dir(tmp_path, monkeypatch):
    """API 响应的 run_name 必须与 basename(run_dir) 一致（多轮调优）。"""
    svc = _fresh_service(tmp_path, monkeypatch)
    _seed_multi_iteration(svc, tmp_path)
    resp = _client().get("/api/training/recent-runs")
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["run_name"] == "autotune_x_iter02"
    assert os.path.basename(item["run_dir"]) == item["run_name"]


def test_candidate_scan_is_bounded(tmp_path, monkeypatch):
    """Never scan the full experiment table: older valid runs beyond the bounded
    candidate window are not resurrected when the newest candidates are invalid."""
    svc = _fresh_service(tmp_path, monkeypatch)
    for i in range(25):
        bad = tmp_path / f"train_bad{i}"
        bad.mkdir(parents=True, exist_ok=True)
        _seed_experiment(
            svc, f"manual:bad{i}", f"train_bad{i}", "manual",
            finished_at=f"2026-08-02T00:{i // 60:02d}:{i % 60:02d}Z", run_dir=bad,
        )
    _seed_experiment(svc, "manual:old", "train_old", "manual",
                     finished_at="2026-08-01T00:00:00Z",
                     run_dir=_make_run_dir(tmp_path, "train_old"))
    items = svc.recent_training_runs(limit=4)
    assert all("old" not in i["run_name"] for i in items)
    assert len(items) <= 4


def test_sqlite_unavailable_raises_stable_domain_error(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    (log_dir / "auto_tune.db").write_bytes(b"\x00\x01\x02 not sqlite at all " * 8)
    svc = app_mod._local_index_service()
    with pytest.raises(LocalIndexError):
        svc.recent_training_runs(limit=4)


# ── service: response field contract ──


def test_response_only_contains_public_fields(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    run_dir = _make_run_dir(tmp_path, "train54")
    _seed_experiment(svc, "manual:54", "train54", "manual",
                     finished_at="2026-08-28T10:20:30Z", run_dir=run_dir,
                     model="yolov8n.pt", map50=0.81206)
    items = svc.recent_training_runs(limit=4)
    item = items[0]
    assert set(item) == {
        "run_id", "run_name", "source", "model_name", "finished_at", "map50", "run_dir",
    }
    assert item["run_id"] == "manual:54"
    assert item["run_name"] == "train54"
    assert item["source"] == "manual"
    assert item["model_name"] == "yolov8n.pt"
    assert item["finished_at"] == "2026-08-28T10:20:30Z"
    assert item["map50"] == 0.81206
    assert item["run_dir"] == str(run_dir)


# ── API: GET /api/training/recent-runs ──


def _seed_through_api(tmp_path, monkeypatch, n=4):
    svc = _fresh_service(tmp_path, monkeypatch)
    for i in range(n):
        _seed_experiment(
            svc, f"manual:{i}", f"train{i}", "manual",
            finished_at=f"2026-08-0{i + 1}T00:00:00Z",
            run_dir=_make_run_dir(tmp_path, f"train{i}"),
        )
    return svc


def test_api_returns_items_and_source(tmp_path, monkeypatch):
    _seed_through_api(tmp_path, monkeypatch, n=3)
    resp = _client().get("/api/training/recent-runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "sqlite"
    assert len(data["items"]) == 3
    assert data["items"][0]["run_name"] == "train2"


def test_api_limit_param(tmp_path, monkeypatch):
    _seed_through_api(tmp_path, monkeypatch, n=4)
    assert len(_client().get("/api/training/recent-runs?limit=1").json()["items"]) == 1
    assert len(_client().get("/api/training/recent-runs?limit=4").json()["items"]) == 4


@pytest.mark.parametrize("bad", ["0", "-1", "5", "abc", "3.5", "true"])
def test_api_invalid_limit_returns_400(tmp_path, monkeypatch, bad):
    _seed_through_api(tmp_path, monkeypatch, n=2)
    resp = _client().get(f"/api/training/recent-runs?limit={bad}")
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "INVALID_QUERY"


def test_api_run_dir_is_full_validated_path(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    run_dir = _make_run_dir(tmp_path, "train54")
    _seed_experiment(svc, "manual:54", "train54", "manual",
                     finished_at="2026-08-28T10:20:30Z", run_dir=run_dir,
                     model="yolov8n.pt", map50=0.81206)
    resp = _client().get("/api/training/recent-runs")
    item = resp.json()["items"][0]
    assert item["run_dir"] == str(run_dir)
    assert os.path.isdir(item["run_dir"])
    assert os.path.exists(os.path.join(item["run_dir"], "args.yaml"))
    assert os.path.exists(os.path.join(item["run_dir"], "results.csv"))


def test_api_response_leaks_no_forbidden_fields(tmp_path, monkeypatch):
    svc = _fresh_service(tmp_path, monkeypatch)
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    audit = tmp_path / "audits"
    audit.mkdir(exist_ok=True)
    run_dir = _make_run_dir(tmp_path, "train54")
    record = {
        "run_id": "tuning:sess:54", "run_name": "autotune_54", "source": "tuning",
        "status": "completed", "finished_at": "2026-08-28T10:20:30Z",
        "params": {
            "model": "yolov8n.pt", "task": "detect",
            "data": str(dataset / "data.yaml"),
            "weights": str(run_dir / "weights" / "best.pt"),
            "command": "yolo train data=...",
        },
        "metrics": {"mAP50": 0.8},
        "artifacts": {"run_dir": str(run_dir), "report_path": str(run_dir / "report.json")},
        "audit_path": str(audit / "tuning_audit_x.json"),
    }
    svc.index_experiment(record)
    resp = _client().get("/api/training/recent-runs")
    assert resp.status_code == 200
    data = resp.json()
    # run_dir itself is the documented business field and must be present as the
    # full validated local path (parsed JSON, not a JSON-escaped string blob).
    assert os.path.normcase(os.path.normpath(data["items"][0]["run_dir"])) == (
        os.path.normcase(os.path.normpath(str(run_dir)))
    )
    blob = json.dumps(data)
    for forbidden in [
        str(dataset), str(audit), str(run_dir / "weights"), str(run_dir / "report.json"),
        "tuning_audit_x.json", "data.yaml", "command", "weights", "params", "report",
    ]:
        assert forbidden not in blob, f"forbidden content leaked: {forbidden}"
    assert set(data["items"][0]) == {
        "run_id", "run_name", "source", "model_name", "finished_at", "map50", "run_dir",
    }


def test_api_sqlite_unavailable_stable_503(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    (log_dir / "auto_tune.db").write_bytes(b"\x00\x01\x02 not sqlite at all " * 8)
    resp = _client().get("/api/training/recent-runs")
    assert resp.status_code == 503
    data = resp.json()
    assert data["error_code"] in ("LOCAL_INDEX_CORRUPT", "LOCAL_INDEX_UNAVAILABLE")
    blob = json.dumps(data).lower()
    assert "sqlite" not in blob
    assert "traceback" not in blob
    assert str(log_dir) not in blob


def test_api_experiments_redaction_contract_unchanged(tmp_path, monkeypatch):
    """The generic /api/experiments list keeps redacting run_dir to a basename;
    P3 must not change that contract."""
    svc = _fresh_service(tmp_path, monkeypatch)
    run_dir = _make_run_dir(tmp_path, "train54")
    _seed_experiment(svc, "manual:54", "train54", "manual",
                     finished_at="2026-08-28T10:20:30Z", run_dir=run_dir)
    resp = _client().get("/api/experiments")
    items = resp.json()["items"]
    assert str(run_dir) not in json.dumps(items)
    run_dirs = [a.get("path") for a in items[0].get("artifacts", []) if a.get("kind") == "run_dir"]
    assert run_dirs and run_dirs[0] == "train54"
