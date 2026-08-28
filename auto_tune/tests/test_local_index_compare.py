"""Studio S2.4: basic experiment comparison (2-5 runs, one baseline).

Comparison consumes only fact fields, never creates labels/baselines/releases,
keeps missing metrics as null (never 0), marks incomparable mixes honestly, and
outputs only factual summaries (highest metric / shortest duration / most
complete artifacts) — never a "best model" conclusion.
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.local_index.compare import (
    compare_experiments,
    is_noise_param,
)
from auto_tune.modules.local_index.models import LocalIndexConfig
from auto_tune.modules.local_index.service import LocalIndexService


def _cfg(tmp_path, **kw):
    return LocalIndexConfig(
        database_path=tmp_path / "auto_tune.db",
        backup_dir=tmp_path / "db_backups",
        backup_max_files=kw.get("backup_max_files", 3),
        busy_timeout_ms=kw.get("busy_timeout_ms", 200),
    )


def _service(tmp_path):
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    return svc


def _exp(run_id, source="manual", status="completed", metrics=None, run_dir=None,
         params=None, tuning=None, decision=None, data_yaml=None, started_at=None,
         finished_at="2026-08-01T00:00:00Z", task_type="detect", audit_path=None):
    rec = {
        "run_id": run_id,
        "run_name": run_id.rsplit(":", 1)[-1],
        "source": source,
        "status": status,
        "analysis_status": "completed",
        "metrics": dict(metrics or {}),
        "params": dict(params or {}),
        "finished_at": finished_at,
        "artifacts": {"run_dir": run_dir},
        "audit_path": audit_path,
        "started_at": started_at,
    }
    if data_yaml:
        rec["params"]["data"] = data_yaml
    if task_type:
        rec["params"]["task"] = task_type
    rec["params"].setdefault("model", "yolov8n.pt")
    if tuning is not None:
        rec["tuning"] = tuning
    if decision is not None:
        rec["decision"] = decision
    return rec


def _make_run_dir(tmp_path, name):
    run_dir = tmp_path / "detect" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.1\n", encoding="utf-8")
    (run_dir / "args.yaml").write_text("model: yolov8n.pt\n", encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"weights")
    return str(run_dir)


def _dataset(svc, ds_path, snapshot_id):
    return svc.index_dataset({
        "source_dataset_path": str(ds_path),
        "data_yaml_path": os.path.join(str(ds_path), "data.yaml"),
        "snapshot_id": snapshot_id,
        "snapshot_valid": True,
    })


# ── validation ──


def test_compare_rejects_too_few(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:1"))
    with pytest.raises(Exception) as exc:
        svc.compare_experiments(["manual:1"], "manual:1")
    assert getattr(exc.value, "error_code", "") == "LOCAL_INDEX_COMPARE_INVALID"


def test_compare_rejects_too_many(tmp_path):
    svc = _service(tmp_path)
    run_ids = []
    for i in range(6):
        svc.index_experiment(_exp(f"manual:{i}"))
        run_ids.append(f"manual:{i}")
    with pytest.raises(Exception) as exc:
        svc.compare_experiments(run_ids, "manual:0")
    assert getattr(exc.value, "error_code", "") == "LOCAL_INDEX_COMPARE_INVALID"


def test_compare_rejects_baseline_not_in_set(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:1"))
    svc.index_experiment(_exp("manual:2"))
    with pytest.raises(Exception) as exc:
        svc.compare_experiments(["manual:1", "manual:2"], "manual:99")
    assert getattr(exc.value, "error_code", "") == "LOCAL_INDEX_COMPARE_INVALID"


def test_compare_rejects_missing_experiment(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:1"))
    with pytest.raises(Exception) as exc:
        svc.compare_experiments(["manual:1", "manual:nope"], "manual:1")
    assert getattr(exc.value, "error_code", "") == "NOT_FOUND"


def test_compare_dedupes_run_ids(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:1"))
    svc.index_experiment(_exp("manual:2"))
    result = svc.compare_experiments(["manual:1", "manual:2", "manual:2"], "manual:1")
    assert result["run_ids"] == ["manual:1", "manual:2"]


# ── projection ──


def test_compare_identity_and_metrics(tmp_path):
    svc = _service(tmp_path)
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    svc.index_experiment(_exp(
        "manual:a", data_yaml=data_yaml,
        metrics={"mAP50": 0.4, "mAP50_95": 0.2, "precision": 0.5, "recall": 0.4},
        finished_at="2026-08-01T00:00:00Z"))
    svc.index_experiment(_exp(
        "manual:b", data_yaml=data_yaml,
        metrics={"mAP50": 0.8, "mAP50_95": 0.5, "precision": 0.7, "recall": 0.6},
        finished_at="2026-08-02T00:00:00Z"))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["baseline_run_id"] == "manual:a"
    assert result["comparable"] is True
    assert result["identity"][0]["model_name"] is not None
    assert result["identity"][0]["dataset_id"] == ds.dataset_id
    assert result["metrics"]["mAP50"]["manual:b"] == 0.8
    assert result["relative"]["mAP50"]["manual:b"] == pytest.approx(1.0)
    assert result["relative"]["mAP50"]["manual:a"] == 0.0


def test_compare_missing_metrics_stay_null_never_zero(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", metrics={"mAP50": 0.5}))
    svc.index_experiment(_exp("manual:b", metrics={}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["metrics"]["mAP50_95"]["manual:b"] is None
    assert result["relative"]["mAP50_95"]["manual:b"] is None
    assert result["metrics"]["mAP50"]["manual:b"] is None


def test_compare_baseline_zero_relative_null(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", metrics={"mAP50": 0.0}))
    svc.index_experiment(_exp("manual:b", metrics={"mAP50": 0.5}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["relative"]["mAP50"]["manual:b"] is None


def test_compare_param_diff_folds_common_and_ignores_noise(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp(
        "manual:a",
        params={"model": "yolov8n.pt", "epochs": 100, "batch": 16,
                "name": "noise", "save_dir": "/tmp/x", "project": "p",
                "_epochs": {"configured": 100}}))
    svc.index_experiment(_exp(
        "manual:b",
        params={"model": "yolov8n.pt", "epochs": 100, "batch": 32,
                "name": "noise2", "save_dir": "/tmp/y", "project": "p",
                "_epochs": {"configured": 100}}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    params = result["parameters"]
    assert params["common"]["model"] == "yolov8n.pt"
    assert params["common"]["epochs"] == 100
    assert "name" not in params["common"]
    assert "_epochs" not in params["common"]
    assert params["differences"]["manual:a"].get("batch") == 16
    assert params["differences"]["manual:b"].get("batch") == 32


def test_compare_different_dataset_not_comparable_with_warning(tmp_path):
    svc = _service(tmp_path)
    d1 = _dataset(svc, tmp_path / "d1", "snap1")
    d2 = _dataset(svc, tmp_path / "d2", "snap2")
    svc.index_experiment(_exp(
        "manual:a", data_yaml=os.path.join(str(tmp_path / "d1"), "data.yaml")))
    svc.index_experiment(_exp(
        "manual:b", data_yaml=os.path.join(str(tmp_path / "d2"), "data.yaml")))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is False
    assert any("dataset" in w for w in result["warnings"])
    assert result["summary"] is None
    assert result["identity"][0]["dataset_id"] == d1.dataset_id


def test_compare_different_task_not_comparable(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", task_type="detect"))
    svc.index_experiment(_exp("manual:b", task_type="classify"))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is False
    assert any("task" in w for w in result["warnings"])
    assert result["summary"] is None


def test_compare_summary_factual_only(tmp_path):
    svc = _service(tmp_path)
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    ra = _make_run_dir(tmp_path, "a")
    rb = _make_run_dir(tmp_path, "b")
    svc.index_experiment(_exp(
        "manual:a", data_yaml=data_yaml, metrics={"mAP50": 0.4}, run_dir=ra,
        started_at="2026-08-01T00:00:00Z", finished_at="2026-08-01T01:00:00Z"))
    svc.index_experiment(_exp(
        "manual:b", data_yaml=data_yaml, metrics={"mAP50": 0.8}, run_dir=rb,
        started_at="2026-08-01T00:00:00Z", finished_at="2026-08-01T00:30:00Z"))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is True
    summary = result["summary"]
    assert summary["highest_mAP50"] == "manual:b"
    assert summary["shortest_duration"] == "manual:b"
    assert "best_model" not in summary
    assert result["training"]["manual:b"]["duration_seconds"] == 1800


def test_compare_tuning_facts(tmp_path):
    svc = _service(tmp_path)
    audit = tmp_path / "log" / "tuning_audit_x.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("{}", encoding="utf-8")
    tuning = {"guardrails": {"valid": True}, "decision": {"diagnosis": "overfit"}}
    svc.index_experiment(_exp(
        "tuning:u1:at1", source="tuning", tuning=tuning, audit_path=str(audit),
        finished_at="2026-08-01T00:00:00Z"))
    svc.index_experiment(_exp(
        "tuning:u2:at2", source="tuning", tuning={"guardrails": {"valid": False}},
        finished_at="2026-08-02T00:00:00Z"))

    result = svc.compare_experiments(["tuning:u1:at1", "tuning:u2:at2"], "tuning:u1:at1")

    assert result["tuning"]["tuning:u1:at1"]["guardrails_valid"] is True
    assert result["tuning"]["tuning:u1:at1"]["has_audit"] is True
    assert result["tuning"]["tuning:u2:at2"]["guardrails_valid"] is False


def test_compare_artifact_completeness(tmp_path):
    svc = _service(tmp_path)
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    ra = _make_run_dir(tmp_path, "a")
    rb = tmp_path / "detect" / "b"  # intentionally missing run_dir
    svc.index_experiment(_exp("manual:a", data_yaml=data_yaml, metrics={"mAP50": 0.5},
                              run_dir=ra, finished_at="2026-08-01T00:00:00Z"))
    svc.index_experiment(_exp("manual:b", data_yaml=data_yaml, metrics={"mAP50": 0.5},
                              run_dir=str(rb), finished_at="2026-08-02T00:00:00Z"))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["artifacts"]["manual:a"]["exists"] > result["artifacts"]["manual:b"]["exists"]
    assert result["summary"]["most_complete_artifacts"] == "manual:a"


def test_is_noise_param():
    assert is_noise_param("name") is True
    assert is_noise_param("project") is True
    assert is_noise_param("save_dir") is True
    assert is_noise_param("_epochs") is True
    assert is_noise_param("_legacy_record_run_id") is True
    assert is_noise_param("model") is False
    assert is_noise_param("epochs") is False


# ── API ──


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


def _client():
    from auto_tune.ui import app as app_mod

    return TestClient(app_mod.app)


def test_compare_route_requires_csrf(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().post("/api/experiments/compare", json={
        "run_ids": ["manual:1", "manual:2"], "baseline_run_id": "manual:1"})
    assert resp.status_code == 403


def test_compare_route_ok(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    svc.index_experiment(_exp("manual:a", data_yaml=data_yaml, metrics={"mAP50": 0.4},
                              finished_at="2026-08-01T00:00:00Z"))
    svc.index_experiment(_exp("manual:b", data_yaml=data_yaml, metrics={"mAP50": 0.8},
                              finished_at="2026-08-02T00:00:00Z"))
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}

    resp = _client().post("/api/experiments/compare", headers=headers, json={
        "run_ids": ["manual:a", "manual:b"], "baseline_run_id": "manual:a"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["comparable"] is True
    assert data["summary"]["highest_mAP50"] == "manual:b"


def test_compare_route_invalid_body(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}
    resp = _client().post("/api/experiments/compare", headers=headers, json={
        "run_ids": ["manual:a"], "baseline_run_id": "manual:a"})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "LOCAL_INDEX_COMPARE_INVALID"


# ── 返修 7: 未知 task_type / dataset_id / 指标口径不得判定 comparable ──


def test_compare_unknown_task_type_not_comparable(tmp_path):
    svc = _service(tmp_path)
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    svc.index_experiment(_exp("manual:a", data_yaml=data_yaml, task_type=None,
                              metrics={"mAP50": 0.4}))
    svc.index_experiment(_exp("manual:b", data_yaml=data_yaml, task_type=None,
                              metrics={"mAP50": 0.8}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is False
    assert any("task" in w for w in result["warnings"])
    assert result["summary"] is None


def test_compare_unknown_dataset_not_comparable(tmp_path):
    svc = _service(tmp_path)
    # No dataset registered: both experiments carry dataset_id=None.
    svc.index_experiment(_exp("manual:a", metrics={"mAP50": 0.4}))
    svc.index_experiment(_exp("manual:b", metrics={"mAP50": 0.8}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is False
    assert any("dataset" in w for w in result["warnings"])
    assert result["summary"] is None


def test_compare_insufficient_metric_scope_not_comparable(tmp_path):
    svc = _service(tmp_path)
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    # Only one experiment carries mAP50; the other only mAP50_95.
    svc.index_experiment(_exp("manual:a", data_yaml=data_yaml, metrics={"mAP50": 0.4}))
    svc.index_experiment(_exp("manual:b", data_yaml=data_yaml,
                              metrics={"mAP50_95": 0.3}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is False
    assert any("metric" in w for w in result["warnings"])
    assert result["summary"] is None


def test_compare_never_mixes_detect_and_classify_in_best(tmp_path):
    svc = _service(tmp_path)
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    svc.index_experiment(_exp("manual:a", data_yaml=data_yaml, task_type="detect",
                              metrics={"mAP50": 0.9}))
    svc.index_experiment(_exp("manual:b", data_yaml=data_yaml, task_type="classify",
                              metrics={"mAP50": 0.7}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is False
    assert any("task" in w for w in result["warnings"])
    # No best conclusion is produced across mixed tasks.
    assert result["summary"] is None


# ── 返修 5: 空/unknown/无法识别 task_type 与明确任务指标矩阵 ──


def test_compare_unknown_string_task_type_not_comparable(tmp_path):
    svc = _service(tmp_path)
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    svc.index_experiment(_exp("manual:a", data_yaml=data_yaml, task_type="unknown",
                              metrics={"mAP50": 0.4}))
    svc.index_experiment(_exp("manual:b", data_yaml=data_yaml, task_type="unknown",
                              metrics={"mAP50": 0.8}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is False
    assert any("task" in w for w in result["warnings"])
    assert result["summary"] is None


def test_compare_empty_task_type_not_comparable(tmp_path):
    svc = _service(tmp_path)
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    # Both experiments carry an empty task string in params.
    svc.index_experiment(_exp("manual:a", data_yaml=data_yaml, task_type=None,
                              params={"task": ""}, metrics={"mAP50": 0.4}))
    svc.index_experiment(_exp("manual:b", data_yaml=data_yaml, task_type=None,
                              params={"task": ""}, metrics={"mAP50": 0.8}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is False
    assert any("task" in w for w in result["warnings"])
    assert result["summary"] is None


def test_compare_unsupported_task_type_not_comparable(tmp_path):
    svc = _service(tmp_path)
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    svc.index_experiment(_exp("manual:a", data_yaml=data_yaml, task_type="recognize",
                              metrics={"mAP50": 0.4}))
    svc.index_experiment(_exp("manual:b", data_yaml=data_yaml, task_type="recognize",
                              metrics={"mAP50": 0.8}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is False
    assert any("not supported" in w for w in result["warnings"])
    assert result["summary"] is None


def test_compare_metric_scope_unconfirmable_not_comparable(tmp_path):
    svc = _service(tmp_path)
    ds = _dataset(svc, tmp_path / "d", "snap1")
    data_yaml = os.path.join(str(tmp_path / "d"), "data.yaml")
    # Both detect, but neither exposes mAP50 — the metric scope cannot be
    # confirmed for the supported matrix.
    svc.index_experiment(_exp("manual:a", data_yaml=data_yaml, task_type="detect",
                              metrics={"precision": 0.5}))
    svc.index_experiment(_exp("manual:b", data_yaml=data_yaml, task_type="detect",
                              metrics={"recall": 0.6}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["comparable"] is False
    assert any("metric" in w or "mAP50" in w for w in result["warnings"])
    assert result["summary"] is None


# ── 返修 1: 比较参数差异排除路径型参数 ──


def test_compare_param_diff_excludes_path_params(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", params={
        "data": "D:/x/data.yaml", "epochs": 100, "batch": 16}))
    svc.index_experiment(_exp("manual:b", params={
        "data": "D:/x/data.yaml", "epochs": 100, "batch": 32}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    params = result["parameters"]
    assert "data" not in params["common"]
    assert "data" not in params["differences"]["manual:a"]
    assert "data" not in params["differences"]["manual:b"]
    assert params["common"]["epochs"] == 100
    assert params["differences"]["manual:b"].get("batch") == 32


def test_is_noise_param_path_typed():
    assert is_noise_param("data") is True
    assert is_noise_param("data_yaml_path") is True
    assert is_noise_param("run_dir") is True
    assert is_noise_param("report_path") is True
    assert is_noise_param("epochs") is False


# ─────────────────────────────────────────────────────────────
# 返修5: 严格验证比较指标 — 仅有限数值可用；无效值 unavailable + comparable=false + 稳定 warning
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_value", [
    "unknown", "", "nan", "N/A", "abc", "0.5x",
])
def test_compare_non_numeric_metric_unavailable(tmp_path, bad_value):
    """非数字字符串指标（unknown/空/非法）→ unavailable，comparable=false + 稳定 warning。"""
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", metrics={"mAP50": 0.5}))
    svc.index_experiment(_exp("manual:b", metrics={"mAP50": bad_value}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["metrics"]["mAP50"]["manual:b"] is None
    assert result["comparable"] is False
    assert any("finite" in w or "not finite" in w for w in result["warnings"])
    assert result["summary"] is None


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_compare_nan_inf_metric_unavailable(tmp_path, bad_value):
    """NaN / ±Infinity 指标 → unavailable，comparable=false，不产生异常。"""
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", metrics={"mAP50": 0.5}))
    svc.index_experiment(_exp("manual:b", metrics={"mAP50": bad_value}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["metrics"]["mAP50"]["manual:b"] is None
    assert result["comparable"] is False
    assert any("finite" in w for w in result["warnings"])


def test_compare_bool_metric_not_finite(tmp_path):
    """布尔值不是有限数值指标，按 unavailable 处理。"""
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", metrics={"mAP50": 0.5}))
    svc.index_experiment(_exp("manual:b", metrics={"mAP50": True}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["metrics"]["mAP50"]["manual:b"] is None
    assert result["comparable"] is False


def test_compare_none_metric_unavailable_no_relative(tmp_path):
    """None 指标保持 null，不参与相对变化，不产生异常。"""
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", metrics={"mAP50": 0.5}))
    svc.index_experiment(_exp("manual:b", metrics={"mAP50": None}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["metrics"]["mAP50"]["manual:b"] is None
    assert result["relative"]["mAP50"]["manual:b"] is None


# ─────────────────────────────────────────────────────────────
# 返修6: 统一路径参数策略 — projection 与 compare 复用 is_path_key
# ─────────────────────────────────────────────────────────────


def test_is_noise_param_extra_path_keys():
    """labels/images/cache 等路径键同样被排除（返修 6 补齐路径键清单）。"""
    for key in ("labels", "images", "cache", "labels_path", "images_path",
                "cache_path", "source_root", "snapshot_path", "manifest_path"):
        assert is_noise_param(key) is True, key


def test_compare_param_diff_excludes_nested_path_params(tmp_path):
    """嵌套路径型参数不能出现在 parameter_diff（common 或 differences）。"""
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", params={
        "epochs": 100,
        "data": "D:/x/data.yaml",
        "nested": {"labels_path": r"E:\data\labels\train", "ok": 1},
    }))
    svc.index_experiment(_exp("manual:b", params={
        "epochs": 100,
        "data": "D:/x/data.yaml",
        "nested": {"labels_path": r"E:\data\labels\train", "ok": 1},
    }))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    params = result["parameters"]
    blob = json.dumps(params)
    assert r"E:\data\labels\train" not in blob
    assert "data" not in json.dumps(params["common"])
    assert "labels_path" not in json.dumps(params["common"])
    assert params["common"]["epochs"] == 100
    # The nested business value that is not path-typed survives.
    assert params["common"].get("nested", {}).get("ok") == 1


def test_compare_model_name_preserved_as_non_path(tmp_path):
    """模型名称等正常非路径值继续保留在 parameter_diff 中。"""
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", params={
        "model": "yolov8n.pt", "epochs": 100}))
    svc.index_experiment(_exp("manual:b", params={
        "model": "yolov8n.pt", "epochs": 100}))

    result = svc.compare_experiments(["manual:a", "manual:b"], "manual:a")

    assert result["parameters"]["common"]["model"] == "yolov8n.pt"
    assert result["parameters"]["common"]["epochs"] == 100
