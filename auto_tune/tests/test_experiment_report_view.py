"""Bugfix P5: bounded, read-only, run_id-bound training report view projection.

The report view builds a minimal display model from a SQLite experiment (run_id
is the only identity) plus the *registered* report artifact whose content
references the same run_name. It never globs, never calls the LLM/vision
analyzers, never recomputes metrics, never returns full paths/commands/logs/
credentials and never writes any file.
"""

import json
import os
from pathlib import Path

import pytest

from auto_tune.modules.presentation.experiment_views import (
    ArtifactIdentityMismatchError,
    ArtifactTooLargeError,
    ArtifactUnavailableError,
    ReportNotAvailableError,
    build_report_view,
)


def _roots(tmp_path) -> tuple:
    return (str(Path(tmp_path) / "log"),)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _report_payload(run_name="train1", **overrides):
    payload = {
        "module": "train_analyzer",
        "version": "1.0",
        "analysis_timestamp": "2026-08-01T00:10:00Z",
        "detect_dir": "detect/train1",
        "total_runs": 1,
        "runs": {
            run_name: {
                "name": run_name,
                "args": {
                    "model": "yolov8n.pt", "epochs": 50, "batch": 16, "imgsz": 640,
                    "optimizer": "AdamW", "lr0": 0.001, "device": "0",
                    "data": "/secret/ds/data.yaml", "project": "detect",
                },
                "results": {
                    "total_epochs": 50, "best_epoch": 43,
                    "final_metrics": {
                        "metrics/mAP50(B)": 0.8121,
                        "metrics/mAP50-95(B)": 0.3676,
                        "metrics/precision(B)": 0.7769,
                        "metrics/recall(B)": 0.6571,
                    },
                },
                "issues": [
                    {"issue": "overfitting", "severity": "high", "description": "val loss rises late"},
                    {"issue": "plateau", "severity": "medium", "description": "mAP50 flat for 10 epochs"},
                ],
            }
        },
        "summary": {"best_overall_run": run_name},
        "comparison": {"best_run": run_name},
        "llm_analysis": {"diagnosis": "overfitting detected", "action": "reduce lr0"},
        "vision_analysis": {
            "confusion_matrix_analysis": {"analysis": "diagonal is strong"},
            "error_crop_analysis": {"analysis": "small objects missed"},
        },
    }
    payload.update(overrides)
    return payload


def _experiment(run_id="manual:train1", run_name="train1", source="manual",
                report_path=None, status="completed", metrics=None, params=None,
                started_at="2026-08-01T00:00:00Z",
                finished_at="2026-08-01T01:00:00Z", analysis_status="completed",
                artifacts=None, model_name="yolov8n.pt", task_type="detect"):
    default_metrics = {"mAP50": 0.8121, "mAP50_95": 0.3676,
                       "precision": 0.7769, "recall": 0.6571}
    metrics = dict(metrics) if metrics is not None else default_metrics
    params = dict(params or {"model": "yolov8n.pt", "_epochs": {
        "configured": 50, "completed": 50, "best": 43}})
    artifacts = list(artifacts or [])
    if report_path:
        artifacts.append({"kind": "report", "path": report_path, "exists_state": "exists"})
    return {
        "run_id": run_id,
        "run_name": run_name,
        "source": source,
        "status": status,
        "phase": "terminal",
        "model_name": model_name,
        "task_type": task_type,
        "started_at": started_at,
        "finished_at": finished_at,
        "updated_at": finished_at,
        "params": params,
        "metrics": metrics,
        "analysis_status": analysis_status,
        "error": None,
        "artifacts": artifacts,
    }


# ── 1. run_id 精确绑定报告 ──


def test_report_view_bound_by_run_id(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _report_payload("train1"))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert view["run_id"] == "manual:train1"
    assert view["run_name"] == "train1"
    assert view["source"] == "manual"
    assert view["status"] == "completed"
    assert view["task_type"] == "detect"
    assert view["model_name"] == "yolov8n.pt"
    assert view["analysis_status"] == "completed"


# ── 2. 同名/相似 run_name 不串线 ──


def test_report_view_similar_run_names_do_not_cross_wire(tmp_path):
    log_dir = Path(tmp_path) / "log"
    # Only train1's report exists; a different run must not be served its content.
    report_path = _write(log_dir / "train1_report.json", _report_payload("train1"))
    exp = _experiment(run_id="manual:train2", run_name="train2", report_path=report_path)

    with pytest.raises(ArtifactIdentityMismatchError):
        build_report_view(exp, artifact_roots=_roots(tmp_path))


def test_report_view_selects_matching_registered_artifact_among_many(tmp_path):
    log_dir = Path(tmp_path) / "log"
    _write(log_dir / "at_iter01_report.json", _report_payload("autotune_x_iter01"))
    last_report = _write(log_dir / "autotune_x_iter02_report.json", _report_payload("autotune_x_iter02"))
    exp = _experiment(
        run_id="tuning:u1", run_name="autotune_x_iter02", source="tuning",
        report_path=None,
        artifacts=[
            {"kind": "report", "path": str(log_dir / "at_iter01_report.json"), "exists_state": "missing"},
            {"kind": "report", "path": last_report, "exists_state": "exists"},
        ],
    )

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))
    assert view["run_name"] == "autotune_x_iter02"
    assert view["metrics"]["mAP50"] == 0.8121


# ── 3. 不使用 glob ──


def test_report_view_module_does_not_use_glob():
    from auto_tune.modules.presentation import experiment_views as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "glob" not in source


# ── 4/5. 不调用 _ensure_report_llm / _ensure_report_vision ──


def test_report_view_never_calls_llm_or_vision(tmp_path):
    from auto_tune.modules.presentation import experiment_views as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("_ensure_report_llm", "_ensure_report_vision",
                      "llm_analyzer", "vision_analyzer", "multimodal_consult"):
        assert forbidden not in source


# ── 6. 报告文件缺失 ──


def test_report_missing_registered_file_is_honest_error(tmp_path):
    log_dir = Path(tmp_path) / "log"
    missing = str(log_dir / "train1_report.json")
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=missing)

    with pytest.raises(ArtifactUnavailableError):
        build_report_view(exp, artifact_roots=_roots(tmp_path))


def test_report_no_registered_artifact_raises_not_available(tmp_path):
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=None)

    with pytest.raises(ReportNotAvailableError):
        build_report_view(exp, artifact_roots=_roots(tmp_path))


# ── 7. 损坏 JSON ──


def test_report_corrupt_json_is_invalid(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = str(log_dir / "train1_report.json")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train1_report.json").write_text("{ not valid json !!!", encoding="utf-8")
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    with pytest.raises(Exception) as excinfo:
        build_report_view(exp, artifact_roots=_roots(tmp_path))
    assert getattr(excinfo.value, "error_code", "") == "REPORT_INVALID"


# ── 8. 超过 16 MiB ──


def test_report_too_large(tmp_path, monkeypatch):
    from auto_tune.modules.presentation import experiment_views as views_mod

    monkeypatch.setattr(views_mod, "MAX_ARTIFACT_BYTES", 64)
    log_dir = Path(tmp_path) / "log"
    report_path = str(log_dir / "train1_report.json")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train1_report.json").write_text("x" * 200, encoding="utf-8")
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    with pytest.raises(ArtifactTooLargeError):
        build_report_view(exp, artifact_roots=_roots(tmp_path))


# ── 9. 读取期间增长超过上限 ──


def test_report_grows_past_limit_during_read(tmp_path, monkeypatch):
    from auto_tune.modules.presentation import experiment_views as views_mod

    monkeypatch.setattr(views_mod, "MAX_ARTIFACT_BYTES", 64)
    log_dir = Path(tmp_path) / "log"
    report_path = str(log_dir / "train1_report.json")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train1_report.json").write_text("x" * 200, encoding="utf-8")
    # The declared size fast-path guard is bypassed so the chunked stream itself
    # must stop once the accumulated bytes exceed the cap.
    monkeypatch.setattr(views_mod.os.path, "getsize", lambda p: 10)
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    with pytest.raises(ArtifactTooLargeError):
        build_report_view(exp, artifact_roots=_roots(tmp_path))


# ── 10. symlink/reparse 拒绝 ──


def test_report_reparse_point_rejected(tmp_path, monkeypatch):
    from auto_tune.modules.presentation import experiment_views as views_mod

    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _report_payload("train1"))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)
    monkeypatch.setattr(views_mod, "_is_reparse_point", lambda p: True)

    with pytest.raises(ArtifactUnavailableError):
        build_report_view(exp, artifact_roots=_roots(tmp_path))


# ── 11. 越出受控产物根目录拒绝 ──


def test_report_outside_controlled_root_rejected(tmp_path):
    outside = str(tmp_path / "outside" / "train1_report.json")
    _write(Path(outside), _report_payload("train1"))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=outside)

    with pytest.raises(ArtifactUnavailableError):
        build_report_view(exp, artifact_roots=_roots(tmp_path))


# ── 12. 身份不匹配拒绝 ──


def test_report_identity_mismatch_raises(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "other_report.json", _report_payload("other_run"))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    with pytest.raises(ArtifactIdentityMismatchError):
        build_report_view(exp, artifact_roots=_roots(tmp_path))


# ── 13. 只返回最小字段 ──


def test_report_view_returns_minimal_shape(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _report_payload("train1"))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert set(view) == {
        "run_id", "run_name", "source", "status", "task_type", "model_name",
        "dataset", "timing", "metrics", "epochs", "parameters", "issues",
        "ai_analysis", "vision_analysis", "artifacts", "analysis_status",
    }


# ── 14. 不泄漏路径、命令、日志、凭据 ──


def test_report_view_never_leaks_paths_commands_logs_credentials(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _report_payload("train1"))
    exp = _experiment(
        run_id="manual:train1", run_name="train1", report_path=report_path,
        params={"data": "/secret/ds/data.yaml", "model": "yolov8n.pt",
                "_epochs": {"configured": 50, "completed": 50, "best": 43}},
    )

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))
    blob = json.dumps(view)

    # The full dataset path never leaks; "data" is not a core displayed param.
    assert "/secret/" not in blob
    assert "/secret/ds/" not in blob
    assert "data" not in view["parameters"]
    assert "api_key" not in blob.lower()
    assert "apikey" not in blob.lower()
    assert "token" not in blob.lower()
    assert "password" not in blob.lower()
    assert "secret" not in blob.lower()
    assert "command" not in blob.lower()
    assert "traceback" not in blob.lower()


# ── 15. 指标、epoch、参数和 artifact 投影正确 ──


def test_report_view_metrics_epochs_parameters_artifacts(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _report_payload("train1"))
    exp = _experiment(
        run_id="manual:train1", run_name="train1", report_path=report_path,
        artifacts=[
            {"kind": "report", "path": report_path, "exists_state": "exists"},
            {"kind": "best_pt", "path": str(log_dir / "best.pt"), "exists_state": "exists"},
            {"kind": "last_pt", "path": str(log_dir / "last.pt"), "exists_state": "missing"},
        ],
    )

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert view["metrics"]["mAP50"] == 0.8121
    assert view["metrics"]["mAP50_95"] == 0.3676
    assert view["metrics"]["precision"] == 0.7769
    assert view["metrics"]["recall"] == 0.6571
    assert view["epochs"]["configured"] == 50
    assert view["epochs"]["completed"] == 50
    assert view["epochs"]["best"] == 43
    assert view["parameters"]["model"] == "yolov8n.pt"
    assert view["parameters"]["batch"] == 16
    # Path-typed params (data/project) are not core displayed parameters.
    assert "data" not in view["parameters"]
    assert "project" not in view["parameters"]
    kinds = {a["kind"] for a in view["artifacts"]}
    assert "report" in kinds and "best_pt" in kinds and "last_pt" in kinds
    names = {a["name"] for a in view["artifacts"]}
    assert "train1_report.json" in names and "best.pt" in names
    states = {a["kind"]: a["exists_state"] for a in view["artifacts"]}
    assert states["best_pt"] == "exists"
    assert states["last_pt"] == "missing"


def test_report_view_issues_and_ai_analysis(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _report_payload("train1"))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert len(view["issues"]) == 2
    assert view["issues"][0]["issue"] == "overfitting"
    assert view["issues"][0]["severity"] == "high"
    assert view["ai_analysis"]["content_origin"] == "stored"


# ── 16. 缺失字段诚实返回 null/空集合 ──


def test_report_view_missing_fields_are_null_or_empty(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _report_payload(
        "train1", runs={}, llm_analysis=None, vision_analysis=None
    ))
    exp = _experiment(
        run_id="manual:train1", run_name="train1", report_path=report_path,
        metrics={}, params={"model": "yolov8n.pt"},
        started_at=None, finished_at=None,
    )

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert view["metrics"] == {}
    assert view["epochs"]["configured"] is None
    assert view["epochs"]["completed"] is None
    assert view["epochs"]["best"] is None
    assert view["issues"] == []
    assert view["ai_analysis"] is None
    assert view["vision_analysis"] is None
    assert view["timing"]["duration_seconds"] is None
    assert view["dataset"] is None


def test_report_view_dataset_projection(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _report_payload("train1"))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)
    dataset = {
        "dataset_id": "ds_abc", "display_name": "myds", "snapshot_id": "snap9",
        "validation_status": "valid", "canonical_path": "/secret/ds",
    }

    view = build_report_view(exp, dataset=dataset, artifact_roots=_roots(tmp_path))

    assert view["dataset"]["dataset_id"] == "ds_abc"
    assert view["dataset"]["display_name"] == "myds"
    assert view["dataset"]["snapshot_id"] == "snap9"
    assert view["dataset"]["validation_status"] == "valid"
    blob = json.dumps(view)
    assert "/secret/" not in blob
    assert "canonical_path" not in json.dumps(view["dataset"])


def test_report_view_timing_duration(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _report_payload("train1"))
    exp = _experiment(
        run_id="manual:train1", run_name="train1", report_path=report_path,
        started_at="2026-08-01T00:00:00Z", finished_at="2026-08-01T00:30:00Z",
    )

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert view["timing"]["started_at"] == "2026-08-01T00:00:00Z"
    assert view["timing"]["finished_at"] == "2026-08-01T00:30:00Z"
    assert abs(view["timing"]["duration_seconds"] - 1800.0) < 1e-6


def test_report_view_parameters_are_whitelisted_only(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _report_payload(
        "train1", runs={"train1": {
            "name": "train1",
            "args": {"model": "yolov8n.pt", "epochs": 50, "batch": 16, "imgsz": 640,
                     "optimizer": "AdamW", "lr0": 0.001, "device": "0",
                     "cache": True, "half": True, "verbose": True, "workers": 8},
            "results": {"total_epochs": 50, "best_epoch": 43},
            "issues": [],
        }}
    ))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert set(view["parameters"]) == {"model", "epochs", "batch", "imgsz",
                                       "optimizer", "lr0", "device"}


# ── 返修：训练问题字段映射（生产 type/detail，兼容旧 issue/description）──


def _issues_report(run_name="train1", issues=()):
    """Report modeled on the real production schema (type/severity/detail)."""
    return _report_payload(run_name, runs={run_name: {
        "name": run_name,
        "args": {"model": "yolov8n.pt", "epochs": 50, "batch": 16},
        "results": {"total_epochs": 50, "best_epoch": 43},
        "issues": list(issues),
    }})


def test_report_issues_uses_production_type_detail_schema(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _issues_report(
        "train1",
        issues=[
            {"type": "overfitting", "severity": "medium", "detail": "Validation loss is rising."},
        ],
    ))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert view["issues"] == [
        {"issue": "overfitting", "severity": "medium", "description": "Validation loss is rising."},
    ]


def test_report_issues_backward_compat_issue_description(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _issues_report(
        "train1",
        issues=[
            {"issue": "overfitting", "severity": "high", "description": "old format"},
        ],
    ))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert view["issues"] == [
        {"issue": "overfitting", "severity": "high", "description": "old format"},
    ]


def test_report_issues_mixed_prefers_type_detail(tmp_path):
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _issues_report(
        "train1",
        issues=[
            {"type": "plateau", "severity": "low", "detail": "saturated",
             "issue": "old_type", "description": "old_desc"},
        ],
    ))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    # Production type/detail win when both are present.
    assert view["issues"] == [
        {"issue": "plateau", "severity": "low", "description": "saturated"},
    ]


def test_report_issues_real_production_schema_contract(tmp_path):
    """Read-only contract against the real autotune_52580523_iter02 report issue
    schema (type/severity/detail), reproduced as an equivalent fixture so the
    test is stable even when the log/ artifact is absent."""
    log_dir = Path(tmp_path) / "log"
    report_path = _write(log_dir / "train1_report.json", _issues_report(
        "train1",
        issues=[
            {"type": "overfitting", "severity": "medium",
             "detail": "Validation loss is rising while training loss descends."},
            {"type": "plateau", "severity": "low",
             "detail": "mAP50 has saturated — further training unlikely to improve."},
            {"type": "unstable_training", "severity": "medium",
             "detail": "Validation loss shows rising trend — potential divergence."},
        ],
    ))
    exp = _experiment(run_id="manual:train1", run_name="train1", report_path=report_path)

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert [i["issue"] for i in view["issues"]] == [
        "overfitting", "plateau", "unstable_training",
    ]
    assert [i["severity"] for i in view["issues"]] == ["medium", "low", "medium"]
    # Original detail text is preserved verbatim (no translation).
    assert view["issues"][2]["description"] == \
        "Validation loss shows rising trend — potential divergence."


def test_report_issues_real_file_read_only_contract(tmp_path, monkeypatch):
    """When the real production report exists in log/, it is a read-only source:
    the projection never rewrites it and maps its production issue schema."""
    import os as real_os
    from pathlib import Path as RealPath

    real_file = RealPath("log") / "autotune_52580523_iter02_report.json"
    if not real_file.is_file():
        pytest.skip("real production report artifact not present")

    log_dir = Path(tmp_path) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(real_file), str(log_dir / "train1_report.json"))
    exp = _experiment(
        run_id="manual:train1", run_name="autotune_52580523_iter02",
        report_path=str(log_dir / "train1_report.json"),
    )

    view = build_report_view(exp, artifact_roots=_roots(tmp_path))

    assert view["issues"] and all("type" not in i for i in view["issues"])
    assert {i["issue"] for i in view["issues"]} == {
        "overfitting", "plateau", "unstable_training",
    }
    assert all(i["severity"] in ("low", "medium", "high") for i in view["issues"])
    assert all(i["description"] for i in view["issues"])
