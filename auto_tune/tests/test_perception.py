"""Bugfix P1.1 — perception report selection and availability contract.

Module C must only select schema-valid Module A / Module B reports, must never
match ``latest_dataset.json`` (a snapshot pointer) as a Module A report, and
must report unavailable facts explicitly instead of disguising them as 0.
"""

import json
import os
import time

from auto_tune.modules.agent_engine.perception import (
    PERCEPTION_DATASET_REPORT_INVALID,
    PERCEPTION_DATASET_REPORT_UNAVAILABLE,
    PERCEPTION_TRAINING_REPORT_INVALID,
    PERCEPTION_TRAINING_REPORT_UNAVAILABLE,
    build_perception,
    find_module_a_report,
    find_module_b_report,
    perception_blocking_error,
    select_module_a_report,
    select_module_b_report,
)


def _module_a_report(total_images=290, total_annotations=128, module="dataset_analyzer", **overrides):
    data = {
        "module": module,
        "version": "1.0",
        "total_images": total_images,
        "total_annotations": total_annotations,
        "label_coverage": {"label_rate": 0.44},
        "class_balance": {"is_balanced": True, "long_tail_classes": [], "imbalance_ratio": 1.0},
        "image_quality": {"blur_ratio": 0.1},
        "bbox_analysis": {"tiny_bbox_ratio": 0.0, "small_bbox_ratio": 0.2},
        "spatial_bias": {"center_concentration_score": 0.0},
        "summary": {"dataset_quality_score": 0.9, "key_issues": []},
    }
    data.update(overrides)
    return data


def _module_b_report(run_name="train53", map50=0.5, module="train_analyzer", **overrides):
    data = {
        "module": module,
        "version": "1.0",
        "runs": {
            run_name: {
                "name": run_name,
                "args": {"epochs": 30, "patience": 20, "lr0": 0.01},
                "results": {"final_metrics": {"metrics/mAP50(B)": map50}},
            }
        },
        "summary": {"best_mAP50": map50, "best_overall_run": run_name},
    }
    data.update(overrides)
    return data


def _write(log_dir, name, payload):
    path = log_dir / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _set_mtime(path, stamp):
    os.utime(path, (stamp, stamp))


# ── Module A selection ──────────────────────────────────────────────────────


def test_latest_dataset_json_newer_still_not_selected(tmp_path):
    """latest_dataset.json is a snapshot pointer, never a Module A report —
    even when it is the most recently modified candidate."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "dataset_report_ds_1.json", _module_a_report())
    latest = _write(log_dir, "latest_dataset.json", {"upload_id": "ds_1", "train_count": 290})
    _set_mtime(latest, time.time() + 1000)

    report, source = select_module_a_report(str(log_dir))
    assert report is not None
    assert report["total_images"] == 290
    assert source["status"] == "available"
    assert source["basename"] == "dataset_report_ds_1.json"


def test_valid_dataset_report_parses_total_images(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "dataset_report_ds_1787809084.json", _module_a_report(total_images=290))

    report, source = select_module_a_report(str(log_dir))
    assert source["status"] == "available"
    assert report["total_images"] == 290
    assert report["total_annotations"] == 128


def test_corrupt_latest_skipped_second_valid_selected(tmp_path):
    """A corrupt newest candidate is skipped; the next valid report is used."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "dataset_report_a.json").write_text("this is not json {", encoding="utf-8")
    _write(log_dir, "dataset_report_b.json", _module_a_report(total_images=290))

    report, source = select_module_a_report(str(log_dir))
    assert source["status"] == "available"
    assert report["total_images"] == 290
    assert source["basename"] == "dataset_report_b.json"


def test_schema_invalid_file_rejected(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "dataset_report_bad.json", {
        "module": "dataset_analyzer",
        "total_images": 290,  # missing summary/class_balance/bbox_analysis
    })

    report, source = select_module_a_report(str(log_dir))
    assert report is None
    assert source["status"] == "schema_invalid"
    assert source["error_code"] == PERCEPTION_DATASET_REPORT_INVALID


def test_all_reports_invalid_returns_unavailable_not_zero(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "dataset_report_bad.json", {"not": "a report"})

    report, source = select_module_a_report(str(log_dir))
    assert report is None
    assert source["error_code"] == PERCEPTION_DATASET_REPORT_INVALID


def test_no_candidates_unavailable(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "latest_dataset.json", {"train_count": 290})

    report, source = select_module_a_report(str(log_dir))
    assert report is None
    assert source["status"] == "unavailable"
    assert source["error_code"] == PERCEPTION_DATASET_REPORT_UNAVAILABLE


def test_real_zero_total_images_is_kept_not_missing(tmp_path):
    """A legal report with total_images=0 is a real fact, not a missing field."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "dataset_report_ds_0.json", _module_a_report(total_images=0, total_annotations=0))

    report, source = select_module_a_report(str(log_dir))
    assert source["status"] == "available"
    assert report["total_images"] == 0
    assert report["total_annotations"] == 0


def test_same_mtime_deterministic_basename_tiebreak(tmp_path):
    """When mtimes are identical the basename order is deterministic."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    a = _write(log_dir, "dataset_report_a.json", _module_a_report(total_images=100))
    b = _write(log_dir, "dataset_report_b.json", _module_a_report(total_images=200))
    stamp = 1787809000.0
    _set_mtime(a, stamp)
    _set_mtime(b, stamp)

    report, source = select_module_a_report(str(log_dir))
    # determinism: the chosen basename is repeatable across calls
    second_report, second_source = select_module_a_report(str(log_dir))
    assert source["basename"] == second_source["basename"]
    assert source["status"] == "available"


def test_mtime_tiebreak_prefers_newer_report(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    old = _write(log_dir, "dataset_report_old.json", _module_a_report(total_images=100))
    new = _write(log_dir, "dataset_report_new.json", _module_a_report(total_images=290))
    _set_mtime(old, 1787809000.0)
    _set_mtime(new, 1787809100.0)

    report, source = select_module_a_report(str(log_dir))
    assert report["total_images"] == 290
    assert source["basename"] == "dataset_report_new.json"


# ── Module B selection ──────────────────────────────────────────────────────


def test_module_b_exact_report_for_reference_run(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "train53_report.json", _module_b_report(run_name="train53", map50=0.477))

    report, source = select_module_b_report(str(log_dir), "train53")
    assert source["status"] == "available"
    assert source["basename"] == "train53_report.json"
    assert "train53" in report["runs"]


def test_newer_report_for_other_run_does_not_override_exact(tmp_path):
    """A newer report for train52 must never shadow the train53 report."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "train52_report.json", _module_b_report(run_name="train52", map50=0.99))
    _write(log_dir, "train53_report.json", _module_b_report(run_name="train53", map50=0.477))

    report, source = select_module_b_report(str(log_dir), "train53")
    assert source["basename"] == "train53_report.json"
    assert "train53" in report["runs"]
    assert "train52" not in report["runs"]


def test_module_b_no_match_returns_unavailable(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "train52_report.json", _module_b_report(run_name="train52", map50=0.99))

    report, source = select_module_b_report(str(log_dir), "train53")
    assert report is None
    assert source["status"] == "unavailable"
    assert source["error_code"] == PERCEPTION_TRAINING_REPORT_UNAVAILABLE


def test_module_b_corrupt_or_mismatched_stable_error(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "train53_report.json").write_text("corrupt {", encoding="utf-8")

    report, source = select_module_b_report(str(log_dir), "train53")
    assert report is None
    assert source["error_code"] == PERCEPTION_TRAINING_REPORT_INVALID


def test_module_b_schema_invalid_rejected(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "train53_report.json", {"module": "train_analyzer", "runs": {}})

    report, source = select_module_b_report(str(log_dir), "train53")
    assert report is None
    assert source["error_code"] == PERCEPTION_TRAINING_REPORT_INVALID


def test_find_module_a_backward_compat_returns_report_or_none(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "dataset_report_ds_1.json", _module_a_report(total_images=290))
    assert find_module_a_report(str(log_dir))["total_images"] == 290

    empty = tmp_path / "empty"
    empty.mkdir()
    assert find_module_a_report(str(empty)) is None


def test_find_module_b_backward_compat(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "train53_report.json", _module_b_report(run_name="train53"))
    assert find_module_b_report("train53", str(log_dir)) is not None
    assert find_module_b_report("train52", str(log_dir)) is None


# ── build_perception availability contract ──────────────────────────────────


def test_build_perception_available_sources(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "dataset_report_ds_1.json", _module_a_report(total_images=290))
    _write(log_dir, "train53_report.json", _module_b_report(run_name="train53", map50=0.477))

    perception = build_perception(log_dir=str(log_dir), reference_run="train53")
    sources = perception["sources"]
    assert sources["dataset_report"]["status"] == "available"
    assert sources["dataset_report"]["basename"] == "dataset_report_ds_1.json"
    assert sources["training_report"]["status"] == "available"
    assert sources["training_report"]["basename"] == "train53_report.json"
    assert perception["dataset"]["total_images"] == 290
    assert perception["training"]["per_run"]["train53"]["mAP50"] == 0.477
    assert perception["training"].get("reference_run") == "train53"


def test_build_perception_unavailable_does_not_fake_zero(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    _write(log_dir, "latest_dataset.json", {"train_count": 290})
    _write(log_dir, "train53_report.json", _module_b_report(run_name="train53"))

    perception = build_perception(log_dir=str(log_dir), reference_run="train53")
    sources = perception["sources"]
    assert sources["dataset_report"]["status"] == "unavailable"
    assert perception["dataset"] == {}  # never {"total_images": 0}
    assert perception["training"]["reference_run"] == "train53"


# ── gate / blocking errors ──────────────────────────────────────────────────


def test_perception_blocking_error_dataset_unavailable():
    perception = {
        "dataset": {},
        "sources": {
            "dataset_report": {"status": "unavailable", "error_code": PERCEPTION_DATASET_REPORT_UNAVAILABLE},
            "training_report": {"status": "available"},
        },
    }
    code, _ = perception_blocking_error(perception)
    assert code == PERCEPTION_DATASET_REPORT_UNAVAILABLE


def test_perception_blocking_error_dataset_invalid():
    perception = {
        "sources": {
            "dataset_report": {"status": "schema_invalid", "error_code": PERCEPTION_DATASET_REPORT_INVALID},
            "training_report": {"status": "available"},
        },
    }
    code, _ = perception_blocking_error(perception)
    assert code == PERCEPTION_DATASET_REPORT_INVALID


def test_perception_blocking_error_training_unavailable():
    perception = {
        "sources": {
            "dataset_report": {"status": "available"},
            "training_report": {"status": "unavailable", "error_code": PERCEPTION_TRAINING_REPORT_UNAVAILABLE},
        },
    }
    code, _ = perception_blocking_error(perception)
    assert code == PERCEPTION_TRAINING_REPORT_UNAVAILABLE


def test_perception_blocking_error_training_invalid():
    perception = {
        "sources": {
            "dataset_report": {"status": "available"},
            "training_report": {"status": "corrupt", "error_code": PERCEPTION_TRAINING_REPORT_INVALID},
        },
    }
    code, _ = perception_blocking_error(perception)
    assert code == PERCEPTION_TRAINING_REPORT_INVALID


def test_perception_blocking_error_none_when_available():
    perception = {
        "sources": {
            "dataset_report": {"status": "available"},
            "training_report": {"status": "available"},
        },
    }
    assert perception_blocking_error(perception) == (None, None)


def test_perception_blocking_error_absent_sources_opt_out():
    """Hand-built perception without the sources contract (unit mocks of
    downstream stages) does not trigger the gate."""
    perception = {"dataset": {"total_images": 10}}
    assert perception_blocking_error(perception) == (None, None)
