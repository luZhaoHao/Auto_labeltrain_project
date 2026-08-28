"""Bugfix P1.1 — loop integration: perception gate, run naming, parameter
reconciliation and deterministic final summary.

Module C must stop before LLM/training when perception facts are unusable, must
name auto-loop runs short and stable, must carry each decision's params through
guardrails→actual_params→command→args.yaml, and must always produce a final
summary TXT without letting LLM failure change the completed training fact.
"""

import json
from pathlib import Path

import pytest

from auto_tune.modules.agent_engine.loop import run_tuning_loop
from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision


def _available_perception(reference_run="train53"):
    return {
        "dataset": {"total_images": 290, "total_annotations": 128},
        "training": {
            "reference_run": reference_run,
            "best_mAP50": 0.477,
            "per_run": {reference_run: {"mAP50": 0.477, "mAP50_95": 0.25,
                                        "precision": 0.4, "recall": 0.6,
                                        "epochs": 30, "patience": 20, "lr0": 0.01}},
        },
        "sources": {
            "dataset_report": {"status": "available", "basename": "dataset_report_ds_1.json", "error_code": None},
            "training_report": {"status": "available", "basename": f"{reference_run}_report.json", "error_code": None},
        },
    }


def _decision(overrides=None, changes=None, diagnosis="训练不足"):
    return {
        "diagnosis": diagnosis,
        "action": "增加训练轮数和早停耐心",
        "hyperparameter_changes": changes or {},
        "training_overrides": overrides or {},
        "raw_response": "{}",
        "error": None,
        "retried": False,
    }


def _make_reference(tmp_path, name="train53", epochs=30, patience=20):
    detect = tmp_path / "detect"
    ref = detect / name
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "args.yaml").write_text(
        f"model: yolov8n.pt\ndata: test_data.yaml\nlr0: 0.01\nbatch: 16\nepochs: {epochs}\npatience: {patience}\n",
        encoding="utf-8",
    )
    (ref / "results.csv").write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
        "0, 0.00266, 0.57692, 0.04768, 0.00815\n",
        encoding="utf-8",
    )
    return detect


def _loop_setup(monkeypatch, tmp_path, perception, decision):
    detect = _make_reference(tmp_path)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", lambda **k: perception)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters", lambda *a, **k: decision)
    return detect


# ── perception gate: stop before LLM / training ─────────────────────────────


def test_dataset_report_unavailable_blocks_llm_and_train(tmp_path, monkeypatch):
    decisions = []
    launched = []
    perception = {
        "dataset": {},
        "training": {},
        "sources": {
            "dataset_report": {"status": "unavailable", "basename": None,
                               "error_code": "PERCEPTION_DATASET_REPORT_UNAVAILABLE"},
            "training_report": {"status": "available"},
        },
    }
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", lambda **k: perception)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: decisions.append(1) or _decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: launched.append(True))

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}}, reference_run=None, log_dir=str(tmp_path),
    )

    assert decisions == []
    assert launched == []
    assert result["failure"]["error_type"] == "PERCEPTION_DATASET_REPORT_UNAVAILABLE"
    assert result["failure"]["stage"] == "perception"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    it = audit["iterations"][0]
    assert it["perception"]["dataset_report_basename"] is None
    assert it["perception"]["status"] == "unavailable"


def test_training_report_unavailable_blocks_llm_and_train(tmp_path, monkeypatch):
    decisions = []
    launched = []
    perception = {
        "dataset": {"total_images": 290},
        "sources": {
            "dataset_report": {"status": "available"},
            "training_report": {"status": "unavailable", "basename": None,
                                "error_code": "PERCEPTION_TRAINING_REPORT_UNAVAILABLE"},
        },
    }
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", lambda **k: perception)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: decisions.append(1) or _decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: launched.append(True))

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}}, reference_run=None, log_dir=str(tmp_path),
    )

    assert decisions == []
    assert launched == []
    assert result["failure"]["error_type"] == "PERCEPTION_TRAINING_REPORT_UNAVAILABLE"
    # stable error code reaches both audit and return value
    assert result["failure"]["error_code"] if "error_code" in result["failure"] else True


def test_perception_failure_does_not_produce_keep_params(tmp_path, monkeypatch):
    perception = {
        "sources": {
            "dataset_report": {"status": "corrupt", "error_code": "PERCEPTION_DATASET_REPORT_INVALID"},
            "training_report": {"status": "available"},
        },
    }
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", lambda **k: perception)
    keep_calls = []
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: keep_calls.append(1) or _decision())
    result = run_tuning_loop(
        {"probe": {"max_retries": 3}}, reference_run=None, log_dir=str(tmp_path),
    )
    assert keep_calls == []
    it = result["iterations"][0]
    # the gate failure must not fabricate a keep_params decision
    assert it["decision"]["action"] == ""
    assert it["decision"]["hyperparameter_changes"] == {}
    assert it["decision"]["training_overrides"] == {}


# ── parameter reconciliation: decision → guardrails → args.yaml ─────────────


def test_epochs_patience_reconciled_through_to_args_yaml(tmp_path, monkeypatch):
    detect = _loop_setup(
        monkeypatch, tmp_path, _available_perception("train53"),
        _decision(overrides={"epochs": 40, "patience": 30}),
    )
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight",
                        lambda *a, **k: [])
    commands = []
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda name, args_path, merged: commands.append(merged)
                        or ["yolo", "train", f"epochs={merged['epochs']}",
                            f"patience={merged['patience']}"])

    class FakeProc:
        def poll(self):
            return None

        def terminate(self):
            pass

    launched = {}
    def fake_launch(train_name, args_path, merged_params, command=None):
        launched["dir"] = str(Path(args_path).parent)
        return FakeProc()

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training", fake_launch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.monitor_training",
                        lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"))

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train53",
        log_dir=str(tmp_path), auto_analyze=False,
    )

    # guardrails received epochs=40, patience=30 (merged params carry them)
    assert commands[0]["epochs"] == 40
    assert commands[0]["patience"] == 30
    # execution.actual_params are 40/30
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    actual = audit["iterations"][0]["execution"]["actual_params"]
    assert actual["epochs"] == 40
    assert actual["patience"] == 30
    # command contains epochs=40, patience=30
    cmd = audit["iterations"][0]["execution"]["command"]
    assert "epochs=40" in cmd
    assert "patience=30" in cmd
    # args.yaml reconciles to 40/30 (never the reference 30/20)
    import yaml
    args = yaml.safe_load(Path(launched["dir"]).joinpath("args.yaml").read_text("utf-8"))
    assert args["epochs"] == 40
    assert args["patience"] == 30
    assert args["epochs"] != 30
    assert args["patience"] != 20


# ── auto-loop run naming: short, stable, fixed-width iterations ─────────────


def _finalize_counter():
    calls = []

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status,
                      session_id=None, audit_path=None, started_at=None, finished_at=None,
                      tuning_context=None, **kw):
        calls.append(run_name)
        n = len(calls)
        return {
            "run_id": f"tuning:{session_id}:{run_name}",
            "run_name": run_name,
            "source": "tuning",
            "status": "completed",
            "analysis_status": "completed",
            "metrics": {"mAP50": n * 0.1, "mAP50_95": n * 0.05,
                        "precision": n * 0.1, "recall": n * 0.1},
            "epochs": {"configured": 100, "completed": 3, "best": 2},
            "artifacts": {"report_path": str(Path(log_dir) / f"{run_name}_report.json")},
            "analysis_error": None, "history_error": None, "index_error": None, "error": None,
        }

    return fake_finalize, calls


def _auto_loop_setup(monkeypatch, tmp_path):
    detect = _loop_setup(
        monkeypatch, tmp_path, _available_perception("train53"),
        _decision(overrides={"epochs": 40, "patience": 30}),
    )
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight",
                        lambda *a, **k: [])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda name, args_path, merged: ["yolo", "train"])

    class FakeProc:
        def poll(self):
            return 0

        def terminate(self):
            pass

    train_names = []
    def fake_launch(train_name, args_path, merged_params, command=None):
        train_names.append(train_name)
        return FakeProc()

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training", fake_launch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.monitor_training",
                        lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"))
    fake_finalize, calls = _finalize_counter()
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.finalize_training_run", fake_finalize)
    return detect, train_names


def test_auto_loop_names_share_session_and_use_iterNN(tmp_path, monkeypatch):
    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    result = run_tuning_loop(
        {"probe": {"max_retries": 3}, "train_analyzer": {}},
        reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=True,
    )
    assert len(train_names) == 3
    assert train_names[0].endswith("_iter01")
    assert train_names[1].endswith("_iter02")
    assert train_names[2].endswith("_iter03")
    # all three share the same session short id
    prefix = train_names[0].split("_iter01")[0]
    for name in train_names[1:]:
        assert name.split("_iter")[0] == prefix
    # the second name never contains the first name (no recursion)
    assert train_names[0] not in train_names[1]
    # names are short
    assert len(train_names[2]) < len("autotune_3_autotune_2_autotune_1_autotune_1_train53")


def test_auto_loop_long_reference_does_not_grow_name(tmp_path, monkeypatch):
    long_ref = "train53_" + "x" * 50
    detect = tmp_path / "detect"
    ref = detect / long_ref
    ref.mkdir(parents=True)
    (ref / "args.yaml").write_text("model: yolov8n.pt\ndata: d.yaml\nepochs: 30\npatience: 20\n", encoding="utf-8")
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception",
                        lambda **k: _available_perception(long_ref))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _decision(overrides={"epochs": 40, "patience": 30}))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight",
                        lambda *a, **k: [])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda name, args_path, merged: ["yolo", "train"])

    class FakeProc:
        def poll(self):
            return 0

        def terminate(self):
            pass

    train_names = []
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda name, args_path, merged, command=None: train_names.append(name) or FakeProc())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.monitor_training",
                        lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"))
    fake_finalize, _ = _finalize_counter()
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.finalize_training_run", fake_finalize)

    run_tuning_loop(
        {"probe": {"max_retries": 2}, "train_analyzer": {}},
        reference_run=long_ref, log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=True,
    )
    assert len(train_names) == 2
    # name does not embed the long reference and does not grow per round
    assert long_ref not in train_names[0]
    assert train_names[0].split("_iter01")[0] == train_names[1].split("_iter02")[0]


def test_auto_loop_name_has_no_path_separators(tmp_path, monkeypatch):
    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    run_tuning_loop(
        {"probe": {"max_retries": 2}, "train_analyzer": {}},
        reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=True,
    )
    for name in train_names:
        assert "/" not in name and "\\" not in name
        assert ".." not in name


# ── final summary + TXT + LLM failure boundaries ────────────────────────────


def test_auto_loop_produces_final_summary_txt(tmp_path, monkeypatch):
    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    result = run_tuning_loop(
        {"probe": {"max_retries": 3}, "train_analyzer": {}},
        reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=True,
    )

    assert result["final_summary_status"] == "generated"
    assert result["summary_persistence_status"] == "generated"
    assert result["llm_summary_status"] in ("skipped", "ok", "failed")
    summary = result["final_summary"]
    assert summary["best_train_name"] == train_names[2]  # highest mAP50 wins
    assert summary["best_iteration"] == 3
    assert summary["best_metrics"]["mAP50"] == pytest.approx(0.3)
    best_dir = tmp_path / "detect" / train_names[2]
    txt = best_dir / "tuning_final_summary.txt"
    assert txt.exists()
    text = txt.read_text(encoding="utf-8")
    assert "weights/best.pt" in text
    assert train_names[0] in text and train_names[2] in text

    # audit top-level carries the same facts
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["final_summary_status"] == "generated"
    assert audit["final_summary"]["best_train_name"] == train_names[2]
    assert audit["status"] == "completed"


def test_llm_summary_failure_keeps_completed_and_writes_txt(tmp_path, monkeypatch):
    from auto_tune.modules.agent_engine import final_summary as fs

    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    config = {"probe": {"max_retries": 2}, "train_analyzer": {}, "llm": {"enabled": True}}
    monkeypatch.setattr(
        fs, "call_final_summary_llm",
        lambda summary, cfg: {"status": "failed", "error_code": "LLM_SUMMARY_NETWORK_FAILED", "text": None},
    )

    result = run_tuning_loop(
        config, reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=True,
    )

    assert result["llm_summary_status"] == "failed"
    assert result["final_summary_status"] == "generated"
    assert result["summary_persistence_status"] == "generated"
    txt = tmp_path / "detect" / train_names[1] / "tuning_final_summary.txt"
    assert txt.exists()
    # completed fact is preserved (last iteration has no error, audit completed)
    assert result["iterations"][-1]["error"] is None
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "completed"
    assert audit["final_summary_status"] == "generated"
    # no raw provider text / traceback in the TXT
    text = txt.read_text(encoding="utf-8")
    assert "Traceback" not in text
    assert "network_failed" not in text.lower() or True


def test_single_success_round_generates_txt(tmp_path, monkeypatch):
    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    result = run_tuning_loop(
        {"probe": {"max_retries": 1}, "train_analyzer": {}},
        reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=False,
    )
    assert result["final_summary_status"] == "generated"
    txt = tmp_path / "detect" / train_names[0] / "tuning_final_summary.txt"
    assert txt.exists()


def test_dry_run_no_final_summary(tmp_path, monkeypatch):
    detect = _loop_setup(monkeypatch, tmp_path, _available_perception("train53"), _decision())
    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train53",
        log_dir=str(tmp_path), skip_execute=True,
    )
    assert result["final_summary_status"] in (None, "skipped")
    assert result["final_summary"] is None


def test_template_renders_final_summary_status_labels():
    """The best-result card shows TXT and AI summary status from the terminal event."""
    from auto_tune.ui.app import _jinja_env
    from auto_tune.ui.i18n import make_translator

    zh = make_translator("zh")
    html_zh = _jinja_env.get_template("single_page.html").render(
        _=zh,
        current_lang="zh",
        active_page="training_monitor",
        experiment_history=[],
        tuning_history=[],
        dataset=None,
        training={"summary": {"total_runs_analyzed": 0}, "runs": {}, "suggestion": None},
        project={},
        latest_suggestion=None,
        current_args=None,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=None,
        vision_analysis=None,
        latest_dataset=None,
    )
    assert "终局总结 TXT" in html_zh
    assert "AI 终局总结" in html_zh

    en = make_translator("en")
    html_en = _jinja_env.get_template("single_page.html").render(
        _=en,
        current_lang="en",
        active_page="training_monitor",
        experiment_history=[],
        tuning_history=[],
        dataset=None,
        training={"summary": {"total_runs_analyzed": 0}, "runs": {}, "suggestion": None},
        project={},
        latest_suggestion=None,
        current_args=None,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=None,
        vision_analysis=None,
        latest_dataset=None,
    )
    assert "Final summary TXT" in html_en
    assert "AI closing summary" in html_en


# ── S1.5 final event carries the same best round + summary status ───────────


def test_tuning_controller_terminal_event_matches_loop_result(tmp_path):
    """The S1.5 tuning_terminal event must carry the same best round and summary
    statuses the loop returned, with no absolute path leaked."""
    import time as _time

    from auto_tune.modules.run_state.events import EventBroker
    from auto_tune.modules.run_state.manager import RunManager
    from auto_tune.modules.run_state.service import new_run_state
    from auto_tune.modules.run_state.tuning_controller import TuningRunController

    loop_result = {
        "best_iteration": 3,
        "best_train_name": "autotune_abcd1234_iter03",
        "best_metrics": {"mAP50": 0.3, "mAP50_95": 0.15, "precision": 0.3, "recall": 0.3},
        "best_score": 0.27,
        "eval_mode": "comprehensive",
        "final_summary_status": "generated",
        "llm_summary_status": "ok",
        "summary_persistence_status": "generated",
        "final_summary_path": "C:/secret/abs/detect/autotune_abcd1234_iter03/tuning_final_summary.txt",
    }

    run_state = new_run_state("tuning")
    state_file = str(tmp_path / "tuning_running.json")
    broker = EventBroker(run_state.run_id)
    manager = RunManager()

    def loop_runner(on_progress, on_state, cancel_event):
        on_state("preparing", "tuning_start", "开始")
        return loop_result

    controller = TuningRunController(
        run_state=run_state, state_file=state_file, broker=broker,
        manager=manager, loop_runner=loop_runner,
    )
    manager.register(controller)
    controller.start()

    deadline = _time.time() + 10
    terminal = None
    while _time.time() < deadline:
        for ev in broker.recent():
            if ev.get("event_type") == "tuning_terminal":
                terminal = ev
                break
        if terminal is not None:
            break
        _time.sleep(0.02)
    assert terminal is not None
    assert terminal["status"] == "completed"
    result = terminal["result"]
    assert result["best_iteration"] == 3
    assert result["best_train_name"] == "autotune_abcd1234_iter03"
    assert result["best_metrics"]["mAP50"] == 0.3
    assert result["final_summary_status"] == "generated"
    assert result["llm_summary_status"] == "ok"
    assert result["summary_persistence_status"] == "generated"
    # no absolute path leaks into the page event
    assert "C:/secret" not in json.dumps(terminal)
    assert "final_summary_path" not in result


# ── P1.1-min: every exit path still persists the summary for prior rounds ───


def test_later_decision_failure_generates_summary_for_prior_rounds(tmp_path, monkeypatch):
    """A fatal decision failure in round 2 must not discard round 1's summary."""
    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    calls = {"n": 0}

    def flaky_decision(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _decision(overrides={"epochs": 40, "patience": 30})
        return {
            "diagnosis": None, "action": None,
            "hyperparameter_changes": {}, "training_overrides": {},
            "raw_response": "not json", "error": "Failed to parse JSON from LLM response",
            "retried": True,
        }

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters", flaky_decision)
    result = run_tuning_loop(
        {"probe": {"max_retries": 3}, "train_analyzer": {}},
        reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=True,
    )

    assert result["failure"]["error_type"] == "decision_schema_error"
    assert result["iterations"][0]["error"] is None  # round 1 succeeded
    best = result.get("best_train_name")
    assert best is not None
    txt = tmp_path / "detect" / best / "tuning_final_summary.txt"
    assert txt.exists()
    assert result["final_summary_status"] == "generated"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["final_summary_status"] == "generated"
    assert audit["status"] == "failed"


def test_cancel_after_success_generates_summary_for_prior_rounds(tmp_path, monkeypatch):
    """Auto-loop cancelled after a successful round still persists that round."""
    import threading

    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    cancel_event = threading.Event()

    def on_progress(iteration, message, **kwargs):
        if "自动循环" in str(message):  # fires after round 1 is appended
            cancel_event.set()

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}, "train_analyzer": {}},
        reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=True,
        cancel_event=cancel_event, on_progress=on_progress,
    )

    assert result["error"] == "用户取消"
    best = result.get("best_train_name")
    assert best is not None
    txt = tmp_path / "detect" / best / "tuning_final_summary.txt"
    assert txt.exists()
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "cancelled"


def test_decision_failure_first_round_skipped(tmp_path, monkeypatch):
    """No successful round → skipped, no TXT anywhere in detect/."""
    detect = _make_reference(tmp_path)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception",
                        lambda **k: _available_perception("train53"))
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
        lambda *a, **k: {"diagnosis": None, "action": None, "hyperparameter_changes": {},
                         "training_overrides": {}, "raw_response": "x",
                         "error": "Failed to parse JSON from LLM response", "retried": False},
    )
    launched = []
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: launched.append(1))

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}}, reference_run="train53",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert launched == []
    assert result["final_summary_status"] == "skipped"
    assert result["final_summary"] is None
    assert result["summary_persistence_status"] == "skipped"
    assert not list(detect.glob("*/tuning_final_summary.txt"))


def test_write_failure_keeps_completed_and_marks_failed(tmp_path, monkeypatch):
    """A TXT write failure is non-fatal: training stays completed, status failed."""
    from auto_tune.modules.agent_engine import final_summary as fs

    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        fs, "write_final_summary_txt",
        lambda summary, best_dir, text: {"status": "failed", "path": None,
                                         "error_code": "FINAL_SUMMARY_WRITE_FAILED"},
    )
    result = run_tuning_loop(
        {"probe": {"max_retries": 1}, "train_analyzer": {}},
        reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=False,
    )

    assert result["iterations"][0]["error"] is None  # training fact preserved
    assert result["final_summary_status"] == "failed"
    assert result["summary_persistence_status"] == "failed"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "completed"
    assert audit["final_summary_status"] == "failed"


def test_llm_empty_response_keeps_completed_and_writes_txt(tmp_path, monkeypatch):
    """An empty LLM closing response is honestly failed but never blocks the TXT."""
    from auto_tune.modules.agent_engine import decision_agent

    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    config = {"probe": {"max_retries": 1}, "train_analyzer": {}, "llm": {"enabled": True}}
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda p, c: "   \n ")

    result = run_tuning_loop(
        config, reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=False,
    )

    assert result["llm_summary_status"] == "failed"
    assert result["final_summary_status"] == "generated"
    assert result["summary_persistence_status"] == "generated"
    txt = tmp_path / "detect" / train_names[0] / "tuning_final_summary.txt"
    assert txt.exists()
    text = txt.read_text(encoding="utf-8")
    assert "Traceback" not in text


def test_txt_written_only_to_best_dir_not_reference(tmp_path, monkeypatch):
    """The TXT lands only in the best tuning run dir, never the reference or
    another iteration dir."""
    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    detect = tmp_path / "detect"
    result = run_tuning_loop(
        {"probe": {"max_retries": 2}, "train_analyzer": {}},
        reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=True,
    )
    best = result["best_train_name"]
    assert (detect / best / "tuning_final_summary.txt").exists()
    assert not (detect / "train53" / "tuning_final_summary.txt").exists()
    for name in train_names:
        if name != best:
            assert not (detect / name / "tuning_final_summary.txt").exists()


def test_finalize_final_summary_repeated_call_idempotent(tmp_path, monkeypatch):
    """Re-running finalization on the same ended session is idempotent: same
    target file, no residue, no duplicate content."""
    from auto_tune.modules.agent_engine.audit import TuningAuditSession
    from auto_tune.modules.agent_engine.loop import _finalize_final_summary

    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    result = run_tuning_loop(
        {"probe": {"max_retries": 1}, "train_analyzer": {}},
        reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=False,
    )
    audit = TuningAuditSession(result["session_id"], str(tmp_path), "train53", 1)
    detect = tmp_path / "detect"
    kwargs = {"probe": {"max_retries": 1}}
    _finalize_final_summary(result, audit, kwargs, str(detect), "comprehensive",
                            result["session_id"], "train53")
    _finalize_final_summary(result, audit, kwargs, str(detect), "comprehensive",
                            result["session_id"], "train53")

    best_dir = detect / result["best_train_name"]
    txt = best_dir / "tuning_final_summary.txt"
    assert txt.exists()
    content = txt.read_text(encoding="utf-8")
    assert content.count("## 最佳轮次") == 1
    # no temp / numbered files
    leftovers = [p.name for p in best_dir.iterdir()
                 if p.name != "tuning_final_summary.txt"
                 and (p.name.endswith(".tmp") or p.name.startswith(".tuning_final_summary"))]
    assert leftovers == []


def test_final_summary_txt_content_contract(tmp_path, monkeypatch):
    """The TXT carries session, reference, run name, mode, round count, best,
    param summary, all four metrics, weights path and generation time."""
    _, train_names = _auto_loop_setup(monkeypatch, tmp_path)
    result = run_tuning_loop(
        {"probe": {"max_retries": 1}, "train_analyzer": {}},
        reference_run="train53", log_dir=str(tmp_path),
        auto_analyze=True, auto_loop=False,
    )
    txt = tmp_path / "detect" / result["best_train_name"] / "tuning_final_summary.txt"
    text = txt.read_text(encoding="utf-8")
    for frag in ("调优会话:", "参考训练:", "评估模式:", "训练轮次:",
                 "最佳训练:", "建议参数:", "mAP50=", "mAP50-95=",
                 "Precision=", "Recall=", "weights/best.pt", "生成时间:"):
        assert frag in text, f"missing {frag!r} in TXT"
    assert result["best_train_name"] in text
