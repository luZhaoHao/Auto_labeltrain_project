"""F1.2-C: both adjustment routes must run inside the container delivery layout.

The container gives the Studio six mounted directories and nothing else. These
tests drive the smallest complete HPO flow and the smallest complete LLM
tuning flow over exactly that layout — creation, parameter/decision validation,
one execution attempt, terminal state and artifact association — and assert
that every artifact lands in a mounted directory so a container restart can
still see it.

Training and the LLM are stubbed, as in the module suites; the subject under
test is the delivery layout and its persistence, not training itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from auto_tune.delivery.preflight import (
    bootstrap_config,
    ensure_directories,
    resolve_paths,
)
from auto_tune.modules.agent_engine import loop
from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import (
    Evidence,
    HpoError,
    HpoService,
    ResultInput,
    StudyConfig,
)
from auto_tune.modules.hpo.execution import HpoRunner
from auto_tune.modules.hpo.execution_adapter import CollectedOutcome
from auto_tune.modules.hpo.execution_models import ExecutionConfig, MetricDiagnostics
from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision

TEMPLATE = "project:\n  name: 交付模板\nlocal_index:\n  database_path: log/auto_tune.db\n"


# ── the container layout, as mounted by compose.yaml ────────────────────────


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    """A host directory tree shaped exactly like the six compose mounts."""
    data = tmp_path / "docker-data"
    for name in ("config", "log", "detect", "runs", "models/weights", "datasets"):
        (data / name).mkdir(parents=True, exist_ok=True)
    template = tmp_path / "app" / "auto_tune" / "config.template.yaml"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(TEMPLATE, encoding="utf-8")

    monkeypatch.setenv("AUTO_TUNE_APP_ROOT", str(data))
    monkeypatch.setenv("AUTO_TUNE_CONFIG_PATH", str(data / "config" / "config.yaml"))
    monkeypatch.setenv("AUTO_TUNE_DATASETS_DIR", str(data / "datasets"))
    monkeypatch.setenv("AUTO_TUNE_TEMPLATE_PATH", str(template))

    paths = resolve_paths()
    ensure_directories(paths)
    bootstrap_config(paths, create=True)
    return paths


def _mounted(paths) -> Path:
    return paths.app_root


def _outside_layout(paths, candidate, tmp_path) -> bool:
    roots = [paths.persistent_dirs["config"], paths.persistent_dirs["log"],
             paths.persistent_dirs["detect"], paths.persistent_dirs["runs"],
             paths.persistent_dirs["models/weights"], paths.datasets_dir]
    resolved = Path(candidate).resolve()
    return not any(resolved.is_relative_to(root.resolve()) for root in roots)


# ── HPO: create → validate → execute → terminal state → artifacts ───────────


class _FakeAdapter:
    """Stands in for the training subprocess; writes nothing real."""

    launch_count = 0

    def __init__(self, output_root, log_root):
        self.output_root = Path(output_root)
        self.log_root = Path(log_root)

    def prepare(self, study, trial, config):
        from auto_tune.modules.agent_engine.executor import build_yolo_command

        run_relpath = f"{study.study_id}/{trial.trial_id}"
        effective = {
            "task": "detect", "workers": 0, "resume": False,
            "deterministic": True, "patience": 0, "val": True, "save": True,
            "plots": False, "amp": False,
            "model": study.model_binding.model_path,
            "data": study.snapshot_binding.data_yaml_path,
            "epochs": study.config.epochs,
            "seed": study.config.seed,
            "batch": config.batch, "imgsz": config.imgsz, "device": config.device,
        }
        effective.update(trial.candidate_params)
        run_dir = self.output_root / run_relpath
        command = build_yolo_command(trial.trial_id, str(run_dir / "args.yaml"),
                                     dict(effective))
        # the stub creates the directory the real training would create
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.csv").write_text(
            "epoch, metrics/mAP50-95(B)\n0, 0.7\n", encoding="utf-8")
        return {
            "trial_number": trial.number,
            "trial_id": trial.trial_id,
            "candidate_params": dict(trial.candidate_params),
            "effective_params": effective,
            "command": command,
            "run_relpath": run_relpath,
            "args_sha256": "0" * 64,
        }

    def validate_launch(self, prepared):
        return None

    def launch(self, prepared):
        _FakeAdapter.launch_count += 1

        class _Proc:
            pid = 4242

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        return _Proc()

    def collect(self, study, attempt):
        evidence = Evidence(
            run_id=attempt.run_id,
            artifact_relpath=f"{attempt.run_relpath}/results.csv",
            artifact_sha256="0" * 64, epoch=1)
        return CollectedOutcome(
            result=ResultInput(state="SUCCESS", value=0.7, evidence=evidence),
            diagnostics=MetricDiagnostics(total_rows=1, excluded_rows=0))

    def finalize(self, study, attempt):
        return {"status": "completed"}

    def detect_oom(self, run_dir):
        return False


@pytest.fixture
def hpo_inputs(delivery):
    """A minimal legal dataset in the dataset mount and a weight in the store."""
    source = _mounted(delivery) / "datasets" / "raw"
    source.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        Image.new("RGB", (16, 16)).save(source / f"{index}.jpg")
        (source / f"{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n",
                                             encoding="utf-8")
    snapshot = create_dataset_snapshot(source, _mounted(delivery) / "log" / "snapshots",
                                       val_ratio=0.5, seed=42,
                                       class_names={0: "part"})
    model = delivery.persistent_dirs["models/weights"] / "best.pt"
    model.write_bytes(b"delivery-flow-test-not-a-real-model")
    return snapshot.snapshot_path, model


@pytest.fixture
def hpo(delivery, hpo_inputs, monkeypatch):
    monkeypatch.setattr("auto_tune.modules.hpo.execution.ExecutionAdapter",
                        _FakeAdapter)
    monkeypatch.setattr("auto_tune.modules.hpo.execution._capture_process_identity",
                        lambda pid: f"tok:{pid}")
    _FakeAdapter.launch_count = 0
    snapshot, model = hpo_inputs
    storage = _mounted(delivery) / "log" / "hpo" / "studies"
    service = HpoService(storage)
    runner = HpoRunner(storage, _mounted(delivery) / "detect",
                       _mounted(delivery) / "log")
    return delivery, service, runner, snapshot, model


def test_hpo_study_is_bound_to_the_mounted_snapshot_and_weight(hpo):
    delivery, service, _, snapshot, model = hpo

    study = service.create_study(StudyConfig(budget=1, epochs=1),
                                 snapshot_dir=snapshot, model_path=model)

    assert Path(study.snapshot_binding.snapshot_path).is_relative_to(
        _mounted(delivery))
    assert Path(study.model_binding.model_path).resolve() == model.resolve()
    assert service.load_study(study.study_id).trials == []


def test_hpo_rejects_an_invalid_weight_without_touching_the_layout(hpo):
    delivery, service, _, snapshot, _ = hpo
    bogus = _mounted(delivery) / "models" / "weights" / "notes.txt"
    bogus.write_text("not a weight", encoding="utf-8")

    with pytest.raises(HpoError):
        service.create_study(StudyConfig(budget=1, epochs=1),
                             snapshot_dir=snapshot, model_path=bogus)


def test_hpo_route_reaches_a_terminal_state_with_artifacts_in_the_mounts(hpo, tmp_path):
    delivery, service, runner, snapshot, model = hpo
    study = service.create_study(StudyConfig(budget=1, epochs=1),
                                 snapshot_dir=snapshot, model_path=model)
    runner.prepare(study.study_id, ExecutionConfig(batch=2, imgsz=64,
                                                   device="0", timeout_seconds=120))

    record = runner.run(study.study_id)

    assert record.status == "COMPLETED"
    assert _FakeAdapter.launch_count == 1
    assert all(attempt.phase == "FINALIZED" for attempt in record.attempts)

    loaded = service.load_study(study.study_id)
    assert [trial.state for trial in loaded.trials] == ["SUCCESS"]
    assert loaded.trials[0].result.value == 0.7

    # the objective is bound to the artifact the attempt really produced
    attempt = record.attempts[0]
    artifact = _mounted(delivery) / "detect" / attempt.run_relpath / "results.csv"
    assert artifact.is_file()
    assert loaded.trials[0].result.evidence.artifact_relpath.endswith("results.csv")

    # everything the flow wrote stays inside the mounted directories
    for produced in (_mounted(delivery) / "log" / "hpo" / "studies").rglob("*"):
        assert not _outside_layout(delivery, produced, tmp_path)


def test_hpo_state_survives_a_container_restart(delivery, hpo_inputs, monkeypatch):
    """A new service over the same mounts sees the finished study."""
    monkeypatch.setattr("auto_tune.modules.hpo.execution.ExecutionAdapter",
                        _FakeAdapter)
    monkeypatch.setattr("auto_tune.modules.hpo.execution._capture_process_identity",
                        lambda pid: f"tok:{pid}")
    snapshot, model = hpo_inputs
    storage = _mounted(delivery) / "log" / "hpo" / "studies"
    service = HpoService(storage)
    study = service.create_study(StudyConfig(budget=1, epochs=1),
                                 snapshot_dir=snapshot, model_path=model)
    HpoRunner(storage, _mounted(delivery) / "detect",
              _mounted(delivery) / "log").prepare(
        study.study_id, ExecutionConfig(batch=2, imgsz=64, device="0",
                                        timeout_seconds=120))
    HpoRunner(storage, _mounted(delivery) / "detect",
              _mounted(delivery) / "log").run(study.study_id)

    restarted = HpoService(storage)
    loaded = restarted.load_study(study.study_id)
    record = HpoRunner(storage, _mounted(delivery) / "detect",
                       _mounted(delivery) / "log").status(study.study_id)

    assert [trial.state for trial in loaded.trials] == ["SUCCESS"]
    assert record.status == "COMPLETED"


# ── LLM tuning: decision → validation → execution → terminal state ──────────


def _reference_run(detect: Path, name: str = "train54") -> Path:
    ref = detect / name
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "args.yaml").write_text(
        "model: yolov8n.pt\ndata: test_data.yaml\nlr0: 0.01\nbatch: 16\n"
        "epochs: 100\n", encoding="utf-8")
    (ref / "results.csv").write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), "
        "metrics/mAP50-95(B)\n0, 0.5, 0.4, 0.06, 0.02\n", encoding="utf-8")
    return detect


def _fact_package() -> dict:
    return {
        "schema_version": "1.0",
        "fact_package_id": "sha256:delivery",
        "task": "detect",
        "reference_run": "train54",
        "sources": {
            "dataset_report": "dataset_report_1.json",
            "training_report": "train54_report.json",
            "metrics": "results.csv",
            "params": "args.yaml",
        },
        "facts": [{"fact_id": "training.params.lr0", "value": 0.01,
                   "source": "params"}],
    }


def _controlled_llm_decision() -> dict:
    """The deterministic response the stub LLM returns, in the Q1 contract."""
    return {
        "diagnosis": "stub diagnosis",
        "action": "apply changes",
        "hyperparameter_changes": {"lr0": 0.001},
        "training_overrides": {},
        "raw_response": '{"lr0": 0.001}',
        "error": None,
        "retried": False,
        "schema_version": "1.0",
        "fact_package_id": "sha256:delivery",
        "evidence_ids": {"lr0": ["training.params.lr0"]},
        "validation": {
            "valid": True, "error_code": None, "error_detail": None,
            "retried": False, "referenced_fact_ids": ["training.params.lr0"],
        },
        "semantic_validation": {
            "valid": True, "error_code": None, "reason_code": None,
            "retried": False, "parameters": [],
        },
    }


def test_llm_tuning_route_reaches_a_terminal_state_in_the_mounts(delivery, monkeypatch):
    detect = _reference_run(_mounted(delivery) / "detect")
    llm_calls = []
    launches = []

    monkeypatch.setattr(loop, "find_detect_dir", lambda: str(detect))
    monkeypatch.setattr(loop, "build_perception",
                        lambda **kwargs: {"dataset": {"total_images": 10}})
    monkeypatch.setattr(loop, "build_tuning_fact_package",
                        lambda *a, **k: _fact_package())
    monkeypatch.setattr(
        loop, "decide_hyperparameters",
        lambda *a, **k: llm_calls.append(k.get("config", a)) or _controlled_llm_decision())
    monkeypatch.setattr(loop, "validate_training_preflight", lambda *a, **k: [])
    monkeypatch.setattr(loop, "build_yolo_command",
                        lambda name, args_path, merged: ["python", "-m",
                                                         "ultralytics", "train"])

    class _Proc:
        def poll(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr(loop, "launch_training",
                        lambda *a, **k: launches.append(1) or _Proc())
    monkeypatch.setattr(loop, "monitor_training",
                        lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"))
    monkeypatch.setattr(
        loop, "finalize_training_run",
        lambda run_dir, run_name, source, config, **kw: {
            "run_id": f"tuning:{kw.get('session_id')}:{run_name}",
            "run_name": run_name, "source": "tuning", "status": "completed",
            "analysis_status": "completed",
            "metrics": {"mAP50": 0.08, "mAP50_95": 0.03},
            "epochs": {"configured": 100, "completed": 3, "best": 2},
            "artifacts": {"report_path": None},
            "analysis_error": None, "history_error": None,
            "index_error": None, "error": None,
        })

    result = loop.run_tuning_loop({"probe": {"max_retries": 1}},
                                  reference_run="train54",
                                  log_dir=str(_mounted(delivery) / "log"))

    assert len(llm_calls) == 1, "the delivered route calls the model once"
    assert len(launches) == 1, "the validated decision launches exactly one run"

    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    iteration = audit["iterations"][0]
    assert iteration["decision_validation"]["valid"] is True
    assert iteration["execution"]["actual_params"]["lr0"] == 0.001
    assert audit["status"] == "completed"

    # the audit and the run artifacts live in the mounted log directory
    assert Path(result["audit_path"]).resolve().is_relative_to(
        (_mounted(delivery) / "log").resolve())
    assert Path(result["audit_path"]).is_file()


def test_llm_tuning_route_refuses_a_decision_the_semantics_reject(delivery, monkeypatch):
    detect = _reference_run(_mounted(delivery) / "detect")
    commands = []
    launches = []

    monkeypatch.setattr(loop, "find_detect_dir", lambda: str(detect))
    monkeypatch.setattr(loop, "build_perception",
                        lambda **kwargs: {"dataset": {"total_images": 10}})
    monkeypatch.setattr(loop, "build_tuning_fact_package",
                        lambda *a, **k: _fact_package())
    rejected = _controlled_llm_decision()
    rejected["error"] = "DECISION_SEMANTIC_UNSUPPORTED"
    rejected["hyperparameter_changes"] = {}
    rejected["semantic_validation"] = {
        "valid": False, "error_code": "DECISION_SEMANTIC_UNSUPPORTED",
        "reason_code": "NO_SUPPORTING_RULE", "retried": True, "parameters": [],
    }
    monkeypatch.setattr(loop, "decide_hyperparameters", lambda *a, **k: rejected)
    monkeypatch.setattr(loop, "build_yolo_command",
                        lambda *a, **k: commands.append(1) or ["yolo"])
    monkeypatch.setattr(loop, "launch_training",
                        lambda *a, **k: launches.append(1))

    result = loop.run_tuning_loop({"probe": {"max_retries": 1}},
                                  reference_run="train54",
                                  log_dir=str(_mounted(delivery) / "log"))

    assert commands == [], "a rejected decision never builds a command"
    assert launches == [], "a rejected decision never starts training"
    assert result["failure"]["error_type"] == "decision_semantic_error"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["iterations"][0]["semantic_validation"]["retried"] is True
    assert Path(result["audit_path"]).resolve().is_relative_to(
        (_mounted(delivery) / "log").resolve())


# ── restart: the operator's configuration is never rewritten ────────────────


def test_container_restart_keeps_the_operator_configuration(delivery):
    operator_config = "project:\n  name: 客户项目\nllm:\n  model: deepseek-flash\n"
    delivery.config_path.write_text(operator_config, encoding="utf-8")

    created = bootstrap_config(delivery, create=True)

    assert created is False
    assert delivery.config_path.read_text(encoding="utf-8") == operator_config
