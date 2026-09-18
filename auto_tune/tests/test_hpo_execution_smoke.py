"""H1.2 验收入口 smoke 测试 — 验证 CLI→公共服务映射，不启动真实训练。

普通 pytest 只验证参数映射与非法输入零启动；真实短训练由 Codex 在独立验收时用
显式命令运行 ``auto_tune/scripts/verify_hpo_execution.py``。
"""

import uuid
from pathlib import Path

import pytest
from PIL import Image

from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import HpoService
from auto_tune.modules.hpo.execution_models import ExecutionConfig
from auto_tune.scripts import verify_hpo_execution as verify


class FakeHpoRunner:
    prep_configs = []
    run_calls = 0

    def __init__(self, storage_root, output_root, log_root):
        FakeHpoRunner.storage_root = storage_root
        FakeHpoRunner.output_root = output_root
        FakeHpoRunner.log_root = log_root

    def prepare(self, study_id, config):
        FakeHpoRunner.prep_configs.append((study_id, config))

    def run(self, study_id):
        FakeHpoRunner.run_calls += 1
        from types import SimpleNamespace
        return SimpleNamespace(status="COMPLETED", stop_reason=None)


@pytest.fixture
def reset_fake():
    FakeHpoRunner.prep_configs = []
    FakeHpoRunner.run_calls = 0
    yield
    FakeHpoRunner.prep_configs = []
    FakeHpoRunner.run_calls = 0


@pytest.fixture
def real_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for n in range(4):
        Image.new("RGB", (16, 16)).save(source / f"{n}.jpg")
        (source / f"{n}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    snapshot = create_dataset_snapshot(source, tmp_path / "snapshots",
                                       val_ratio=0.5, seed=42,
                                       class_names={0: "part"})
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"hpo-smoke-test-not-a-real-model")
    return snapshot.snapshot_path, model


def _args(tmp_path, snapshot, model):
    return [
        "--snapshot-dir", str(snapshot),
        "--model-path", str(model),
        "--storage-root", str(tmp_path / "hpo"),
        "--output-root", str(tmp_path / "out"),
        "--log-root", str(tmp_path / "log"),
        "--sampler", "random",
        "--budget", "2",
        "--epochs", "1",
        "--batch", "2",
        "--imgsz", "64",
        "--device", "cpu",
    ]


def test_help_lists_all_arguments(capsys):
    with pytest.raises(SystemExit) as exc:
        verify.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for token in ("--snapshot-dir", "--model-path", "--storage-root",
                  "--output-root", "--log-root", "--sampler", "--budget",
                  "--epochs", "--batch", "--imgsz", "--device"):
        assert token in out


def test_missing_local_paths_zero_launch(tmp_path, monkeypatch, reset_fake):
    monkeypatch.setattr(verify, "HpoRunner", FakeHpoRunner)
    code = verify.main([
        "--snapshot-dir", str(tmp_path / "nope"),
        "--model-path", str(tmp_path / "missing.pt"),
    ])
    assert code == 2
    assert FakeHpoRunner.run_calls == 0


def test_cli_maps_to_public_services(tmp_path, real_inputs, monkeypatch,
                                     reset_fake, capsys):
    snapshot, model = real_inputs
    monkeypatch.setattr(verify, "HpoRunner", FakeHpoRunner)
    argv = _args(tmp_path, snapshot, model)
    code = verify.main(argv)
    # FakeHpoRunner 不产生真实 trial → 无有效结果，CLI 应诚实返回非 0。
    assert code == 1
    assert FakeHpoRunner.run_calls == 1
    assert len(FakeHpoRunner.prep_configs) == 1
    _, exec_config = FakeHpoRunner.prep_configs[0]
    assert isinstance(exec_config, ExecutionConfig)
    assert exec_config.batch == 2
    assert exec_config.imgsz == 64
    assert exec_config.device == "cpu"
    # CLI 创建的 study config（sampler/budget/epochs）与参数一致。
    service = HpoService(tmp_path / "hpo")
    study_id = list((tmp_path / "hpo").glob("hpo_*"))[0].name
    study = service.load_study(study_id)
    assert study.config.sampler == "random"
    assert study.config.budget == 2
    assert study.config.epochs == 1
    captured = capsys.readouterr()
    assert "study_id=" in captured.out or "no" in captured.err.lower()
