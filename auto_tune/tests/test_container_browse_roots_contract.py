"""The container browse roots, end to end: the dataset share and ``detect``.

In the container the folder pickers must reach exactly two places: the dataset
share, which the product only ever reads, and ``/opt/auto-tune/detect``, which is
where ``find_detect_dir()`` really writes every training run. Offering the
retired ``/opt/auto-tune/runs`` mount as well would only hide the mismatch
between what the browser offers and where the product writes.

This suite walks the chain the product really runs — the controlled environment
value, the policy it builds, the root listing, browsing a real ``detect/train1``,
``/api/training/analyze-folder``, and the recent-training shortcut's verification
— instead of asserting one link of it in isolation.

The container namespace is a *POSIX* namespace: only a POSIX host can spell
``/opt/auto-tune/detect`` and a real directory with the same string. The
namespace itself is pinned portably by the tests at the top; the whole chain is
then run for real on a POSIX host, and on a Windows host with the same
authoritative-root mechanism spelled in host paths, so every wiring point is
still exercised here.
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.delivery.runtime import CONTAINER_INPUT_ROOTS, INPUT_ALLOWED_ROOTS_ENV
from auto_tune.ui import app as app_mod

SHIPPED_ROOTS = ";".join(CONTAINER_INPUT_ROOTS)

SAMPLE_CSV = (
    "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),"
    "metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n"
    "1,1.5,3.0,2.0,0.1,0.2,0.05,0.01,1.6,3.1,2.1\n"
)


def _client():
    return TestClient(app_mod.app)


def _make_run_dir(parent, name="train1"):
    run_dir = Path(parent) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
    (run_dir / "results.csv").write_text(SAMPLE_CSV, encoding="utf-8")
    return run_dir


def _stub_module_b(monkeypatch):
    """Module B is not what this suite tests; the input gate is."""
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.results_parser.load_training_run",
        lambda d: {"name": Path(d).name,
                   "results": {"total_epochs": 1, "best_epoch": 1}, "args": {}},
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
        lambda r, c: {"best_mAP50": None, "average_mAP50": None,
                      "runs_with_issues": 0, "common_issues": []},
    )
    monkeypatch.setitem(app_mod.APP_CONFIG.setdefault("llm", {}), "enabled", False)
    monkeypatch.setitem(app_mod.APP_CONFIG.setdefault("vision", {}), "enabled", False)


def _use_tmp_log(monkeypatch, tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_mod, "LATEST_DATASET_PATH", log_dir / "latest_dataset.json")
    return log_dir


def _seed_indexed_run(run_dir, run_name="train1"):
    """One completed detect experiment carrying its real run_dir artifact."""
    svc = app_mod._local_index_service()
    svc.initialize()
    svc.index_experiment({
        "run_id": "manual:00000000-0000-4000-8000-000000000001",
        "run_name": run_name,
        "source": "manual",
        "status": "completed",
        "finished_at": "2026-09-28T00:00:00Z",
        "started_at": "2026-09-27T00:00:00Z",
        "params": {"model": "yolov8n.pt", "task": "detect"},
        "metrics": {"mAP50": 0.8},
        "artifacts": {"run_dir": str(run_dir)},
    })
    return svc


# ── the declared namespace: portable, and the same on every host ──


def test_the_shipped_roots_are_the_dataset_share_and_the_detect_directory():
    from auto_tune.delivery.preflight import CONTAINER_APP_ROOT, CONTAINER_DATASETS_DIR

    assert CONTAINER_INPUT_ROOTS == (
        CONTAINER_DATASETS_DIR.as_posix(),
        (CONTAINER_APP_ROOT / "detect").as_posix(),
    )


def test_the_retired_runs_directory_is_never_offered(monkeypatch):
    from auto_tune.delivery.preflight import CONTAINER_APP_ROOT

    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, SHIPPED_ROOTS)
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", {"allowed_roots": []})

    resp = _client().post("/api/browse-folder", json={"path": ""})

    assert resp.status_code == 200
    offered = [entry["path"] for entry in resp.json()["entries"]]
    retired = str(Path((CONTAINER_APP_ROOT / "runs").as_posix()).resolve(strict=False))
    assert retired not in offered
    assert offered == [str(Path(root).resolve(strict=False))
                       for root in CONTAINER_INPUT_ROOTS]


def test_the_environment_roots_bound_the_policy_the_api_builds(monkeypatch):
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, SHIPPED_ROOTS)
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", {"allowed_roots": []})

    policy = app_mod._load_input_policy()

    assert [str(root).replace("\\", "/") for root in policy.allowed_roots] == \
        list(CONTAINER_INPUT_ROOTS)


# ── the whole chain, on the namespace the container really uses ──


@pytest.mark.skipif(
    os.name == "nt",
    reason="a container-absolute path and a real Windows directory cannot be spelled alike",
)
def test_browse_analyze_and_verify_a_real_training_run(tmp_path, monkeypatch):
    """The chain the container runs, with the real container spelling.

    ``/opt/auto-tune/detect/train1`` may be browsed, accepted by the training
    analysis endpoint and offered by the recent-training shortcut, because
    ``detect`` is a declared root. For the same chain spelled in host paths on a
    Windows host, see the test below.
    """
    from auto_tune.delivery import runtime

    datasets = tmp_path / "datasets"
    datasets.mkdir()
    detect = tmp_path / "auto-tune" / "detect"
    run_dir = _make_run_dir(detect)
    _use_tmp_log(monkeypatch, tmp_path)
    _stub_module_b(monkeypatch)

    monkeypatch.setattr(runtime, "CONTAINER_INPUT_ROOTS",
                        (datasets.as_posix(), detect.as_posix()))
    monkeypatch.setenv(
        INPUT_ALLOWED_ROOTS_ENV, f"{datasets.as_posix()};{detect.as_posix()}")
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", {"allowed_roots": []})

    _assert_the_whole_chain(run_dir, datasets, run_name="train1")


@pytest.mark.skipif(
    os.name != "nt",
    reason="the host-path spelling of the same chain is the Windows variant",
)
def test_browse_analyze_and_verify_a_real_training_run_with_host_paths(
    tmp_path, monkeypatch
):
    """The same chain with the authoritative roots spelled as host paths.

    A Windows host cannot spell the container namespace, so the same mechanism —
    an authoritative root set that replaces the configured one — is exercised
    with host directories. What the roots are is pinned by the namespace tests
    above and by the Docker delivery suites.
    """
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    detect = tmp_path / "auto-tune" / "detect"
    run_dir = _make_run_dir(detect)
    _use_tmp_log(monkeypatch, tmp_path)
    _stub_module_b(monkeypatch)

    monkeypatch.setattr(app_mod, "resolve_input_allowed_roots",
                        lambda: (datasets.resolve(), detect.resolve()))
    monkeypatch.setitem(app_mod.APP_CONFIG, "input_safety", {"allowed_roots": []})

    _assert_the_whole_chain(run_dir, datasets, run_name="train1")


def _assert_the_whole_chain(run_dir, datasets, *, run_name):
    client = _client()

    roots = client.post("/api/browse-folder", json={"path": ""})
    assert roots.status_code == 200
    offered = [entry["path"] for entry in roots.json()["entries"]]
    assert offered == [str(Path(datasets).resolve()), str(Path(run_dir).parent.resolve())]

    detect_listing = client.post("/api/browse-folder", json={"path": str(run_dir.parent)})
    assert detect_listing.status_code == 200
    assert run_name in [entry["name"] for entry in detect_listing.json()["entries"]]

    train_listing = client.post("/api/browse-folder", json={"path": str(run_dir)})
    assert train_listing.status_code == 200
    assert train_listing.json()["path"] == str(Path(run_dir).resolve())

    analyzed = client.post("/api/training/analyze-folder", json={"path": str(run_dir)})
    assert analyzed.status_code == 200, analyzed.text
    body = analyzed.json()
    assert body["status"] == "success"
    assert body["input_scan"]["member_count"] >= 2

    svc = _seed_indexed_run(run_dir, run_name=run_name)
    items = svc.recent_training_runs(limit=4, policy=app_mod._load_input_policy())
    assert [item["run_name"] for item in items] == [run_name], (
        "the recent-training shortcut must not filter a run under a declared root")


# ── the dataset share keeps its read-only contract ──


class _RecordingProbes:
    """The start-up's filesystem, recording every call it makes."""

    def __init__(self, info):
        self.calls: list[tuple] = []
        self._info = info

    def lstat(self, path):
        return self._info

    def mkdir(self, path, mode):
        self.calls.append(("mkdir", path, mode))

    def chmod(self, path, mode):
        self.calls.append(("chmod", path, mode))

    def chown(self, path, uid, gid):
        self.calls.append(("chown", path, uid, gid))


def test_the_dataset_share_is_still_only_read_never_repermissioned():
    """Adding ``detect`` as a browse root must not widen how the dataset share is
    treated: it stays an input, and the one privileged step makes no permission
    call on it at all — not even to make it usable."""
    from auto_tune.delivery import container_entrypoint as ce

    share = Path("/data/datasets")
    host_owned = os.stat_result((0o040755, 0, 0, 1, 1000, 1000, 0, 0, 0, 0))
    probes = _RecordingProbes(host_owned)

    created = ce.initialise_directories(
        [share], datasets_directory=share, probes=probes)

    assert created == []
    assert probes.calls == [], "the dataset share is read-only for the product"


def test_a_dataset_under_the_share_is_still_an_acceptable_read_input(monkeypatch, tmp_path):
    """The share stays usable as an input for the dataset analysis."""
    dataset = tmp_path / "datasets" / "ds"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "train" / "img0.jpg").write_bytes(b"xxx")
    _use_tmp_log(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "auto_tune.modules.dataset_analyzer.analyzer.analyze_dataset",
        lambda dir_, yaml_, cfg: {"status": "ok", "quality_score": 0.5},
    )
    monkeypatch.setattr(
        app_mod, "resolve_input_allowed_roots",
        lambda: ((tmp_path / "datasets").resolve(), (tmp_path / "detect").resolve()),
    )

    resp = _client().post("/api/dataset/analyze-folder", json={"path": str(dataset)})

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "success"


def test_a_directory_outside_the_declared_roots_is_still_refused(monkeypatch, tmp_path):
    """A declared root is a *browse* root, not a bypass: the allowed-root check
    still applies to every other directory."""
    _make_run_dir(tmp_path / "detect")
    outside = tmp_path / "outside"
    outside.mkdir()
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    monkeypatch.setattr(
        app_mod, "resolve_input_allowed_roots",
        lambda: (datasets.resolve(), (tmp_path / "detect").resolve()),
    )

    resp = _client().post("/api/training/analyze-folder", json={"path": str(outside)})

    assert resp.status_code == 403
    assert resp.json()["error_code"] == "INPUT_PATH_NOT_ALLOWED"


def test_the_recent_training_shortcut_still_refuses_an_out_of_root_run(tmp_path, monkeypatch):
    """The verification that exposes a run under a declared root also keeps
    refusing one outside them: an unverifiable run_dir is skipped, never shown."""
    _make_run_dir(tmp_path / "detect")
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    monkeypatch.setattr(
        app_mod, "resolve_input_allowed_roots",
        lambda: (datasets.resolve(), (tmp_path / "detect").resolve()),
    )
    outside = _make_run_dir(tmp_path / "outside", name="train9")
    svc = _seed_indexed_run(outside, run_name="train9")

    items = svc.recent_training_runs(limit=4, policy=app_mod._load_input_policy())

    assert items == []
