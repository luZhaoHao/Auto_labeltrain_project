"""Tests for safe archive extraction and the disabled legacy upload APIs."""

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from auto_tune.ui import app as app_mod
from auto_tune.ui.app import _safe_extract_zip


def test_safe_extract_rejects_parent_path(tmp_path):
    """Catches ZIP members escaping the upload directory."""
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "bad")

    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError, match="unsafe"):
        _safe_extract_zip(zf, tmp_path / "output")

    assert not (tmp_path / "escaped.txt").exists()


def test_safe_extract_allows_normal_dataset_files(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dataset/images/train/a.jpg", b"image")
        zf.writestr("dataset/data.yaml", "train: images/train")

    output = tmp_path / "output"
    with zipfile.ZipFile(archive) as zf:
        _safe_extract_zip(zf, output)

    assert (output / "dataset" / "data.yaml").read_text() == "train: images/train"


# ── Task 4: legacy upload APIs are disabled with 410 ──


def _post_upload_with_read_guard(monkeypatch, url, filename, content):
    """POST an upload whose file body must not be read; assert 410 + error code."""
    import starlette.datastructures as sd

    def guard_read(self, *args, **kwargs):
        raise AssertionError("legacy upload handler must not read the request body")

    monkeypatch.setattr(sd.UploadFile, "read", guard_read)
    client = TestClient(app_mod.app)
    resp = client.post(url, files={"file": (filename, content)})
    assert resp.status_code == 410
    data = resp.json()
    assert data["error_code"] == "LEGACY_UPLOAD_DISABLED"
    assert isinstance(data["error"], str) and data["error"]
    return data


def test_dataset_upload_disabled_returns_410_without_reading_body(monkeypatch):
    data = _post_upload_with_read_guard(
        monkeypatch, "/api/dataset/upload", "dataset.zip", b"PK\x03\x04fake-zip"
    )
    assert "目录" in data["error"]


def test_training_analyze_disabled_returns_410_without_reading_body(monkeypatch):
    data = _post_upload_with_read_guard(
        monkeypatch, "/api/training/analyze", "train.zip", b"PK\x03\x04fake-zip"
    )
    assert "目录" in data["error"]


def test_training_analyze_json_disabled_returns_410(monkeypatch):
    data = _post_upload_with_read_guard(
        monkeypatch, "/api/training/analyze", "report.json", b'{"runs": {}}'
    )
    assert "目录" in data["error"]

