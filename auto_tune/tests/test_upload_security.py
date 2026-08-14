"""Tests for safe archive extraction."""

import io
import zipfile

import pytest

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

