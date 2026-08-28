"""Bugfix P2 Task 5: tuning-start confirmation UI and outbound sanitization.

Renders the real single_page.html and asserts the reference-dataset confirmation
block is present, localized through i18n, and free of absolute paths.
"""

from pathlib import Path

from auto_tune.ui.app import _jinja_env
from auto_tune.ui.i18n import make_translator


def _render(lang="zh"):
    translator = make_translator(lang)
    return _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang=lang,
        active_page="dashboard",
        experiment_history=[],
        tuning_history=[],
        dataset=None,
        training={
            "summary": {
                "total_runs_analyzed": 0,
                "best_mAP50": None,
                "best_overall_run": None,
                "average_mAP50": None,
                "runs_with_issues": 0,
            },
            "runs": {},
            "suggestion": None,
        },
        project={},
        latest_suggestion=None,
        current_args=None,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=None,
        vision_analysis=None,
        latest_dataset=None,
        ai_config={},
        csrf_token="",
    )


def test_template_renders_reference_dataset_confirmation_blocks():
    html = _render(lang="zh")
    assert 'id="referenceDatasetInfo"' in html
    assert 'id="referenceDatasetError"' in html
    assert 'id="refDsRun"' in html
    assert 'id="refDsName"' in html
    assert 'id="refDsSnapshot"' in html
    assert 'id="refDsSource"' in html
    # zh labels present
    assert "参考训练" in html
    assert "关联数据集" in html
    assert "解析来源" in html


def test_template_reference_block_localized_in_english():
    html = _render(lang="en")
    assert "Reference Training" in html
    assert "Bound Dataset" in html
    assert "Resolution Source" in html
    assert "Reference dataset bound" in html
    assert "Reference dataset unresolved" in html


def test_template_reference_block_has_no_absolute_paths():
    html = _render(lang="zh")
    # The static confirmation block must never carry snapshot paths.
    block_start = html.index('id="referenceDatasetInfo"')
    block_end = html.index('id="paramChanges"')
    block = html[block_start:block_end]
    assert "dataset_snapshots" not in block
    assert "C:\\" not in block
    assert "/log/" not in block


def test_reference_dataset_public_projection_is_display_safe():
    from auto_tune.modules.reference_dataset import ReferenceDatasetResolution
    from auto_tune.ui.app import _reference_dataset_public

    res = ReferenceDatasetResolution(
        reference_run="train52",
        dataset_id="d" * 64,
        snapshot_id="a" * 64,
        data_yaml_path=Path("C:/secret/log/dataset_snapshots/aaaaaaaa/data.yaml"),
        resolution_source="reference_args",
        dataset_display_name="defect photos",
    )
    proj = _reference_dataset_public(res)
    assert proj["reference_run"] == "train52"
    assert proj["dataset_display_name"] == "defect photos"
    assert proj["snapshot_short_id"] == "a" * 8
    assert proj["resolution_source"] == "reference_args"
    text = str(proj)
    assert "dataset_snapshots" not in text
    assert "C:\\secret" not in text
    assert "data.yaml" not in text
