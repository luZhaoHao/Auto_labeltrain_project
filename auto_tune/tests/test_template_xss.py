"""Server-side template XSS tests for the S1.3 AI service settings block.

Renders the real single_page.html through the app's Jinja2 environment and
asserts that customer-editable config values are HTML-escaped inside
attributes and that the CSRF token is JSON-safe inside the script block.
"""

from auto_tune.ui.app import _jinja_env
from auto_tune.ui.i18n import make_translator


MALICIOUS_CONFIG = {
    "text": {
        "purpose": "text",
        "enabled": True,
        "provider": 'x" onfocus="alert(1)',
        "model": "<script>alert(1)</script>",
        "endpoint": "y' onerror='alert(2)",
        "allow_private_endpoint": False,
        "default_endpoint": 'z" autofocus onfocus="alert(3)',
        "migration_required": False,
    },
    "vision": {
        "purpose": "vision",
        "enabled": True,
        "provider": 'x" onfocus="alert(4)',
        "model": "<script>alert(4)</script>",
        "endpoint": "https://dashscope.example/v1/chat/completions",
        "allow_private_endpoint": False,
        "default_endpoint": "https://default.example/v1/chat/completions",
        "migration_required": False,
    },
}


def _render(ai_config=MALICIOUS_CONFIG, csrf_token=""):
    translator = make_translator("zh")
    return _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
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
        ai_config=ai_config,
        csrf_token=csrf_token,
    )


def test_ai_config_values_do_not_break_out_of_attributes():
    html = _render()

    # A double-quote injection must not create a new executable attribute.
    assert 'value="x" onfocus="alert(1)' not in html
    assert 'placeholder="z" autofocus' not in html
    assert 'onfocus="alert(3)' not in html
    # The escaped entity form proves the value is escaped, not emitted raw.
    assert "x&#34; onfocus=&#34;alert(1)" in html
    assert "z&#34; autofocus onfocus=&#34;alert(3)" in html

    # A single-quote injection must not become a real attribute.
    assert " onerror='alert(2)" not in html
    assert "y&#39; onerror=&#39;alert(2)" in html


def test_ai_config_script_tags_are_escaped():
    html = _render()

    assert "<script>alert(1)</script>" not in html
    assert "<script>alert(4)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;script&gt;alert(4)&lt;/script&gt;" in html


def test_csrf_token_is_json_safe_in_script_block():
    token = 'abc"</script><script>alert(9)</script>'
    html = _render(csrf_token=token)

    # The token is serialized through tojson: the embedded double quote is
    # backslash-escaped and < / > are \u-escaped, so the script string cannot
    # be terminated early or broken out of.
    assert 'window._CSRF_TOKEN = "abc\\"' in html  # internal quote is \"
    assert 'window._CSRF_TOKEN = "abc"' not in html  # no bare-quote termination
    assert '"</script>' not in html
    assert "<script>alert(9)</script>" not in html
    assert "\\u003c" in html
