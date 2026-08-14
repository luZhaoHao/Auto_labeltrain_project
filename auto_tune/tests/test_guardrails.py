"""Regression tests for the Module C parameter safety boundary."""

from auto_tune.modules.agent_engine.guardrails import (
    sanitize_tuning_parameters,
    validate_and_clamp,
)


def test_out_of_range_value_is_clamped_in_executable_params():
    """Catches the bug where warnings were recorded but raw values executed."""
    result = validate_and_clamp({"lr0": 2.0})

    assert result.valid is True
    assert result.params["lr0"] == 0.1
    assert result.clamped["lr0"] == 0.1


def test_changes_and_training_overrides_share_one_validation_path():
    """Catches training_overrides bypassing the first guardrail pass."""
    result = sanitize_tuning_parameters(
        {"lr0": 0.002},
        {"epochs": 5000, "batch": 0, "optimizer": "AdamW"},
    )

    assert result.valid is True
    assert result.params == {
        "lr0": 0.002,
        "epochs": 1000,
        "batch": 1,
        "optimizer": "AdamW",
    }


def test_auto_optimizer_cannot_claim_explicit_lr_is_applied():
    """Catches UI/audit claiming lr0 changed when Ultralytics ignores it."""
    result = sanitize_tuning_parameters(
        {"lr0": 0.0025},
        {"optimizer": "auto"},
    )

    assert result.valid is False
    assert any("optimizer=auto" in error for error in result.errors)


def test_unknown_llm_parameter_is_rejected():
    """Catches unsupported Prompt parameters silently reaching reports."""
    result = sanitize_tuning_parameters({"fl_gamma": 1.5}, {})

    assert result.valid is False
    assert result.params == {}
    assert any("Unknown parameter" in error for error in result.errors)


def test_numeric_string_is_normalized_before_range_validation():
    """Catches JSON numeric strings causing comparison TypeError."""
    result = sanitize_tuning_parameters(
        {"lr0": "0.005"},
        {"epochs": "50", "optimizer": "AdamW"},
    )

    assert result.valid is True
    assert result.params["lr0"] == 0.005
    assert result.params["epochs"] == 50

