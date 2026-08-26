"""Module C auto-tuning display helpers."""
import json
from pathlib import Path


def get_tuning_history(log_dir: str = "log") -> list:
    """Load auto-tuning history."""
    path = Path(log_dir) / "tuning_history.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_tuning_status() -> dict:
    """Project the persisted auto-tuning run state into the unified public shape.

    A legacy ``{status: running}`` file carries no verifiable run identity, so
    it projects to ``unknown`` — it is never reported as still running.
    """
    from auto_tune.modules.run_state.service import project_public_state, read_run_state

    status_file = Path("log") / "tuning_running.json"
    state = read_run_state(status_file, run_kind="tuning")
    return project_public_state(state)
