"""Start the Auto-Tune server (helper script)."""
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
)
from auto_tune.ui.app import start_server
start_server()
