import os
from pathlib import Path

from askcos_context.v1.services.config.base import ContextConfig

RESOURCES_DIR = Path(os.environ.get("ASKCOS_DATA_DIR", "askcos_context/resources"))
CONTEXT_DIR = RESOURCES_DIR / "models" / "context" / "v1"

DEFAULT_CONFIG = ContextConfig(
    CONTEXT_DIR / "model.json",
    CONTEXT_DIR,
    CONTEXT_DIR / "weights.h5",
    CONTEXT_DIR / "ehs_solvent_scores.csv"
)