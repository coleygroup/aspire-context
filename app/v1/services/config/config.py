import os
from pathlib import Path

from app.v1.services.config.base import ContextConfig

RESOURCES_DIR = Path(os.environ.get("ASKCOS_DATA_DIR", "app/resources"))
CONTEXT_DIR = RESOURCES_DIR / "models" / "context" / "v1"

DEFAULT_CONFIG = ContextConfig(
    CONTEXT_DIR / "model.json",
    CONTEXT_DIR,
    CONTEXT_DIR / "weights.h5",
    CONTEXT_DIR / "ehs_solvent_scores.csv",
)
