import os
from pathlib import Path

from fastapi import Depends

from askcos_context.v1.config.base import ContextConfig
from askcos_context.v1.services import NeuralNetContextRecommender


def get_context_config() -> ContextConfig:
    RESOURCES_DIR = Path(os.environ.get("ASKCOS_DATA_DIR", "askcos_context/resources"))
    CONTEXT_DIR = RESOURCES_DIR / "models" / "context" / "v1"

    return ContextConfig(
        CONTEXT_DIR / "model.json",
        CONTEXT_DIR,
        CONTEXT_DIR / "weights.h5",
        CONTEXT_DIR / "ehs_solvent_scores.csv"
    )


def get_model(config: ContextConfig = Depends(get_context_config)):
    return NeuralNetContextRecommender(config=config).load()
