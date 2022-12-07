from functools import cache
import os
from pathlib import Path

from fastapi import Depends

from askcos_context.v1.services.predictor import NeuralNetContextRecommender
from askcos_context.v1.services.config import ContextConfig


def get_context_config() -> ContextConfig:
    _DEFAULT_PATH = "askcos_context/resources/models/context/v1"
    CONTEXT_DIR = Path(os.getenv("ASKCOS_CONTEXT_V1_DIR", _DEFAULT_PATH))

    return ContextConfig(
        CONTEXT_DIR / "model.json",
        CONTEXT_DIR,
        CONTEXT_DIR / "weights.h5",
        CONTEXT_DIR / "ehs_solvent_scores.csv",
    )


@cache
def get_model(config: ContextConfig = Depends(get_context_config)):
    return NeuralNetContextRecommender(config=config).load()
