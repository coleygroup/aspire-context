from functools import cache
import os
from pathlib import Path

from fastapi import Depends

from app.v1.services.predictor import NeuralNetContextRecommender
from app.v1.services.config import ContextConfig


def get_context_config() -> ContextConfig:
    _DEFAULT_PATH = "app/resources/models/context/v1"
    CONTEXT_DIR = Path(os.getenv("app_V1_DIR", _DEFAULT_PATH))

    return ContextConfig(
        CONTEXT_DIR / "model.json",
        CONTEXT_DIR,
        CONTEXT_DIR / "weights.h5",
        CONTEXT_DIR / "ehs_solvent_scores.csv",
    )


@cache
def get_model(config: ContextConfig = Depends(get_context_config)):
    return NeuralNetContextRecommender(config=config).load()
