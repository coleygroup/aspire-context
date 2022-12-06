from enum import Enum, auto

from fastapi import FastAPI

from askcos_context.common.schemas.request import (
    RecommendConditionRequest,
    RecommendConditionResponse
)
from askcos_context.common.utils.utils import AutoName
from askcos_context.v1.api.endpoints import router as v1_router
from askcos_context.v2.services import ReactionContextRecommenderWLN, ReactionContextRecommenderFP


class ModelType(AutoName):
    GRAPH = auto()
    FP = auto()


app = FastAPI()
# v1api = FastAPI()
v2api = FastAPI()

graph_model = ReactionContextRecommenderWLN().load_models()
fp_model = ReactionContextRecommenderFP().load_models()


# @v1api.post("/condition", response_model=RecommendConditionResponse)
# def recommend(request: RecommendConditionRequest):
#     conditions = v1_model.get_n_conditions(request.smiles, request.n_conditions)
    
#     return conditions


@v2api.post("/condition/{model_type}", response_model=RecommendConditionResponse)
def recommend(model_type: ModelType, request: RecommendConditionRequest):
    match model_type:
        case ModelType.GRAPH:
            model = graph_model
        case ModelType.FP:
            model = fp_model

    conditions = model.recommend(request.smiles, request.reagents, request.n_conditions)
    
    return conditions

app.include_router(v1_router, prefix="/api/v1")
# app.mount("/v2", v2api)
