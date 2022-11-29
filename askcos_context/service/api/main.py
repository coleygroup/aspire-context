from enum import Enum

from fastapi import FastAPI

from askcos_context.data import RecommendConditionRequest, RecommendConditionResponse
from askcos_context.service.v2 import ReactionContextRecommenderWLN, ReactionContextRecommenderFP


class ModelType(str, Enum):
    GRAPH = "graph"
    FP = "fp"


app = FastAPI()
v1api = FastAPI()
v2api = FastAPI()

graph_model = ReactionContextRecommenderWLN().load_models()
fp_model = ReactionContextRecommenderFP().load_models()


@v2api.post("/condition/{model_type}", response_model=RecommendConditionResponse)
def recommend(model_type: ModelType, request: RecommendConditionRequest):
    match model_type:
        case ModelType.GRAPH:
            model = graph_model
        case ModelType.FP:
            model = fp_model

    conditions = model.predict(request.smiles, request.beam_size, request.reagents)
    
    return conditions


app.mount("/v2", v2api)
