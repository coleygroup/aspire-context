from enum import Enum

from fastapi import FastAPI

from askcos_context.common.schemas import RecommendConditionRequest, RecommendConditionResponse
from askcos_context.v1.service import NeuralNetContextRecommender
from askcos_context.v2.service import ReactionContextRecommenderWLN, ReactionContextRecommenderFP


class ModelType(str, Enum):
    GRAPH = "graph"
    FP = "fp"


app = FastAPI()
v1api = FastAPI()
v2api = FastAPI()

v1_model = NeuralNetContextRecommender().load()
graph_model = ReactionContextRecommenderWLN().load_models()
fp_model = ReactionContextRecommenderFP().load_models()


@v1api.post("/condition", response_model=RecommendConditionResponse)
def recommend(request: RecommendConditionRequest):
    conditions = v1_model.get_n_conditions(request.smiles, request.n_conditions)
    
    return conditions


@v2api.post("/condition/{model_type}", response_model=RecommendConditionResponse)
def recommend(model_type: ModelType, request: RecommendConditionRequest):
    match model_type:
        case ModelType.GRAPH:
            model = graph_model
        case ModelType.FP:
            model = fp_model

    conditions = model.predict(request.smiles, request.n_conditions, request.reagents)
    
    return conditions

app.mount("/v1", v1api)
app.mount("/v2", v2api)
