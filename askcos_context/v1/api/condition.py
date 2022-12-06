from fastapi import APIRouter

from askcos_context.common.schemas import RecommendConditionRequest, RecommendConditionResponse
from askcos_context.v1.services import NeuralNetContextRecommender

api_router = APIRouter()

model = NeuralNetContextRecommender().load()


@api_router.post("/condition", response_model=RecommendConditionResponse)
def recommend(request: RecommendConditionRequest):
    conditions = model.get_n_conditions(request.smiles, request.n_conditions)
    
    return conditions
