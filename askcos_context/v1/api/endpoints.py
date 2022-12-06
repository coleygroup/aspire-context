from fastapi import APIRouter, Depends

from askcos_context.common.schemas.request import (
    RecommendConditionRequest,
    RecommendConditionResponse
)
from askcos_context.v1.api.deps import get_model
from askcos_context.v1.services import NeuralNetContextRecommender

router = APIRouter()


@router.post("/condition", response_model=RecommendConditionResponse)
def recommend(
    request: RecommendConditionRequest,
    model: NeuralNetContextRecommender = Depends(get_model)
):
    conditions = model.recommend(request.smiles, request.reagents, request.n_conditions)
    
    return conditions
