from fastapi import APIRouter, Depends

from app.common.schemas.condition import Agent, ConditionRecommendation, Role
from app.common.schemas.request import RecommendConditionRequest, RecommendConditionResponse
from app.v1.api.deps import get_model
from app.v1.services import NeuralNetContextRecommender

router = APIRouter()


@router.post(
    "/condition", response_model=RecommendConditionResponse, response_model_exclude_unset=True
)
def recommend(
    request: RecommendConditionRequest, model: NeuralNetContextRecommender = Depends(get_model)
):
    reactants = [Agent(smi_or_name=smi, role=Role.REACTANT) for smi in request.smiles.split(".")]
    conditions, scores = model.recommend(request.smiles, request.reagents, request.n_conditions)

    recommendations = []
    for condition, score in zip(conditions, scores):
        temp, reagents, solvents, catalyst, *_ = condition
        reagents = [Agent(smi_or_name=smi, role=Role.REAGENT) for smi in reagents.split(".")]
        solvents = [Agent(smi_or_name=smi, role=Role.SOLVENT) for smi in solvents.split(".")]
        catalyst = Agent(smi_or_name=catalyst, role=Role.CATALYST)
        agents = [*reactants, *reagents, *solvents, catalyst]

        rec = ConditionRecommendation(agents=agents, temperature=temp, score=score)
        recommendations.append(rec)

    return recommendations
