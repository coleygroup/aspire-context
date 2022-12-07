from fastapi import APIRouter, Depends

from askcos_context.common.schemas.condition import Agent, ConditionRecommendation, Role
from askcos_context.common.schemas.request import RecommendConditionRequest, RecommendConditionResponse
from askcos_context.v2.api.deps import ModelType, get_models
from askcos_context.v2.services.recommender import ReactionContextRecommenderBase

router = APIRouter()


@router.post(
    "/condition/{model_type}",
    response_model=RecommendConditionResponse,
    response_model_exclude_unset=True
)
def recommend(
    model_type: ModelType,
    request: RecommendConditionRequest,
    models: dict[str, ReactionContextRecommenderBase] = Depends(get_models)
):
    model = models[model_type.value]

    recommendations = []
    for condition in model.recommend(request.smiles, request.reagents, request.n_conditions):
        print(condition)
        reactants = [
            Agent(smi_or_name=smi, role=Role.REACTANT, amt=amt)
            for smi, amt in condition["reactants"].items()
        ]
        reagents = [
            Agent(smi_or_name=smi, amt=amt)
            for smi, amt in condition["reagents"].items()
        ]
        rec = ConditionRecommendation(
            agents=[*reactants, *reagents],
            temperature=condition["temperature"],
            score=condition["reagents_score"]
        )
        recommendations.append(rec)

    return recommendations