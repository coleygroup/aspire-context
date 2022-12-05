from pydantic import BaseModel

from askcos_context.common.schemas import ReactionConditions


class RecommendConditionRequest(BaseModel):
    smiles: str
    reagents: list[str] | None = None
    n_conditions: int = 10


class RecommendConditionResponse(list[ReactionConditions]):
    pass
