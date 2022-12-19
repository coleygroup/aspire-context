from pydantic import BaseModel

from app.common.schemas.condition import ConditionRecommendation


class RecommendConditionRequest(BaseModel):
    smiles: str
    reagents: list[str] | None = None
    n_conditions: int = 10


RecommendConditionResponse = list[ConditionRecommendation]
