from pydantic import BaseModel

from askcos_context.data.data import ReactionConditions


class RecommendConditionRequest(BaseModel):
    smiles: str
    reagents: list[str] | None = None
    beam_size: int = 10

class RecommendConditionResponse(list[ReactionConditions]):
    pass
