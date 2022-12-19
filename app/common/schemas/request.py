from pydantic import BaseModel

from app.common.schemas.condition import ConditionRecommendation


class RecommendConditionRequest(BaseModel):
    smiles: str
    reagents: list[str] | None = None
    n_conditions: int = 10

    class Config:
        schema_extra = {
            "example": {
                "smiles": "CC1(C)OBOC1(C)C.Cc1ccc(Br)cc1>>Cc1cccc(B2OC(C)(C)C(C)(C)O2)c1",
                "reagents": [],
                "n_conditions": 10
            }
        }
    
RecommendConditionResponse = list[ConditionRecommendation]
