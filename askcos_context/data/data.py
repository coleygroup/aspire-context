from pydantic import BaseModel


class ReactionConditions(BaseModel):
    reagents: dict[str, float]
    reactants: dict[str, float]
    temperature: float
    reagents_score: float
