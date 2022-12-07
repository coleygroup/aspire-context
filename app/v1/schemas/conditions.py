from pydantic import BaseModel


class ReactionConditions(BaseModel):
    temperature: float
    solvent: str
    reagents: str
    catalyst: str
