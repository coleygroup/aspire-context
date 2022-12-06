from enum import auto
from pydantic import BaseModel

from askcos_context.common.utils.utils import AutoName


class Role(AutoName):
    REACTANT = auto()
    REAGENT = auto()
    PRODUCT = auto()
    SOLVENT = auto()
    CATALYST = auto()


class Agent(BaseModel):
    name: str | None
    smi: str | None
    amt: float | None
    role: Role

    
class ConditionRecommendation(BaseModel):
    agents: list[Agent]
    temperature: float | None
    score: float | None