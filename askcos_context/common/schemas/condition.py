from enum import auto
from pydantic import BaseModel

from askcos_context.common.utils.utils import AutoName


class Role(AutoName):
    REACTANT = auto()
    REAGENT = auto()
    SOLVENT = auto()
    CATALYST = auto()


class Agent(BaseModel):
    smi_or_name: str | None
    role: Role
    amt: float | None = None

    
class ConditionRecommendation(BaseModel):
    agents: list[Agent]
    temperature: float | None
    score: float | None