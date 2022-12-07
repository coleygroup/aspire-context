from enum import auto
from pydantic import BaseModel

from app.common.utils.utils import AutoName


class Role(AutoName):
    REACTANT = auto()
    REAGENT = auto()
    SOLVENT = auto()
    CATALYST = auto()
    UNKNOWN = auto()


class Agent(BaseModel):
    smi_or_name: str | None
    role: Role = Role.UNKNOWN
    amt: float | None = None


class ConditionRecommendation(BaseModel):
    agents: list[Agent]
    temperature: float | None
    score: float | None
