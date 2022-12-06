from enum import Enum, auto

from pydantic import BaseModel


class Role(Enum):
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