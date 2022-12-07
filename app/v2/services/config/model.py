from dataclasses import dataclass
from os import PathLike


@dataclass(frozen=True, unsafe_hash=True)
class ModelConfig:
    reagents_path: PathLike
    reagents_model_path: PathLike
    temperature_model_path: PathLike
    reagent_quantity_model_path: PathLike
    reactant_quantity_model_path: PathLike


@dataclass(frozen=True, unsafe_hash=True)
class FpModelConfig(ModelConfig):
    length: int
    radius: int


@dataclass(frozen=True, unsafe_hash=True)
class GraphModelConfig(ModelConfig):
    encoder_path: PathLike
    condensed_graph: bool
