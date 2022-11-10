from dataclasses import dataclass
from os import PathLike


@dataclass
class ModelConfig:
    reagents_path: PathLike
    reagents_model_path: PathLike
    temperature_model_path: PathLike
    reagent_quantity_model_path: PathLike
    reactant_quantity_model_path: PathLike


@dataclass
class FpModelConfig(ModelConfig):
    length: int
    radius: int


@dataclass
class GraphModelConfig(ModelConfig):
    encoder_path: PathLike
    condensed_graph: bool