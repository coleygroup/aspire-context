from dataclasses import dataclass
from os import PathLike

from askcos_context.v2.services.config.model import ModelConfig


@dataclass(frozen=True, unsafe_hash=True)
class ContextConfig:
    reagent_conv_rules_path: PathLike
    default_models: dict[str, str]
    model_configs: dict[str, ModelConfig]
