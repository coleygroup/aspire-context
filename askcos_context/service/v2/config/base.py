from dataclasses import dataclass
from os import PathLike

from askcos_context.service.v2.config.model import ModelConfig


@dataclass(frozen=True)
class ContextConfig:
    reagent_conv_rules_path: PathLike
    default_models: dict[str, str]
    model_configs: dict[str, ModelConfig]
