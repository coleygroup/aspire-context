from dataclasses import dataclass
from os import PathLike


@dataclass(frozen=True)
class ContextConfig:
    model_path: PathLike
    info_path: str
    weights_path: PathLike
    ehs_score_path: PathLike
