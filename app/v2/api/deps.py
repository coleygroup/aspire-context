from enum import auto
from functools import cache
import os
from pathlib import Path

from fastapi import Depends
from app.common.utils.utils import AutoName

from app.v2.services.recommender import (
    ReactionContextRecommenderWLN,
    ReactionContextRecommenderFP,
)
from app.v2.services.config import FpModelConfig, GraphModelConfig


class ModelType(AutoName):
    FP = auto()
    GRAPH = auto()


@cache
def get_context_dir() -> Path:
    _DEFAULT_PATH = "app/resources/models/context/v2"
    return Path(os.environ.get("app_V2_DIR", _DEFAULT_PATH))


@cache
def get_reagent_rules_path(context_dir=Depends(get_context_dir)) -> Path:
    return context_dir / "stage0" / "reagent_conv_rules.json"


@cache
def get_stage_dirs(context_dir=Depends(get_context_dir)) -> tuple[Path, Path, Path, Path]:
    stage_0_dir = context_dir / "stage0"
    stage_1_dir = context_dir / "stage1"
    stage_2_dir = context_dir / "stage2"
    stage_3_dir = context_dir / "stage3"

    return stage_0_dir, stage_1_dir, stage_2_dir, stage_3_dir


@cache
def get_fp_big_config(
    stage_dirs: tuple[Path, Path, Path, Path] = Depends(get_stage_dirs)
) -> FpModelConfig:
    stage_0_dir, stage_1_dir, stage_2_dir, stage_3_dir = stage_dirs

    return FpModelConfig(
        length=16384,
        radius=3,
        reagents_path=stage_0_dir / "reagents_list_minocc100.json",
        reagents_model_path=(
            stage_1_dir
            / "fp_multicategorical_50_input_reagents_fplength16384_fpradius3"
            / "model-densegraph-04-4.18.hdf5.final-tf.20191118"
        ),
        temperature_model_path=(
            stage_2_dir
            / "50_temperature_regression_fp_baseline"
            / "model-densegraph-24-0.02.hdf5.final-tf.20191118"
        ),
        reagent_quantity_model_path=(
            stage_3_dir
            / "50_amount_regression_fp_baseline"
            / "model-densegraph-12-0.00.hdf5.final-tf.20191118"
        ),
        reactant_quantity_model_path=(
            stage_3_dir
            / "50_amount_reactant_regression_fp_baseline_dense2048_3"
            / "model-densegraph-24-0.05.hdf5.final-tf.20191118"
        ),
    )


@cache
def get_fp_config(
    stage_dirs: tuple[Path, Path, Path, Path] = Depends(get_stage_dirs)
) -> FpModelConfig:
    stage_0_dir, stage_1_dir, stage_2_dir, stage_3_dir = stage_dirs

    return FpModelConfig(
        length=2048,
        radius=3,
        reagents_path=stage_0_dir / "reagents_list_minocc100.json",
        reagents_model_path=(
            stage_1_dir
            / "fp_multicategorical_50_input_reagents_fplength2048_fpradius3"
            / "model-densegraph-04-4.27.hdf5.final-tf.20191118"
        ),
        temperature_model_path=(
            stage_2_dir
            / "50_temperature_regression_fp_baseline_fp2048"
            / "model-densegraph-40-0.02.hdf5.final-tf.20191118"
        ),
        reagent_quantity_model_path=(
            stage_3_dir
            / "50_amount_regression_fp_baseline_fp2048"
            / "model-densegraph-48-0.00.hdf5.final-tf.20191118"
        ),
        reactant_quantity_model_path=(
            stage_3_dir
            / "50_amount_reactant_regression_fp_baseline_fp2048_dense512"
            / "model-densegraph-04-0.05.hdf5.final-tf.20191118"
        ),
    )


@cache
def get_graph_config(
    stage_dirs: tuple[Path, Path, Path, Path] = Depends(get_stage_dirs)
) -> GraphModelConfig:
    stage_0_dir, stage_1_dir, stage_2_dir, stage_3_dir = stage_dirs

    return GraphModelConfig(
        encoder_path=stage_0_dir / "feature-statistics-final-s-natom50.pickle",
        condensed_graph=True,
        reagents_path=stage_0_dir / "reagents_list_minocc100.json",
        reagents_model_path=(
            stage_1_dir
            / "50_multicategorical_input_reagents_wlnlen512_wlnstep3"
            / "model-densegraph-08-4.08.hdf5.final-tf.20191118"
        ),
        temperature_model_path=(
            stage_2_dir
            / "50_temperature_regression"
            / "model-densegraph-16-0.02.hdf5.final-tf.20191118"
        ),
        reagent_quantity_model_path=(
            stage_3_dir / "50_amount_regression" / "model-densegraph-08-0.00.hdf5.final-tf.20191118"
        ),
        reactant_quantity_model_path=(
            stage_3_dir
            / "50_amount_reactant_regression_dense2048_3"
            / "model-densegraph-08-0.05.hdf5.final-tf.20191118"
        ),
    )


@cache
def get_fp_model(config: FpModelConfig = Depends(get_fp_config)):
    return ReactionContextRecommenderFP(None, config=config).load_models()


@cache
def get_fp_big_model(config: FpModelConfig = Depends(get_fp_big_config)):
    return ReactionContextRecommenderFP(None, config=config).load_models()


@cache
def get_graph_model(config: GraphModelConfig = Depends(get_graph_config)):
    return ReactionContextRecommenderWLN(None, config=config).load_models()


@cache
def get_models(
    fp_model: ReactionContextRecommenderFP = Depends(get_fp_model),
    graph_model: ReactionContextRecommenderWLN = Depends(get_graph_model),
):
    # fp_big_config = config.model_configs[config.default_models["fp-big"]]
    # fp_big_model = ReactionContextRecommenderFP(None, config=fp_big_config).load_models()

    return {
        ModelType.FP.value: fp_model,
        ModelType.GRAPH.value: graph_model
        # ModelType.FP_BIG.value: fp_big_model,
    }
