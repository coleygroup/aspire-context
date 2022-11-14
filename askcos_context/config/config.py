import os
from pathlib import Path

from context.config.base import ContextConfig
from context.config.model import FpModelConfig, GraphModelConfig

DATA_DIR = Path(os.environ.get("ASKCOS_DATA_DIR", "data"))
MODEL_DIR = DATA_DIR / "models"
CONTEXT_DIR = MODEL_DIR / "context" / "v2"
STAGE_1_DIR = CONTEXT_DIR / "stage1"
STAGE_2_DIR = CONTEXT_DIR / "stage2"
STAGE_3_DIR = CONTEXT_DIR / "stage3"

DEFAULT_FP_CONFIG = FpModelConfig(
    length=16384,
    radius=3,
    reagents_path=CONTEXT_DIR / "stage0" / "reagents_list_minocc100.json",
    reagents_model_path=(
        STAGE_1_DIR /
        "fp_multicategorical_50_input_reagents_fplength16384_fpradius3" /
        "model-densegraph-04-4.18.hdf5.final-tf.20191118",
    ),
    temperature_model_path=(
        STAGE_2_DIR /
        "50_temperature_regression_fp_baseline" /
        "model-densegraph-24-0.02.hdf5.final-tf.20191118",
    ),
    reagent_quantity_model_path=(
        STAGE_3_DIR /
        "50_amount_regression_fp_baseline" /
        "model-densegraph-12-0.00.hdf5.final-tf.20191118",
    ),
    reactants_amount_model=(
        CONTEXT_DIR /
        "stage3" /
        "50_amount_reactant_regression_fp_baseline_dense2048_3" /
        "model-densegraph-24-0.05.hdf5.final-tf.20191118",
    ),
)

DEFAULT_FP_SMALL_CONFIG = FpModelConfig(
    length=2048,
    radius=3,
    reagents_path=CONTEXT_DIR / "stage0" / "reagents_list_minocc100.json",
    reagents_model_path=(
        STAGE_1_DIR /
        "fp_multicategorical_50_input_reagents_fplength2048_fpradius3" /
        "model-densegraph-04-4.27.hdf5.final-tf.20191118",
    ),
    temperature_model_path=(
        STAGE_2_DIR /
        "50_temperature_regression_fp_baseline_fp2048" /
        "model-densegraph-40-0.02.hdf5.final-tf.20191118",
    ),
    reagent_quantity_model_path=(
        STAGE_3_DIR /
        "50_amount_regression_fp_baseline_fp2048" /
        "model-densegraph-48-0.00.hdf5.final-tf.20191118",
    ),
    reactants_amount_model_path=(
        STAGE_3_DIR /
        "50_amount_reactant_regression_fp_baseline_fp2048_dense512",
        "model-densegraph-04-0.05.hdf5.final-tf.20191118",
    )
)

DEFAULT_GRAPH_CONFIG = GraphModelConfig(
    encoder_path=CONTEXT_DIR / "stage0" / "feature-statistics-final-s-natom50.pickle",
    condensed_graph=True,
    reagents_path=CONTEXT_DIR / "stage0" / "reagents_list_minocc100.json",
    reagents_model_path=(
        STAGE_1_DIR /
        "50_multicategorical_input_reagents_wlnlen512_wlnstep3" /
        "model-densegraph-08-4.08.hdf5.final-tf.20191118"
    ),
    temperature_mode_pathl=(
        STAGE_2_DIR /
        "50_temperature_regression" /
        "model-densegraph-16-0.02.hd5.final-tf.20191118",
    ),
    reagent_quantity_model_path=(
        STAGE_3_DIR /
        "50_amount_regression" /
        "model-densegraph-08-0.0.hdf5.final-tf.20191118",
    ),
    reactants_amount_model_path=(
        STAGE_3_DIR /
        "50_amount_reactant_regression_dense2048_3" /
        "model-densegraph-08-0.05.hdf5.final-tf.2019118",
    ),
)

DEFAULT_CONFIG = ContextConfig(
    reagent_conv_rules_path=CONTEXT_DIR / "stage0" / "reagent_conv_rules.json",
    default_models={"graph": "graph-20191118", "fp": "fp-small-20191118"},
    model_configs={
        "fp-20191118": DEFAULT_FP_CONFIG,
        "fp-small-20191118": DEFAULT_FP_SMALL_CONFIG,
        "graph-20191118": DEFAULT_GRAPH_CONFIG
    }
)