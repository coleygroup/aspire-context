import logging
from os import PathLike
from pathlib import Path
import pickle

import numpy as np
from rdkit import Chem
from scipy import stats
import tensorflow as tf

from askcos_context.common.services.recommender import ReactionContextRecommender
from askcos_context.v1.services import utils
from askcos_context.v1.config import ContextConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class NeuralNetContextRecommender(ReactionContextRecommender):
    """Reaction condition predictor using neural network architecture"""

    def __init__(self, with_smiles: bool = False, config: ContextConfig = DEFAULT_CONFIG, **kwargs):
        """Initializes Neural Network predictor.

        Args:
            with_smiles (bool, optional): Remove predictions which only have
                a name and no SMILES string (default: {False})
        """
        # Full neural network model
        self.nnModel = None

        # Index to label dictionaries
        self.c1_dict = None
        self.s1_dict = None
        self.s2_dict = None
        self.r1_dict = None
        self.r2_dict = None

        # Input/output dimensions (should equal size of dictionaries)
        self.c1_dim = None
        self.r1_dim = None
        self.r2_dim = None
        self.s1_dim = None
        self.s2_dim = None

        # Functions for evaluating sub-component of full model
        self.fp_func = None
        self.c1_func = None
        self.s1_func = None
        self.s2_func = None
        self.r1_func = None
        self.r2_func = None
        self.T_func = None

        self.with_smiles = with_smiles
        self.fp_size = None # 2048
        self.ehs_dict = {}

        self.model_path = config.model_path
        self.info_path = config.info_path
        self.weights_path = config.weights_path
        self.ehs_score_path = config.ehs_score_path

    def validate_paths(self):
        """Check that the configured paths exist.

        Raises
        ------
        ValueError
            if any of the paths do not exist
        """
        paths = [
            self.model_path,
            self.info_path,
            self.weights_path,
            self.ehs_score_path,
        ]

        for path in paths:
            if path is None or not Path(path).exists():
                raise ValueError(f"Missing path for {self.__class__.__name__}: {path}")

    def load(self):
        """Load the configured model"""
        self.load_nn_model(self.model_path, self.info_path, self.weights_path)
        self.load_ehs_dictionary(self.ehs_score_path)
        
        logger.info("Neural network context recommender has been loaded.")

        return self

    def load_nn_model(self, model_path: PathLike, info_path: PathLike, weights_path: PathLike):
        """Loads specified Neural Network model.

        Parameters
        ----------
        model_path : PathLike
        info_path : PathLike
        weights_path : PathLike
        """
        if not model_path:
            logger.error(
                "Cannot load neural net context recommender without a specific path to the model. Exiting..."
            )
        if not info_path:
            logger.error(
                "Cannot load neural net context recommender without a specific path to the model info. Exiting..."
            )

        # load json and create model
        with open(model_path, "r") as f:
            loaded_model_json = f.read()

        self.nnModel = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        self.nnModel.load_weights(weights_path)
        # get fp_size based on the model
        self.fp_size = self.nnModel.input_shape[0][1]

        # load label dictionaries
        info_path = Path(info_path)
        r1_dict_file = info_path / "r1_dict.pickle"
        r2_dict_file = info_path / "r2_dict.pickle"
        s1_dict_file = info_path / "s1_dict.pickle"
        s2_dict_file = info_path / "s2_dict.pickle"
        c1_dict_file = info_path / "c1_dict.pickle"

        with open(r1_dict_file, "rb") as R1_DICT_F:
            self.r1_dict = pickle.load(R1_DICT_F)

        with open(r2_dict_file, "rb") as R2_DICT_F:
            self.r2_dict = pickle.load(R2_DICT_F)

        with open(s1_dict_file, "rb") as S1_DICT_F:
            self.s1_dict = pickle.load(S1_DICT_F)

        with open(s2_dict_file, "rb") as S2_DICT_F:
            self.s2_dict = pickle.load(S2_DICT_F)

        with open(c1_dict_file, "rb") as C1_DICT_F:
            self.c1_dict = pickle.load(C1_DICT_F)

        # extract input/output dimensions from model
        self.c1_dim = self.nnModel.input_shape[2][1]
        self.r1_dim = self.nnModel.input_shape[3][1]
        self.r2_dim = self.nnModel.input_shape[4][1]
        self.s1_dim = self.nnModel.input_shape[5][1]
        self.s2_dim = self.nnModel.input_shape[6][1]

        (
            input_pfp,
            input_rxnfp,
            input_c1,
            input_r1,
            input_r2,
            input_s1,
            input_s2,
        ) = self.nnModel.inputs

        # add an intermediate input to the model which feeds into dropout_1
        # [input_pfp, input_rxnfp] -> fp_transform1 -> fp_transform2 -> dropout_1
        # this allows computing the first 2 layers once
        # the new input must be fed through the remainder of the model
        h2 = self.nnModel.get_layer("fp_transform2").output
        input_h2 = tf.keras.Input(shape=(1000,), name="input_h2")

        h2_dropout = self.nnModel.get_layer("dropout_1")(input_h2)

        c1_h1 = self.nnModel.get_layer("c1_h1")(h2_dropout)
        c1_h2 = self.nnModel.get_layer("c1_h2")(c1_h1)
        c1_output = self.nnModel.get_layer("c1")(c1_h2)
        c1_dense = self.nnModel.get_layer("c1_dense")(input_c1)

        concat_fp_c1 = self.nnModel.get_layer("concat_fp_c1")([h2_dropout, c1_dense])

        s1_h1 = self.nnModel.get_layer("s1_h1")(concat_fp_c1)
        s1_h2 = self.nnModel.get_layer("s1_h2")(s1_h1)

        s1_output = self.nnModel.get_layer("s1")(s1_h2)
        s1_dense = self.nnModel.get_layer("s1_dense")(input_s1)

        concat_fp_c1_s1 = self.nnModel.get_layer("concat_fp_c1_s1")(
            [h2_dropout, c1_dense, s1_dense]
        )

        s2_h1 = self.nnModel.get_layer("s2_h1")(concat_fp_c1_s1)
        s2_h2 = self.nnModel.get_layer("s2_h2")(s2_h1)

        s2_output = self.nnModel.get_layer("s2")(s2_h2)
        s2_dense = self.nnModel.get_layer("s2_dense")(input_s2)

        concat_fp_c1_s1_s2 = self.nnModel.get_layer("concat_fp_c1_s1_s2")(
            [h2_dropout, c1_dense, s1_dense, s2_dense]
        )

        r1_h1 = self.nnModel.get_layer("r1_h1")(concat_fp_c1_s1_s2)
        r1_h2 = self.nnModel.get_layer("r1_h2")(r1_h1)

        r1_output = self.nnModel.get_layer("r1")(r1_h2)
        r1_dense = self.nnModel.get_layer("r1_dense")(input_r1)

        concat_fp_c1_s1_s2_r1 = self.nnModel.get_layer("concat_fp_c1_s1_s2_r1")(
            [h2_dropout, c1_dense, s1_dense, s2_dense, r1_dense]
        )

        r2_h1 = self.nnModel.get_layer("r2_h1")(concat_fp_c1_s1_s2_r1)
        r2_h2 = self.nnModel.get_layer("r2_h2")(r2_h1)

        r2_output = self.nnModel.get_layer("r2")(r2_h2)
        r2_dense = self.nnModel.get_layer("r2_dense")(input_r2)

        concat_fp_c1_s1_s2_r1_r2 = self.nnModel.get_layer("concat_fp_c1_s1_s2_r1_r2")(
            [h2_dropout, c1_dense, s1_dense, s2_dense, r1_dense, r2_dense]
        )

        T_h1 = self.nnModel.get_layer("T_h1")(concat_fp_c1_s1_s2_r1_r2)

        T_output = self.nnModel.get_layer("T")(T_h1)

        # create functions for each of the sub-model evaluations
        self.fp_func = tf.function(tf.keras.Model([input_pfp, input_rxnfp], [h2]))
        self.c1_func = tf.function(tf.keras.Model([input_h2], [c1_output]))
        self.s1_func = tf.function(tf.keras.Model([input_h2, input_c1], [s1_output]))
        self.s2_func = tf.function(tf.keras.Model([input_h2, input_c1, input_s1], [s2_output]))
        self.r1_func = tf.function(
            tf.keras.Model([input_h2, input_c1, input_s1, input_s2], [r1_output])
        )
        self.r2_func = tf.function(
            tf.keras.Model([input_h2, input_c1, input_s1, input_s2, input_r1], [r2_output])
        )
        self.T_func = tf.function(
            tf.keras.Model([input_h2, input_c1, input_s1, input_s2, input_r1, input_r2], [T_output])
        )

    def smiles_to_fp(self, smiles):
        """Generates fingerprints for the input reaction SMILES.

        Canonicalizes and removes atom map numbers before generation.

        Args:
            smiles (str): input reaction SMILES

        Returns:
            np.ndarray, np.ndarray: product and reaction fingerprints with dtype of int8
        """
        rsmi, _, psmi = smiles.split(">")
        rct_mol = Chem.MolFromSmiles(rsmi)
        prd_mol = Chem.MolFromSmiles(psmi)
        [
            atom.ClearProp("molAtomMapNumber")
            for atom in rct_mol.GetAtoms()
            if atom.HasProp("molAtomMapNumber")
        ]
        [
            atom.ClearProp("molAtomMapNumber")
            for atom in prd_mol.GetAtoms()
            if atom.HasProp("molAtomMapNumber")
        ]
        rsmi = Chem.MolToSmiles(rct_mol, isomericSmiles=True)
        psmi = Chem.MolToSmiles(prd_mol, isomericSmiles=True)
        pfp, rfp = utils.reac_prod_smi_to_morgan_fp(
            rsmi, psmi, length=self.fp_size, as_column=True, useFeatures=False, useChirality=True
        )
        rxnfp = pfp - rfp
        return pfp, rxnfp

    def get_n_conditions(
        self, smi: str,
        reagents: list[str] | None = None,
        n_conditions: int = 10,
        with_smiles=False,
        return_scores=False,
        return_separate=False,
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        smi : str
            SMILES string for reaction., by default None
        reagents : list[str] | None, default=None
            NOTE: unused, maintained only for signature compatibility
        n_conditions : int, default=10
        with_smiles : bool, default=False
            remove predictions that have only a name and no SMILES string
        return_scores : bool, default=False
            return the scores of the recommendations as well
        return_separate : bool, default=False
            return predictions directly without postprocessing

        Parameters
        ----------
        rxn (str): SMILES string for reaction.
        n (int, optional): Number of condition recommendations to return.
            (default: {10})
        with_smiles (bool, optional): Remove predictions which only have
            a name and no SMILES string (default: {False})
        return_scores (bool, optional): Whether to also return the scores of the
            recommendations. (default: {True})
        return_separate (bool, optional): Return predictions directly without
            postprocessing. (default: {False})
        Returns
        -------
        _type_
            _description_
        """
        self.with_smiles = with_smiles

        try:
            pfp, rxnfp = self.smiles_to_fp(smi)
            c1_input = []
            r1_input = []
            r2_input = []
            s1_input = []
            s2_input = []
            inputs = [pfp, rxnfp, c1_input, r1_input, r2_input, s1_input, s2_input]

            top_combos, top_combo_scores = self.predict_top_combos(
                inputs=inputs, return_categories_only=return_separate
            )

            top_combo_scores = [float(score) for score in top_combo_scores]

            top_combos, top_combo_scores = top_combos[:n_conditions], top_combo_scores[:n_conditions]

            if not return_separate:
                top_combos = self.contexts_ehs_scores(top_combos[:n_conditions])

            if return_scores:
                return top_combos, top_combo_scores
            else:
                return top_combos

        except Exception as e:
            logger.warning(f"Failed for reaction {smi} because {e}. Returning None.")

            return [[]]

    def path_condition(self, n, path):
        """Recommends reaction conditions reaction path with multiple reactions.

        Args:
            n (int): Number of options to use at each step.
            path (list): Reaction SMILES for each step.


            Returns:
                A list of reaction contexts with n options for each step.
        """
        contexts = []

        for rxn in path:
            try:
                pfp, rxnfp = self.smiles_to_fp(rxn)
                c1_input = []
                r1_input = []
                r2_input = []
                s1_input = []
                s2_input = []
                inputs = [pfp, rxnfp, c1_input, r1_input, r2_input, s1_input, s2_input]
                top_combos = self.predict_top_combos(
                    inputs=inputs,
                    c1_rank_thres=1,
                    s1_rank_thres=3,
                    s2_rank_thres=1,
                    r1_rank_thres=4,
                    r2_rank_thres=1,
                )
                contexts.append(top_combos[:n])
            except Exception as e:
                logger.warning(f"Failed for reaction {rxn} because {e}. Returning None.")

        return contexts

    def predict_top_combos(
        self,
        inputs,
        return_categories_only=False,
        c1_rank_thres=2,
        s1_rank_thres=3,
        s2_rank_thres=1,
        r1_rank_thres=3,
        r2_rank_thres=1,
    ):
        """Predicts top combos based on rank thresholds for individual elements.

        Args:
            inputs (list): Input values for model.
            return_categories_only (bool, optional): Whether to only return the
                categories. Used for testing. (default: {False})
            c1_rank_thres (int, optional): Rank threshold for c1 (default: {2})
            s1_rank_thres (int, optional): Rank threshold for s1 (default: {3})
            s2_rank_thres (int, optional): Rank threshold for s2 (default: {1})
            r1_rank_thres (int, optional): Rank threshold for r1 (default: {3})
            r2_rank_thres (int, optional): Rank threshold for r2 (default: {1})

        Returns:
            list, list: Context combinations and overall scores from model
        """
        # this function predicts the top combos based on rank thresholds for
        # individual elements
        context_combos = []
        context_combo_scores = []
        num_combos = c1_rank_thres * s1_rank_thres * s2_rank_thres * r1_rank_thres * r2_rank_thres
        [
            pfp,
            rxnfp,
            c1_input_user,
            r1_input_user,
            r2_input_user,
            s1_input_user,
            s2_input_user,
        ] = inputs

        fp_trans = self.fp_func([pfp, rxnfp])
        if not c1_input_user:
            c1_inputs = [fp_trans]
            c1_pred = self.c1_func(c1_inputs).numpy()
            c1_cdts = c1_pred[0].argsort()[-c1_rank_thres:][::-1]
        else:
            c1_cdts = np.nonzero(c1_input_user)[0]
        # find the name of catalyst
        for c1_cdt in c1_cdts:
            c1_name = self.c1_dict[c1_cdt]
            c1_input = np.zeros([1, self.c1_dim])
            c1_input[0, c1_cdt] = 1
            if not c1_input_user:
                c1_sc = c1_pred[0][c1_cdt]
            else:
                c1_sc = 1
            if not s1_input_user:
                s1_inputs = [fp_trans, c1_input]
                s1_pred = self.s1_func(s1_inputs).numpy()
                s1_cdts = s1_pred[0].argsort()[-s1_rank_thres:][::-1]
            else:
                s1_cdts = np.nonzero(s1_input_user)[0]
            for s1_cdt in s1_cdts:
                s1_name = self.s1_dict[s1_cdt]
                s1_input = np.zeros([1, self.s1_dim])
                s1_input[0, s1_cdt] = 1
                if not s1_input_user:
                    s1_sc = s1_pred[0][s1_cdt]
                else:
                    s1_sc = 1
                if not s2_input_user:
                    s2_inputs = [fp_trans, c1_input, s1_input]
                    s2_pred = self.s2_func(s2_inputs).numpy()
                    s2_cdts = s2_pred[0].argsort()[-s2_rank_thres:][::-1]
                else:
                    s2_cdts = np.nonzero(s2_input_user)[0]
                for s2_cdt in s2_cdts:
                    s2_name = self.s2_dict[s2_cdt]
                    s2_input = np.zeros([1, self.s2_dim])
                    s2_input[0, s2_cdt] = 1
                    if not s2_input_user:
                        s2_sc = s2_pred[0][s2_cdt]
                    else:
                        s2_sc = 1
                    if not r1_input_user:
                        r1_inputs = [fp_trans, c1_input, s1_input, s2_input]
                        r1_pred = self.r1_func(r1_inputs).numpy()
                        r1_cdts = r1_pred[0].argsort()[-r1_rank_thres:][::-1]
                    else:
                        r1_cdts = np.nonzero(r1_input_user)[0]
                    for r1_cdt in r1_cdts:
                        r1_name = self.r1_dict[r1_cdt]
                        r1_input = np.zeros([1, self.r1_dim])
                        r1_input[0, r1_cdt] = 1
                        if not r1_input_user:
                            r1_sc = r1_pred[0][r1_cdt]
                        else:
                            r1_sc = 1
                        if not r2_input_user:
                            r2_inputs = [fp_trans, c1_input, s1_input, s2_input, r1_input]
                            r2_pred = self.r2_func(r2_inputs).numpy()
                            r2_cdts = r2_pred[0].argsort()[-r2_rank_thres:][::-1]
                        else:
                            r2_cdts = np.nonzero(r2_input_user)[0]
                        for r2_cdt in r2_cdts:
                            r2_name = self.r2_dict[r2_cdt]
                            r2_input = np.zeros([1, self.r2_dim])
                            r2_input[0, r2_cdt] = 1
                            if not r2_input_user:
                                r2_sc = r2_pred[0][r2_cdt]
                            else:
                                r2_sc = 1
                            T_inputs = [fp_trans, c1_input, s1_input, s2_input, r1_input, r2_input]
                            T_pred = self.T_func(T_inputs).numpy()
                            # print(c1_name,s1_name,s2_name,r1_name,r2_name)
                            cat_name = [c1_name]
                            if r2_name == "":
                                rgt_name = [r1_name]
                            else:
                                rgt_name = [r1_name, r2_name]
                            if s2_name == "":
                                slv_name = [s1_name]
                            else:
                                slv_name = [s1_name, s2_name]
                            if self.with_smiles:
                                rgt_name = [rgt for rgt in rgt_name if "Reaxys" not in rgt]
                                slv_name = [slv for slv in slv_name if "Reaxys" not in slv]
                                cat_name = [cat for cat in cat_name if "Reaxys" not in cat]
                            # for testing purpose only, output order as training
                            if return_categories_only:
                                context_combos.append(
                                    [
                                        c1_name,
                                        s1_name,
                                        s2_name,
                                        r1_name,
                                        r2_name,
                                        float(T_pred[0][0]),
                                    ]
                                )
                            # else output format compatible with the overall framework
                            else:
                                context_combos.append(
                                    [
                                        float(T_pred[0][0]),
                                        ".".join(slv_name),
                                        ".".join(rgt_name),
                                        ".".join(cat_name),
                                    ]
                                )

                            context_combo_scores.append(c1_sc * s1_sc * s2_sc * r1_sc * r2_sc)
        context_ranks = list(num_combos + 1 - stats.rankdata(context_combo_scores))

        context_combos = [context_combos[context_ranks.index(i + 1)] for i in range(num_combos)]
        context_combo_scores = [
            context_combo_scores[context_ranks.index(i + 1)] for i in range(num_combos)
        ]

        return context_combos, context_combo_scores

    def postprocess(self, context_combos):
        """Postprocess context combos by converting categories to names."""
        output = []
        for c1_name, s1_name, s2_name, r1_name, r2_name, T_pred in context_combos:
            cat_name = [c1_name]
            if r2_name == "":
                rgt_name = [r1_name]
            else:
                rgt_name = [r1_name, r2_name]
            if s2_name == "":
                slv_name = [s1_name]
            else:
                slv_name = [s1_name, s2_name]

            if self.with_smiles:
                rgt_name = [rgt for rgt in rgt_name if "Reaxys" not in rgt]
                slv_name = [slv for slv in slv_name if "Reaxys" not in slv]
                cat_name = [cat for cat in cat_name if "Reaxys" not in cat]

            output.append(
                [float(T_pred), ".".join(slv_name), ".".join(rgt_name), ".".join(cat_name)]
            )

        return output

    def load_ehs_dictionary(self, ehs_score_path):
        """Populates self.ehs_dict with mapping of solvent to EHS scores.

        Assumes CSV input file does not have any entries that are not valid
        ASKCOS solvents.

        Unscored solvents receive a score of 7.
        Otherwise, scores range 1 (best) to 6 (worst).

        Args:
            ehs_score_path (str): path to a csv file pairing valid ASKCOS solvents with an EHS score

        Returns:
            None
        """
        self.ehs_dict = {}
        with open(ehs_score_path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Skip the first line (header)
                    continue
                a = line.strip().split(",")  # Remove whitespace and split by commas
                key = a[2]
                value = a[3]
                if value.isdigit():
                    value = int(value)
                else:
                    value = 7
                self.ehs_dict[key] = value

    def contexts_ehs_scores(self, top_combos):
        """Appends EHS score information to each context object in input list.

        Adds a solvent EHS score and a boolean indicating whether the score is
        the best out of all contexts in the input list.
        """
        # Assign scores and get best score
        best_score = self.combo_ehs_score(top_combos)
        for item in top_combos:
            item.append(item[-1] == best_score)
        return top_combos

    def combo_ehs_score(self, context_combos, best=True):
        """Determines EHS score information for each context in input list.

        Modifies items in input list by appending EHS score.

        Args:
            context_combos (list): list of potential reaction conditions in the format returned by get.n.conditions
            best (bool, optional): if True, returns best solvent score, otherwise returns average score (default: True)

        Returns:
            int if best=True, else float
        """
        scores = []
        for item in context_combos:
            solvent = item[1]
            if solvent in self.ehs_dict:
                score = self.ehs_dict[solvent]
            elif "." in solvent:  # solvent is actually multiple solvents
                solvents = solvent.split(".")
                sub_scores = []
                for s in solvents:
                    if s in self.ehs_dict:
                        sub_scores.append(self.ehs_dict[s])
                if sub_scores:
                    score = sum(sub_scores) / len(sub_scores)
                else:
                    score = None
            else:
                score = None
            item.append(score)
            if score is not None:
                scores.append(score)

        if scores:
            if best:
                # Return best score
                return min(scores)
            else:
                # Return average score
                return sum(scores) / len(scores)
        else:
            return 8


if __name__ == "__main__":
    model = NeuralNetContextRecommender().load()
    print(
        model.get_n_conditions(
            "CC1(C)OBOC1(C)C.Cc1ccc(Br)cc1>>Cc1cccc(B2OC(C)(C)C(C)(C)O2)c1",
            None,
            10,
            with_smiles=False,
            return_scores=True,
        )
    )
