from abc import abstractmethod
import copy
import json
from pathlib import Path
import pickle
from typing import Optional

import numpy as np
import tensorflow as tf

from askcos_context.common.services.recommender import ReactionContextRecommender
from askcos_context.v2.services import search, results_preprocess, utils
from askcos_context.v2.services.config import (
    DEFAULT_CONFIG,
    ModelConfig,
    FpModelConfig,
    GraphModelConfig,
)


def add_batch_dimension(x):
    """
    Insert a new axis at position 0 to the array or each array in the list.
    """
    if isinstance(x, list):
        return [tf.convert_to_tensor(np.expand_dims(i, axis=0), dtype=tf.float32) for i in x]
    else:
        return tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float32)


def get_input_dims(atom, bond, batched=False):
    """
    atom: (batch_size, atom_dim, atom_nfeature)
    bond: (batch_size, atom_dim, atom_dim, bond_nfeature)
    or
    atom: (atom_dim, atom_nfeature)
    bond: (atom_dim, atom_dim, bond_nfeature)
    """
    if batched:
        atom_dim = atom.shape[1]
        atom_nfeature = atom.shape[-1]
        bond_nfeature = bond.shape[-1]
    else:
        atom_dim = atom.shape[0]
        atom_nfeature = atom.shape[-1]
        bond_nfeature = bond.shape[-1]
    return atom_dim, atom_nfeature, bond_nfeature


def load_model(model_dir):
    imported = tf.saved_model.load(export_dir=model_dir, tags="serve")
    imported_function = imported.signatures["serving_default"]
    return imported, imported_function


class ReactionContextRecommenderBase(ReactionContextRecommender):
    def __init__(self, model_name: Optional[str] = None, config: Optional[ModelConfig] = None):
        self.model_name = model_name

        self.max_num_reagents = 4
        self.max_num_reactants = 5

        self.reagent_encoder_path = None
        self.reagent_encoder = None
        self.reagent_decoder = None

        self.reagent_model_path = None
        self.reagent_model_tf = None
        self.reagent_model = None

        self.temperature_model_path = None
        self.temperature_model_tf = None
        self.temperature_model = None

        self.reagent_quantity_model_path = None
        self.reagent_quantity_model_tf = None
        self.reagent_quantity_model = None

        self.reactant_quantity_model_path = None
        self.reactant_quantity_model_tf = None
        self.reactant_quantity_model = None

        if self.model_name is not None:
            try:
                config = DEFAULT_CONFIG.model_configs[self.model_name]
            except KeyError:
                raise KeyError(
                    f"Unrecognized model name, please check global config: {self.model_name}"
                )

        self.reagent_encoder_path = config.reagents_path
        self.reagent_model_path = config.reagents_model_path
        self.temperature_model_path = config.temperature_model_path
        self.reagent_quantity_model_path = config.reagent_quantity_model_path
        self.reactant_quantity_model_path = config.reactant_quantity_model_path

    @abstractmethod
    def encode_condensed_graph(self, smiles, **kwargs):
        """Encode the input SMILES string"""

    @abstractmethod
    def predict_reactant_quantities(self, smiles=None, reagents=None, encoded_reagents=None):
        pass

    def check_model_paths(self, *args):
        """Check that the configured model paths exist.

        Parameters
        ----------
        *args : Iterable[PathLike]
            Additional file paths to check

        Raises
        ------
        ValueError
            if any of the model paths do not exist.
        """
        paths = [
            self.reagent_encoder_path,
            self.reagent_model_path,
            self.temperature_model_path,
            self.reagent_quantity_model_path,
            self.reactant_quantity_model_path,
        ]
        paths.extend(args)

        for path in paths:
            if path is None or not Path(path).exists():
                raise ValueError(f"Missing model(s) for reaction context recommender: {path}")

    def load_models(self):
        # reagent encoder/decoder
        with open(self.reagent_encoder_path, "r") as f:
            self.reagent_encoder = results_preprocess.generate_reagents_encoder2(json.load(f))
        self.reagent_decoder = {v: k for k, v in self.reagent_encoder.items()}

        # reagent predictor
        self.reagent_model_tf, self.reagent_model = load_model(self.reagent_model_path)

        # temperature predictor
        self.temperature_model_tf, self.temperature_model = load_model(self.temperature_model_path)

        # reagent quantity predictor
        self.reagent_quantity_model_tf, self.reagent_quantity_model = load_model(
            self.reagent_quantity_model_path
        )

        # reactant quantity predictor
        self.reactant_quantity_model_tf, self.reactant_quantity_model = load_model(
            self.reactant_quantity_model_path
        )

        return self

    def encode_reagents(self, reagents):
        """
        Args:
            reagents: list of strings, each string is a reagent SMILES

        Returns:
            Sum of one-hot encoding, 0 or 1
        """
        reagents = [{"smiles": r} for r in reagents]
        # for i in reagents:
        #     r.append()
        reagents_onehot = results_preprocess.prepare_reagents2(self.reagent_encoder, reagents)
        reagents_multiclass = results_preprocess.convert_onehots_to_multiclass(reagents_onehot)

        return add_batch_dimension(reagents_multiclass)

    def decode_reagents(self, encoded_reagents):
        return [self.reagent_decoder[i] for i in np.where(np.abs(encoded_reagents - 1.0) < 1e-6)[0]]

    def predict_reagents(
        self,
        smiles: Optional[str] = None,
        beam_size: int = 10,
        is_decode: bool = True,
        encoded_graph=None,
        reagents=None,
    ):
        """
        Returns:
            if is_decode: res = [(reagents_onehot, score)]
            else: res = [([reagents_smiles], score)]
        """
        if encoded_graph is None:
            encoded_graph = self.encode_condensed_graph(smiles)

        encoded_reagents = self.encode_reagents(reagents) if reagents is not None else None
        
        res = search.beam_search(
            self.reagent_model,
            copy.copy(encoded_graph),
            len(self.reagent_encoder),
            max_steps=self.max_num_reagents + 1,
            beam_size=beam_size,
            eos_id=0,
            keepall=False,
            reagents=encoded_reagents,
        )  # keepall=False is beam search
        res_top = search.top_n_results(res, n=beam_size)

        if is_decode:
            return [(self.decode_reagents(res[0]), float(res[1])) for res in res_top]
            # for r in res_top:
            #     res_top_decode.append((self.decode_reagents(r[0]), float(r[1])))
            # return res_top_decode

        return res_top

    def predict_temperature(
        self, smiles=None, reagents=None, encoded_graph=None, encoded_reagents=None
    ):
        if encoded_graph is None:
            encoded_graph = self.encode_condensed_graph(smiles)
        if encoded_reagents is None:
            encoded_reagents = self.encode_reagents(reagents)
        data_input = copy.copy(encoded_graph)
        data_input["Input_reagent"] = encoded_reagents
        y_pred = self.temperature_model(**data_input)["output_regression"].numpy()

        return float(y_pred[0][0] * 273.15)

    def predict_reagent_quantities(
        self, smiles=None, reagents=None, encoded_graph=None, encoded_reagents=None
    ):
        if encoded_graph is None:
            encoded_graph = self.encode_condensed_graph(smiles)
        if encoded_reagents is None:
            encoded_reagents = self.encode_reagents(reagents)
        data_input = copy.copy(encoded_graph)
        data_input["Input_reagent"] = encoded_reagents
        y_pred = self.reagent_quantity_model(**data_input)["multiply_1"].numpy()

        amount = {}
        for i in np.where(encoded_reagents[0, :] > 1e-6)[0]:
            amount[self.reagent_decoder[i]] = float(np.exp(y_pred[0, i]))
        return amount

    def recommend(self, smi: str, reagents: list[str] | None = None, n_conditions: int = 10):
        """
        Parameters
        ----------
        smiles : str
        reagents : list[str] | None, default=None
        n_conditions : int, default=10

        Returns:
        -------
        list[dict]
            a list of condition sets, where each condition set is a dict of the form::
                {
                    'reagents': {'reagent_0_smi': mol_0, ... 'reagent_i_smi': mol_i},
                    'reactants': {'reactant_0_smi': mol_0, ..., 'reactan_j_smi': mol_j},
                    'temperature': T,
                    'reagents_score': score
                }

        Raises
        ------
        Exception
            May throw exceptions due to encoding failures

        Notes
        -----
        * `reagents_score` is the product of individual reagent score, so it is very small.
        """
        encoded_graph = self.encode_condensed_graph(smi)
        reagents = self.predict_reagents(
            smiles=None,
            beam_size=n_conditions,
            is_decode=False,
            encoded_graph=encoded_graph,
            reagents=reagents,
        )

        res = []
        for encoded_reagents, score in reagents:
            res_one = {"reagents_score": float(score)}
            # reagents_smiles = self.decode_reagents(encoded_reagents)
            encoded_reagents = add_batch_dimension(encoded_reagents)
            res_one["temperature"] = self.predict_temperature(
                smiles=None,
                reagents=None,
                encoded_graph=encoded_graph,
                encoded_reagents=encoded_reagents,
            )  # Kelvin
            res_one["reagents"] = self.predict_reagent_quantities(
                smiles=None,
                reagents=None,
                encoded_graph=encoded_graph,
                encoded_reagents=encoded_reagents,
            )
            res_one["reactants"] = self.predict_reactant_quantities(
                smiles=smi, reagents=None, encoded_reagents=encoded_reagents
            )
            # append results
            res.append(res_one)

        return res


class ReactionContextRecommenderWLN(ReactionContextRecommenderBase):
    """
    This predictor requires atom mapped reactions.
    Require number of atoms in all reactants <= 50.
    max_num_reagents   <= 4
    max_num_reactants  <= 5
    """

    def __init__(
        self,
        model_name=DEFAULT_CONFIG.default_models["graph"],
        config: Optional[GraphModelConfig] = None,
        *args, **kwargs
    ):
        super().__init__(model_name, config)

        self.feature_encoder_path = None
        self.feature_encoder = None
        self.condensed_graph = None  # True if atom-mapping is required

        if self.model_name is not None:
            try:
                config = DEFAULT_CONFIG.model_configs[self.model_name]
            except KeyError:
                raise KeyError(
                    f"Unrecognized model name, please check global config: {self.model_name}"
                )

        self.feature_encoder_path = config.encoder_path
        self.condensed_graph = config.condensed_graph

        self.check_model_paths()

    def check_model_paths(self, *args):
        super().check_model_paths(self.feature_encoder_path)

    def load_models(self):
        with open(self.feature_encoder_path, "rb") as f:
            self.feature_encoder = pickle.load(f)

        return super().load_models()

    def encode_condensed_graph(self, smiles, **kwargs):
        # feature
        f = utils.rxn2features(smiles)
        atom, bond, conn = utils.encode_features_atommapped_dense_graph(
            f, self.feature_encoder, isrand=False
        )
        atom, bond, conn = add_batch_dimension([atom, bond, conn])
        return {
            "Input_atom": tf.convert_to_tensor(atom, dtype=tf.float32),
            "Input_bond": tf.convert_to_tensor(bond, dtype=tf.float32),
            "Input_conn": tf.convert_to_tensor(conn, dtype=tf.float32),
        }

    def predict_reactant_quantities(self, smiles=None, reagents=None, encoded_reagents=None):
        """
        Reactants in smiles are splitted by '.'

        Returns:
            dict: {'reactants': mol}

        Raises:
            May throw exceptions due to encoding failures.
        """
        # all reactants and products
        f = utils.rxn2features(smiles)
        (atom1, bond1, conn1, atom2, bond2, conn2) = utils.encode_features_non_mapped(
            f, self.feature_encoder, isrand=False
        )
        atom1, bond1, conn1, atom2, bond2, conn2 = add_batch_dimension(
            [atom1, bond1, conn1, atom2, bond2, conn2]
        )

        # reagents
        if encoded_reagents is None:
            encoded_reagents = self.encode_reagents(reagents)

        data_input = {
            "Input_atom_reactants": atom1,
            "Input_bond_reactants": bond1,
            "Input_conn_reactants": conn1,
            "Input_atom_products": atom2,
            "Input_bond_products": bond2,
            "Input_conn_products": conn2,
            "Input_reagent": encoded_reagents,
        }

        # individual reactants
        reactants = smiles.split(">")[0].split(".")
        if len(reactants) > self.max_num_reactants:
            raise ValueError(
                "Number of reactants ({}) is greater than the allowed maximum {}.".format(
                    len(reactants), self.max_num_reactants
                )
            )
        atom_dim, atom_nfeature, bond_nfeature = get_input_dims(atom1, bond1, batched=True)
        mask = np.zeros((1, self.max_num_reactants), dtype=np.float32)
        for i, r in enumerate(reactants):
            f = utils.smiles2features(r)
            a, b, c = utils.encode_features_onemol(f, self.feature_encoder, isrand=False)
            data_input[f"Input_atom_{i}"] = add_batch_dimension(a)
            data_input[f"Input_bond_{i}"] = add_batch_dimension(b)
            data_input[f"Input_conn_{i}"] = add_batch_dimension(c)
            mask[0, i] = 1

        # append empty vector
        for i in range(len(reactants), self.max_num_reactants):
            data_input[f"Input_atom_{i}"] = tf.zeros((1, atom_dim, atom_nfeature), dtype=tf.float32)
            data_input[f"Input_bond_{i}"] = tf.zeros(
                (1, atom_dim, atom_dim, bond_nfeature), dtype=tf.float32
            )
            data_input[f"Input_conn_{i}"] = tf.zeros((1, atom_dim, atom_dim), dtype=tf.float32)

        data_input["Input_reactant_mask"] = tf.convert_to_tensor(mask, dtype=tf.float32)
        y_pred = self.reactant_quantity_model(**data_input)["multiply_1"].numpy()

        amount = {}
        for i, r in enumerate(reactants):
            amount[r] = float(np.exp(y_pred[0, i]))
        return amount


class ReactionContextRecommenderFP(ReactionContextRecommenderBase):
    """
    This predictor does not require atom mapped reactions.
    Require number of atoms in all reactants <= 50.
    max_num_reagents   <= 4
    max_num_reactants  <= 5
    """

    def __init__(
        self, model_name=DEFAULT_CONFIG.default_models["fp"], config: Optional[FpModelConfig] = None
    ):
        super().__init__(model_name, config)

        if self.model_name is not None:
            try:
                config = DEFAULT_CONFIG.model_configs[self.model_name]
            except KeyError:
                raise KeyError(
                    f"Unrecognized model name, please check global config: {self.model_name}"
                )

        self.radius = config.radius
        self.length = config.length

        self.check_model_paths()

    def encode_condensed_graph(self, smiles, fp_length=None, fp_radius=None):
        if fp_length is None:
            fp_length = self.length
        if fp_radius is None:
            fp_radius = self.radius
        # feature
        smiles_splitted = smiles.split(" ")[0].split(">")
        r_fp = utils.get_morgan_fp(smiles_splitted[0], fp_radius, fp_length)
        p_fp = utils.get_morgan_fp(smiles_splitted[2], fp_radius, fp_length)
        input_fp = np.concatenate([r_fp, p_fp], axis=-1)
        return {"Input_fp": add_batch_dimension(input_fp)}

    def predict_reactant_quantities(self, smiles=None, reagents=None, encoded_reagents=None):
        """
        Reactants in smiles are splitted by '.'

        Returns:
            dict: {'reactants': mol}

        Raises:
            May throw exceptions due to encoding failures.
        """
        # all reactants and products
        encoded_graph = self.encode_condensed_graph(smiles)

        # reagents
        if encoded_reagents is None:
            encoded_reagents = self.encode_reagents(reagents)

        data_input = copy.copy(encoded_graph)
        data_input["Input_reagent"] = encoded_reagents

        # individual reactants
        reactants = smiles.split(">")[0].split(".")
        if len(reactants) > self.max_num_reactants:
            raise ValueError(
                "Number of reactants ({}) is greater than the allowed maximum {}.".format(
                    len(reactants), self.max_num_reactants
                )
            )
        mask = np.zeros(shape=(1, self.max_num_reactants), dtype=np.float32)
        for i, r in enumerate(reactants):
            r_fp = utils.get_morgan_fp(r, self.radius, self.length)
            data_input[f"Input_fp_reactant_{i}"] = add_batch_dimension(r_fp)
            mask[0, i] = 1

        # append empty vector
        for i in range(len(reactants), self.max_num_reactants):
            data_input[f"Input_fp_reactant_{i}"] = tf.zeros((1, self.length), dtype=tf.float32)

        data_input["Input_reactant_mask"] = tf.convert_to_tensor(mask, dtype=tf.float32)
        y_pred = self.reactant_quantity_model(**data_input)["multiply_1"].numpy()

        amount = {}
        for i, r in enumerate(reactants):
            amount[r] = float(np.exp(y_pred[0, i]))
        return amount


def __test_wln():
    predictor_wln = ReactionContextRecommenderWLN()
    predictor_wln.load_models()
    # {'id': 84022, 'db_id': None, 'date': '20190801', 'rxn_smiles': '[N:1]#[C:2][CH:3]([C:4](=O)[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1)[c:11]1[cH:12][cH:13][cH:14][cH:15][cH:16]1.[NH2:17][NH2:18]>CC(=O)O.CCO>[NH2:1][c:2]1[nH:18][n:17][c:4]([c:3]1-[c:11]1[cH:16][cH:15][cH:14][cH:13][cH:12]1)-[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1', 'temperature': None, 'reactants': [{'smiles': 'N#CC(C(=O)c1ccccc1)c1ccccc1', 'conc': None, 'unit': None, 'value': None, 'is_set': True, 'mole': None, 'mass': None, 'volume': None}, {'smiles': 'NN', 'conc': None, 'unit': None, 'value': None, 'is_set': True, 'mole': None, 'mass': None, 'volume': None}], 'products': [{'smiles': 'Nc1[nH]nc(c1c1ccccc1)c1ccccc1', 'yield': None, 'unit': None, 'value': None, 'is_set': True, 'mole': None, 'mass': None, 'volume': None}], 'reagents': [{'smiles': 'CC(=O)O', 'conc': None, 'unit': None, 'value': None, 'is_set': True, 'mole': None, 'mass': None, 'volume': None}, {'smiles': 'CCO', 'conc': None, 'unit': None, 'value': None, 'is_set': True, 'mole': None, 'mass': None, 'volume': None}], 'filepath': 'applications/2019/I20190801_cdx_reactions_wbib.json', 'filelinenum': 1434}
    s = "[N:1]#[C:2][CH:3]([C:4](=O)[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1)[c:11]1[cH:12][cH:13][cH:14][cH:15][cH:16]1.[NH2:17][NH2:18]>>[NH2:1][c:2]1[nH:18][n:17][c:4]([c:3]1-[c:11]1[cH:16][cH:15][cH:14][cH:13][cH:12]1)-[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1"
    reagents_true = ["CC(=O)O", "CCO"]
    reagents = predictor_wln.predict_reagents(smiles=s)
    print("ReactionContextRecommenderWLN")
    print("reagents: ", json.dumps(reagents, indent=4))
    t = predictor_wln.predict_temperature(smiles=s, reagents=reagents_true)
    print("temperature (K): ", t)
    print()

    results = predictor_wln.recommend(smi=s)
    print("ReactionContextRecommenderWLN")
    print("all predictions")
    print(json.dumps(results, indent=4))
    print()

    results = predictor_wln.recommend(smi=s, reagents=["CC(=O)O"])
    print("ReactionContextRecommenderWLN")
    print("preset reagents")
    print(json.dumps(results, indent=4))
    print()

    print("test")
    s = "CC(=O)O.Fc1ccccc1Nc1cccc2ccccc12>>Cc1c2cccc(F)c2nc2c1ccc1ccccc12"
    results = predictor_wln.recommend(smi=s)
    print("ReactionContextRecommenderWLN")
    print("all predictions")
    print(json.dumps(results, indent=4))
    s = "CC(=O)O.Fc1ccccc1Nc1ccccc1>>Cc1c2cccc(F)c2nc2c1cccc2"
    results = predictor_wln.recommend(smi=s)
    print("ReactionContextRecommenderWLN")
    print("smaller rings")
    print(json.dumps(results, indent=4))
    s = "CC(=O)O.c1ccccc1>>CC(=O)c1ccccc1"
    results = predictor_wln.recommend(smi=s)
    print("ReactionContextRecommenderWLN")
    print("benzene rings")
    print(json.dumps(results, indent=4))
    s = "CC(=O)Cl.c1ccccc1>>CC(=O)c1ccccc1"
    results = predictor_wln.recommend(smi=s)
    print("ReactionContextRecommenderWLN")
    print("benzene Cl rings")
    print(json.dumps(results, indent=4))


def __test_fp():
    print("ReactionContextRecommenderFP")
    predictor_fp = ReactionContextRecommenderFP()
    predictor_fp.load_models()
    s = "[N:1]#[C:2][CH:3]([C:4](=O)[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1)[c:11]1[cH:12][cH:13][cH:14][cH:15][cH:16]1.[NH2:17][NH2:18]>>[NH2:1][c:2]1[nH:18][n:17][c:4]([c:3]1-[c:11]1[cH:16][cH:15][cH:14][cH:13][cH:12]1)-[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1"
    reagents_true = ["CC(=O)O", "CCO"]
    reagents = predictor_fp.predict_reagents(smiles=s)
    print("reagents: ", json.dumps(reagents, indent=4))
    t = predictor_fp.predict_temperature(smiles=s, reagents=reagents_true)
    print("temperature (K): ", t)
    print()

    results = predictor_fp.recommend(smi=s)
    print("ReactionContextRecommenderFP")
    print("all predictions")
    print(json.dumps(results, indent=4))
    print()

    results = predictor_fp.recommend(smi=s, reagents=["OCC"])
    print("ReactionContextRecommenderWLN")
    print("preset reagents")
    print(json.dumps(results, indent=4))
    print()


if __name__ == "__main__":
    __test_wln()
    __test_fp()
