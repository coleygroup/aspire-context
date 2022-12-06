import sys
from typing import Optional

import numpy as np
from rdkit.Chem import AllChem as Chem


def get_morgan_fp(smi: str, radius: int, length: int):
    return np.array(
        Chem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi, sanitize=True), radius, nBits=length
        ),
        dtype="float32",
    )


def canonicalize_smiles_rdkit(smi: str) -> Optional[str]:
    try:
        # avoid 'Br[Br-]Br' problem
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False))
    except Exception:
        sys.stderr.write("canonicalize_smiles_rdkit(): fail s=" + smi + "\n")

    return None


def canonicalize_smiles(smi: str) -> str:
    return canonicalize_smiles_rdkit(smi)
