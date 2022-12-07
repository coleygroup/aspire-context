import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

logger = logging.getLogger(__name__)


def mol_smi_to_morgan_fp(
    smi: str,
    radius: int = 2,
    length: int = 2048,
    as_column: bool = False,
    raise_exceptions: bool = False,
    dtype: str = "float32",
    **fp_kwargs,
) -> np.ndarray:
    """
    Create Morgan Fingerprint from molecule SMILES.
    Returns correctly shaped zero vector on errors.

    Args:
        smi (str): input molecule SMILES
        radius (int, optional): fingerprint radius, default 2
        length (int, optional): fingerprint length, default 2048
        as_column (bool, optional): return fingerprint as column vector
        raise_exceptions (bool, optional): raise exceptions instead of returning zero vector
        dtype (str, optional): data type of the generated fingerprint array
        **kwargs: passed to GetMorganFingerprintAsBitVect

    Returns:
        np.array of shape (length,) or (1, length) if as_column = True
    """
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception as e:
        logger.warning(f"Unable to parse SMILES {smi}: {e!s}")
        if raise_exceptions:
            raise
        fp = np.zeros(length, dtype)
    else:
        try:
            fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=length, **fp_kwargs)
            fp = np.empty(length, dtype)
            DataStructs.ConvertToNumpyArray(fp_bit, fp)
        except Exception as e:
            logger.warning(f"Unable to generate fingerprint for {smi}: {e!s}")
            if raise_exceptions:
                raise
            fp = np.zeros(length, dtype)

    if as_column:
        return fp.reshape(1, -1)
    else:
        return fp


def reac_prod_smi_to_morgan_fp(
    reactant: str,
    pdt: str,
    radius: int = 2,
    length: int = 2048,
    as_column: bool = False,
    raise_exceptions: bool = False,
    **fp_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create Morgan Fingerprints from reactant and product SMILES separately.

    Args:
        rsmi (str): reactant molecule SMILES
        psmi (str): product molecule SMILES
        radius (int, optional): fingerprint radius, default 2
        length (int, optional): fingerprint length, default 2048
        as_column (bool, optional): return fingerprints as column vector
        raise_exceptions (bool, optional): raise exceptions instead of returning zero vector
        **kwargs: passed to GetMorganFingerprintAsBitVect

    Returns:
        product np.array of shape (length,) or (1, length) if as_column = True
        reactant np.array of shape (length,) or (1, length) if as_column = True
    """
    params = dict(
        radius=radius, length=length, as_column=as_column, raise_exceptions=raise_exceptions
    )
    rfp = mol_smi_to_morgan_fp(reactant, **params, **fp_kwargs)
    pfp = mol_smi_to_morgan_fp(pdt, **params, **fp_kwargs)

    return pfp, rfp


def rxn_smi_to_morgan_fp(
    rxn,
    radius: int = 2,
    length: int = 2048,
    as_column: bool = False,
    raise_exceptions: bool = False,
    **fp_kwargs,
) -> np.ndarray:
    """
    Create Morgan Fingerprint from reaction SMILES. Ignores agents.

    Args:
        rxn (str): input reaction SMILES
        radius (int, optional): fingerprint radius, default 2
        length (int, optional): fingerprint length, default 2048
        as_column (bool, optional): return fingerprints as column vector
        raise_exceptions (bool, optional): raise exceptions instead of returning zero vector
        **kwargs: passed to GetMorganFingerprintAsBitVect

    Returns:
        np.array of shape (length,) or (1, length) if as_column = True
    """
    rsmi, _, psmi = rxn.split(">")

    params = dict(
        radius=radius, length=length, as_column=as_column, raise_exceptions=raise_exceptions
    )
    rfp = mol_smi_to_morgan_fp(rsmi, **params**fp_kwargs)
    pfp = mol_smi_to_morgan_fp(psmi, **params, **fp_kwargs)

    if pfp is not None and rfp is not None:
        pfp -= rfp

    return pfp
