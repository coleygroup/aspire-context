from abc import ABC, abstractmethod


class ReactionContextRecommender(ABC):
    @abstractmethod
    def recommend(
        self, smi: str, reagents: list[str] | None, n_conditions: int, *args, **kwargs
    ) -> list:
        """Predict reaction conditions for the input reaction SMILES.


        Parameters
        ----------
        smi : str
            the reaction SMILES including reactants and products
        reagents : list[str] | None
            additional reagents not included in the reaction SMILES.
            NOTE: this may be ignored depending on the implementation
        n_conditions : int
            the number of conditions to return
        *args, **kwargs
            additional positional and keyword arguments
            
        Returns
        -------
        list
            a list of conditions, the format of which is implementation-dependent
        """

    def predict(
        self, smiles: str, reagents: list[str] | None, n_conditions: int, *args, **kwargs
    ):
        return self.recommend(smiles, reagents, n_conditions, *args, **kwargs)