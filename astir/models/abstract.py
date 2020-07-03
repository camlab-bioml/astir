from typing import Tuple, List, Dict
import warnings

import torch
import numpy as np

from astir.data import SCDataset


class AstirModel:
    """Abstract class to perform statistical inference to assign. This module is the super class of 
        `CellTypeModel` and `CellStateModel` and is not supposed to be instantiated.
    """

    def __init__(self, dset: SCDataset, random_seed: int, dtype: torch.dtype) -> None:

        if not isinstance(random_seed, int):
            raise NotClassifiableError("Random seed is expected to be an integer.")
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)

        if dtype != torch.float32 and dtype != torch.float64:
            raise NotClassifiableError(
                "dtype must be one of torch.float32 and torch.float64."
            )
        elif dtype != dset.get_dtype():
            raise NotClassifiableError(
                "dtype must be the same as `dset`."
            )
        self._dtype = dtype
        self._data = None
        self._variables = None
        self._losses = None

        self._dset = dset
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_converged = False

    def get_losses(self) -> float:
        """ Getter for losses.

        :return: self.losses
        :rtype: float
        """
        if self._losses is None:
            raise Exception("The model has not been trained yet")
        return self._losses

    def get_scdataset(self) -> SCDataset:
        """Getter for the `SCDataset`.

        :return: `self._dset`
        :rtype: SCDataset
        """
        return self._dset

    def get_data(self):
        return self._data

    def get_variables(self):
        """ Returns all variables

        :return: self._variables
        """
        return self._variables

    def is_converged(self) -> bool:
        """ Returns True if the model converged

        :return: self._is_converged
        """
        return self._is_converged

    def _param_init(self) -> None:
        raise NotImplementedError("AbstractModel is not supposed to be instantiated.")

    def _forward(
        self, Y: torch.Tensor, X: torch.Tensor, design: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("AbstractModel is not supposed to be instantiated.")

    def fit(
        self,
        max_epochs: int,
        learning_rate: float,
        batch_size: int,
        delta_loss: float,
        msg: str,
    ) -> None:
        raise NotImplementedError("AbstractModel is not supposed to be instantiated.")


class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """

    pass
