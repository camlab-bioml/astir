from typing import Tuple, List, Dict
import warnings

import torch
import numpy as np

from .scdataset import SCDataset


class AbstractModel:
    """Abstract class to perform statistical inference to assign
    """

    def __init__(
        self, dset: SCDataset, include_beta: bool, random_seed: int, dtype: torch.dtype,
    ) -> None:

        if not isinstance(random_seed, int):
            raise NotClassifiableError("Random seed is expected to be an integer.")
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)

        if dtype != torch.float32 and dtype != torch.float64:
            raise NotClassifiableError(
                "Dtype must be one of torch.float32 and torch.float64."
            )
        self._dtype = dtype
        self._data = None
        self._variables = None
        self._losses = None

        self._dset = dset
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._include_beta = include_beta
        self._is_converged = False

    def get_losses(self) -> float:
        """ Getter for losses

        :return: self.losses
        :rtype: float
        """
        if self._losses is None:
            raise Exception("The model has not been trained yet")
        return self._losses

    def get_scdataset(self):
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


class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """

    pass
