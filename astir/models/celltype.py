# astir: Automated aSsignmenT sIngle-cell pRoteomics

# VSCode tips for python:
# ctrl+` to open terminal
# cmd+P to go to file

import re
from typing import Tuple, List, Dict
import warnings

import torch
from torch.autograd import Variable
from torch.distributions import Normal, StudentT
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from astir.models.imcdataset import IMCDataSet
from astir.models.recognet import RecognitionNet


class CellTypeModel:
    """Loads a .csv expression file and a .yaml marker file.

    :raises NotClassifiableError: raised when the input gene expression
        data or the marker is not classifiable

    :param assignments: cell type assignment probabilities
    :param losses:losses after optimization
    :param type_dict: dictionary mapping cell type
        to the corresponding genes
    :param state_dict: dictionary mapping cell state
        to the corresponding genes
    :param N: number of rows of data
    :param G: number of cell type genes
    :param C: number of cell types
    :param initializations: initialization parameters
    :param data: parameters that is not to be optimized
    :param variables: parameters that is to be optimized
    :param include_beta: [summery]
    """
    def _param_init(self) -> None:
        """Initialize parameters and design matrices.
        """
        self.initializations = {
            "mu": np.log(self.Y_np.mean(0)).reshape((-1,1)),
            "log_sigma": np.log(self.Y_np.std(0))
        }

        # Add additional columns of mu for anything in the design matrix
        P = self.dset.design.shape[1]
        self.initializations['mu'] = np.column_stack( \
            [self.initializations['mu'], np.zeros((self.G, P-1))])

        ## prior on z
        self.variables = {
            "log_sigma": Variable(torch.from_numpy(
                self.initializations["log_sigma"].copy()), \
                    requires_grad = True),
            "mu": Variable(torch.from_numpy(\
                self.initializations["mu"].copy()), requires_grad = True),
            "log_delta": Variable(0 * torch.ones((self.G,self.C+1)), \
                requires_grad = True)
        }

        self.data = {
            "log_alpha": torch.log(torch.ones(self.C+1) / (self.C+1)),
            "delta": torch.exp(self.variables["log_delta"]),
            "rho": torch.from_numpy(self.marker_mat)
        }

        if self.include_beta:
            self.variables['beta'] = Variable(\
                torch.zeros(self.G, self.C+1), requires_grad=True)

    ## Declare pytorch forward fn
    def _forward(self, Y: torch.Tensor , X: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
        """[summary]

        :param Y: [description]
        :type Y: torch.Tensor
        :param X: [description]
        :type X: torch.Tensor
        :param design: [description]
        :type design: torch.Tensor

        :return: [description]
        :rtype: torch.Tensor
        """
        Y_spread = Y.reshape(-1, self.G, 1).repeat(1, 1, self.C+1)

        delta_tilde = torch.exp(self.variables["log_delta"]) # + np.log(0.5)
        mean =  delta_tilde * self.data["rho"] 
        mean2 = torch.matmul(design, self.variables['mu'].T) ## N x P * P x G
        mean2 = mean2.reshape(-1, self.G, 1).repeat(1, 1, self.C+1)
        mean = mean + mean2

        if self.include_beta:
            with torch.no_grad():
                min_delta = torch.min(delta_tilde, 1).values.reshape((self.G,1))
            mean = mean + min_delta * torch.tanh(self.variables["beta"]) * (1 - self.data["rho"]) 

        dist = Normal(torch.exp(mean), torch.exp(self.variables["log_sigma"]).reshape(-1, 1))
        # dist = StudentT(torch.tensor(1.), torch.exp(mean), torch.exp(self.variables["log_sigma"]).reshape(-1, 1))

        log_p_y = dist.log_prob(Y_spread)
        log_p_y_on_c = log_p_y.sum(1)
        gamma = self.recog.forward(X)
        elbo = ( gamma * (log_p_y_on_c + self.data["log_alpha"] - torch.log(gamma)) ).sum()
        
        return -elbo

    ## Todo: an output function
    def __init__(self, Y_np: np.array, type_dict: Dict,  \
                N: int, G: int, C: int, type_mat: np.array, \
                include_beta = True, design = None
                ) -> None:
        """Initializes an Astir object

        :param df_gex: the input gene expression dataframe
        :type df_gex: pd.DataFrame
        :param marker_dict: the gene marker dictionary
        :type marker_dict: Dict

        :param design: [description], defaults to None
        :type design: [type], optional
        :param random_seed: [description], defaults to 1234
        :type random_seed: int, optional
        :param include_beta: [description], defaults to True
        :type include_beta: bool, optional

        :raises NotClassifiableError: raised when randon seed is not an integer
        """
        self.losses = None # losses after optimization

        self.type_dict = type_dict

        self.N, self.G, self.C = N, G, C

        # Does this model use separate beta?
        self.include_beta = include_beta

        self.marker_mat = type_mat
        self.Y_np = Y_np

        if design is not None:
            if isinstance(design, pd.DataFrame):
                design = design.to_numpy()

        self.dset = IMCDataSet(self.Y_np, design)
        self.recog = RecognitionNet(self.C, self.G)

        self._param_init()

    def fit(self, epochs = 100, learning_rate = 1e-2, 
        batch_size = 1024) -> None:
        """Fit the model.

        :param epochs: [description], defaults to 100
        :type epochs: int, optional
        :param learning_rate: [description], defaults to 1e-2
        :type learning_rate: [type], optional
        :param batch_size: [description], defaults to 1024
        :type batch_size: int, optional
        """
        ## Make dataloader
        dataloader = DataLoader(self.dset, batch_size=min(batch_size, self.N),\
            shuffle=True)
        
        ## Run training loop
        losses = np.empty(epochs)

        ## Construct optimizer
        opt_params = list(self.variables.values()) + \
            list(self.recog.parameters())

        if self.include_beta:
            opt_params = opt_params + [self.variables["beta"]]
        optimizer = torch.optim.Adam(opt_params, lr=learning_rate)

        for ep in range(epochs):
            L = None
            for batch in dataloader:
                Y,X,design = batch
                optimizer.zero_grad()
                L = self._forward(Y, X, design)
                L.backward()
                optimizer.step()
            l = L.detach().numpy()
            losses[ep] = l
            print(l)

        ## Save output
        g = self.recog.forward(self.dset.X).detach().numpy()
        self.losses = losses
        print("Done!")
        return g
    
    def get_losses(self) -> float:
        """ Getter for losses

        :return: self.losses
        :rtype: float
        """
        return self.losses

    def __str__(self) -> str:
        """ String representation for Astir.

        :return: summary for Astir object
        :rtype: str
        """
        return "Astir object with " + str(self.Y_np.shape[1]) + \
            " columns of cell types, " + \
            str(self.Y_np.shape[0]) + " rows."


## NotClassifiableError: an error to be raised when the dataset fails 
# to be analysed.
class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """
    pass
