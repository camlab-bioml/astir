# astir: Automated aSsignmenT sIngle-cell pRoteomics

# VSCode tips for python:
# ctrl+` to open terminal
# cmd+P to go to file

import re
from typing import Tuple, List, Dict
import warnings

import torch
from torch.autograd import Variable
from torch.distributions import Normal, StudentT, MultivariateNormal
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
    def _param_init(self, dset) -> None:
        """Initialize parameters and design matrices.
        """

        # Establish data
        self.data = {
            "log_alpha": torch.log(torch.ones(self.C+1) / (self.C+1)),
            "rho": torch.from_numpy(self.marker_mat)
        }

        # Initialize mu, log_delta
        t = torch.distributions.Normal(torch.tensor(0.), torch.tensor(0.2))
        log_delta_init = t.sample((self.G,self.C+1))

        mu_init = torch.from_numpy(np.log(self.Y_np.mean(0).copy()))
        mu_init = mu_init - (self.data['rho'] * torch.exp(log_delta_init)).mean(1)
        mu_init = mu_init.reshape(-1,1)

        # Create initialization dictionary
        self.initializations = {
            "mu":  mu_init,
            "log_sigma": torch.from_numpy(np.log(self.Y_np.std(0)).copy()),
            "log_delta": log_delta_init,
            "p": torch.zeros(self.G, self.C+1)
        }

        # Add additional columns of mu for anything in the design matrix
        P = dset.design.shape[1]
        self.initializations['mu'] = torch.cat( \
            [self.initializations['mu'], torch.zeros((self.G, P-1)).double()],
            1)
        
        # Create trainable variables
        self.variables = {n: Variable(v, requires_grad=True) for (n,v) in self.initializations.items()}

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

        # now do the variance modelling
        p = torch.sigmoid(self.variables["p"])
        corr_mat = torch.einsum('gc,hc->cgh', self.data['rho'] * p, self.data['rho'] * p) * \
            (1 - torch.eye(self.G)) + torch.eye(self.G)
        self.cov_mat = torch.einsum('i,j->ij', torch.exp(self.variables["log_sigma"]), torch.exp(self.variables["log_sigma"]))
        self.cov_mat = self.cov_mat * corr_mat

        dist = MultivariateNormal(loc=torch.exp(mean).permute(0,2,1), covariance_matrix=self.cov_mat)
        # dist = Normal(torch.exp(mean), torch.exp(self.variables["log_sigma"]).reshape(-1, 1))
        # dist = StudentT(torch.tensor(1.), torch.exp(mean), torch.exp(self.variables["log_sigma"]).reshape(-1, 1))

        log_p_y_on_c = dist.log_prob(Y_spread.permute(0,2,1))
        # log_p_y_on_c = log_p_y.sum(1)
        gamma = self.recog.forward(X)
        elbo = ( gamma * (log_p_y_on_c + self.data["log_alpha"] - torch.log(gamma)) ).sum()
        
        return -elbo

    ## Todo: an output function
    def __init__(self, Y_np: np.array, type_dict: Dict,  \
                N: int, G: int, C: int, type_mat: np.array, \
                include_beta = False, design = None,
                random_seed=1234
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
        if not isinstance(random_seed, int):
            raise NotClassifiableError(\
                "Random seed is expected to be an integer.")
        torch.manual_seed(random_seed)
        
        self.losses = None # losses after optimization

        self.cov_mat = None # temporary -- remove

        self.type_dict = type_dict

        self.Y_np = Y_np
        self.N, self.G, self.C = N, G, C

        # Does this model use separate beta?
        self.include_beta = include_beta

        self.marker_mat = type_mat

        if design is not None:
            if isinstance(design, pd.DataFrame):
                design = design.to_numpy()

        self.recog = RecognitionNet(self.C, self.G)

    def fit(self, dset, max_epochs = 100, learning_rate = 1e-2, 
        batch_size = 1024) -> None:
        """Fit the model.

        :param epochs: [description], defaults to 100
        :type epochs: int, optional
        :param learning_rate: [description], defaults to 1e-2
        :type learning_rate: [type], optional
        :param batch_size: [description], defaults to 1024
        :type batch_size: int, optional
        """
        self._param_init(dset)
        ## Make dataloader
        dataloader = DataLoader(dset, batch_size=min(batch_size, self.N),\
            shuffle=True)
        
        ## Run training loop
        losses = np.empty(0)
        per = 1

        ## Construct optimizer
        opt_params = list(self.variables.values()) + \
            list(self.recog.parameters())

        if self.include_beta:
            opt_params = opt_params + [self.variables["beta"]]
        optimizer = torch.optim.Adam(opt_params, lr=learning_rate)

        for ep in range(max_epochs):
            L = None
            for batch in dataloader:
                Y,X,design = batch
                optimizer.zero_grad()
                L = self._forward(Y, X, design)
                L.backward()
                optimizer.step()
            l = self._forward(dset.Y, dset.X, \
                dset.design).detach().numpy()
            if losses.shape[0] > 0:
                per = abs((l - losses[-1]) / losses[-1])
            losses = np.append(losses, l)
            if per <= 0.0001:
                break
            print(l)

        ## Save output
        g = self.recog.forward(dset.X).detach().numpy()
        self.losses = losses
        print("Done!")
        if per > 0.0001:
            msg = "Maximum epochs reached. More iteration may be needed to" +\
                " complete the training."
            warnings.warn(msg)
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
