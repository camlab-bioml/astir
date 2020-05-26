# astir: Automated aSsignmenT sIngle-cell pRoteomics

# VSCode tips for python:
# ctrl+` to open terminal
# cmd+P to go to file

import re
from typing import Tuple, List, Dict
import warnings
from tqdm import trange


import torch
from torch.autograd import Variable
from torch.distributions import Normal, StudentT, MultivariateNormal
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from astir.models.scdataset import SCDataset
from astir.models.recognet import RecognitionNet


class CellTypeModel:
    """Loads a .csv expression file and a .yaml marker file.

    :raises NotClassifiableError: raised when the input gene expression 
        data or the marker is not classifiable

    :param assignments: cell type assignment probabilities
    :param losses: losses after optimization
    :param type_dict: dictionary mapping cell type
        to the corresponding genes
    :param N: number of rows of data
    :param G: number of cell type genes
    :param C: number of cell types
    :param initializations: initialization parameters
    :param data: parameters that is not to be optimized
    :param variables: parameters that is to be optimized
    :param include_beta: [summery]
    """

    def __init__(
        self,
        dset: SCDataset,
        include_beta=False,
        design=None,
        random_seed=1234,
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
            raise NotClassifiableError("Random seed is expected to be an integer.")
        torch.manual_seed(random_seed)

        self.losses = None  # losses after optimization
        self._is_converged = False
        self.cov_mat = None  # temporary -- remove
        self._data = None
        self._variables = None
        self._losses = None

        self._dset = dset

        # Does this model use separate beta?
        self.include_beta = include_beta

        # if design is not None:
        #     if isinstance(design, pd.DataFrame):
        #         design = design.to_numpy()

        self._recog = RecognitionNet(dset.get_class_amount(), dset.get_protein_amount())

    def _param_init(self) -> None:
        """Initialize parameters and design matrices.
        """
        G = self._dset.get_protein_amount()
        C = self._dset.get_class_amount()

        # Establish data
        self._data = {
            "log_alpha": torch.log(torch.ones(C + 1) / (C + 1)),
            "rho": torch.from_numpy(self._dset.get_marker_mat()),
        }

        # Initialize mu, log_delta
        t = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(0.2))
        log_delta_init = t.sample((G, C + 1))

        # mu_init = torch.from_numpy(np.log(Y_np.mean(0).copy()))
        mu_init = torch.log(self._dset.get_mu())
        mu_init = mu_init - (self._data["rho"] * torch.exp(log_delta_init)).mean(1)
        mu_init = mu_init.reshape(-1, 1)

        # Create initialization dictionary
        initializations = {
            "mu": mu_init,
            # "log_sigma": torch.from_numpy(np.log(Y_np.std(0)).copy()),
            "log_sigma": torch.log(self._dset.get_sigma()),
            "log_delta": log_delta_init,
            "p": torch.zeros(G, C + 1),
        }

        P = self._dset.design.shape[1]
        # Add additional columns of mu for anything in the design matrix
        initializations["mu"] = torch.cat(
            [initializations["mu"], torch.zeros((G, P - 1)).double()], 1
        )

        # Create trainable variables
        self._variables = {
            n: Variable(v, requires_grad=True)
            for (n, v) in initializations.items()
        }

        if self.include_beta:
            self._variables["beta"] = Variable(
                torch.zeros(G, C + 1), requires_grad=True
            )

    ## Declare pytorch forward fn
    def _forward(
        self, Y: torch.Tensor, X: torch.Tensor, design: torch.Tensor
    ) -> torch.Tensor:
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
        G = self._dset.get_protein_amount()
        C = self._dset.get_class_amount()
        Y_spread = Y.reshape(-1, G, 1).repeat(1, 1, C + 1)

        delta_tilde = torch.exp(self._variables["log_delta"])  # + np.log(0.5)
        mean = delta_tilde * self._data["rho"]
        mean2 = torch.matmul(design, self._variables["mu"].T)  ## N x P * P x G
        mean2 = mean2.reshape(-1, G, 1).repeat(1, 1, C + 1)
        mean = mean + mean2

        if self.include_beta:
            with torch.no_grad():
                min_delta = torch.min(delta_tilde, 1).values.reshape((G, 1))
            mean = mean + min_delta * torch.tanh(self._variables["beta"]) * (
                1 - self._data["rho"]
            )

        # now do the variance modelling
        p = torch.sigmoid(self._variables["p"])
        corr_mat = torch.einsum(
            "gc,hc->cgh", self._data["rho"] * p, self._data["rho"] * p
        ) * (1 - torch.eye(G)) + torch.eye(G)
        self.cov_mat = torch.einsum(
            "g,h->gh",
            torch.exp(self._variables["log_sigma"]),
            torch.exp(self._variables["log_sigma"]),
        ) + 1e-6 * torch.eye(G)
        self.cov_mat = self.cov_mat * corr_mat

        dist = MultivariateNormal(
            loc=torch.exp(mean).permute(0, 2, 1), covariance_matrix=self.cov_mat
        )
        # dist = Normal(torch.exp(mean), torch.exp(self.variables["log_sigma"]).reshape(-1, 1))
        # dist = StudentT(torch.tensor(1.), torch.exp(mean), torch.exp(self.variables["log_sigma"]).reshape(-1, 1))

        log_p_y_on_c = dist.log_prob(Y_spread.permute(0, 2, 1))
        # log_p_y_on_c = log_p_y.sum(1)
        gamma = self._recog.forward(X)
        elbo = (
            gamma * (log_p_y_on_c + self._data["log_alpha"] - torch.log(gamma))
        ).sum()

        return -elbo

    def fit(
        self, max_epochs=10, learning_rate=1e-2, batch_size=24, delta_loss=0.001
    ) -> None:
        """Fit the model.

        :param epochs: [description], defaults to 100
        :type epochs: int, optional
        :param learning_rate: [description], defaults to 1e-2
        :type learning_rate: [type], optional
        :param batch_size: [description], defaults to 1024
        :type batch_size: int, optional
        """
        self._param_init()
        ## Make dataloader
        dataloader = DataLoader(self._dset, batch_size=min(batch_size, 
            len(self._dset)), shuffle=True)

        ## Run training loop
        losses = np.empty(0)
        per = 1

        ## Construct optimizer
        opt_params = list(self._variables.values()) + list(self._recog.parameters())

        if self.include_beta:
            opt_params = opt_params + [self._variables["beta"]]
        optimizer = torch.optim.Adam(opt_params, lr=learning_rate)

        iterator = trange(max_epochs, desc = "Training Astir", unit = "epochs", 
            postfix = "(cell type classification)")
        for ep in iterator:
            L = None
            for batch in dataloader:
                Y, X, design = batch
                optimizer.zero_grad()
                L = self._forward(Y, X, design)
                L.backward()
                optimizer.step()
            l = self._forward(self._dset.get_exprs(), self._dset.get_exprs_X(), 
                self._dset.design).detach().numpy()
            if losses.shape[0] > 0:
                per = abs((l - losses[-1]) / losses[-1])
            losses = np.append(losses, l)
            if per <= delta_loss:
                self._is_converged = True
                iterator.close()
                print("Reached convergence -- breaking from training loop")
                break

        ## Save output
        g = self._recog.forward(self._dset.get_exprs_X()).detach().numpy()
        self._losses = losses
        print(f"loss: {losses[-1]} \t % change: {100*per}")
        print("Done!")
        return g

    def get_losses(self) -> float:
        """ Getter for losses

        :return: self.losses
        :rtype: float
        """
        if self._losses is None:
            raise Exception("The type model has not been trained yet")
        return self._losses

    def is_converged(self) -> bool:
        return self._is_converged


## NotClassifiableError: an error to be raised when the dataset fails
# to be analysed.
class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """
    pass
