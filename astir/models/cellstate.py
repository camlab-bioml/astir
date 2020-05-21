"""
Cell State Model
"""

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
import random

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class CellStateModel:
    """Cell state model

    :param df_gex: the input gene expression dataframe
    :param marker_dict: the gene marker dictionary
    :param random_seed: seed number to reproduce results, defaults to 1234
    :param include_beta: model parameter that measures with arbitrary unit,
        by how much protein g contributes to pathway p
    :param alpha_random: adds Gaussian noise to alpha initialization if True
        otherwise alpha is initialized to zeros
    """

    def __init__(
        self,
        Y_np,
        state_dict,
        N,
        G,
        C,
        state_mat,
        design=None,
        include_beta=True,
        alpha_random=True,
        random_seed=42,
    ):

        if not isinstance(random_seed, int):
            raise NotClassifiableError("Random seed is expected to be an integer.")
        # Setting random seeds
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dict = state_dict

        self.N, self.G, self.C = N, G, C

        self.include_beta = include_beta
        self.alpha_random = alpha_random

        self.state_mat = state_mat
        self.Y_np = Y_np
        # Rescale data so that the model is not gene specific
        self.Y_np = self.Y_np / (self.Y_np.std(0))

        self.optimizer = None
        self.losses = None

        # Convergence flag
        self._is_converged = False

        self._param_init()

    def _param_init(self) -> None:
        """ Initialize sets of parameters
        """
        self.initializations = {
            "log_sigma": np.log(self.Y_np.std()).reshape(1),
            "mu": self.Y_np.mean(0).reshape(1, -1),
        }
        # Implement Gaussian noise to alpha?
        if self.alpha_random:
            self.initializations["z"] = np.zeros((self.N, self.C)) + np.random.normal(
                loc=0, scale=0.5
            )
        else:
            self.initializations["z"] = np.zeros((self.N, self.C))

        # Include beta or not
        if self.include_beta:
            self.initializations["log_w"] = np.log(
                np.random.uniform(low=0, high=1.5, size=(self.C, self.G))
            )

        self.variables = {
            n: Variable(torch.from_numpy(i.copy()), requires_grad=True)
            for (n, i) in self.initializations.items()
        }

        self.data = {
            "rho": torch.from_numpy(self.state_mat.T).double().to(self.device),
            "Y": torch.from_numpy(self.Y_np).to(self.device),
        }

    def _forward(self):
        """ One forward pass
        """
        log_sigma = self.variables["log_sigma"]
        mu = self.variables["mu"]
        alpha = self.variables["z"]
        log_beta = self.variables["log_w"]

        rho = self.data["rho"]
        Y = self.data["Y"]

        rho_beta = torch.mul(rho, torch.exp(log_beta))
        mean = mu + torch.matmul(alpha, rho_beta)

        dist = Normal(mean, torch.exp(log_sigma).reshape(1, -1))

        log_p_y = dist.log_prob(Y)
        prior_alpha = Normal(torch.zeros(1), 0.5 * torch.ones(1)).log_prob(alpha)
        prior_sigma = Normal(torch.zeros(1), 0.5 * torch.ones(1)).log_prob(log_sigma)

        loss = log_p_y.sum() + prior_alpha.sum() + prior_sigma.sum()
        return -loss

    def fit(self, n_epochs, lr=1e-2, delta_loss=1e-3, delta_loss_batch=10) -> np.array:
        """ Train loops

        :param n_epochs: number of train loop iterations
        :type n_epochs: int, required
        :param lr: the learning rate, defaults to 0.01
        :type lr: float, optional
        :param delta_loss: stops iteration once the loss rate reaches
            delta_loss, defaults to 0.001
        :type delta_loss: float, optional
        :param delta_loss_batch: the batch size  to consider delta loss,
            defaults to 10
        :type delta_loss_batch: int, optional

        :return: np.array of shape (n_iter,) that contains the losses after
            each iteration where the last element of the numpy array is the loss
            after n_iter iterations
        :rtype: np.array
        """
        if delta_loss_batch >= n_epochs:
            warnings.warn(
                "Delta loss batch size is greater than the number " "of epochs"
            )

        losses = np.empty(n_epochs)

        opt_params = list(self.variables.values())

        # Create an optimizer if there is no optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(opt_params, lr=lr)

        # Returns early if the model has already converged
        if self._is_converged:
            return losses[:0]

        prev_mean = None
        curr_mean = None
        curr_delta_loss = None
        delta_cond_met = False

        for ep in range(n_epochs):
            self.optimizer.zero_grad()

            # Forward pass & Compute loss
            loss = self._forward()

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimizer.step()

            l = loss.detach().numpy()
            losses[ep] = l

            start_index = ep - delta_loss_batch + 1
            if start_index >= 0:
                end_index = start_index + delta_loss_batch
                curr_mean = np.mean(losses[start_index:end_index])
                if prev_mean is not None:
                    curr_delta_loss = (prev_mean - curr_mean) / prev_mean
                    delta_cond_met = 0 < curr_delta_loss < delta_loss
                prev_mean = curr_mean

            if delta_cond_met:
                losses = losses[0 : ep + 1]
                self._is_converged = True
                break

        if self.losses is None:
            self.losses = losses
        else:
            self.losses = np.append(self.losses, losses)

        return losses

    def get_losses(self) -> np.array:
        """ Getter for losses

        :return: a numpy array of losses for each training iteration the
            model runs
        :rtype: np.array
        """
        return self.losses

    def is_converged(self) -> bool:
        """ Returns True if the model converged

        :return: self._is_converged
        :rtype: bool
        """
        return self._is_converged

    def __str__(self) -> str:
        """ String representation for CellStateModel.

        :return: summary for CellStateModel object
        :rtype: str
        """
        return (
            "CellStateModel object with "
            + str(self.Y_np.shape[1])
            + " columns of cell states, "
            + str(self.Y_np.shape[0])
            + " rows."
        )


class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """

    pass


class InvalidInputError(RuntimeError):
    pass
