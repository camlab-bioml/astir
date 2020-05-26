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
from tqdm import trange

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
        self, dset, include_beta=True, alpha_random=True, random_seed=42,
    ):
        if not isinstance(random_seed, int):
            raise NotClassifiableError("Random seed is expected to be an integer.")
        # Setting random seeds
        torch.manual_seed(random_seed)
        self.random_seed = random_seed

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._include_beta = include_beta
        self._alpha_random = alpha_random

        # Rescale data so that the model is not gene specific
        dset.rescale()
        self._dset = dset

        self._optimizer = None
        self._losses = None
        self._variables = None
        self._data = None

        # Convergence flag
        self._is_converged = False

    def _param_init(self) -> None:
        """ Initialize sets of parameters
        """
        N = len(self._dset)
        C = self._dset.get_class_amount()
        initializations = {
            "log_sigma": torch.log(self._dset.get_sigma()),
            "mu": torch.reshape(self._dset.get_mu(), (1, -1)),
        }
        # Implement Gaussian noise to alpha?
        if self._alpha_random:
            # self.initializations["z"] = torch.zeros((len(self._dset), self.C)) + torch.random.normal(
            #     loc=0, scale=0.5)
            d = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(0.5))
            initializations["z"] = d.sample((N, C))
        else:
            initializations["z"] = torch.zeros((N, C))

        # Include beta or not
        if self._include_beta:
            d = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.5))
            initializations["log_w"] = torch.log(
                d.sample((C, self._dset.get_protein_amount()))
            )

        self._variables = {
            n: Variable(i, requires_grad=True) for (n, i) in initializations.items()
        }

        self._data = {
            "rho": self._dset.get_marker_mat().T.to(self._device),
            "Y": self._dset.get_exprs().to(self._device),
        }

    def _forward(self):
        """ One forward pass
        """
        log_sigma = self._variables["log_sigma"]
        mu = self._variables["mu"]
        z = self._variables["z"]
        log_w = self._variables["log_w"]

        rho = self._data["rho"]
        Y = self._data["Y"]
        rho_w = torch.mul(rho, torch.exp(log_w))
        mean = mu + torch.matmul(z, rho_w)

        dist = Normal(mean, torch.exp(log_sigma).reshape(1, -1))

        log_p_y = dist.log_prob(Y)
        prior_alpha = Normal(torch.zeros(1), 0.5 * torch.ones(1)).log_prob(z)
        prior_sigma = Normal(torch.zeros(1), 0.5 * torch.ones(1)).log_prob(log_sigma)

        loss = log_p_y.sum() + prior_alpha.sum() + prior_sigma.sum()
        return -loss

    def fit(
        self, max_epochs, lr=1e-2, delta_loss=0.001, delta_loss_batch=10
    ) -> np.array:
        """ Train loops

        :param max_epochs: number of train loop iterations
        :param lr: the learning rate, defaults to 0.01
        :param delta_loss: stops iteration once the loss rate reaches
            delta_loss, defaults to 0.001
        :param delta_loss_batch: the batch size to consider delta loss,
            defaults to 10

        :return: np.array of shape (n_iter,) that contains the losses after
            each iteration where the last element of the numpy array is the loss
            after n_iter iterations
        :rtype: np.array
        """
        torch.manual_seed(self.random_seed)
        self._param_init()

        if delta_loss_batch >= max_epochs:
            warnings.warn("Delta loss batch size is greater than the number of epochs")

        losses = np.empty(max_epochs)

        opt_params = list(self._variables.values())

        # Create an optimizer if there is no optimizer
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(opt_params, lr=lr)

        # Returns early if the model has already converged
        if self._is_converged:
            return losses[:0]

        prev_mean = None
        curr_mean = None
        curr_delta_loss = None
        delta_cond_met = False

        iterator = trange(
            max_epochs,
            desc="training astir",
            unit="epochs",
        )
        for ep in iterator:
            self._optimizer.zero_grad()

            # Forward pass & Compute loss
            loss = self._forward()

            # Backward pass
            loss.backward()

            # Update parameters
            self._optimizer.step()

            l = loss.detach().numpy()
            losses[ep] = l

            start_index = ep - delta_loss_batch + 1
            if start_index >= 0:
                end_index = start_index + delta_loss_batch
                curr_mean = np.mean(losses[start_index:end_index])
                if prev_mean is not None:
                    curr_delta_loss = abs((prev_mean - curr_mean) / prev_mean)
                    delta_cond_met = 0 <= curr_delta_loss <= delta_loss
                prev_mean = curr_mean

            if delta_cond_met:
                losses = losses[0 : ep + 1]
                self._is_converged = True
                iterator.close()
                print("Reached convergence -- breaking from training loop")
                break
        if self._losses is None:
            self._losses = losses
        else:
            self._losses = np.append(self._losses, losses)

        return losses

    def get_losses(self) -> np.array:
        """ Getter for losses

        :return: a numpy array of losses for each training iteration the
            model runs
        :rtype: np.array
        """
        if self._losses is None:
            raise Exception("The state model has not been trained yet")
        return self._losses

    def is_converged(self) -> bool:
        """ Returns True if the model converged

        :return: self._is_converged
        :rtype: bool
        """
        return self._is_converged


class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """

    pass


class InvalidInputError(RuntimeError):
    pass
