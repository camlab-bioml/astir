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
    """ Cell State Model

    :raises NotClassifiableError: raised when the input gene expression
        data or the marker is not classifiable

    :param initializations: initialization of parameters
    :type initializations: List[Dict[str, np.array]]
    :param data: parameters that are not optimized
    :type data: Dict[str, torch.Tensor]
    :param variables: parameters that are optimized
    :type variables: List[Dict[str, torch.Tensor]]
    """
    def _param_init(self) -> None:
        """ Initialize sets of parameters
        """
        self.initializations = {
            "log_sigma": np.log(self.Y_np.std()).reshape(1),
            "mu": self.Y_np.mean(0).reshape(1, -1)
        }
        # Implement Gaussian noise to alpha?
        if self.alpha_random:
            self.initializations["alpha"] = np.zeros((self.N, self.C)) + \
                                            np.random.normal(loc=0, scale=0.5)
        else:
            self.initializations["alpha"] = np.zeros((self.N, self.C))

        # Include beta or not
        if self.include_beta:
            self.initializations["log_beta"] = np.log(
                np.random.uniform(low=0, high=1.5, size=(self.C, self.G)))

        self.variables = {}
        for param_name, param in self.initializations.items():
            self.variables[param_name] = Variable(
                torch.from_numpy(self.initializations[param_name].copy()),
                requires_grad=True
            )

        self.data = {
            "rho": torch.from_numpy(self.state_mat.T).double().to(self.device),
            "Y": torch.from_numpy(self.Y_np).to(self.device)
        }

    def _forward(self):
        """ One forward pass
        """
        log_sigma = self.variables["log_sigma"]
        mu = self.variables["mu"]
        alpha = self.variables["alpha"]
        log_beta = self.variables["log_beta"]

        rho = self.data["rho"]
        Y = self.data["Y"]

        rho_beta = torch.mul(rho, torch.exp(log_beta))
        mean = mu + torch.matmul(alpha, rho_beta)

        dist = Normal(mean, torch.exp(log_sigma).reshape(1, -1))

        log_p_y = dist.log_prob(Y)
        prior_alpha = Normal(torch.zeros(1),
                             0.5 * torch.ones(1)).log_prob(alpha)
        prior_sigma = Normal(torch.zeros(1),
                             0.5 * torch.ones(1)).log_prob(log_sigma)

        loss = log_p_y.sum() + prior_alpha.sum() + prior_sigma.sum()
        return -loss

    def __init__(self, Y_np, state_dict, N, G, C,
                 state_mat, design=None,
                 include_beta=True, alpha_random=True,
                 random_seed=42):
        """ Initialize a Cell State Model

        :param df_gex: the input gene expression dataframe
        :type df_gex: pd.DataFrame
        :param marker_dict: the gene marker dictionary
        :type marker_dict: Dict
        :param random_seed: seed number to reproduce results, defaults to 1234
        :type random_seed: int, optional
        :param include_beta: model parameter that measures with arbitrary unit,
         by how much protein g contributes to pathway p
        :type include_beta: bool, optional
        :param alpha_random: adds Gaussian noise to alpha initialization if True
        otherwise alpha is initialized to zeros
        :type alpha_random: bool, optional, defaults to True
        """
        if not isinstance(random_seed, int):
            raise NotClassifiableError(\
                "Random seed is expected to be an integer.")
        # Setting random seeds
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'cpu')

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

        # if design is not None:
        #     if isinstance(design, pd.DataFrame):
        #         design = design.to_numpy()
        #
        # self.dset = IMCDataSet(self.CT_np, design)
        #
        # self.recog = RecognitionNet(self.C, self.G)
        #
        self._param_init()
        # print("===== self.Y_np =====")
        # print(self.Y_np)
        # print("===== self.initializations =====")
        # for param_name, param in self.initializations.items():
        #     print("#########", param_name, "#########")
        #     print(self.initializations[param_name])
        #
        # print("===== self.variables =====")
        # for param_name, param in self.variables.items():
        #     print("#########", param_name, "#########")
        #     print(self.variables[param_name])
        #
        # print("===== self.data =====")
        # for param_name, param in self.data.items():
        #     print("#########", param_name, "#########")
        #     print(self.data[param_name])
        #
        # print("\nloss: ", self._forward())

    def fit(self, n_epochs, lr=1e-2, delta_loss=1e-3,
            delta_loss_batch=10) -> np.array:
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
            warnings.warn("Delta loss batch size is greater than the number "
                          "of epochs")

        losses = np.empty(n_epochs)

        opt_params = list(self.variables.values())
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(opt_params, lr=lr)
        else:
            if self.losses.size > delta_loss_batch:
                prev_mean = np.mean(self.losses[-delta_loss_batch-1:-1])
                curr_mean = np.mean(self.losses[-delta_loss_batch:])
                curr_delta_loss = (prev_mean - curr_mean) / prev_mean
                if 0 < curr_delta_loss < delta_loss:
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
                losses = losses[0:ep+1]
                break

        if not delta_cond_met:
            warnings.warn("Reached max iter but not converged")

        if self.losses is None:
            self.losses = losses
        else:
            self.losses = np.append(self.losses, losses)

        return losses


class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """
    pass


class InvalidInputError(RuntimeError):
    pass


# if __name__ == "__main__":
#     import yaml
#
#     pd.set_option("max_rows", None)
#
#     expr_csv = "sce.csv"
#     marker_yaml = "jackson-2020-markers.yml"
#
#     df_gex = pd.read_csv(expr_csv, index_col=0)
#     with open(marker_yaml, 'r') as stream:
#         marker_dict = yaml.safe_load(stream)
#
#     state_dict = marker_dict["cell_states"]
#     marker_genes = sorted(list(
#         set([l for s in state_dict.values() for l in s])))
#     state_names = list(state_dict.keys())
#
#     Y_np = df_gex[marker_genes].to_numpy()
#
#     N = df_gex.shape[0]
#     G = len(marker_genes)
#     C = len(state_names)
#
#     state_mat = np.zeros(shape=(G, C))
#
#     for g, gene in enumerate(marker_genes):
#         for ct, state in enumerate(state_names):
#             if gene in state_dict[state]:
#                 state_mat[g, ct] = 1
#
#     model = CellStateModel(Y_np=Y_np, state_dict=state_dict,
#                            N=N, G=G, C=C,
#                            state_mat=state_mat, design=None,
#                            include_beta=True, alpha_random=True,
#                            random_seed=42)
#
#     losses = model.fit(n_epochs=100, delta_loss=1e-3, delta_loss_batch=10,
#                        lr=0.01)
#     # losses2 = model.fit(n_epochs=100, delta_loss=1e-3, delta_loss_batch=10)
#     # losses = np.append(losses, losses2)
#
#     pf_losses = pd.DataFrame(losses)
#     print(pf_losses)


