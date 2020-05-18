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


#mplements _forward and fit
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

    def _construct_state_mat(self) -> np.array:
        """ Constructs a matrix representing the marker information.

        :return: constructed matrix
        :rtype: np.array
        """
        state_mat = np.zeros(shape=(self.G, self.C))

        for g, gene in enumerate(self.marker_genes):
            for ct, state in enumerate(self.state_names):
                if gene in self.state_dict[state]:
                    state_mat[g, ct] = 1

        return state_mat

    def _sanitize_dict(self, marker_dict: Dict[str, dict]) -> Dict[str, dict]:
        """ Sanitizes the marker dictionary.

        :param marker_dict: dictionary read from the  yaml file
        :type marker_dict: Dict[str, dict]

        :raises NotClassifiableError: raised when the marker dictionary
            doesn't have the required format

        :return: dictionaries for cell state:
        :rtype: Dict[str, dict]
        """
        keys = list(marker_dict.keys())
        if not len(keys) == 2:
            raise NotClassifiableError("Marker file does not follow the "
                                       "required format")
        cs = re.compile("cell[^a-zA-Z0-9]*state", re.IGNORECASE)
        if cs.match(keys[0]):
            state_dict = marker_dict[keys[0]]
        elif cs.match(keys[1]):
            state_dict = marker_dict[keys[1]]
        else:
            raise NotClassifiableError("Can't find cell state dictionary in "
                                       "the marker file.")

        return state_dict

    def _sanitize_gex(self, df_gex: pd.DataFrame) -> Tuple[int, int, int]:
        """ Sanitizes the inputted gene expression dataframe

        :param df_gex: dataframe read from the input .csv file
        :type df_gex: pd.DataFrame

        :raises NotClassifiableError: raised when the input information is
            not sufficient for the classification.

        :return: # of rows (# of cells that were sampled), # of marker genes,
            # of cell states
        :rtype: Tuple[int, int, int]
        """
        N = df_gex.shape[0]
        G = len(self.marker_genes)
        C = len(self.state_names)
        if N <= 0:
            raise NotClassifiableError("Clasification failed. There " + \
                "should be at least one row of data to be classified.")
        if C <= 1:
            raise NotClassifiableError("Classification failed. There " + \
                "should be at least two cell states to classify the data " + \
                "into.")
        return N, G, C

    def _get_classifiable_genes(self, df_gex: pd.DataFrame) -> \
        pd.DataFrame:
        """ Return the classifiable data which contains a subset of genes from
            the marker and the input data.

        :raises NotClassifiableError: raised when there is no overlap between the
            inputted state gene and the marker type or state gene

        :return: classifiable cell type data
            and cell state data
        :rtype: Tuple[pd.DataFrame, pd.Dataframe]
        """
        try:
            Y_np = df_gex[self.marker_genes].to_numpy()
        except(KeyError):
            raise NotClassifiableError("Classification failed. There's no " + \
                "overlap between marked genes and expression genes to " + \
                "classify among cell states.")
        if Y_np.shape[1] < len(self.marker_genes):
            warnings.warn("Classified state genes are less than marked genes.")

        return Y_np

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

    def __init__(self, df_gex: pd.DataFrame, marker_dict: Dict,
                 random_seed=1234, include_beta=True, alpha_random=True):
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

        self.state_dict = self._sanitize_dict(marker_dict)

        self.state_names = list(self.state_dict.keys())

        self.marker_genes = sorted(list(
            set([l for s in self.state_dict.values() for l in s])))

        self.N, self.G, self.C = self._sanitize_gex(df_gex)

        # Read input data
        self.core_names = list(df_gex.index)
        self.gene_names = list(df_gex.columns)

        self.include_beta = include_beta
        self.alpha_random = alpha_random

        self.state_mat = self._construct_state_mat()
        self.Y_np = self._get_classifiable_genes(df_gex)
        # Rescale data so that the model is not gene specific
        self.Y_np = self.Y_np / (self.Y_np.std(0))

        self.optimizer = None

        # if design is not None:
        #     if isinstance(design, pd.DataFrame):
        #         design = design.to_numpy()
        #
        # self.dset = IMCDataSet(self.CT_np, design)
        #
        # self.recog = RecognitionNet(self.C, self.G)
        #
        self._param_init()

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
        :param delta_loss_batch:
        :type delta_loss_batch:

        :return: np.array of shape (n_iter,) that contains the losses after
        each iteration where the last element of the numpy array is the loss
        after n_iter iterations
        :rtype: np.array
        """
        if delta_loss_batch >= n_epochs:
            warnings.warn("Delta loss batch size is greater than the number "
                          "of epochs")

        opt_params = list(self.variables.values())
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(opt_params, lr=lr)

        losses = np.empty(n_epochs)

        prev_mean = None
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

        return losses

    # def _train_param_init(self, max_iter, var_index, conv_rate=1e-3):
    #     """ Train each parameters until convergences
    #
    #     :param max_iter: maximum iteration, if the convergence rate does not
    #     reach by max_iter iterations the train loop terminates. Assume that
    #     max_iter is less than or equal to number of epochs.
    #     :type max_iter: int, required
    #     :param var_index: the index of the parameters
    #     :type var_index: int, required
    #     :param conv_rate: the convergence rate where the parameter stops,
    #     defaults to 0.001
    #     :type conv_rate: float, optional
    #     """
    #     for i in range(self.n_init_params):
    #         # Perform first ten iterations
    #         n_train_iter = 10
    #         count_iter = n_train_iter
    #         losses = self._train_loops(n_iter=n_train_iter, var_index=i)
    #
    #         ratio_loss = 1
    #         prev_mean = np.sum(losses) / n_train_iter
    #
    #         # Find the iteration where the loss converges
    #         print(max_iter)
    #         print(count_iter)
    #         while (not (0 < ratio_loss < conv_rate)) and \
    #                 (not (count_iter > max_iter)):
    #             print((~(count_iter > max_iter)))
    #             loss = self._train_loops(n_iter=1, var_index=i)
    #             losses = np.append(losses, loss)
    #
    #             curr_mean = np.sum(losses[-n_train_iter:]) / n_train_iter
    #             ratio_loss = (prev_mean - curr_mean) / prev_mean
    #             count_iter += 1
    #
    #         self.losses.append(losses)

    # def fit(self, n_epochs=100, conv_rate=1e-3, lr=1e-2,
    #         batch_size=1024) -> None:
    #     """ Fit the model
    #
    #     :param n_epochs: the number of epochs, defaults to 100
    #     :type n_epochs: int, optional
    #     :param lr: the learning rate, defaults to 0.01
    #     :type lr: float, optional
    #     :param batch_size: the batch size, defaults to 1024
    #     :type batch_size: int, optional
    #     """
    #     # Create optimizers
    #     self.optimizers = []
    #     for i in range(self.n_init_params):
    #         opt_params = list(self.variables[i].values())
    #         optimizer = torch.optim.Adam(opt_params, lr=lr)
    #         self.optimizers.append(optimizer)
    #
    #     # Determine which parameter the model should use
    #     losses_after_convergence = np.array([
    #         self._train_param_init(max_iter=n_epochs, var_index=i,
    #                                conv_rate=conv_rate)
    #         for i in range(self.n_init_params)
    #     ])
    #     best_loss_index = np.argmin(losses_after_convergence)
    #     n_iter_done = len(self.losses[best_loss_index])
    #
    #     # Run the remaining training loops
    #     remaining_n_iter = n_epochs - n_iter_done
    #     losses = self._train_loops(n_iter=remaining_n_iter,
    #                                var_index=best_loss_index)
    #     self.losses[best_loss_index] = \
    #         np.append(self.losses[best_loss_index], losses)
    #
    #     # TODO: delete
    #     for i in range(self.n_init_params):
    #         print("##############", i, "##############")
    #         pf = pd.DataFrame(self.losses[i])
    #         print(pf)

        # TODO
        # g = self.recog.forward(self.dset.X).detach().numpy()
        # self.assignments = pd.DataFrame(g)
        # self.assignments.columns = self.cell_types + ['Other']
        # self.assignments.index = self.core_names

    # TODO: write a function that determines which parameter the model should
    #  use
    # TODO: write a function that runs train loops
    # TODO: dataloader make batches
    # TODO: random seed when n_init_params > 1


class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """
    pass


class InvalidInputError(RuntimeError):
    pass


if __name__ == "__main__":
    import yaml

    expr_csv = "sce.csv"
    marker_yaml = "jackson-2020-markers.yml"

    df_gex = pd.read_csv(expr_csv, index_col=0)
    with open(marker_yaml, 'r') as stream:
        marker_dict = yaml.safe_load(stream)
    model = CellStateModel(df_gex, marker_dict, random_seed=42,
                           include_beta=True, alpha_random=True)
    pd.set_option("max_rows", None)
    losses = model.fit(n_epochs=100, delta_loss=1e-3, delta_loss_batch=10,
                       lr=0.01)
    # losses2 = model.fit(n_epochs=100, delta_loss=1e-3, delta_loss_batch=10)
    # losses = np.append(losses, losses2)

    pf_losses = pd.DataFrame(losses)
    print(pf_losses)
