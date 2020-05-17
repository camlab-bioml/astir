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
    def _single_param_init(self) -> Dict[str, np.array]:
        """ Initializes a single set of parameters

        :return: dictionary which contains initializations of parameters
        :rtype: Dict[str, np.array]
        """
        init_params = {
            "log_sigma": np.log(self.Y_np.std()).reshape(1),
            "mu": self.Y_np.mean(0).reshape(1, -1)
        }
        if self.alpha_random:
            init_params["alpha"] = np.zeros((self.N, self.C)) + \
                                   np.random.normal(loc=0, scale=0.5)
        else:
            init_params["alpha"] = np.zeros((self.N, self.C))

        if self.include_beta:
            init_params["log_beta"] = np.log(
                np.random.uniform(low=0, high=1.5, size=(self.C, self.G)))

        return init_params

    def _param_init(self) -> None:
        """ Initialize sets of parameters
        """
        self.initializations = [self._single_param_init()
                                for _ in range(self.n_init_params)]
        self.variables = []
        for i in range(self.n_init_params):
            variables = {}
            for param_name, param in self.initializations[i].items():
                variable = Variable(torch.from_numpy(param.copy()),
                                    requires_grad=True)
                variables[param_name] = variable
            self.variables.append(variables)

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

    def _forward(self, param_index=0):
        """ One forward pass

        :param param_index: the index of the parameters, if not specified
        the function assumes that it only has one set of parameters,
        defaults to 0
        :type param_index: int, optional
        """
        log_sigma = self.variables["log_sigma"][param_index]
        mu = self.variables["mu"][param_index]
        alpha = self.variables["alpha"][param_index]
        log_beta = self.variables["log_beta"][param_index]

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
                 random_seed=1234, include_beta=True, alpha_random=True,
                 n_init_params=1):
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
        :param n_init_params: number of sets of parameters to initialize
        :type n_init_params: int, optional, defaults to 1
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

        self.n_init_params = n_init_params

        self.state_mat = self._construct_state_mat()
        self.Y_np = self._get_classifiable_genes(df_gex)
        # Rescale data so that the model is not gene specific
        self.Y_np = self.Y_np / (self.Y_np.std(0))

        # if design is not None:
        #     if isinstance(design, pd.DataFrame):
        #         design = design.to_numpy()
        #
        # self.dset = IMCDataSet(self.CT_np, design)
        #
        # self.recog = RecognitionNet(self.C, self.G)
        #
        self._param_init()

    def fit(self, n_epochs=100, learning_rate=1e-2, batch_size=1024) -> None:
        """ Fit the model

        :param n_epochs: the number of epochs, defaults to 100
        :type n_epochs: int, optional
        :param learning_rate: the learning rate, defaults to 0.01
        :type learning_rate: float, optional
        :param batch_size: the batch size, defaults to 1024
        :type batch_size: int, optional
        """
        # Determine which parameter the model should use


    # TODO: write a function that determines which parameter the model should
    #  use
    # TODO: write a function that runs train loops
    # TODO: dataloader make batches
    # TODO: random seed when n_init_params > 1


class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """
    pass


if __name__ == "__main__":
    import yaml

    expr_csv = "sce.csv"
    marker_yaml = "jackson-2020-markers.yml"

    df_gex = pd.read_csv(expr_csv, index_col=0)
    with open(marker_yaml, 'r') as stream:
        marker_dict = yaml.safe_load(stream)
    model = CellStateModel(df_gex, marker_dict, random_seed=42,
                           include_beta=True, alpha_random=True,
                           n_init_params=4)

