# astir: Automated aSsignmenT sIngle-cell pRoteomics

# VSCode tips for python:
# ctrl+` to open terminal
# cmd+P to go to file

import re
from typing import Tuple, List, Dict
import warnings

import torch
from torch.autograd import Variable
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


class Astir:
    """ Loads a .csv expression file and a .yaml marker file.
    """

    def _param_init(self) -> None:
        """[summery]
        """
        self.initializations = {
            "mu": np.log(self.CT_np.mean(0)),
            "log_sigma": np.log(self.CT_np.std(0))
        }
        ## prior on z
        log_delta = Variable(0 * torch.ones((self.G,self.C+1)), requires_grad = True)
        self.data = {
            "log_alpha": torch.log(torch.ones(self.C+1) / (self.C+1)),
            "delta": torch.exp(log_delta),
            "rho": torch.from_numpy(self.marker_mat)
        }
        self.variables = {
            "log_sigma": Variable(torch.from_numpy(
                self.initializations["log_sigma"].copy()), requires_grad = True),
            "mu": Variable(torch.from_numpy(self.initializations["mu"].copy()), requires_grad = True),
            "log_delta": log_delta
        }

    def _sanitize_dict(self, marker_dict: Dict[str, dict]) -> Tuple[dict, dict]:
        keys = list(marker_dict.keys())
        if not len(keys) == 2:
            raise NotClassifiableError("Marker file does not follow the " +\
                "required format.")
        ct = re.compile("cell[^a-zA-Z0-9]*type", re.IGNORECASE)
        cs = re.compile("cell[^a-zA-Z0-9]*state", re.IGNORECASE)
        if ct.match(keys[0]):
            type_dict = marker_dict[keys[0]]
        elif ct.match(keys[1]):
            type_dict = marker_dict[keys[1]]
        else:
            raise NotClassifiableError("Can't find cell type dictionary" +\
                " in the marker file.")
        if cs.match(keys[0]):
            state_dict = marker_dict[keys[0]]
        elif cs.match(keys[1]):
            state_dict = marker_dict[keys[1]]
        else:
            raise NotClassifiableError("Can't find cell state dictionary" +\
                " in the marker file.")
        return type_dict, state_dict

    def _sanitize_gex(self, df_gex: pd.DataFrame) -> Tuple[int, int, int, np.array]:
        N = df_gex.shape[0]
        G = len(self.mtype_genes)
        C = len(self.cell_types)
        if N <= 0:
            raise NotClassifiableError("Classification failed. There " + \
                "should be at least one row of data to be classified.")
        if C <= 1:
            raise NotClassifiableError("Classification failed. There " + \
                "should be at least two cell types to classify the data into.")
        return N, G, C

    def _get_classifiable_genes(self):
        ## This should also be done to cell_states
        try:
            CT_np = self.df_gex[self.mtype_genes].to_numpy()
        except(KeyError):
            raise NotClassifiableError("Classification failed. There's no " + \
                "overlap between marked genes and expression genes to " + \
                "classify among cell types.")
        try:
            CS_np = self.df_gex[self.mstate_genes].to_numpy()
        except(KeyError):
            raise NotClassifiableError("Classification failed. There's no " + 
                "overlap between marked genes and expression genes to " + 
                "classify among cell types.")
        if CT_np.shape[1] < len(self.mtype_genes):
            warnings.warn("Classified type genes are less than marked genes.")
        if CS_np.shape[1] < len(self.mstate_genes):
            warnings.warn("Classified state genes are less than marked genes.")
        if CT_np.shape[1] + CS_np.shape[1] < len(self.expression_genes):
            warnings.warn("Classified type and state genes are less than the expression genes in the input data.")
        
        return CT_np, CS_np

    def _construct_marker_mat(self) -> np.array:
        """[summary]

        Returns:
            [type] -- [description]
        """
        marker_mat = np.zeros(shape = (self.G,self.C+1))
        for g in range(self.G):
            for ct in range(self.C):
                gene = self.mtype_genes[g]
                cell_type = self.cell_types[ct]
                if gene in self.type_dict[cell_type]:
                    marker_mat[g,ct] = 1
        # print(marker_mat)

        return marker_mat

    ## Declare pytorch forward fn
    def _forward(self, Y: torch.Tensor , X: torch.Tensor) -> torch.Tensor:
        """[summary]

        Arguments:
            Y {[type]} -- [description]
            X {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        
        Y_spread = Y.reshape(-1, self.G, 1).repeat(1, 1, self.C+1)
        
        mean = torch.exp(torch.exp(self.variables["log_delta"]) * self.data["rho"]\
             + self.variables["mu"].reshape(-1, 1))
        dist = Normal(mean, torch.exp(self.variables["log_sigma"]).reshape(-1, 1))
        

        log_p_y = dist.log_prob(Y_spread)

        log_p_y_on_c = log_p_y.sum(1)
        
        gamma = self.recog.forward(X)

        elbo = ( gamma * (log_p_y_on_c + self.data["log_alpha"] - torch.log(gamma)) ).sum()
        
        return -elbo

    ## Todo: an output function
    def __init__(self, df_gex: pd.DataFrame, marker_dict: Dict, \
        random_seed = 1234) -> None:
        """[summary]

        Arguments:
            df_gex {[type]} -- [description]
            marker_dict {[type]} -- [description]

        Keyword Arguments:
            random_seed {int} -- [description] (default: {1234})

        Raises:
            NotClassifiableError: [description]
        """
        #Todo: fix problem with random seed
        if not isinstance(random_seed, int):
            raise NotClassifiableError(\
                "Random seed is expected to be an integer.")
        torch.manual_seed(random_seed)

        self.assignments = None # cell type assignment probabilities
        self.losses = None # losses after optimization

        self.type_dict, self.state_dict = self._sanitize_dict(marker_dict)

        self.cell_types = list(self.type_dict.keys())
        self.mtype_genes = list(set([l for s in self.type_dict.values() \
            for l in s]))
        self.mstate_genes = list(set([l for s in self.state_dict.values() \
            for l in s]))

        self.N, self.G, self.C = self._sanitize_gex(df_gex)

        # Read input data
        self.df_gex = df_gex
        self.core_names = list(df_gex.index)
        self.expression_genes = list(df_gex.columns)

        self.marker_mat = self._construct_marker_mat()
        self.CT_np, self.CS_np = self._get_classifiable_genes()

        self.dset = IMCDataSet(self.CT_np)
        self.recog = RecognitionNet(self.C, self.G)

        self._param_init()

    def fit(self, epochs = 100, learning_rate = 1e-2, 
        batch_size = 1024) -> None:
        """[summary]

        Keyword Arguments:
            epochs {int} -- [description] (default: {100})
            learning_rate {[type]} -- [description] (default: {1e-2})
            batch_size {int} -- [description] (default: {1024})
        """
        ## Make dataloader
        dataloader = DataLoader(self.dset, batch_size=min(batch_size, self.N),\
            shuffle=True)
        ## Run training loop
        losses = np.empty(epochs)
        ## Construct optimizer
        optimizer = torch.optim.Adam(list(self.variables.values()) + \
            list(self.recog.parameters()), lr=learning_rate)

        for ep in range(epochs):
            L = None
            for batch in dataloader:
                Y,X = batch
                optimizer.zero_grad()
                L = self._forward(Y, X)
                L.backward()
                optimizer.step()
            l = L.detach().numpy()
            losses[ep] = l
            print(l)

        ## Save output
        g = self.recog.forward(self.dset.X).detach().numpy()
        self.assignments = pd.DataFrame(g)
        self.assignments.columns = self.cell_types + ['Other']
        self.assignments.index = self.core_names
        self.losses = losses
        print("Done!")

    def get_assignments(self) -> pd.DataFrame:
        """[summary]

        Returns:
            [type] -- [description]
        """
        return self.assignments
    
    def get_losses(self) -> float:
        """[summary]

        Returns:
            [type] -- [description]
        """
        return self.losses

    def to_csv(self, output_csv: str) -> None:
        """[summary]

        Arguments:
            output_csv {[type]} -- [description]
        """
        self.assignments.to_csv(output_csv)

    def __str__(self) -> str:
        return "Astir object with " + str(self.CT_np.shape[1]) + \
            " columns of cell types, " + str(self.CS_np.shape[1]) + \
            " columns of cell states and " + \
            str(self.CT_np.shape[0]) + " rows."


## NotClassifiableError: an error to be raised when the dataset fails 
# to be analysed.
class NotClassifiableError(RuntimeError):
    pass


## Dataset class: for loading IMC datasets
class IMCDataSet(Dataset):
    
    def __init__(self, Y_np: np.array) -> None:
             
        self.Y = torch.from_numpy(Y_np)
        X = StandardScaler().fit_transform(Y_np)
        self.X = torch.from_numpy(X)
    
    def __len__(self) -> int:
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return self.Y[idx,:], self.X[idx,:]

## Recognition network
class RecognitionNet(nn.Module):
    def __init__(self, C: int, G: int, h=6) -> None:
        super(RecognitionNet, self).__init__()
        self.hidden_1 = nn.Linear(G, h).double()
        self.hidden_2 = nn.Linear(h, C+1).double()

    def forward(self, x):
        x = self.hidden_1(x)
        x = F.relu(x)
        x = self.hidden_2(x)
        x = F.softmax(x, dim=1)
        return x

