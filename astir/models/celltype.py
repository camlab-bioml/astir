
import re
from typing import Tuple, List, Dict
import warnings
from tqdm import trange

import torch
from torch.autograd import Variable
from torch.distributions import Normal, StudentT, MultivariateNormal, LowRankMultivariateNormal
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import loompy

from sklearn.preprocessing import StandardScaler
from scipy import stats


from astir.models.scdataset import SCDataset
from astir.models.recognet import RecognitionNet


class CellTypeModel:
    """Class to perform statistical inference to assign
        cells to cell types

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
        self, dset: SCDataset, include_beta=False, design=None, random_seed=1234,
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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Does this model use separate beta?
        self.include_beta = include_beta

        # if design is not None:
        #     if isinstance(design, pd.DataFrame):
        #         design = design.to_numpy()

        self._recog = RecognitionNet(dset.get_n_classes(), dset.get_n_features()).to(self._device)
        self._param_init()

    def _param_init(self) -> None:
        """Initialize parameters and design matrices.
        """
        G = self._dset.get_n_features()
        C = self._dset.get_n_classes()

        # Establish data
        self._data = {
            "log_alpha": torch.log(torch.ones(C + 1) / (C + 1)).to(self._device),
            "rho": self._dset.get_marker_mat().to(self._device),
        }
        
        # Initialize mu, log_delta
        t = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(0.2))
        log_delta_init = t.sample((G, C + 1))

        mu_init = torch.log(self._dset.get_mu()).to(self._device)
        # mu_init = self._dset.get_mu()

        mu_init = mu_init - (self._data["rho"] * torch.exp(log_delta_init).to(self._device)).mean(1)
        mu_init = mu_init.reshape(-1, 1)

        # Create initialization dictionary
        initializations = {
            "mu": mu_init,
            "log_sigma": torch.log(self._dset.get_sigma()).to(self._device),
            "log_delta": log_delta_init,
            "p": torch.zeros(G, C + 1).to(self._device),
        }

        P = self._dset.design.shape[1]
        # Add additional columns of mu for anything in the design matrix
        initializations["mu"] = torch.cat(
            [initializations["mu"], torch.zeros((G, P - 1)).double().to(self._device)], 1
        ).to(self._device)

        # Create trainable variables
        self._variables = {
            n: Variable(v.clone(), requires_grad=True).to(self._device).detach() for (n, v) in initializations.items()
        }

        if self.include_beta:
            self._variables["beta"] = Variable(
                torch.zeros(G, C + 1).to(self._device).detach(), requires_grad=True
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
        G = self._dset.get_n_features()
        C = self._dset.get_n_classes()
        N = Y.shape[0]
        
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

        sigma = torch.exp(self._variables["log_sigma"])
        v1 = (self._data["rho"] * p).T * sigma
        v2 = torch.pow(sigma,2) * (1 - torch.pow(self._data["rho"] * p, 2)).T

        v1 = v1.reshape(1, C+1, G, 1).repeat(N, 1, 1, 1) # extra 1 is the "rank"
        v2 = v2.reshape(1, C+1, G).repeat(N, 1, 1)


        dist = LowRankMultivariateNormal(
            loc=torch.exp(mean).permute(0, 2, 1),
            cov_factor = v1,
            cov_diag = v2
        )


        log_p_y_on_c = dist.log_prob(Y_spread.permute(0, 2, 1))

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
        ## Make dataloader
        dataloader = DataLoader(
            self._dset, batch_size=min(batch_size, len(self._dset)), shuffle=True
        )

        ## Run training loop
        losses = np.empty(0)
        per = 1

        ## Construct optimizer
        opt_params = \
                        list(self._variables.values()) + \
                        list(self._recog.parameters())

        if self.include_beta:
            opt_params = opt_params + [self._variables["beta"]]
        optimizer = torch.optim.Adam(opt_params, lr=learning_rate)

        _, exprs_X, _ = self._dset[torch.arange(len(self._dset))] # calls dset.get_item

        iterator = trange(max_epochs, desc="training astir", unit="epochs")
        for ep in iterator:
            L = None
            for batch in dataloader:
                Y, X, design = batch
                optimizer.zero_grad()
                L = self._forward(Y, X, design)
                L.backward()
                optimizer.step()
            l = (
                self._forward(
                    self._dset.get_exprs(), exprs_X, self._dset.design
                )
                .detach()
                .cpu().numpy()
            )
            if losses.shape[0] > 0:
                per = abs((l - losses[-1]) / losses[-1])
            losses = np.append(losses, l)
            if per <= delta_loss:
                self._is_converged = True
                iterator.close()
                print("Reached convergence -- breaking from training loop")
                break

        ## Save output
        g = self._recog.forward(exprs_X).detach().cpu().numpy()
        if self._losses is None:
            self._losses = losses
        else:
            self._losses = np.append(self._losses, losses)
        # self.save_model(max_epochs, learning_rate, batch_size, delta_loss)
        print("Done!")
        return g

    def predict(self, dset):
        _, exprs_X, _ = dset[torch.arange(len(dset))]
        g = self._recog(exprs_X)
        return g

    def save_model(self, max_epochs, learning_rate, batch_size, delta_loss):
        row_attrs = {"epochs": list(range(len(self._losses)))}
        params_attr = {"parameters": list(self._variables.keys()) + list(self._data.keys())}
        params_val = np.array(list(self._variables.values()) + list(self._data.values()))
        info_attrs = {"run_info": ["max_epochs", "learning_rate", "batch_size", "delta_loss"]}
        info_val = np.array([max_epochs, learning_rate, batch_size, delta_loss])

    def get_losses(self) -> float:
        """ Getter for losses

        :return: self.losses
        :rtype: float
        """
        if self._losses is None:
            raise Exception("The type model has not been trained yet")
        return self._losses
    
    def get_scdataset(self):
        return self._dset

    def is_converged(self) -> bool:
        return self._is_converged

    

    def _compare_marker_between_types(self, curr_type, celltype_to_compare, marker, cell_types, alpha=0.05):
        """For a given cell type and two proteins, ensure marker
        is expressed at higher level using t-test

        """
        current_marker_ind = np.array(self._dset.get_features()) == marker
        
        cells_x = np.array(cell_types) == curr_type
        cells_y = np.array(cell_types) == celltype_to_compare

        x = self._dset.get_exprs().detach().cpu().numpy()[cells_x, current_marker_ind]
        y = self._dset.get_exprs().detach().cpu().numpy()[cells_y, current_marker_ind]

        stat = np.NaN
        pval = np.Inf
        note = 'Only 1 cell in a type: comparison not possible'

        if len(x) > 1 and len(y) > 1:
            tt = stats.ttest_ind(x, y)
            stat = tt.statistic
            pval = tt.pvalue
            note = None

        if not (stat > 0 and pval < alpha):
            rdict = {
                'current_marker': marker,
                'curr_type': curr_type,
                'celltype_to_compare': celltype_to_compare,
                'mean_A': x.mean(),
                'mean_Y': y.mean(),
                'p-val': pval,
                'note': note
            }
            
            return rdict
        
        return None

    def diagnostics(self, cell_type_assignments: list, alpha: float) -> pd.DataFrame:
        """Run diagnostics on cell type assignments

        See :meth:`astir.Astir.diagnostics_celltype` for full documentation
        """
        problems = []

        # Want to construct a data frame that models rho with
        # cell type names on the columns and feature names on the rows
        g_df = pd.DataFrame(self._data['rho'].detach().cpu().numpy())
        g_df.columns = self._dset.get_classes() + ["Other"]
        g_df.index = self._dset.get_features()


        for curr_type in self._dset.get_classes():
            if not curr_type in cell_type_assignments:
                continue

                

            current_markers = g_df.index[g_df[curr_type] == 1]

            for current_marker in current_markers:
                # find all the cell types that shouldn't highly express this marker
                celltypes_to_compare = g_df.columns[g_df.loc[current_marker] == 0]

                for celltype_to_compare in celltypes_to_compare:
                    if not celltype_to_compare in cell_type_assignments:
                        continue
                    
                    is_problem = self._compare_marker_between_types(curr_type, 
                        celltype_to_compare, 
                        current_marker, 
                        cell_type_assignments, 
                        alpha)
                    
                    if is_problem is not None:
                        problems.append(is_problem)

        col_names = ['feature', 'should be expressed higher in', 'than', 'mean cell type 1', 'mean cell type 2', 'p-value', 'note']
        df_issues = None
        if len(problems) > 0:
            df_issues = pd.DataFrame(problems)
            df_issues.columns = col_names
        else:
            df_issues = pd.DataFrame(columns=col_names)
        
        return df_issues
            


## NotClassifiableError: an error to be raised when the dataset fails
# to be analysed.
class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """

    pass
