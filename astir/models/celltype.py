import re
from typing import Tuple, List, Dict
import warnings

import h5py
from tqdm import trange

import torch
from torch.autograd import Variable
from torch.distributions import (
    Normal,
    StudentT,
    MultivariateNormal,
    LowRankMultivariateNormal,
)
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from scipy import stats

from .abstract import AstirModel
from astir.data import SCDataset
from .celltype_recognet import TypeRecognitionNet


class CellTypeModel(AstirModel):
    """Class to perform statistical inference to assign cells to cell types.

    :param dset: the input gene expression dataframe
    :type dset: SCDataset
    :param random_seed: the random seed for parameter initialization, defaults to 1234
    :type random_seed: int, optional
    :param dtype: the data type of parameters, should be the same as `dset`, defaults to torch.float64
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        dset: SCDataset,
        random_seed: int = 1234,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(dset, random_seed, dtype)

        self.losses = None  # losses after optimization
        self.cov_mat = None  # temporary -- remove
        self._assignment = None

        self._recog = TypeRecognitionNet(dset.get_n_classes(), dset.get_n_features()).to(
            self._device, dtype=dtype
        )
        self._param_init()

    def _param_init(self) -> None:
        """ Initializes parameters and design matrices.
        """
        G = self._dset.get_n_features()
        C = self._dset.get_n_classes()

        # Establish data
        self._data = {
            "log_alpha": torch.log(torch.ones(C + 1, dtype=self._dtype) / (C + 1)).to(
                self._device
            ),
            "rho": self._dset.get_marker_mat().to(self._device),
        }
        # Initialize mu, log_delta
        delta_init_mean = torch.log(
            torch.log(torch.tensor(3.0, dtype=self._dtype))
        )  # the log of the log of this is the multiplier
        t = torch.distributions.Normal(
            delta_init_mean.clone().detach().to(self._dtype),
            torch.tensor(0.1, dtype=self._dtype),
        )
        log_delta_init = t.sample((G, C + 1))

        mu_init = torch.log(self._dset.get_mu()).to(self._device)

        mu_init = mu_init - (
            self._data["rho"] * torch.exp(log_delta_init).to(self._device)
        ).mean(1)
        mu_init = mu_init.reshape(-1, 1)

        # Create initialization dictionary
        initializations = {
            "mu": mu_init,
            "log_sigma": torch.log(self._dset.get_sigma()).to(self._device),
            "log_delta": log_delta_init,
            "p": torch.zeros((G, C + 1), dtype=self._dtype, device=self._device),
        }

        P = self._dset.get_design().shape[1]
        # Add additional columns of mu for anything in the design matrix
        initializations["mu"] = torch.cat(
            [
                initializations["mu"],
                torch.zeros((G, P - 1), dtype=self._dtype, device=self._device),
            ],
            1,
        )

        # Create trainable variables
        self._variables = {}
        for (n, v) in initializations.items():
            self._variables[n] = Variable(v.clone()).to(self._device)
            self._variables[n].requires_grad = True

    # @profile
    ## Declare pytorch forward fn
    def _forward(
        self, Y: torch.Tensor, X: torch.Tensor, design: torch.Tensor
    ) -> torch.Tensor:
        """One forward pass.

        :param Y: a sample from the dataset
        :type Y: torch.Tensor
        :param X: normalized sample data
        :type X: torch.Tensor
        :param design: the corresponding row of design matrix
        :type design: torch.Tensor
        :return: the cost (elbo) of the current pass
        :rtype: torch.Tensor
        """
        G = self._dset.get_n_features()
        C = self._dset.get_n_classes()
        N = Y.shape[0]

        Y_spread = Y.reshape(-1, G, 1).repeat(1, 1, C + 1)

        delta_tilde = torch.exp(self._variables["log_delta"])  # + np.log(0.5)
        mean = delta_tilde * self._data["rho"]
        mean2 = torch.sparse.mm(design, self._variables["mu"].T)  ## N x P * P x G
        mean2 = mean2.reshape(-1, G, 1).repeat(1, 1, C + 1)
        mean = mean + mean2

        # now do the variance modelling
        p = torch.sigmoid(self._variables["p"])

        sigma = torch.exp(self._variables["log_sigma"])
        v1 = (self._data["rho"] * p).T * sigma
        v2 = torch.pow(sigma, 2) * (1 - torch.pow(self._data["rho"] * p, 2)).T

        v1 = v1.reshape(1, C + 1, G, 1).repeat(N, 1, 1, 1)  # extra 1 is the "rank"
        v2 = v2.reshape(1, C + 1, G).repeat(N, 1, 1) + 1e-6

        dist = LowRankMultivariateNormal(
            loc=torch.exp(mean).permute(0, 2, 1), cov_factor=v1, cov_diag=v2
        )

        log_p_y_on_c = dist.log_prob(Y_spread.permute(0, 2, 1))

        gamma = self._recog.forward(X)
        elbo = (
            gamma * (log_p_y_on_c + self._data["log_alpha"] - torch.log(gamma))
        ).sum()

        return -elbo

    def fit(
        self,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
        msg: str = "",
    ) -> None:
        """ Runs train loops until the convergence reaches delta_loss for\ 
            delta_loss_batch sizes or for max_epochs number of times

        :param max_epochs: number of train loop iterations, defaults to 50
        :param learning_rate: the learning rate, defaults to 0.01
        :param batch_size: the batch size, defaults to 128
        :param delta_loss: stops iteration once the loss rate reaches\ 
            delta_loss, defaults to 0.001
        :param msg: iterator bar message, defaults to empty string
        """
        # Make dataloader
        dataloader = DataLoader(
            self._dset, batch_size=min(batch_size, len(self._dset)), shuffle=True
        )

        # Run training loop
        losses = []
        per = 1

        # Construct optimizer
        opt_params = list(self._variables.values()) + list(self._recog.parameters())

        optimizer = torch.optim.Adam(opt_params, lr=learning_rate)

        _, exprs_X, _ = self._dset[:]  # calls dset.get_item

        iterator = trange(
            max_epochs,
            desc="training restart" + msg,
            unit="epochs",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
        )
        for ep in iterator:
            L = None
            loss = torch.tensor(0.0, dtype=self._dtype)
            for batch in dataloader:
                Y, X, design = batch
                optimizer.zero_grad()
                L = self._forward(Y, X, design)
                L.backward()
                optimizer.step()
                with torch.no_grad():
                    loss = loss + L
            if len(losses) > 0:
                per = abs((loss - losses[-1]) / losses[-1])
            losses.append(loss)
            iterator.set_postfix_str("current loss: " + str(round(float(loss), 1)))
            if per <= delta_loss:
                self._is_converged = True
                iterator.close()
                break

        # Save output
        self._assignment = pd.DataFrame(self._recog.forward(exprs_X).detach().cpu().numpy())
        self._assignment.columns = self._dset.get_classes() + ["Other"]
        self._assignment.index = self._dset.get_cell_names()

        if self._losses is None:
            self._losses = torch.tensor(losses)
        else:
            self._losses = torch.cat(
                (self._losses.view(self._losses.shape[0]), torch.tensor(losses)), dim=0
            )

    def predict(self, new_dset: pd.DataFrame) -> np.array:
        """Feed `new_dset` to the recognition net to get a prediction.

        :param new_dset: the dataset to be predicted
        :type new_dset: pd.DataFrame
        :return: the resulting cell type assignment
        :rtype: np.array
        """
        _, exprs_X, _ = new_dset[:]
        g = pd.DataFrame(self._recog.forward(exprs_X).detach().cpu().numpy())
        return g

    def get_assignment(self) -> np.array:
        """Get the final assignment of the dataset.

        :return: the final assignment of the dataset
        :rtype: np.array
        """
        if self._assignment is None:
            raise Exception("The type model has not been trained yet")
        return self._assignment

    def get_recognet(self) -> TypeRecognitionNet:
        """ Getter for the recognition net.

        :return: the trained recognition net
        """
        return self._recog

    def _compare_marker_between_types(
        self, curr_type, celltype_to_compare, marker, cell_types, alpha: float = 0.05
    ):
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
        note = "Only 1 cell in a type: comparison not possible"

        if len(x) > 1 and len(y) > 1:
            tt = stats.ttest_ind(x, y)
            stat = tt.statistic
            pval = tt.pvalue
            note = None

        if not (stat > 0 and pval < alpha):
            rdict = {
                "current_marker": marker,
                "curr_type": curr_type,
                "celltype_to_compare": celltype_to_compare,
                "mean_A": x.mean(),
                "mean_Y": y.mean(),
                "p-val": pval,
                "note": note,
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
        g_df = pd.DataFrame(self._data["rho"].detach().cpu().numpy())
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

                    is_problem = self._compare_marker_between_types(
                        curr_type,
                        celltype_to_compare,
                        current_marker,
                        cell_type_assignments,
                        alpha,
                    )

                    if is_problem is not None:
                        problems.append(is_problem)

        col_names = [
            "feature",
            "should be expressed higher in",
            "than",
            "mean cell type 1",
            "mean cell type 2",
            "p-value",
            "note",
        ]
        df_issues = None
        if len(problems) > 0:
            df_issues = pd.DataFrame(problems)
            df_issues.columns = col_names
        else:
            df_issues = pd.DataFrame(columns=col_names)

        return df_issues
