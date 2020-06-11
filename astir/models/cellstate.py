"""
Cell State Model
"""

from typing import Tuple, List, Dict, Union
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml

from .scdataset import SCDataset
from tqdm import trange
from torch.autograd import Variable
from torch.utils.data import DataLoader


class CellStateModel:
    """Class to perform statistical inference to on the activation
        of states (pathways) across cells

    :param df_gex: the input gene expression dataframe
    :param marker_dict: the gene marker dictionary
    :param random_seed: seed number to reproduce results, defaults to 1234
    :param include_beta: model parameter that measures with arbitrary unit,
        by how much feature g contributes to pathway p
    :param alpha_random: adds Gaussian noise to alpha initialization if True
        otherwise alpha is initialized to zeros
    """

    def __init__(self,
                 dset: SCDataset,
                 include_beta: bool = True,
                 alpha_random: bool = True,
                 random_seed: int = 42) -> None:
        if not isinstance(random_seed, int):
            raise NotClassifiableError("Random seed is expected to be an integer.")
        # Setting random seeds
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._include_beta = include_beta
        self._alpha_random = alpha_random

        self._dset = dset

        self._optimizer = None
        self._losses = np.empty(0)
        self._param_init()

        # Convergence flag
        self._is_converged = False

    def _param_init(self) -> None:
        """ Initializes sets of parameters
        """
        N = len(self._dset)
        C = self._dset.get_n_classes()

        initializations = {
            "log_sigma": torch.log(self._dset.get_sigma().mean()),
            "mu": torch.reshape(self._dset.get_mu(), (1, -1)),
        }

        # Include beta or not
        d = torch.distributions.Uniform(torch.tensor(0.0),
                                        torch.tensor(1.5))
        initializations["log_w"] = torch.log(
            d.sample((C, self._dset.get_n_features()))
        )

        self._variables = {
            n: i.to(self._device).detach().clone().requires_grad_()
            for (n, i) in initializations.items()
        }

        self._data = {
            "rho": self._dset.get_marker_mat().T.to(self._device),
            "Y": self._dset.get_exprs().to(self._device),
        }

        self._models = {
            "main": nn.Sequential(
                nn.Linear(self._dset.get_n_features(), 2 * C),
                nn.ReLU(),
                nn.Linear(2 * C, 2 * C),
                nn.ReLU()
            ).to(self._device),
            "model_mu": nn.Linear(2 * C, C).to(self._device),
            "model_std": nn.Linear(2 * C, C).to(self._device)
        }

    def _loss_fn(self,
                 mu_z: torch.Tensor,
                 std_z: torch.Tensor,
                 z_sample: torch.Tensor,
                 y_in: torch.Tensor) -> torch.Tensor:
        """ Returns the calculated loss

        :param mu_z: the predicted mean of z
        :param std_z: the predicted standard deviation of z
        :param z_sample: the sampled z values
        :param y_in: the input data

        :return: the loss
        """
        S = y_in.shape[0]

        # log posterior q(z) approx p(z|y)
        q_z_dist = torch.distributions.Normal(loc=mu_z, scale=torch.exp(std_z))
        log_q_z = q_z_dist.log_prob(z_sample)

        # log likelihood p(y|z)
        rho_w = torch.mul(self._data["rho"],
                          torch.exp(self._variables["log_w"]))
        mean = self._variables["mu"] + torch.matmul(z_sample, rho_w)
        std = torch.exp(self._variables["log_sigma"]).reshape(1, -1)
        p_y_given_z_dist = torch.distributions.Normal(loc=mean, scale=std)
        log_p_y_given_z = p_y_given_z_dist.log_prob(y_in)

        # log prior p(z)
        p_z_dist = torch.distributions.Normal(0, 1)
        log_p_z = p_z_dist.log_prob(z_sample)

        loss = (1 / S) * (torch.sum(log_q_z) - torch.sum(log_p_y_given_z)
                          - torch.sum(log_p_z))

        return loss

    def _forward(self, y_in: torch.Tensor) -> Tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor]:
        """ One forward pass

        :param y_in: dataset to do forward pass on

        :return: mu_z, std_z, z_sample
        """
        hidden = self._models["main"](y_in)
        mu_z = self._models["model_mu"](hidden)
        std_z = self._models["model_std"](hidden)

        std = torch.exp(std_z)
        eps = torch.randn_like(std)
        z_sample = eps * std + mu_z

        return mu_z, std_z, z_sample

    def fit(self,
            max_epochs: int,
            lr:float = 1e-3,
            batch_size: int = 64,
            delta_loss: float = 1e-3,
            delta_loss_batch: int = 10
    ) -> np.array:
        """ Runs train loops until the convergence reaches delta_loss for
        delta_loss_batch sizes or for max_epochs number of times

        :param max_epochs: number of train loop iterations
        :param lr: the learning rate, defaults to 0.01
        :param batch_size: the batch size
        :param delta_loss: stops iteration once the loss rate reaches
        delta_loss, defaults to 0.001
        :param delta_loss_batch: the batch size to consider delta loss,
        defaults to 10

        :return: np.array of shape (n_iter,) that contains the losses after
        each iteration where the last element of the numpy array is the loss
        after n_iter iterations
        """
        losses = np.empty(max_epochs)

        # Returns early if the model has already converged
        if self._is_converged:
            return losses[:0]

        if delta_loss_batch >= max_epochs:
            warnings.warn("Delta loss batch size is greater than the number of epochs")

        # Create an optimizer if there is no optimizer
        if self._optimizer is None:
            opt_params = list(self._models["main"].parameters()) + \
                         list(self._models["model_mu"].parameters()) + \
                         list(self._models["model_std"].parameters()) + \
                         list(self._variables.values())
            self._optimizer = torch.optim.Adam(opt_params, lr=lr)

        if self._losses.shape[0] >= delta_loss_batch:
            prev_mean = np.mean(self._losses[-delta_loss_batch:])
        else:
            prev_mean = None

        delta_cond_met = False

        iterator = trange(max_epochs, desc="training astir", unit="epochs",)
        train_iterator = DataLoader(self._dset,
                                    batch_size=min(batch_size, len(self._dset)))
        for ep in iterator:
            for i, (y_in, x_in, _) in enumerate(train_iterator):
                self._optimizer.zero_grad()

                mu_z, std_z, z_samples = \
                    self._forward(x_in.float().to(self._device))

                loss = self._loss_fn(mu_z, std_z,
                                     z_samples, x_in.to(self._device))
                loss.backward()

                self._optimizer.step()

            losses[ep] = loss.cpu().detach().numpy()

            start_index = ep - delta_loss_batch + 1
            end_index = start_index + delta_loss_batch
            if start_index >= 0:
                curr_mean = np.mean(losses[start_index:end_index])
            elif self._losses.shape[0] >= -start_index:
                last_ten_losses = np.append(self._losses[start_index:],
                                            losses[:end_index])
                curr_mean = np.mean(last_ten_losses)
            else:
                curr_mean = None

            if prev_mean is not None:
                curr_delta_loss = (prev_mean - curr_mean) / prev_mean
                delta_cond_met = 0 <= curr_delta_loss < delta_loss

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

    def get_final_mu_z(
            self,
            new_dset: SCDataset = None
    ) -> torch.Tensor:
        """ Returns the mean of the predicted z values for each core

        :param new_dset: returns the predicted z values of this dataset on
        the existing model. If None, it predicts using the existing dataset

        :return: the mean of the predicted z values for each core
        """
        if new_dset is None:
            _, x_in, _ = self._dset[:] # should be the scaled one
        else:
            _, x_in, _ = new_dset[:]
        final_mu_z, _, _ = self._forward(x_in.float())

        return final_mu_z

    def diagnostics(self):
        g = self.get_final_mu_z().detach().cpu().numpy()

        state_assignment = pd.DataFrame(g)
        state_assignment.columns = self._dset.get_classes()
        state_assignment.index = self._dset.get_features()

        marker_mat = self._dset.get_marker_mat().T
        Y = self._dset.get_exprs()
        G = self._dset.get_n_features()
        C = self._dset.get_n_classes()
        corr_mat = np.zeros((C, G))
        for c, state in enumerate(self._dset.get_classes()):
            for g, feature in enumerate(self._dset.get_features()):
                states = state_assignment[state]
                protein = Y[:,g]
                corr_mat[c, g] = np.corrcoef(protein, states)[0, 1]

        problems = []

        for c, state in enumerate(self._dset.get_classes()):
            min_marker_corr = np.inf
            marker_name = None
            for g, feature in enumerate(self._dset.get_features()):
                if marker_mat[c, g] == 1 and corr_mat[c, g] < min_marker_corr:
                    min_marker_corr = corr_mat[c, g]
                    marker_name = feature

            print(marker_name, min_marker_corr)

            if marker_name is None:
                continue

            for g, feature in enumerate(self._dset.get_features()):
                if marker_mat[c, g] == 0 and corr_mat[c, g] >= min_marker_corr:
                    print(state, feature)
                    problem = {
                        'current_marker': feature,
                        'curr_state': state,
                        'cellstate_to_compare': marker_name,
                    }
                    problems.append(problem)

        col_names = ['feature',
                     'should have higher correlation with cell state',
                     'than this feature']

        if len(problems) > 0:
            df_issues = pd.DataFrame(problems)
            df_issues.columns = col_names
        else:
            df_issues = pd.DataFrame(columns=col_names)

        return df_issues


    def get_losses(self) -> np.array:
        """ Getter for losses

        :return: a numpy array of losses for each training iteration the
            model runs
        """
        if self._losses is None:
            raise Exception("The state model has not been trained yet")
        return self._losses

    def is_converged(self) -> bool:
        """ Returns True if the model converged

        :return: self._is_converged
        """
        return self._is_converged


class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """

    pass
