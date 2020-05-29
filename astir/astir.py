# astir: Automated aSsignmenT sIngle-cell pRoteomics

# VSCode tips for python:
# ctrl+` to open terminal
# cmd+P to go to file

import re
from typing import Tuple, List, Dict
import warnings

import torch
from torch.autograd import Variable
from torch.distributions import Normal, StudentT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from .models.celltype import CellTypeModel
from .models.cellstate import CellStateModel
from .models.recognet import RecognitionNet
from .models.scdataset import SCDataset


class Astir:
    r"""Create an Astir object

    :param df_gex: A `pd.DataFrame` holding single-cell expression data, 
        where rows are cells and columns are features. Column names refer to
        feature names and row names refer to cell identifiers.
    :param marker_dict: A dictionary holding cell type and state information
    :param design: An (optional) `pd.DataFrame` that represents a design matrix for the samples
    :param random_seed: The random seed to set
    :param include_beta: Deprecated

    :raises NotClassifiableError: raised when the input gene expression
        data or the marker is not classifiable

    """

    def __init__(
        self,
        input_expr: pd.DataFrame,
        marker_dict: Dict,
        design=None,
        random_seed=1234,
        include_beta=True,
    ) -> None:

        if not isinstance(random_seed, int):
            raise NotClassifiableError("Random seed is expected to be an integer.")
        torch.manual_seed(random_seed)
        self.random_seed = random_seed

        self._type_ast, self._state_ast = None, None
        self._type_assignments, self._state_assignments = None, None

        type_dict, state_dict = self._sanitize_dict(marker_dict)

        if design is not None:
            if isinstance(design, pd.DataFrame):
                design = design.to_numpy()

        if isinstance(input_expr, tuple):
            self._type_dset, self._state_dset = input_expr[0], input_expr[1]
        else:
            self._type_dset = SCDataset(
                expr_input=input_expr,
                marker_dict=type_dict,
                design=design,
                include_other_column=True,
            )
            self._state_dset = SCDataset(
                expr_input=input_expr,
                marker_dict=state_dict,
                design=design,
                include_other_column=False,
            )

        self._design = design
        self._include_beta = include_beta

    def _sanitize_dict(self, marker_dict: Dict[str, dict]) -> Tuple[dict, dict]:
        """Sanitizes the marker dictionary.

        :param marker_dict: dictionary read from the yaml file
        :type marker_dict: Dict[str, dict]

        :raises NotClassifiableError: raized when the marker dictionary doesn't 
             have required format

        :return: dictionaries for cell type and state.
        :rtype: Tuple[dict, dict]
        """
        keys = list(marker_dict.keys())
        if not len(keys) == 2:
            raise NotClassifiableError(
                "Marker file does not follow the required format."
            )
        ct = re.compile("cell[^a-zA-Z0-9]*type", re.IGNORECASE)
        cs = re.compile("cell[^a-zA-Z0-9]*state", re.IGNORECASE)
        if ct.match(keys[0]):
            type_dict = marker_dict[keys[0]]
        elif ct.match(keys[1]):
            type_dict = marker_dict[keys[1]]
        else:
            raise NotClassifiableError(
                "Can't find cell type dictionary" + " in the marker file."
            )
        if cs.match(keys[0]):
            state_dict = marker_dict[keys[0]]
        elif cs.match(keys[1]):
            state_dict = marker_dict[keys[1]]
        else:
            raise NotClassifiableError(
                "Can't find cell state dictionary" + " in the marker file."
            )
        return type_dict, state_dict

    def fit_type(
        self,
        max_epochs=10,
        learning_rate=1e-2,
        batch_size=24,
        delta_loss=1e-3,
        n_init=5,
    ) -> None:
        """Run Variational Bayes to infer cell types

        :param max_epochs: Maximum number of epochs to train
        :param learning_rate: ADAM optimizer learning rate
        :param batch_size: Minibatch size
        :param delta_loss: stops iteration once the loss rate reaches
            delta_loss, defaults to 0.001
        :param n_inits: Number of random initializations
        """
        np.random.seed(self.random_seed)
        seeds = np.random.randint(1, 100000000, n_init)
        type_models = [
            CellTypeModel(self._type_dset, self._include_beta, self._design, int(seed))
            for seed in seeds
        ]
        gs = []
        for i in range(n_init):
            print(
                "---------- Astir Training "
                + str(i + 1)
                + "/"
                + str(n_init)
                + " ----------"
            )
            gs.append(
                type_models[i].fit(max_epochs, learning_rate, batch_size, delta_loss)
            )
        if max_epochs >= 2:
            losses = [m.get_losses()[-2:].mean() for m in type_models]
        else:
            losses = [m.get_losses()[0] for m in type_models]

        best_ind = np.argmin(losses)
        self._type_ast = type_models[best_ind]
        if not self._type_ast.is_converged():
            msg = (
                "Maximum epochs reached. More iteration may be needed to"
                + " complete the training."
            )
            warnings.warn(msg)

        # plt.plot(self._type_ast.get_losses())
        # plt.ylabel('losses')
        # plt.show()

        self._type_assignments = pd.DataFrame(gs[best_ind])
        self._type_assignments.columns = self._type_dset.get_classes() + ["Other"]
        self._type_assignments.index = self._type_dset.get_cells()

    def fit_state(
        self,
        max_epochs=100,
        learning_rate=1e-2,
        n_init=5,
        delta_loss=1e-3,
        delta_loss_batch=10,
    ):
        """Run Variational Bayes to infer cell states

        :param max_epochs: number of epochs, defaults to 100
        :param learning_rate: the learning rate, defaults to 1e-2
        :param n_init: the number of initial parameters to compare,
            defaults to 5
        :param delta_loss: stops iteration once the loss rate reaches
            delta_loss, defaults to 0.001
        :param delta_loss_batch: the batch size  to consider delta loss,
            defaults to 10
        """
        cellstate_models = []
        cellstate_losses = []

        if delta_loss_batch >= max_epochs:
            warnings.warn("Delta loss batch size is greater than the number of epochs")

        for i in range(n_init):
            # Initializing a model
            model = CellStateModel(
                dset=self._state_dset,
                include_beta=True,
                alpha_random=True,
                random_seed=(self.random_seed + i),
            )

            print(
                "---------- Astir Training "
                + str(i + 1)
                + "/"
                + str(n_init)
                + " ----------"
            )
            # Fitting the model
            n_init_epochs = min(max_epochs, 100)
            losses = model.fit(
                max_epochs=n_init_epochs,
                lr=learning_rate,
                delta_loss=delta_loss,
                delta_loss_batch=delta_loss_batch,
            )

            cellstate_losses.append(losses)
            cellstate_models.append(model)

        last_delta_losses_mean = np.array(
            [losses[-delta_loss_batch:].mean() for losses in cellstate_losses]
        )

        best_model_index = np.argmin(last_delta_losses_mean)

        self._state_ast = cellstate_models[best_model_index]
        n_epochs_done = cellstate_losses[best_model_index].size
        n_epoch_remaining = max(max_epochs - n_epochs_done, 0)

        self._state_ast.fit(
            max_epochs=n_epoch_remaining,
            lr=learning_rate,
            delta_loss=delta_loss,
            delta_loss_batch=delta_loss_batch,
        )

        # Warns the user if the model has not converged
        if not self._state_ast.is_converged():
            msg = (
                "Maximum epochs reached. More iteration may be needed to"
                + " complete the training."
            )
            warnings.warn(msg)

        g = self._state_ast._variables["z"].detach().numpy()

        self._state_assignments = pd.DataFrame(g)
        self._state_assignments.columns = self._state_dset.get_classes()
        self._state_assignments.index = self._state_dset.get_cells()

    def get_type_dataset(self):
        return self._type_dset

    def get_state_dataset(self):
        return self._state_dset

    def get_celltype_assignments(self) -> pd.DataFrame:
        """[summary]

        :return: self.assignments
        :rtype: pd.DataFrame
        """
        if self._type_assignments is None:
            raise Exception("The type model has not been trained yet")
        return self._type_assignments

    def get_cellstate_assignments(self) -> pd.DataFrame:
        """ Gets state assignment output from training state model

        :return: state assignments
        :rtype: pd.DataFrame
        """
        if self._state_assignments is None:
            raise Exception("The state model has not been trained yet")
        return self._state_assignments

    def get_type_losses(self) -> np.array:
        """[summary]

        :return: self.losses
        :rtype: np.array
        """
        if self._type_ast is None:
            raise Exception("The type model has not been trained yet")
        return self._type_ast.get_losses()

    def get_state_losses(self) -> np.array:
        """ Getter for losses

        :return: a numpy array of losses for each training iteration the
        model runs
        :rtype: np.array
        """
        if self._state_ast is None:
            raise Exception("The state model has not been trained yet")
        return self._state_ast.get_losses()

    def type_to_csv(self, output_csv: str) -> None:
        """[summary]

        :param output_csv: name for the output .csv file
        :type output_csv: str
        """
        if self._type_assignments is None:
            raise Exception("The type model has not been trained yet")
        self._type_assignments.to_csv(output_csv)

    def state_to_csv(self, output_csv: str) -> None:
        """ Writes state assignment output from training state model in csv
        file

        :param output_csv: path to output csv
        :type output_csv: str, required
        """
        if self._state_assignments is None:
            raise Exception("The state model has not been trained yet")
        self._state_assignments.to_csv(output_csv)

    def __str__(self) -> str:
        return (
            "Astir object with "
            + str(self._type_dset.get_n_classes())
            + " columns of cell types, "
            + str(self._state_dset.get_n_classes())
            + " columns of cell states and "
            + str(len(self._type_dset))
            + " rows."
        )


## NotClassifiableError: an error to be raised when the dataset fails
# to be analysed.
class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """

    pass


# KC removed
# :param assignments: cell type assignment probabilities
# :param losses: losses after optimization
# :param type_dict: dictionary mapping cell type
#     to the corresponding genes
# :param state_dict: dictionary mapping cell state
#     to the corresponding genes
# :param cell_types: list of all cell types from marker
# :param mtype_genes: list of all cell type genes from marker
# :param mstate_genes: list of all cell state genes from marker
# :param N: number of rows of data
# :param G: number of cell type genes
# :param C: number of cell types
# :param initializations: initialization parameters
# :param data: parameters that is not to be optimized
# :param variables: parameters that is to be optimized
# :param include_beta: [summery]
