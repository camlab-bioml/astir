# astir: Automated aSsignmenT sIngle-cell pRoteomics

# VSCode tips for python:
# ctrl+` to open terminal
# cmd+P to go to file

import re
from typing import Tuple, List, Dict, Union
import warnings

import torch

import pandas as pd
import numpy as np

from .models.celltype import CellTypeModel
from .models.cellstate import CellStateModel
from .models.scdataset import SCDataset


class Astir:
    """Create an Astir object

    :param df_gex: A `pd.DataFrame` holding single-cell expression data,
        where rows are cells and columns are features. Column names refer to
        feature names and row names refer to cell identifiers.
    :param marker_dict: A dictionary holding cell type and state information
    :param design: An (optional) `pd.DataFrame` that represents a
        design matrix for the samples
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
        include_beta=False,
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
        """ Sanitizes the marker dictionary.

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
        max_epochs=50,
        learning_rate=5e-3,
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
        n_init_epochs = min(max_epochs, 1)
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
                type_models[i].fit(n_init_epochs, learning_rate, batch_size, delta_loss)
            )
        if max_epochs >= 2:
            losses = [m.get_losses()[-2:].mean() for m in type_models]
        else:
            losses = [m.get_losses()[0] for m in type_models]

        best_ind = np.argmin(losses)
        self._type_ast = type_models[best_ind]

        n_epoch_remaining = max_epochs - 1
        self._type_ast.fit(n_epoch_remaining, learning_rate, batch_size, delta_loss)
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
        self._type_assignments.index = self._type_dset.get_cell_names()

    def fit_state(
        self,
        max_epochs=50,
        learning_rate=1e-3,
        n_init=5,
        batch_size=24,
        delta_loss=1e-3,
        delta_loss_batch=10,
    ) -> None:
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
            n_init_epochs = min(max_epochs, batch_size)
            losses = model.fit(
                max_epochs=n_init_epochs,
                lr=learning_rate,
                batch_size=batch_size,
                delta_loss=delta_loss,
                delta_loss_batch=delta_loss_batch,
            )

            cellstate_losses.append(losses)
            cellstate_models.append(model)

        last_delta_losses_mean = np.array(
            [losses[-delta_loss_batch:].mean() for losses in cellstate_losses]
        )

        best_model_index = int(np.argmin(last_delta_losses_mean))

        self._state_ast = cellstate_models[best_model_index]
        n_epochs_done = cellstate_losses[best_model_index].size
        n_epoch_remaining = max(max_epochs - n_epochs_done, 0)

        self._state_ast.fit(
            max_epochs=n_epoch_remaining,
            lr=learning_rate,
            batch_size=batch_size,
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

        g = self._state_ast.get_final_mu_z().detach().cpu().numpy()

        self._state_assignments = pd.DataFrame(g)
        self._state_assignments.columns = self._state_dset.get_classes()
        self._state_assignments.index = self._state_dset.get_cell_names()

    def get_type_dataset(self):
        return self._type_dset

    def get_state_dataset(self):
        return self._state_dset

    def get_type_model(self):
        return self._type_ast

    def get_state_model(self):
        return self._state_ast

    def get_celltype_probabilities(self) -> pd.DataFrame:
        """[summary]

        :return: self.assignments
        :rtype: pd.DataFrame
        """
        if self._type_assignments is None:
            raise Exception("The type model has not been trained yet")
        return self._type_assignments

    def _most_likely_celltype(self, row, threshold, cell_types):
        """Given a row of the assignment matrix
        return the most likely cell type

        """
        row = row.to_numpy()
        max_prob = np.max(row)

        if max_prob < threshold:
            return "Unknown"

        return cell_types[np.argmax(row)]

    def get_celltypes(self, threshold=0.7) -> pd.DataFrame:
        """
        Get the most likely cell types

        A cell is assigned to a cell type if the probability is greater than threshold.
        If no cell types have a probability higher than threshold, then "Unknown" is returned

        :param threshold: The probability threshold above which a cell is assigned to a cell type.
        :return: A data frame with most likely cell types for each 
        """
        probs = self.get_celltype_probabilities()
        cell_types = list(probs.columns)

        cell_type_assignments = probs.apply(
            self._most_likely_celltype,
            axis=1,
            threshold=threshold,
            cell_types=cell_types,
        )
        cell_type_assignments = pd.DataFrame(cell_type_assignments)
        cell_type_assignments.columns = ["cell_type"]

        return cell_type_assignments

    def get_cellstates(self) -> pd.DataFrame:
        """ Get cell state activations

        :return: state assignments
        :rtype: pd.DataFrame
        """
        if self._state_assignments is None:
            raise Exception("The state model has not been trained yet")
        return self._state_assignments

    def predict_celltypes(self, dset=None):
        if self._type_ast is None:
            raise Exception("The type model has not been trained yet")
        if not self._type_ast.is_converged():
            msg = "The state model has not been trained for enough epochs yet"
            warnings.warn(msg)
        if dset is None:
            dset = self.get_type_dataset()
        g = self._type_ast.predict(dset).detach().cpu().numpy()

        type_assignments = pd.DataFrame(g)
        type_assignments.columns = dset.get_classes() + ["others"]
        type_assignments.index = dset.get_cell_names()
        return type_assignments

    def predict_cellstates(self, new_dset: SCDataset) -> pd.DataFrame:
        """ Get the prediction cell state activations on a dataset on an
        existing model

        :param new_dset: the dataset to predict cell state activations

        :return: the prediction of cell state activations
        """
        if not self._state_ast.is_converged():
            msg = "The state model has not been trained for enough epochs yet"
            warnings.warn(msg)

        g = self._state_ast.get_final_mu_z(new_dset).detach().cpu().numpy()

        state_assignments = pd.DataFrame(g)
        state_assignments.columns = self._state_dset.get_classes()
        state_assignments.index = self._state_dset.get_cell_names()

        return state_assignments

    def get_type_losses(self) -> np.array:
        """[summary]

        :return: self.losses
        :rtype: np.array
        """
        if self._type_ast is None:
            raise Exception("The type model has not been trained yet")
        return self._type_ast.get_losses()

    def get_state_losses(self) -> np.array:
        """Getter for losses

        :return: a numpy array of losses for each training iteration the
        model runs
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
            + " cell types, "
            + str(self._state_dset.get_n_classes())
            + " cell states, and "
            + str(len(self._type_dset))
            + " cells."
        )

    def diagnostics_celltype(
        self, threshold: float = 0.7, alpha: float = 0.01
    ) -> pd.DataFrame:
        """Run diagnostics on cell type assignments

        This performs a basic test that cell types express their markers at
        higher levels than in other cell types. This function performs the following steps:

        1. Iterates through every cell type and every marker for that cell type
        2. Given a cell type *c* and marker *g*, find the set of cell types *D* that don't have *g* as a marker
        3. For each cell type *d* in *D*, perform a t-test between the expression of marker *g* in *c* vs *d*
        4. If *g* is not expressed significantly higher (at significance *alpha*), output a diagnostic explaining this for further investigation.

        :param threshold: The threshold at which cell types are assigned (see `get_celltypes`)
        :param alpha: The significance threshold for t-tests for determining over-expression
        :return: Either a :class:`pd.DataFrame` listing cell types whose markers
            aren't expressed signficantly higher.
        """
        celltypes = list(self.get_celltypes(threshold=threshold)["cell_type"])
        return self.get_type_model().diagnostics(celltypes, alpha=alpha)

    def diagnostics_cellstate(self) -> pd.DataFrame:
        """ Run diagnostics on cell state assignments

        This performs a basic test by comparing the correlation values
        between all marker genes and all non marker genes. It detects where the
        non marker gene has higher correlation values than the smallest
        correlation values of marker genes.

        :return: diagnostics
        """
        return self.get_state_model().diagnostics()

    def normalize(self, percentile_lower: int = 1, percentile_upper: int = 99) -> None:
        """Normalize the expression data

        This performs a two-step normalization:
        1. A `log(1+x)` transformation to the data
        2. Winsorizes to (:param:`percentile_lower`, :param:`percentile_upper`)

        :param percentile_lower: Lower percentile for winsorization
        :param percentile_upper: Upper percentile for winsorization
        """
        self.get_type_dataset().normalize(percentile_lower, percentile_upper)
        self.get_state_dataset().normalize(percentile_lower, percentile_upper)


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
