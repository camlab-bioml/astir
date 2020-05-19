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

from sklearn.preprocessing import StandardScaler

from astir.models.celltype import CellTypeModel
from astir.models.cellstate import CellStateModel
from astir.models.imcdataset import IMCDataSet
from astir.models.recognet import RecognitionNet


class Astir:
    """Loads a .csv expression file and a .yaml marker file.

    :raises NotClassifiableError: raised when the input gene expression
        data or the marker is not classifiable

    :param assignments: cell type assignment probabilities
    :param losses:losses after optimization
    :param type_dict: dictionary mapping cell type
        to the corresponding genes
    :param state_dict: dictionary mapping cell state
        to the corresponding genes
    :param cell_types: list of all cell types from marker
    :param mtype_genes: list of all cell type genes from marker
    :param mstate_genes: list of all cell state genes from marker
    :param N: number of rows of data
    :param G: number of cell type genes
    :param C: number of cell types
    :param initializations: initialization parameters
    :param data: parameters that is not to be optimized
    :param variables: parameters that is to be optimized
    :param include_beta: [summery]
    """

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

    def _sanitize_gex(self, df_gex: pd.DataFrame) -> Tuple[int, int, int]:
        """ Sanitizes the inputed gene expression dataframe.

        :param df_gex: dataframe read from the input .csv file
        :type df_gex: pd.DataFrame

        :raises NotClassifiableError: raised when the input information is not 
             sufficient for the classification.

        :return: # of rows, # of marker genes, # of cell
             types
        :rtype: Tuple[int, int, int]
        """
        N = df_gex.shape[0]
        G_t = len(self._mtype_genes)
        G_s = len(self._mstate_genes)
        C_t = len(self._cell_types)
        C_s = len(self._cell_states)
        if N <= 0:
            raise NotClassifiableError("Classification failed. There " + \
                "should be at least one row of data to be classified.")
        if C_t <= 1:
            raise NotClassifiableError("Classification failed. There " + \
                "should be at least two cell types to classify the data into.")
        if C_s <= 1:
            raise NotClassifiableError("Classification failed. There " + \
                "should be at least two cell states to classify the data into.")
        return N, G_t, G_s, C_t, C_s

    def _get_classifiable_genes(self, df_gex: pd.DataFrame) -> \
            Tuple[pd.DataFrame, pd.DataFrame]:
        """Return the classifiable data which contains a subset of genes from
            the marker and the input data.

        :raises NotClassifiableError: raised when there is no overlap between the
            inputed type or state gene and the marker type or state gene

        :return: classifiable cell type data
            and cell state data
        :rtype: Tuple[pd.Dataframe, pd.Dataframe]
        """
        ## This should also be done to cell_states
        try:
            CT_np = df_gex[self._mtype_genes].to_numpy()
        except(KeyError):
            raise NotClassifiableError("Classification failed. There's no " + \
                "overlap between marked genes and expression genes to " + \
                "classify among cell types.")
        try:
            CS_np = df_gex[self._mstate_genes].to_numpy()
        except(KeyError):
            raise NotClassifiableError("Classification failed. There's no " + 
                "overlap between marked genes and expression genes to " + 
                "classify among cell types.")
        if CT_np.shape[1] < len(self._mtype_genes):
            warnings.warn("Classified type genes are less than marked genes.")
        if CS_np.shape[1] < len(self._mstate_genes):
            warnings.warn("Classified state genes are less than marked genes.")
        if CT_np.shape[1] + CS_np.shape[1] < len(self._expression_genes):
            warnings.warn("Classified type and state genes are less than the expression genes in the input data.")
        
        return CT_np, CS_np

    def _construct_type_mat(self) -> np.array:
        """ Constructs a matrix representing the marker information.

        :return: constructed matrix
        :rtype: np.array
        """
        type_mat = np.zeros(shape = (self._G_t,self._C_t+1))
        for g in range(self._G_t):
            for ct in range(self._C_t):
                gene = self._mtype_genes[g]
                cell_type = self._cell_types[ct]
                if gene in self._type_dict[cell_type]:
                    type_mat[g,ct] = 1

        return type_mat

    def _construct_state_mat(self) -> np.array:
        """ Constructs a matrix representing the marker information.

        :return: constructed matrix
        :rtype: np.array
        """
        state_mat = np.zeros(shape=(self._G_s, self._C_s))

        for g, gene in enumerate(self._mstate_genes):
            for ct, state in enumerate(self._cell_states):
                if gene in self._state_dict[state]:
                    state_mat[g, ct] = 1

        return state_mat

    def __init__(self, 
                df_gex: pd.DataFrame, marker_dict: Dict,
                design = None,
                random_seed = 1234,
                include_beta = True) -> None:
        """Initializes an Astir object

        :param df_gex: the input gene expression dataframe
        :type df_gex: pd.DataFrame
        :param marker_dict: the gene marker dictionary
        :type marker_dict: Dict

        :param design: [description], defaults to None
        :type design: [type], optional
        :param random_seed: seed number to reproduce results, defaults to 1234
        :type random_seed: int, optional
        :param include_beta: [description], defaults to True
        :type include_beta: bool, optional

        :raises NotClassifiableError: raised when randon seed is not an integer
        """
        if not isinstance(random_seed, int):
            raise NotClassifiableError(\
                "Random seed is expected to be an integer.")
        torch.manual_seed(random_seed)

        self._design = design
        self._include_beta = include_beta

        self._type_assignments = None
        self._state_assignments = None

        self._type_dict, self._state_dict = self._sanitize_dict(marker_dict)

        self._cell_types = list(self._type_dict.keys())
        self._cell_states = list(self._state_dict.keys())
    
        self._mtype_genes = list(set([l for s in self._type_dict.values() \
            for l in s]))
        self._mstate_genes = list(set([l for s in self._state_dict.values() \
            for l in s]))

        self._N, self._G_t, self._G_s, self._C_t, self._C_s = \
            self._sanitize_gex(df_gex)

        # Read input data
        self._core_names = list(df_gex.index)
        self._expression_genes = list(df_gex.columns)

        self._type_mat = self._construct_type_mat()
        self._state_mat = self._construct_state_mat()
        self._CT_np, self._CS_np = self._get_classifiable_genes(df_gex)

        # self._type_ast = CellTypeModel(self._CT_np, self._type_dict, \
        #     self._N, self._G_t, self._C_t, type_mat, include_beta, design)
        self._state_ast = \
            CellStateModel(Y_np=self._CS_np, state_dict=self._state_dict,
                           N=self._N, G=self._G_s, C=self._C_s,
                           state_mat=self._state_mat, design=None,
                           include_beta=True, alpha_random=True,
                           random_seed=random_seed)

    def fit_type(self, epochs = 100, learning_rate = 1e-2, 
        batch_size = 1024, num_repeats = 5) -> None:
        seeds = np.random.randint(1, 100000000, num_repeats)
        type_models = [CellTypeModel(self._CT_np, self._type_dict, \
                self._N, self._G_t, self._C_t, self._type_mat, \
                self._include_beta, self._design, int(seed)) for seed in seeds]
        gs = [m.fit(epochs, learning_rate, batch_size) for m in type_models]
        losses = [m.get_losses()[-10:].mean() for m in type_models]

        best_ind = np.argmin(losses)
        self._type_ast = type_models[best_ind]
        g = gs[best_ind]

        self._type_assignments = pd.DataFrame(g)
        self._type_assignments.columns = self._cell_types + ['Other']
        self._type_assignments.index = self._core_names
    
    # def fit_state(self, epochs = 100, learning_rate = 1e-2, 
    #     batch_size = 1024) -> None:
    #         self._state_ast.fit(epochs, learning_rate, batch_size)

    def get_celltypes(self) -> pd.DataFrame:
        """[summary]

        :return: self.assignments
        :rtype: pd.DataFrame
        """
        return self._type_assignments

    # def get_cellstates(self) -> pd.DataFrame:
    #     """[summary]

    #     Returns:
    #         [type] -- [description]
    #     """
    #     return self._state_ast.get_assignments()
    
    def get_type_losses(self) -> float:
        """[summary]

        :return: self.losses
        :rtype: float
        """
        return self._type_ast.get_losses()

    # def get_state_losses(self) -> float:
    #     """[summary]

    #     Returns:
    #         [type] -- [description]
    #     """
    #     return self._state_ast.get_losses()

    def type_to_csv(self, output_csv: str) -> None:
        """[summary]

        :param output_csv: name for the output .csv file
        :type output_csv: str
        """
        self._type_assignments.to_csv(output_csv)

    # def state_to_csv(self, output_csv: str) -> None:
    #     """[summary]

    #     Arguments:
    #         output_csv {[type]} -- [description]
    #     """
    #     self._state_ast.to_csv(output_csv)
    
    def __str__(self) -> str:
        return "Astir object with " + str(self._CT_np.shape[1]) + \
            " columns of cell types, " + str(self._CS_np.shape[1]) + \
            " columns of cell states and " + \
            str(self._CT_np.shape[0]) + " rows."


## NotClassifiableError: an error to be raised when the dataset fails 
# to be analysed.
class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """
    pass

