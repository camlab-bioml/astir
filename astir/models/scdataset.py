import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import warnings


class SCDataset(Dataset):
    """Container for single-cell proteomic data in the form of 
    a pytorch dataset

    :param expr_input: Input expression data. See details TODO
    :param marker_dict: Marker dictionary containing cell type and 
        information. See details :TODO:
    :param design: A design matrix
    :param include_other_column: Should an additional 'other' column be included?
    """
    def __init__(
        self,
        expr_input,
        marker_dict: Dict[str, str],
        design: np.array,
        include_other_column: bool,
    ) -> None:
        self._marker_dict = marker_dict
        self._m_features = sorted(list(set([l for s in marker_dict.values()
                                           for l in s])))
        self._classes = list(marker_dict.keys())

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## sanitize features
        if len(self._classes) <= 1:
            raise NotClassifiableError(
                "Classification failed. There "
                + "should be at least two cell classes to classify the data into."
            )
        self._marker_mat = self._construct_marker_mat(
            include_other_column=include_other_column
        )

        if isinstance(expr_input, pd.DataFrame):
            self._exprs = self._process_df_input(expr_input)
            self._expr_features = list(expr_input.columns)
            self._cell_names = list(expr_input.index)
        elif isinstance(expr_input, tuple):
            self._expr_features = expr_input[1]
            self._cell_names = expr_input[2]
            self._exprs = self._process_np_input(expr_input[0])
        self.design = self._fix_design(design)
        ## sanitize df
        if self._exprs.shape[0] <= 0:
            raise NotClassifiableError(
                "Classification failed. There "
                + "should be at least one row of data to be classified."
            )
        self._exprs_mean = self._exprs.mean(0)
        self._exprs_std = self._exprs.std(0)

    def _process_df_input(self, df_input):
        """Processes input as pd.DataFrame and convert it into torch.Tensor

        :param df_input: the input
        :type df_input: pd.DataFrame
        :raises NotClassifiableError: raised when there is no overlap between the 
            data and the marker
        :return: the processed input as a torch.Tensor
        :rtype: torch.Tensor
        """
        try:
            Y_np = df_input[self._m_features].to_numpy()
        except (KeyError):
            raise NotClassifiableError(
                "Classification failed. There's no "
                + "overlap between marked features and expression features for "
                + "the classification of cell type/state."
            )
        return torch.from_numpy(Y_np).float().to(self._device)

    def _process_np_input(self, np_input):
        """Process the input as Tuple[np.array, np.array, np.array] and convert it 
            to torch.Tensor.
        :param np_input: input as a tuple. np_input[0] is the input data. np.input[1]
            is the 
        :type np_input: Tuple[np.array, np.array, np.array]
        :raises NotClassifiableError: raised when there is no overlap between marked
            features and expression feature.
        :return: the processed input as a torch.Tensor
        :rtype: torch.Tensor
        """
        ind = [
            self._expr_features.index(name)
            for name in self._m_features
            if name in self._expr_features
        ]
        if len(ind) <= 0:
            raise NotClassifiableError(
                "Classification failed. There's no "
                + "overlap between marked features and expression features for "
                + "the classification of cell type/state."
            )
        if len(ind) < len(self._m_features):
            warnings.warn("Classified features are less than marked features.")
        Y_np = []
        for cell in np_input[0]:
            temp = [cell[i] for i in ind]
            Y_np.append(np.array(temp))
        Y_np = np.concatenate([Y_np], axis=0)
        return torch.from_numpy(Y_np).to(self._device)

    def _construct_marker_mat(self, include_other_column: bool) -> torch.Tensor:
        """ Construct a marker matrix.
        :param include_other_column: indicates whether or not include other columns.
        :type include_other_column: bool
        :return: A marker matrix. The rows are features and the coloumns are 
            the corresponding classes (type/state).
        :rtype: torch.Tensor
        """
        G = self.get_n_features()
        C = self.get_n_classes()

        marker_mat = torch.zeros((G, C + 1 if include_other_column else C)).to(self._device)
        for g, feature in enumerate(self._m_features):
            for c, cell_class in enumerate(self._classes):
                if feature in self._marker_dict[cell_class]:
                    marker_mat[g, c] = 1.0
        return marker_mat

    def __len__(self) -> int:
        # N
        return self._exprs.shape[0]

    def __getitem__(self, idx):
        y = self._exprs[idx, :]
        x = (y - self._exprs_mean) / self._exprs_std
        return y, x, self.design[idx, :]

    def _fix_design(self, design: np.array) -> torch.tensor:
        d = None
        if design is None:
            d = torch.ones((self._exprs.shape[0], 1)).float().to(self._device)
        else:
            d = torch.from_numpy(design).float().to(self._device)

        if d.shape[0] != self._exprs.shape[0]:
            raise NotClassifiableError(
                "Number of rows of design matrix "
                + "must equal number of rows of expression data"
            )
        return d

    def rescale(self):
        self._exprs = self._exprs / (self.get_sigma())

    def get_exprs(self):
        """Return the expression data as a :class:`torch.Tensor`

        """
        return self._exprs
    
    def get_exprs_df(self):
        """Return the expression data as a :class:`pandas.DataFrame`

        """
        df = pd.DataFrame(self.get_exprs().detach().numpy())
        df.index = self.get_cell_names()
        df.columns = self.get_features()
        return df

    def get_marker_mat(self):
        """Return the marker matrix as a :class:`torch.Tensor`

        """
        return self._marker_mat

    def get_mu(self):
        """Get the mean expression of each protein as a :class:`torch.Tensor`
        """
        return self._exprs_mean

    def get_sigma(self):
        return self._exprs_std

    def get_n_classes(self) -> int:
        """Get the number of 'classes': either the number of cell types or cell states 

        """
        return len(self._classes)

    def get_n_cells(self) -> int:
        """Get the number of cells: either the number of cell types or cell states 

        """
        return len(self.get_cell_names())

    def get_n_features(self) -> int:
        """Get the number of features (proteins)

        """
        return len(self._m_features)

    def get_features(self):
        return self._m_features

    def get_cell_names(self):
        return self._cell_names

    def get_classes(self):
        return self._classes
    
    def normalize(self, percentile_lower:int = 1, percentile_upper:int = 99) -> None:
        """Normalize the expression data

        This performs a two-step normalization:
        1. A `log(1+x)` transformation to the data
        2. Winsorizes to (:param:`percentile_lower`, :param:`percentile_upper`)
        """

        with torch.no_grad():
            exprs = self.get_exprs().numpy()
            exprs = np.log1p(exprs)
            q_low = np.percentile(exprs, (percentile_lower), axis=0)
            q_high = np.percentile(exprs, (percentile_upper), axis=0)

            for g in range(exprs.shape[1]):
                exprs[:,g][exprs[:,g] < q_low[g]] = q_low[g]
                exprs[:,g][exprs[:,g] > q_high[g]] = q_high[g]

            self._exprs = torch.tensor(exprs)

class NotClassifiableError(RuntimeError):
    pass
