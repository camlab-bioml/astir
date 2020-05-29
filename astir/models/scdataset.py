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
        self._m_features = list(set([l for s in marker_dict.values() for l in s]))
        self._classes = list(marker_dict.keys())
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
            self._core_names = list(expr_input.index)
        elif isinstance(expr_input, tuple):
            self._expr_features = expr_input[1]
            self._core_names = expr_input[2]
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
        try:
            Y_np = df_input[self._m_features].to_numpy()
        except (KeyError):
            raise NotClassifiableError(
                "Classification failed. There's no "
                + "overlap between marked features and expression features for "
                + "the classification of cell type/state."
            )
        return torch.from_numpy(Y_np)

    def _process_np_input(self, np_input):
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
        for cell in np_input:
            temp = [cell[i] for i in ind]
            Y_np.append(np.array(temp))
        Y_np = np.concatenate([Y_np], axis=0)
        return torch.from_numpy(Y_np)

    def _construct_marker_mat(self, include_other_column: bool) -> torch.Tensor:
        G = self.get_n_features()
        C = self.get_n_classes()

        marker_mat = torch.zeros((G, C + 1 if include_other_column else C))
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
            d = torch.ones((self._exprs.shape[0], 1)).double()
        else:
            d = torch.from_numpy(design).double()

        if d.shape[0] != self._exprs.shape[0]:
            raise NotClassifiableError(
                "Number of rows of design matrix "
                + "must equal number of rows of expression data"
            )
        return d

    def rescale(self):
        self._exprs = self._exprs / (self.get_sigma())

    def get_exprs(self):
        return self._exprs

    def get_marker_mat(self):
        return self._marker_mat

    def get_mu(self):
        return self._exprs_mean

    def get_sigma(self):
        return self._exprs_std

    def get_n_classes(self):
        ## C
        return len(self._classes)

    def get_n_features(self):
        ## G
        return len(self._m_features)

    def get_features(self):
        return self._m_features

    def get_cells(self):
        return self._core_names

    def get_classes(self):
        return self._classes


class NotClassifiableError(RuntimeError):
    pass
