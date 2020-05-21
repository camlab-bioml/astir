import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.preprocessing import StandardScaler


## Dataset class: for loading IMC datasets
class IMCDataSet(Dataset):
    """Pytorch holder for numpy data

    """
    
    def __init__(self, Y_np: np.array, design: np.array) -> None:
        self.Y = torch.from_numpy(Y_np)
        X = StandardScaler().fit_transform(Y_np)
        self.X = torch.from_numpy(X)
        self.design = self._fix_design(design)
    
    def __len__(self) -> int:
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return self.Y[idx,:], self.X[idx,:], self.design[idx,:]
    
    def _fix_design(self, design: np.array) -> torch.tensor:
        
        d = None
        if design is None:
            d = torch.ones((self.Y.shape[0],1)).double()
        else:
            d = torch.from_numpy(design).double()


        if d.shape[0] != self.Y.shape[0]:
            raise NotClassifiableError("Number of rows of design matrix " + \
                "must equal number of rows of expression data")
            
        return d

class NotClassifiableError(RuntimeError):
    pass