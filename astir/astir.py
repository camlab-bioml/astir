# astir: Automated aSsignmenT sIngle-cell pRoteomics

# VSCode tips for python:
# ctrl+` to open terminal
# cmd+P to go to file


import torch
from torch.autograd import Variable
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler

class Astir:
    def _param_init(self, G, C):
        log_sigma_init = np.log(self.Y_np.std(0))
        mu_init = np.log(self.Y_np.mean(0))

        ## prior on z
        self.log_alpha = torch.log(torch.ones(C+1) / (C+1))

        self.log_sigma = Variable(torch.from_numpy(log_sigma_init.copy()), requires_grad = True)
        self.mu = Variable(torch.from_numpy(mu_init.copy()), requires_grad = True)
        self.log_delta = Variable(0 * torch.ones((G,C+1)), requires_grad = True)
        self.delta = torch.exp(self.log_delta)
        self.rho = torch.from_numpy(self.marker_mat)

        ## Construct optimizer
        learning_rate = 1e-2
        params = [self.log_sigma, self.mu, self.log_delta] + list(self.recog.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
    
    def _construct_marker_mat(self, G, C):
        marker_mat = np.zeros(shape = (G,C+1))
        for g in range(G):
            for ct in range(C):
                gene = self.genes[g]
                cell_type = self.cell_types[ct]
                if gene in self.markers[cell_type]:
                    marker_mat[g,ct] = 1
        return marker_mat

    ## Declare pytorch forward fn
    def _forward(self, log_delta, log_sigma, mu, log_alpha, rho, Y, X):
        
        Y_spread = Y.reshape(-1, self.G, 1).repeat(1, 1, self.C+1)
        
        mean = torch.exp(torch.exp(log_delta) * rho + mu.reshape(-1, 1))
        dist = Normal(mean, torch.exp(log_sigma).reshape(-1, 1))
        

        log_p_y = dist.log_prob(Y_spread)

        log_p_y_on_c = log_p_y.sum(1)
        
        gamma = self.recog.forward(X)

        elbo = ( gamma * (log_p_y_on_c + log_alpha - torch.log(gamma)) ).sum()
        
        return -elbo

    def __init__(self, expr_csv, marker_yaml, output_csv, epochs = 100, 
        learning_rate = 1e-2, batch_size = 1024, random_seed = 1234):
        self.output_csv = output_csv
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        #Todo: fix problem with random seed
        torch.manual_seed(random_seed)

        # Read input data
        self.df_gex = pd.read_csv(expr_csv, index_col = 0)
        self.core_names = list(self.df_gex.index)
        self.gene_names = list(self.df_gex.columns)
        
        with open(marker_yaml, 'r') as stream:
            markers_states = yaml.safe_load(stream)
            
        self.markers = markers_states['cell_types']

        self.cell_types = list(self.markers.keys())
        self.genes = [l for s in self.markers.values() for l in s]
        self.genes = list(set(self.genes)) # Make unique

        ## Infer dimensions
        self.N = self.df_gex.shape[0]
        self.G = len(self.genes)
        self.C = len(self.cell_types)

        self.marker_mat = self._construct_marker_mat(self.G, self.C)
        self.Y_np = self.df_gex[self.genes].to_numpy()

        self.dset = IMCDataSet(self.Y_np)
        self.recog = RecognitionNet(self.C, self.G)

        self._param_init(self.G, self.C)

    def fit(self):
        ## Make dataloader
        dataloader = DataLoader(self.dset, batch_size=min(self.batch_size, self.N), shuffle=True)

        ## Run training loop
        epochs = self.epochs
        losses = np.empty(epochs)

        for ep in range(epochs):
            L = None
            
            for batch in dataloader:
                Y,X = batch
                self.optimizer.zero_grad()
                L = self._forward(self.log_delta, self.log_sigma, self.mu, self.log_alpha, self.rho, Y, X)
                L.backward()
                self.optimizer.step()
            
            l = L.detach().numpy()
            losses[ep] = l
            print(l)

        ## Save output
        g = self.recog.forward(self.dset.X).detach().numpy()

        assignments = pd.DataFrame(g)
        assignments.columns = self.cell_types + ['Other']
        assignments.index = self.core_names

        assignments.to_csv(self.output_csv)

        print("Done!")


## Dataset class: for loading IMC datasets
class IMCDataSet(Dataset):
    
    def __init__(self, Y_np):
             
        self.Y = torch.from_numpy(Y_np)
        X = StandardScaler().fit_transform(Y_np)
        self.X = torch.from_numpy(X)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return self.Y[idx,:], self.X[idx,:]

## Recognition network
class RecognitionNet(nn.Module):
    def __init__(self, C, G, h=6):
        super(RecognitionNet, self).__init__()
        self.hidden_1 = nn.Linear(G, h).double()
        self.hidden_2 = nn.Linear(h, C+1).double()

    def forward(self, x):
        x = self.hidden_1(x)
        x = F.relu(x)
        x = self.hidden_2(x)
        x = F.softmax(x, dim=1)
        return x


# ast = Astir("./BaselTMA_SP43_115_X4Y8.csv", "jackson-2020-markers.yml", "./BaselTMA_SP43_115_X4Y8_assignment.csv")
# ast.fit()
