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

from sklearn.preprocessing import StandardScaler


class Astir:

    def _param_init(self):
        log_sigma_init = np.log(self.Y_np.std(0))
        mu_init = np.log(self.Y_np.mean(0))

        ## prior on z
        self.log_alpha = torch.log(torch.ones(self.C+1) / (self.C+1))

        self.log_sigma = Variable(torch.from_numpy(log_sigma_init.copy()), requires_grad = True)
        self.mu = Variable(torch.from_numpy(mu_init.copy()), requires_grad = True)
        self.log_delta = Variable(0 * torch.ones((self.G,self.C+1)), requires_grad = True)
        self.delta = torch.exp(self.log_delta)
        self.rho = torch.from_numpy(self.marker_mat)
    
    def _construct_marker_mat(self):
        marker_mat = np.zeros(shape = (self.G,self.C+1))
        for g in range(self.G):
            for ct in range(self.C):
                gene = self.marker_genes[g]
                cell_type = self.cell_types[ct]
                if gene in self.marker_dict[cell_type]:
                    marker_mat[g,ct] = 1
        print(marker_mat)
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

    ## Todo: an output function
    def __init__(self, df_gex, marker_dict, random_seed = 1234):
        #Todo: fix problem with random seed
        torch.manual_seed(random_seed)

        self.marker_dict = marker_dict['cell_types']

        # Read input data
        self.df_gex = df_gex
        self.core_names = list(df_gex.index)
        self.expression_genes = list(df_gex.columns)
            

        self.cell_types = list(self.marker_dict.keys())
        self.marker_genes = [l for s in self.marker_dict.values() for l in s]
        self.marker_genes = list(set(self.marker_genes)) # Make unique

        ## Infer dimensions
        self.N = self.df_gex.shape[0]
        self.G = len(self.marker_genes)
        self.C = len(self.cell_types)

        self.marker_mat = self._construct_marker_mat()
        self.Y_np = self.df_gex[self.marker_genes].to_numpy()

        self.dset = IMCDataSet(self.Y_np)
        self.recog = RecognitionNet(self.C, self.G)

        self._param_init()

    def fit(self, epochs = 100, 
        learning_rate = 1e-2, batch_size = 1024):
        ## Make dataloader
        dataloader = DataLoader(self.dset, batch_size=min(batch_size, self.N), shuffle=True)

        ## Run training loop
        losses = np.empty(epochs)

        ## Construct optimizer
        params = [self.log_sigma, self.mu, self.log_delta] + list(self.recog.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        for ep in range(epochs):
            L = None
            
            for batch in dataloader:
                Y,X = batch
                optimizer.zero_grad()
                L = self._forward(self.log_delta, self.log_sigma, self.mu, self.log_alpha, self.rho, Y, X)
                L.backward()
                optimizer.step()
            
            l = L.detach().numpy()
            losses[ep] = l
            print(l)

        ## Save output
        g = self.recog.forward(self.dset.X).detach().numpy()

        assignments = pd.DataFrame(g)
        assignments.columns = self.cell_types + ['Other']
        assignments.index = self.core_names

        print("Done!")

    def output_csv(self, output_csv):
        ## Save output
        g = self.recog.forward(self.dset.X).detach().numpy()

        assignments = pd.DataFrame(g)
        assignments.columns = self.cell_types + ['Other']
        assignments.index = self.core_names

        assignments.to_csv(output_csv)

    # def __str__()


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

