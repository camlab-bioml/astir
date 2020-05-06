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

# import argparse

# parser = argparse.ArgumentParser(description='Run astir')

# ## argparse treats the options we give it as strings by default

# parser.add_argument("expr_csv", help="CSV expression matrix with cells as rows and proteins as columns. First column is cell ID")
# parser.add_argument("marker_yaml", help="YAML file of cell markers")
# parser.add_argument("output_csv", help="Output CSV of cell assignment probabilities")
# parser.add_argument("--epochs", 
#                     help="Number of training epochs",
#                     type=int,
#                     default=100)
# parser.add_argument("--learning_rate",
#                     help="Learning rate",
#                     type=float,
#                     default=1e-2)
# parser.add_argument("--batch_size",
#                     help="Batch size",
#                     type=int,
#                     default=1024)
# parser.add_argument("--random_seed",
#                     help="Random seed",
#                     type=int,
#                     default=1234)

# args = parser.parse_args()

# torch.manual_seed(args.random_seed)

# print((args.expr_csv, args.marker_yaml, args.output_csv))

# ## Load the gene expression data
# df_gex = pd.read_csv('./BaselTMA_SP43_115_X4Y8.csv', index_col = 0)

# core_names = list(df_gex.index)
# gene_names = list(df_gex.columns)

# ## Load the marker info data
# with open("jackson-2020-markers.yml", 'r') as stream:
#     markers_states = yaml.safe_load(stream)
    
# markers = markers_states['cell_types']

# cell_types = list(markers.keys())
# genes = [l for s in markers.values() for l in s]
# genes = list(set(genes)) # Make unique

# ## Infer dimensions
# N = df_gex.shape[0]
# G = len(genes)
# C = len(cell_types)

# ## Construct marker matrix
# marker_mat = np.zeros(shape = (G,C+1))

# for g in range(len(genes)):
#     for ct in range(len(cell_types)):
#         gene = genes[g]
#         cell_type = cell_types[ct]
        
#         if gene in markers[cell_type]:
#             marker_mat[g,ct] = 1

# Y_np = df_gex[genes].to_numpy()

class Astir:

    def _param_init(self):
        log_sigma_init = np.log(self.Y_np.std(0))
        mu_init = np.log(self.Y_np.mean(0))

        ## prior on z
        self.log_alpha = torch.log(torch.ones(C+1) / (C+1))


        self.log_sigma = Variable(torch.from_numpy(log_sigma_init.copy()), requires_grad = True)
        self.mu = Variable(torch.from_numpy(mu_init.copy()), requires_grad = True)
        self.log_delta = Variable(0 * torch.ones((G,C+1)), requires_grad = True)
        self.delta = torch.exp(log_delta)
        self.rho = torch.from_numpy(marker_mat)

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
    def _forward(log_delta, log_sigma, mu, log_alpha, rho, Y, X):
        
        Y_spread = Y.reshape(-1, self.G, 1).repeat(1, 1, self.C+1)
        
        mean = torch.exp(torch.exp(log_delta) * rho + mu.reshape(-1, 1))
        dist = Normal(mean, torch.exp(log_sigma).reshape(-1, 1))
        

        log_p_y = dist.log_prob(Y_spread)

        log_p_y_on_c = log_p_y.sum(1)
        
        gamma = recog.forward(X)

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
        self.core_names = list(df_gex.index)
        self.gene_names = list(df_gex.columns)
        
        with open(marker_yaml, 'r') as stream:
            markers_states = yaml.safe_load(stream)
            
        self.markers = markers_states['cell_types']

        self.cell_types = list(self.markers.keys())
        self.genes = [l for s in self.markers.values() for l in s]
        self.genes = list(set(self.genes)) # Make unique

        ## Infer dimensions
        N = self.df_gex.shape[0]
        G = len(self.genes)
        C = len(self.cell_types)

        self.marker_mat = self._construct_marker_mat(G, C)
        self.Y_np = self.df_gex[self.genes].to_numpy()

        self.dset = IMCDataSet(self.Y_np)
        self.recog = RecognitionNet(C, G)

        self._param_init()

    def fit(self):
        ## Make dataloader
        dataloader = DataLoader(self.dset, batch_size=min(self.batch_size,N), shuffle=True)

        ## Run training loop
        epochs = self.epochs
        losses = np.empty(epochs)

        for ep in range(epochs):
            L = None
            
            for batch in dataloader:
                Y,X = batch
                optimizer.zero_grad()
                L = forward(self.log_delta, self.log_sigma, self.mu, self.log_alpha, self.rho, Y, X)
                L.backward()
                self.optimizer.step()
            
            l = L.detach().numpy()
            losses[ep] = l
            print(l)

        ## Save output
        g = self.recog.forward(dset.X).detach().numpy()

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

# ## Parameters and initialization

# log_sigma_init = np.log(Y_np.std(0))
# mu_init = np.log(Y_np.mean(0))

# ## prior on z
# log_alpha = torch.log(torch.ones(C+1) / (C+1))

# recog = RecognitionNet(C, G)

# log_sigma = Variable(torch.from_numpy(log_sigma_init.copy()), requires_grad = True)
# mu = Variable(torch.from_numpy(mu_init.copy()), requires_grad = True)
# log_delta = Variable(0 * torch.ones((G,C+1)), requires_grad = True)
# delta = torch.exp(log_delta)
# rho = torch.from_numpy(marker_mat)

# ## Construct optimizer
# learning_rate = 1e-2
# params = [log_sigma, mu, log_delta] + list(recog.parameters())
# optimizer = torch.optim.Adam(params, lr=args.learning_rate)

# ## Declare pytorch forward fn
# def forward(log_delta, log_sigma, mu, log_alpha, rho, Y, X):
    
#     Y_spread = Y.reshape(-1, G, 1).repeat(1, 1, C+1)
    
#     mean = torch.exp(torch.exp(log_delta) * rho + mu.reshape(-1, 1))
#     dist = Normal(mean, torch.exp(log_sigma).reshape(-1, 1))
    

#     log_p_y = dist.log_prob(Y_spread)

#     log_p_y_on_c = log_p_y.sum(1)
    
#     gamma = recog.forward(X)

#     elbo = ( gamma * (log_p_y_on_c + log_alpha - torch.log(gamma)) ).sum()
    
#     return -elbo


# ## Make dataloader
# dset = IMCDataSet(Y_np)
# dataloader = DataLoader(dset, batch_size=min(args.batch_size,N), shuffle=True)


# ## Run training loop
# epochs = args.epochs
# losses = np.empty(epochs)

# for ep in range(epochs):
#     L = None
    
#     for batch in dataloader:
#         Y,X = batch
#         optimizer.zero_grad()
#         L = forward(log_delta, log_sigma, mu, log_alpha, rho, Y, X)
#         L.backward()
#         optimizer.step()
    
#     l = L.detach().numpy()
#     losses[ep] = l
#     print(l)

# ## Save output
# g = recog.forward(dset.X).detach().numpy()

# assignments = pd.DataFrame(g)
# assignments.columns = cell_types + ['Other']
# assignments.index = core_names

# assignments.to_csv("output_csv")

# print("Done!")