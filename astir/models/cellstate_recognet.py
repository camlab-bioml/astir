import torch
import torch.nn.functional as F
import torch.nn as nn


# The recognition net
class StateRecognitionNet(nn.Module):
    def __init__(self, C: int, G: int) -> None:
        super(StateRecognitionNet, self).__init__()
        self.input = nn.Linear(G, 2 * C).float()
        self.hidden = nn.Linear(2 * C, 2 * C).float()
        self.output_mu = nn.Linear(2 * C, C).float()
        self.output_std = nn.Linear(2 * C, C).float()

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)

        mu_z = self.output_mu(x)
        std_z = self.output_std(x)
        return mu_z, std_z
