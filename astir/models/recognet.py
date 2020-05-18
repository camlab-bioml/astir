import torch
import torch.nn.functional as F
import torch.nn as nn


# The recognition net
class RecognitionNet(nn.Module):
    def __init__(self, C: int, G: int, h=6) -> None:
        super(RecognitionNet, self).__init__()
        self.hidden_1 = nn.Linear(G, h).double()
        self.hidden_2 = nn.Linear(h, C+1).double()

    def forward(self, x):
        x = self.hidden_1(x)
        x = F.relu(x)
        x = self.hidden_2(x)
        x = F.softmax(x, dim=1)
        return x