import torch
import torch.nn.functional as F
import torch.nn as nn


# The recognition net
class TypeRecognitionNet(nn.Module):
    def __init__(self, C: int, G: int, hidden_size=10) -> None:
        super(TypeRecognitionNet, self).__init__()
        self.hidden_1 = nn.Linear(G, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, C + 1)

    def forward(self, x):
        x = self.hidden_1(x)
        x = F.relu(x)
        x = self.hidden_2(x)
        x = F.softmax(x, dim=1)
        return x
