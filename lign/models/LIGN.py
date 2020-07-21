import torch.nn as nn
import torch as th
import torch.nn.functional as F
from .layers import GCN

class LIGN_cnn(nn.Module):
    def __init__(self, out_feats):
        super(LIGN_cnn, self).__init__()
        self.gcn1 = GCN(func = sum_neighs_data, module_post = nn.Conv2d(3, 6, 5))
        self.gcn2 = GCN(func = sum_neighs_data, module_post = nn.Conv2d(6, 16, 5))
        self.gcn3 = GCN(func = sum_neighs_data, module_post = nn.Linear(16 * 5 * 5, 150))
        self.gcn4 = GCN(func = sum_neighs_data, module_post = nn.Linear(150, 84))
        self.gcn5 = GCN(func = sum_neighs_data, module_post = nn.Linear(84, out_feats))
        self.pool = GCN(func = sum_neighs_data, module_post = nn.MaxPool2d(2, 2))

    def forward(self, g, features):
        x = self.pool(F.relu(self.gcn1(g, features)))
        x = self.pool(F.relu(self.gcn2(g, x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.gcn3(g, x))
        x = F.relu(self.gcn4(g, x))
        
        return th.tanh(self.gcn5(g, x)) 

def sum_neighs_data(neighs):
    out = neighs[0]
    for neigh in neighs[1:]:
        out = out + neighs
    return out