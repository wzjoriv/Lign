import torch.nn as nn
import torch as th
import torch.nn.functional as F

from lign.nn import GCN, DyLinear
from lign.utils.functions import sum_tensors

class Base(nn.Module):  ## base, feature extractor
    def __init__(self, out_feats):
        super(Base, self).__init__()
        self.unit1 = GCN(nn.Linear(1433, 1000), aggregation=sum_tensors)
        self.unit2 = GCN(nn.Linear(1000, 500), aggregation=sum_tensors)
        self.unit3 = nn.Linear(500, out_feats)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.25)

    def forward(self, g, features):
        x = F.relu(self.unit1(g, features))
        x = self.drop1(F.relu(self.unit2(g, x)))
        x = F.relu(self.unit3(x))

        return self.drop2(x)


class Classifier(nn.Module): ## temporality layer for training
    def __init__(self, in_fea, out_fea, device = 'cuda'):
        super(Classifier, self).__init__()
        self.DyLinear = DyLinear(in_fea, out_fea, device=device) # dynamic linear dense layer
    
    def forward(self, features):
        x = F.log_softmax(self.DyLinear(features), dim=1)
        return x