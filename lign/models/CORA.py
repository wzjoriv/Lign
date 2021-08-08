import torch.nn as nn
import torch as th
import torch.nn.functional as F

from lign.nn import GCN, DyLinear
from lign.utils.functions import sum_data

class Base(nn.Module):  ## base, feature extractor
    def __init__(self, out_feats):
        super(Base, self).__init__()
        self.unit1 = GCN(nn.Linear(1433, 700), aggregation=sum_data)
        self.unit2 = GCN(nn.Linear(700, 400), aggregation=sum_data)
        self.unit3 = GCN(nn.Linear(400, 100), aggregation=sum_data, inclusion=nn.Linear(100, out_feats))
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.25)

    def forward(self, g, features):
        x = self.drop1(F.relu(self.unit1(g, features)))
        x = self.drop2(F.relu(self.unit2(g, x)))
        x = F.relu(self.unit3(g, x))

        return x


class Classifier(nn.Module): ## temporality layer for training
    def __init__(self, in_fea, out_fea, device = 'cuda'):
        super(Classifier, self).__init__()
        self.DyLinear = DyLinear(in_fea, out_fea, device=device) # dynamic linear dense layer
    
    def forward(self, features):
        x = F.log_softmax(self.DyLinear(features), dim=1)
        return x