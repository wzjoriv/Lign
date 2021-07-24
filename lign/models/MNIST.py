import torch as th
import torch.nn as nn
import torch.nn.functional as F

from lign.nn import DyLinear

class Base(nn.Module):  ## base, feature extractor
    def __init__(self, out_feats):
        super(Base, self).__init__()
        self.unit1 = nn.Conv2d(1, 32, 3, 1)
        self.unit2 = nn.Conv2d(32, 64, 3, 1)
        self.unit3 = nn.Linear(9216, out_feats)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, features):
        x = F.relu(self.unit1(features))

        x = F.relu(self.unit2(x))
        x = F.max_pool2d(x, 2)
        x = th.flatten(self.drop1(x), 1)
        x = F.relu(self.unit3(x))

        return self.drop2(x)


class Classifier(nn.Module): ## temporality layer for training
    def __init__(self, in_fea, out_fea, device = 'cuda'):
        super(Classifier, self).__init__()
        self.DyLinear = DyLinear(in_fea, out_fea, device=device) # dynamic linear dense layer
    
    def forward(self, features):
        x = F.log_softmax(self.DyLinear(features), dim=1)
        return x