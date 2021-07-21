import torch.nn as nn
import torch as th
import torch.nn.functional as F

from lign.nn import DyLinear, Module

class Base(Module):  ## base, feature extractor
    def __init__(self, out_feats):
        super(Base, self).__init__()
        self.unit1 = nn.Conv2d(3, 6, 5)
        self.unit2 = nn.Conv2d(6, 16, 5)
        self.unit3 = nn.Linear(16 * 5 * 5, 150)
        self.unit4 = nn.Linear(150, 84)
        self.unit5 = nn.Linear(84, out_feats)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, g, features, save=False):
        x = self.pool(g, F.relu(self.unit1(g, features)))
        x = self.pool(g, F.relu(self.unit2(g, x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.unit3(g, x))
        x = F.relu(self.unit4(g, x))
        
        return th.tanh(self.unit5(g, x))


class Classifier(Module): ## temporality layer for training
    def __init__(self, in_fea, out_fea, device = 'cuda'):
        super(Classifier, self).__init__()
        self.DyLinear = DyLinear(in_fea, out_fea, device=device) # dynamic linear dense layer
    
    def forward(self, features):
        x = F.log_softmax(self.DyLinear(features), dim=1)
        return x