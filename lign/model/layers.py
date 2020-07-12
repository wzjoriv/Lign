import torch as th
import torch.nn as nn


#gcn layer in network
class GCN(nn.Module):
    def __init__(self, graph, data, func = None, module_pre = None, module_post = None):
        super(GCN, self).__init__()
        self.g = graph
        self.data = data
        self.func = func
        self.module_pre = module_pre
        self.module_post = module_post

    def forward(self):
        g.push(func = func, data = data)
        return g.ndata.pop('h')