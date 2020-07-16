import torch as th
import torch.nn as nn


#gcn layer in network
class GCN(nn.Module):
    def __init__(self, graph, func, module_post = None, module_pre = None):
        super(GCN, self).__init__()
        self.g = graph
        self.func = func
        self.module_pre = module_pre
        self.module_post = module_post

    def forward(self, data):
        self.g.set_data("h", data)

        if self.module_pre:
            self.g.apply(self.module_pre, "h")

        self.g.push(func = self.func, data = "h")
        
        if self.module_post:
            self.g.apply(self.module_post, "h")

        return self.g.pop_data("h")