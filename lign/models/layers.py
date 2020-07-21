import torch as th
import torch.nn as nn

#gcn layer in network
class GCN(nn.Module):
    def __init__(self, func = None, module_post = None, module_pre = None):
        super(GCN, self).__init__()
        self.func = func
        self.module_pre = module_pre
        self.module_post = module_post

    def forward(self, g, data):
        g.set_data("h", data)

        if self.module_pre:
            g.apply(self.module_pre, "hidden")

        if self.func:
            g.push(func = self.func, data = "hidden")
        
        if self.module_post:
            g.apply(self.module_post, "hidden")

        return g.pop_data("hidden")

class G_LSTM(nn.Module):
    def __init__(self):
        super(G_LSTM, self).__init__()

    def forward(self, g, data):
        g.set_data("hidden", data)

        #cool

        return g.pop_data("hidden")