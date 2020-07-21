import torch as th
import torch.nn as nn

#gcn layer in network
class GCN(nn.Module):
    def __init__(self, post_mod, func = None, pre_mod = None):
        super(GCN, self).__init__()
        self.func = func
        self.pre_mod = pre_mod
        self.post_mod = post_mod

    def forward(self, g, data):
        g.set_data(".hidden", data)

        if self.pre_mod:
            g.apply(self.pre_mod, ".hidden")

        if self.func:
            g.push(func = self.func, data = ".hidden")
        
        g.apply(self.post_mod, ".hidden")

        return g.pop_data(".hidden")

class G_LSTM(nn.Module):
    def __init__(self):
        super(G_LSTM, self).__init__()

    def forward(self, g, data):
        g.set_data(".hidden", data)

        #cool

        return g.pop_data(".hidden")