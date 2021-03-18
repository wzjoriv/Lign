import torch as th
import torch.nn as nn
import torch.nn.functional as F

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

class ADDON(nn.Module):
    def __init__(self, in_fea, out_fea, base = None, device = 'cuda'):
        super(ADDON, self).__init__()
        self.base = base
        self.device = device
        self.in_fea = in_fea
        self.out_fea = out_fea
        self.weight = nn.Parameter(th.randn(out_fea, in_fea)).to(self.device)
    
    def forward(self, g, x):
        if self.base:
            x = self.base(g, x)
        x = F.linear(x, self.weight)
        return x
    
    def update_size(self, size):
        if size <= self.out_fea:
            print("New size needs to be bigger than current output size")
        else:
            with th.no_grad():
                self.weight = nn.Parameter(th.cat((self.weight, th.randn(size - self.out_fea, self.in_fea).to(self.device)), 0)).to(self.device)