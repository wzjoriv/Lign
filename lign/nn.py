import torch as th
import torch.nn as nn
import torch.nn.functional as F

# gcn layer in network
class GCN(nn.Module):
    def __init__(self, discovery, aggregation = None, inclusion = None):
        super(GCN, self).__init__()
        self.aggregation = aggregation
        self.discovery = discovery
        self.inclusion = inclusion

    def forward(self, g, data):
        g.set_data("__hidden__", data)

        g.apply(self.discovery, "__hidden__")

        if self.aggregation:
            g.push(func = self.aggregation, data = "__hidden__")
        
        if self.inclusion:
            g.apply(self.inclusion, "__hidden__")

        return g.pop_data("__hidden__")

# dynamic Linear Layer
class DyLinear(nn.Module):
    def __init__(self, in_fea, out_fea, device = 'cuda'):
        super(DyLinear, self).__init__()
        self.device = device
        self.in_fea = in_fea
        self.out_fea = out_fea
        self.weight = nn.Parameter(th.randn(out_fea, in_fea)).to(self.device)
    
    def forward(self, x):
        x = F.linear(x, self.weight)
        return x

    def update_size(self, size): #slow; doesn't matter much since perform infrequenly
        if size <= self.out_fea:
            raise ValueError(f"New size ({size}) needs to be bigger than current output size ({self.out_fea})")
        else:
            new_weight = th.cat(
                                (self.weight, 
                                th.randn(size - self.out_fea, self.in_fea).to(self.device)
                                ), 0)
            self.weight = nn.Parameter(new_weight).to(self.device)
            self.out_fea = size