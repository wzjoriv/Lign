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
        g.set_data(".hidden", data)

        g.apply(self.discovery, ".hidden")

        if self.agregation:
            g.push(agregation = self.agregation, data = ".hidden")
        
        if self.inclusion:
            g.apply(self.inclusion, ".hidden")

        return g.pop_data(".hidden")

# dynamic Linear Layer
class DyLinear(nn.Module):
    def __init__(self, in_fea, out_fea, base = None, device = 'cuda'):
        super(DyLinear, self).__init__()
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
    
    def update_size(self, size): #slow; doesn't matter much since perform infrequenly
        if size <= self.out_fea:
            raise RuntimeWarning("New size needs to be bigger than current output size")
        else:
            with th.no_grad():
                self.weight = nn.Parameter(th.cat((self.weight, th.randn(size - self.out_fea, self.in_fea).to(self.device)), 0)).to(self.device)