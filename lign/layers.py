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
        g.set_data("_hidden_", data)

        g.apply(self.discovery, "_hidden_")

        if self.aggregation:
            g.push(aggregation = self.aggregation, data = "_hidden_")
        
        if self.inclusion:
            g.apply(self.inclusion, "._hidden_")

        return g.pop_data("_hidden_")

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
            raise RuntimeError("New size needs to be bigger than current output size")
        else:
            with th.no_grad():
                self.out_fea = size
                self.weight = nn.Parameter(th.cat((self.weight, th.randn(size - self.out_fea, self.in_fea).to(self.device)), 0)).to(self.device)