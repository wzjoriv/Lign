import torch as th
import torch.nn as nn


#gcn layer in network
class GCN(nn.Module):
    def __init__(self, layer, data, ):
        super(GCN, self).__init__()
        self.apply_layer = layer

    def forward(self, g, data, func):
        g.pull(func = func)
        return g.ndata.pop('h')

### modified Linear layer for graphs
class G_Linear(nn.Module):
    def __init__(self, in_feats, out_feats, activation, data=h):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.data = data

    def forward(self, node):
        h = self.linear(node.data[self.data])
        if self.activation is not None:
            h = self.activation(h)
        return {self.data: h}