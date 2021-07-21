import torch as th
from lign.nn import GCN

def is_gcn(module):
    return issubclass(module.__class__, GCN)

def has_gcn(network):
    for module in network.modules():
        if is_gcn(module):
            return True
    
    return False

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = th.pow(x - y, p).sum(2)
    
    return dist

def get_equals_filter(i):
    return lambda x: x == i

def filter_tags(data, tags, graph):
    fils = [get_equals_filter(i) for i in tags]

    out = graph.filter(fils, data)
    return out

def filter_k_from_tags(data, tags, graph, k = 3):
    out = []
    labs = []

    for tag in tags:
        labs.extend([tag] * k)
        out.extend(graph.filter(get_equals_filter(tag), data)[:k])

    return th.LongTensor(out), th.LongTensor(labs)

def randomize_tensor(tensor):
    return tensor[th.randperm(len(tensor))]

def same_label(y): # return matrix of nodes that have same label
    s = y.size(0)
    y_expand = y.unsqueeze(0).expand(s, s)
    Y = y_expand.eq(y_expand.t())
    return Y

def onehot_encode(data, labels): # onehot encoding
    final = (data == labels[0]) * 0
    for lab in range(1, len(labels)):
        final |= (data == labels[lab]) * lab
    
    return final

def sum_neighs_data(neighs):
    out = neighs[0]
    for neigh in neighs[1:]:
        out = out + neigh
    return out
