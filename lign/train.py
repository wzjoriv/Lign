
from .model import layers as ly
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from .utils import loss as ls

def filter(data, labels, graph):
    fils = [lambda x: x == i for i in labels]

    out = graph.filter(fils, data)
    return randomize(out)

def randomize(tensor):
    return tensor[th.randperm(len(tensor))]

def unsuperv(model, opt, graph, tag_in, tag_out, vec_size, lam, labels, epochs=100, subgraph_size = 200, clustering = None):
    pass

def superv(model, opt, graph, tag_in, tag_out, vec_size, lam, labels, epochs=100, subgraph_size = 200):
    
    labels_len = len(labels)
    temp_ly = ly.GCN(module_post = nn.Linear(vec_size, labels_len))
    nodes = filter(tag_out, labels, graph)

    for i in range(epochs):
        sub = graph.subgraph(nodes[:subgraph_size])

        inp = sub.get_parent_data(tag_in)
        outp = sub.get_parent_data(tag_out)

        out1 = model(sub, inp)
        out2 = temp_ly(sub, out1)
        out2 = th.log_softmax(out2, 1)

        loss = ls.pairwaise_loss(out1, outp) * lam + F.nll_loss(out2, outp)

        loss.backward()


