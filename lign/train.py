
from .models import layers as ly
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from .utils import loss as ls
from .models import LIGN as lg
from torch.cuda.amp import autocast, GradScaler

def filter(data, labels, graph):
    fils = [lambda x: x == i for i in labels]

    out = graph.filter(fils, data)
    return out

def randomize(tensor):
    return tensor[th.randperm(len(tensor))]

def unsuperv(model, opt, graph, tag_in, tag_out, vec_size, lam, labels, device = (th.device('cpu'), None), epochs=100, subgraph_size = 200, clustering = None):
    pass

def superv(model, opt, graph, tag_in, tag_out, vec_size, Lambda, labels, device = (th.device('cpu'), None), epochs=100, subgraph_size = 200):
    
    labels_len = len(labels)
    temp_ly = ly.GCN(func = lg.sum_neighs_data, module_post = nn.Linear(vec_size, labels_len))
    nodes = filter(tag_out, labels, graph)
    scaler = device(1)
    amp_enable = device(1) != None

    for i in range(epochs):
        opt.zero_grad()

        nodes = randomize(nodes)
        sub = graph.subgraph(nodes[:subgraph_size])

        inp = sub.get_parent_data(tag_in).to(device(0))
        outp = sub.get_parent_data(tag_out).to(device(0))

        if amp_enable:
            with autocast():
                out1 = model(sub, inp)
                out2 = temp_ly(sub, out1)
                out2 = th.log_softmax(out2, 1)
                loss = ls.pairwaise_loss(out1, outp) * Lambda + F.nll_loss(out2, outp)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
        else:
            out1 = model(sub, inp)
            out2 = temp_ly(sub, out1)
            out2 = th.log_softmax(out2, 1)
            loss = ls.pairwaise_loss(out1, outp) * Lambda + F.nll_loss(out2, outp)

            loss.backward()
            opt.step()


