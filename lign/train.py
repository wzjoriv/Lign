from math import floor

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from lign import layers as ly
from lign.models import LIGN as lg
from lign.utils import clustering as cl
from lign.utils import loss as ls


def randomize(tensor):
    return tensor[th.randperm(len(tensor))]

def norm_labels(inp, labels):
    out = (inp == labels[0]) * 0

    for i in range(1, len(labels)):
        out |= (inp == labels[i]) * i

    return out

""" def semi_superv(model, opt, graph, tag_in, tag_out, vec_size, labels, Lambda = 0.0001, device = (th.device('cpu'), None), lossF = nn.CrossEntropyLoss(), epochs=100, addon = None, subgraph_size = 200, cluster = (cl.NN(), 3)):
    
    labels_len = len(labels)
    scaler = device[1]
    amp_enable = device[1] != None

    if addon == None:
        temp_ly = ly.GCN(func = lg.sum_neighs_data, post_mod = nn.Linear(vec_size, labels_len)).to(device[0])
    else:
        temp_ly = addon(vec_size, labels_len).to(device[0])

    opt2 = th.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    with th.no_grad():
        nodes = cl.filter(tag_out, labels, graph)
        tr_nodes, tr_labs = cl.filter_k(tag_out, labels, graph, cluster[1])

    cluster = cluster[0]
    nodes_len = len(nodes)

    model.train()
    for i in range(epochs):
        ### train clustering
        with th.no_grad():
            sub = graph.subgraph(tr_nodes)
            inp = sub.get_parent_data(tag_in).to(device[0])
            cluster.train(model(sub, inp), tr_labs.to(device[0]))

            nodes = randomize(nodes)

        for batch in range(0, nodes_len, subgraph_size):
            with th.no_grad():
                sub = graph.subgraph(nodes[batch:min(nodes_len, batch + subgraph_size)])
                inp = sub.get_parent_data(tag_in).to(device[0])
                outp = norm_labels(cluster(model(sub, inp)), labels).to(device[0])

            opt.zero_grad()
            opt2.zero_grad()
            
            if amp_enable:
                with th.cuda.amp.autocast():
                    out1 = model(sub, inp)
                    out2 = temp_ly(sub, out1)
                    loss = ls.distance_loss(out1, outp) * Lambda + lossF(out2, outp)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.step(opt2)
                scaler.update()
                
            else:
                out1 = model(sub, inp)
                out2 = temp_ly(sub, out1)
                loss = ls.distance_loss(out1, outp) * Lambda + lossF(out2, outp)

                loss.backward()
                opt.step()
                opt2.step() """

def superv(model, opt, graph, tag_in, tag_out, vec_size, labels, Lambda = 0.0001, device = (th.device('cpu'), None), lossF = nn.CrossEntropyLoss(), epochs=100, addon = None, subgraph_size = 200):
    
    labels_len = len(labels)
    scaler = device[1]
    amp_enable = device[1] != None

    if addon == None:
        temp_ly = ly.GCN(func = lg.sum_neighs_data, post_mod = nn.Linear(vec_size, labels_len)).to(device[0])
    else:
        temp_ly = addon(vec_size, labels_len).to(device[0])

    opt2 = th.optim.Adam(temp_ly.parameters())

    with th.no_grad():
        nodes = cl.filter(tag_out, labels, graph)
    
    
    nodes_len = len(nodes)

    model.train()
    for i in range(epochs):

        nodes = randomize(nodes)
        for batch in range(0, nodes_len, subgraph_size):
            with th.no_grad():
                sub = graph.subgraph(nodes[batch:min(nodes_len, batch + subgraph_size)])

                inp = sub.get_parent_data(tag_in).to(device[0])
                outp = norm_labels(sub.get_parent_data(tag_out), labels).to(device[0])

            opt.zero_grad()
            opt2.zero_grad()

            if amp_enable:
                with th.cuda.amp.autocast():
                    out1 = model(sub, inp)
                    out2 = temp_ly(sub, out1)
                    loss = ls.distance_loss(out1, outp) * Lambda + lossF(out2, outp)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.step(opt2)
                scaler.update()
                
            else:
                out1 = model(sub, inp)
                out2 = temp_ly(sub, out1)
                loss = ls.distance_loss(out1, outp) * Lambda - lossF(out2, outp)

                loss.backward()
                opt.step()
                opt2.step()


