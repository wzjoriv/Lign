
from .models import layers as ly
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from .utils import loss as ls, clustering as cl
from .models import LIGN as lg

def randomize(tensor):
    return tensor[th.randperm(len(tensor))]

def norm_labels(inp, labels):
    out = (inp == labels[0]) * 0

    for i in range(1, len(labels)):
        out |= (inp == labels[i]) * i

    return out

def semi_superv(model, opt, graph, tag_in, tag_out, vec_size, labels, Lambda = 0.0001, device = (th.device('cpu'), None), epochs=100, subgraph_size = 200, cluster = (cl.NN(), 3)):
    
    labels_len = len(labels)
    temp_ly = ly.GCN(func = lg.sum_neighs_data, post_mod = nn.Linear(vec_size, labels_len))
    nodes = cl.filter(tag_out, labels, graph)
    scaler = device[1]
    amp_enable = device[1] != None

    tr_nodes, tr_labs = cl.filter_k(tag_out, labels, graph, cluster[1])
    cluster = cluster[0]

    for i in range(epochs):
        opt.zero_grad()

        ### train clustering
        sub = graph.subgraph(tr_nodes)
        inp = sub.get_parent_data(tag_in)
        cluster.train(model(sub, inp), tr_labs)


        nodes = randomize(nodes)
        sub = graph.subgraph(nodes[:subgraph_size])
        inp = sub.get_parent_data(tag_in).to(device[0])
        outp = norm_labels(cluster(inp), labels).to(device[0])

        if amp_enable:
            with th.cuda.amp.autocast():
                out1 = model(sub, inp)
                out2 = temp_ly(sub, out1)
                out2 = th.log_softmax(out2, 1)
                loss = ls.distance_loss(out1, outp) * Lambda + F.nll_loss(out2, outp)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
        else:
            out1 = model(sub, inp)
            out2 = temp_ly(sub, out1)
            out2 = th.log_softmax(out2, 1)
            loss = ls.distance_loss(out1, outp) * Lambda + F.nll_loss(out2, outp)

            loss.backward()
            opt.step()

def superv(model, opt, graph, tag_in, tag_out, vec_size, labels, Lambda = 0.0001, device = (th.device('cpu'), None), epochs=100, subgraph_size = 200):
    
    labels_len = len(labels)
    temp_ly = ly.GCN(func = lg.sum_neighs_data, post_mod = nn.Linear(vec_size, labels_len))
    nodes = cl.filter(tag_out, labels, graph)
    print(nodes)
    scaler = device[1]
    amp_enable = device[1] != None

    for i in range(epochs):
        opt.zero_grad()

        nodes = randomize(nodes)
        sub = graph.subgraph(nodes[:subgraph_size])

        inp = sub.get_parent_data(tag_in).to(device[0])
        outp = norm_labels(sub.get_parent_data(tag_out), labels).to(device[0])

        if amp_enable:
            with th.cuda.amp.autocast():
                out1 = model(sub, inp)
                out2 = temp_ly(sub, out1)
                out2 = th.log_softmax(out2, 1)
                loss = ls.distance_loss(out1, outp) * Lambda + F.nll_loss(out2, outp)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
        else:
            out1 = model(sub, inp)
            out2 = temp_ly(sub, out1)
            out2 = th.log_softmax(out2, 1)
            loss = ls.distance_loss(out1, outp) * Lambda + F.nll_loss(out2, outp)

            loss.backward()
            opt.step()


