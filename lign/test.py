from .train import norm_labels
from .utils import clustering as cl
import torch as th
from .utils import io

def validate(model, graph, train, tag_in, tag_out, labels, metrics = ['accuracy'], cluster = (cl.NN(), 3), device = th.device('cpu')):
    tr_nodes, tr_labs = cl.filter_k(tag_out, labels, train, cluster[1])
    sub = train.subgraph(tr_nodes)
    inp = sub.get_parent_data(tag_in).to(device)
    
    cluster = cluster[0]
    cluster.train(model(sub, inp), tr_labs.to(device))

    ts_nodes = cl.filter(tag_out, labels, graph)
    graph = graph.subgraph(ts_nodes)

    inp = graph.get_parent_data(tag_in).to(device)
    outp_t = graph.get_parent_data(tag_out).to(device)

    rep_vec = model(graph, inp)
    outp_p = cluster(rep_vec)

    out = {}
    metrics = io.to_iter(metrics)
    for metric in metrics:
        if metric == 'accuracy':
            print(outp_p.eq(outp_t).sum())
            out[metric] = outp_p.eq(outp_t).sum().item() * 100.0/outp_p.size(0)

    return out

def accuracy(model, graph, train, tag_in, tag_out, labels, cluster = (cl.NN(), 3), device = th.device('cpu')):

    out = validate(model, graph, train, tag_in, tag_out, labels, metrics = 'accuracy', cluster = cluster, device = device)

    return out['accuracy']