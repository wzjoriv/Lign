
import torch as th

from lign.utils import io, clustering as cl, functions as fn

def validate(model, graph, train, tag_in, tag_out, labels, metrics = ['accuracy'], cluster = (cl.NN(), 3), device = th.device('cpu')):

    model.eval()
    with th.no_grad():
        tr_nodes, tr_labs = fn.filter_k_from_tags(tag_out, labels, train, cluster[1])
        sub = train.sub_graph(tr_nodes)
        inp = sub.get_parent_data(tag_in).to(device)
        
        cluster = cluster[0]
        tr_vec = model(sub, inp) if fn.has_gcn(model) else model(inp)
        cluster.train(tr_vec, tr_labs.to(device))

        ts_nodes = fn.filter_tags(tag_out, labels, graph)
        graph = graph.sub_graph(ts_nodes)

        inp = graph.get_parent_data(tag_in).to(device)
        outp_t = graph.get_parent_data(tag_out).to(device)

        rep_vec = model(sub, inp) if fn.has_gcn(model) else model(inp)
        outp_p = cluster(rep_vec)

    out = {}
    metrics = io.to_iter(metrics)
    for metric in metrics:
        if metric == 'accuracy':
            out[metric] = (outp_p == outp_t).sum().item() * 100.0/outp_p.size(0)
    
    return out

def accuracy(model, graph, train, tag_in, tag_out, labels, cluster = (cl.NN(), 3), device = th.device('cpu')):

    out = validate(model, graph, train, tag_in, tag_out, labels, metrics = 'accuracy', cluster = cluster, device = device)

    return out['accuracy']