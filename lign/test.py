
import torch as th

from lign.utils import io, clustering as cl, functions as fn

def validate(model, graph, train, tag_in, tag_out, labels, metrics = ['accuracy'], sv_img = None, cluster = (cl.NN(), 3), device = th.device('cpu'), save_img = False):

    model.eval()
    with th.no_grad():
        tr_nodes, tr_labs = fn.filter_k_from_tags(tag_out, labels, train, cluster[1])
        sub = train.sub_graph(tr_nodes)
        inp = sub.get_parent_data(tag_in).to(device)
        
        cluster = cluster[0]
        cluster.train(model(sub, inp), tr_labs.to(device))

        ts_nodes = fn.filter_tags(tag_out, labels, graph)
        graph = graph.sub_graph(ts_nodes)

        inp = graph.get_parent_data(tag_in).to(device)
        outp_t = graph.get_parent_data(tag_out).to(device)

        rep_vec = model(graph, inp, save=save_img)
        outp_p = cluster(rep_vec)

    out = {}
    metrics = io.to_iter(metrics)
    for metric in metrics:
        if metric == 'accuracy':
            out[metric] = (outp_p == outp_t).sum().item() * 100.0/outp_p.size(0)
    
    return out

def accuracy(model, graph, train, tag_in, tag_out, labels, cluster = (cl.NN(), 3), sv_img = None, device = th.device('cpu'), save_img = False):

    out = validate(model, graph, train, tag_in, tag_out, labels, metrics = 'accuracy', cluster = cluster, sv_img=sv_img, device = device, save_img = save_img)

    return out['accuracy']