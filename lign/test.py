
import torch as th

from lign.utils import io, clustering as cl, functions as fn

def validate(
            model, graphs, labels, 
            tags = ('x', 'label'), metrics = ['accuracy'], cluster = (cl.NN(), 3), 
            sub_graph_size = 200, device = th.device('cpu'), kipf_approach=False
        ):

    if kipf_approach:
        full_graph, train_graph, test_graph = graphs
    else:
        train_graph, test_graph = graphs
    
    tag_in, tag_out = tags

    model.eval()
    is_gcn = fn.has_gcn(model)
    with th.no_grad():

        # Get sample to compare

        tr_nodes, tr_labs = fn.filter_k_from_tags(tag_out, labels, train_graph, cluster[1])
        tr_nodes = tr_nodes.tolist()
        
        if kipf_approach:
            tr_nodes = train_graph.child_to_parent_index(tr_nodes)
        else:
            full_graph = train_graph

        inp = full_graph.get_data(tag_in).to(device) if is_gcn else full_graph.get_data(tag_in, nodes=tr_nodes).to(device)
        
        cluster_m = cluster[0]
        
        tr_vec = model(full_graph, inp)[tr_nodes] if is_gcn else model(inp)
        cluster_m.train(tr_vec, tr_labs.to(device))

        # Infer testing nodes
        ts_nodes = fn.filter_tags(tag_out, labels, test_graph)

        if kipf_approach:
            ts_nodes = test_graph.child_to_parent_index(ts_nodes)
        else:
            full_graph = test_graph

        inp = full_graph.get_data(tag_in).to(device) if is_gcn else full_graph.get_data(tag_in, nodes=ts_nodes).to(device)
        ts_labs = full_graph.get_data(tag_out, nodes=ts_nodes).to(device)

        rep_vec = model(full_graph, inp)[ts_nodes] if is_gcn else model(inp)
        outp_p = th.zeros(rep_vec.size(0), dtype=rep_vec.dtype, device=rep_vec.device)

        for i in range(0, len(rep_vec), sub_graph_size):
            outp_p[i:min(len(rep_vec), i + sub_graph_size)] = cluster_m(rep_vec[i:min(len(rep_vec), i + sub_graph_size)])

    out = {}
    metrics = io.to_iter(metrics)
    for metric in metrics:
        if str(metric) == 'accuracy':
            out[metric] = (outp_p == ts_labs).sum().item() * 100.0/outp_p.size(0)
        else:
            out[metric.__name__] = metric(outp_p, ts_labs)
    
    return out

def accuracy(
            model, graphs, labels, 
            tags = ('x', 'label'), cluster = (cl.NN(), 3), 
            sub_graph_size = 200, device = th.device('cpu'), kipf_approach=False
        ):

    out = validate(model, graphs, labels, tags,
            metrics = 'accuracy', cluster = cluster, sub_graph_size = sub_graph_size,
            device = device, kipf_approach=kipf_approach)

    return out['accuracy']