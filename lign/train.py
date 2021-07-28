import torch as th
import torch.nn as nn
import torch.nn.functional as F

from lign.utils import functions as fn
from lign.utils.clustering import KMeans, KNN

def growing_exemplar(
            models, graph, labels, opt, 
            tags = ('x', 'label'), examplar_n = 20, device = (th.device('cpu'), None), 
            lossF = nn.CrossEntropyLoss(), epochs=200, sub_graph_size = 128
        ):

    tag_in, tag_out = tags

    tr_nodes, _ = fn.filter_k_from_tags(tag_out, labels, graph, examplar_n)

    sub = graph.sub_graph(tr_nodes, get_data=True)

    superv(models, sub, labels, opt, 
            tags = (tag_in, tag_out), device = device, lossF = lossF, epochs = epochs, sub_graph_size = sub_graph_size)

def fixed_exemplar(
            models, graph, labels, opt, 
            tags = ('x', 'label'), examplar_n = 2000, device = (th.device('cpu'), None), 
            lossF = nn.CrossEntropyLoss(), epochs=200, sub_graph_size = 128
        ):

    growing_exemplar(models, graph, labels, opt, 
            tags = tags, examplar_n=int(examplar_n/len(labels)), device = device, lossF=lossF, epochs=epochs, sub_graph_size = sub_graph_size)

def unsuperv(
            models, graph, labels, opt, 
            tags = ('x', 'label'), cluster = KMeans(), device = (th.device('cpu'), None), 
            lossF = nn.CrossEntropyLoss(), epochs=1000, sub_graph_size = 200
        ):

    tag_in, tag_out = tags

    nodes = fn.filter_tags(tag_out, labels, graph)

    dt = graph.get_data(tag_in)

    cluster.k = len(labels)
    cluster.train(dt[nodes])

    data = cluster(dt)
    graph.set_data('_p_label_', data)

    superv(models, graph, labels, opt, 
            tags = (tag_in, '_p_label_'), device = device, lossF = lossF, epochs = epochs, sub_graph_size = sub_graph_size)

    graph.pop_data('_p_label_')

def semi_superv(
            models, graph, labels, opt, 
            tags = ('x', 'label'), k = 5, cluster = KNN(), device = (th.device('cpu'), None), 
            lossF = nn.CrossEntropyLoss(), epochs=1000, sub_graph_size = 200
        ):

    tag_in, tag_out = tags

    tr_nodes, tr_labs = fn.filter_k_from_tags(tag_out, labels, graph, k)

    dt = graph.get_data(tag_in)

    cluster.train(dt[tr_nodes], tr_labs)

    data = cluster(dt)
    graph.set_data('_p_label_', data)

    superv(models, graph, labels, opt, 
            tags = (tag_in, '_p_label_'), device = device, lossF = lossF, epochs = epochs, sub_graph_size = sub_graph_size)

    graph.pop_data('_p_label_')

def superv(
            models, graph, labels, opt, 
            tags = ('x', 'label'), device = (th.device('cpu'), None), 
            lossF = nn.CrossEntropyLoss(), epochs=1000, sub_graph_size = 200, kipf_approach=False
        ):

    if kipf_approach:
        full_graph = graph[0]
        train_graph = graph[1]
    else:
        full_graph = graph
        train_graph = graph
    
    base, classifier = models
    tag_in, tag_out = tags

    scaler = device[1]
    amp_enable = device[1] != None

    is_base_gcn = fn.has_gcn(base)
    is_classifier_gcn = fn.has_gcn(classifier)

    nodes = fn.filter_tags(tag_out, labels, train_graph)
    
    nodes_len = len(nodes)

    # training
    base.train()
    classifier.train()
    for _ in range(epochs):

        opt.zero_grad()

        nodes = fn.randomize_tensor(nodes)
        for batch in range(0, nodes_len, sub_graph_size):
            with th.no_grad():
                b_nodes = nodes[batch:min(nodes_len, batch + sub_graph_size)]
                sub = train_graph.sub_graph(b_nodes)

                if kipf_approach:
                    b_nodes = train_graph.child_to_parent_index(b_nodes)

                inp = full_graph.get_data(tag_in).to(device[0]) if is_base_gcn else sub.get_parent_data(tag_in).to(device[0])
                outp = fn.onehot_encode(sub.get_parent_data(tag_out), labels).to(device[0])

            opt.zero_grad()

            if amp_enable:
                with th.cuda.amp.autocast():
                    out = base(full_graph, inp) if is_base_gcn else base(inp)
                    if is_base_gcn:
                        out = classifier(full_graph, out)[b_nodes] if is_classifier_gcn else classifier(out[b_nodes])
                    else:
                        out = classifier(sub, out) if is_classifier_gcn else classifier(out)
                    loss = lossF(out, outp)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
            else:
                out = base(full_graph, inp) if is_base_gcn else base(inp)
                if is_base_gcn:
                    out = classifier(full_graph, out)[b_nodes] if is_classifier_gcn else classifier(out[b_nodes])
                else:
                    out = classifier(sub, out) if is_classifier_gcn else classifier(out)
                loss = lossF(out, outp)

                loss.backward()
                opt.step()