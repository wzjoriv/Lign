import torch as th
import torch.nn as nn
import torch.nn.functional as F

from lign.utils import functions as fn
from lign.utils.clustering import KMeans, KNN

def unsuperv(
            models, graph, labels, opt, 
            tags = ('x', 'label'), cluster = KMeans(), device = (th.device('cpu'), None), 
            lossF = nn.CrossEntropyLoss(), epochs=1000, subgraph_size = 200
        ):

    tag_in, tag_out = tags

    nodes = fn.filter_tags(tag_out, labels, graph)

    dt = graph.get_data(tag_in)

    cluster.k = len(labels)
    cluster.train(dt[nodes])

    data = cluster(dt)
    graph.set_data('_p_label_', data)

    superv(models, graph, labels, opt, 
            tags = (tag_in, '_p_label_'), device = device, lossF = lossF, epochs = epochs, subgraph_size = subgraph_size)

    graph.pop_data('_p_label_')

def semi_superv(
            models, graph, labels, opt, 
            tags = ('x', 'label'), k = 5, cluster = KNN(), device = (th.device('cpu'), None), 
            lossF = nn.CrossEntropyLoss(), epochs=1000, subgraph_size = 200
        ):

    tag_in, tag_out = tags

    tr_nodes, tr_labs = fn.filter_k_from_tags(tag_out, labels, graph, k)

    dt = graph.get_data(tag_in)

    cluster.train(dt[tr_nodes], tr_labs)

    data = cluster(dt)
    graph.set_data('_p_label_', data)

    superv(models, graph, labels, opt, 
            tags = (tag_in, '_p_label_'), device = device, lossF = lossF, epochs = epochs, subgraph_size = subgraph_size)

    graph.pop_data('_p_label_')

def superv(
            models, graph, labels, opt, 
            tags = ('x', 'label'), device = (th.device('cpu'), None), 
            lossF = nn.CrossEntropyLoss(), epochs=1000, subgraph_size = 200
        ):
    
    base, classifier = models
    tag_in, tag_out = tags

    scaler = device[1]
    amp_enable = device[1] != None

    is_base_gcn = fn.has_gcn(base)
    is_classifier_gcn = fn.has_gcn(classifier)

    nodes = fn.filter_tags(tag_out, labels, graph)
    
    nodes_len = len(nodes)

    # training
    base.train()
    classifier.train()
    for _ in range(epochs):

        opt.zero_grad()

        nodes = fn.randomize_tensor(nodes)
        for batch in range(0, nodes_len, subgraph_size):
            with th.no_grad():
                b_nodes = nodes[batch:min(nodes_len, batch + subgraph_size)]
                sub = graph.subgraph(b_nodes)

                inp = graph.get_data(tag_in).to(device[0]) if is_base_gcn else sub.get_parent_data(tag_in).to(device[0])
                outp = fn.onehot_encode(sub.get_parent_data(tag_out), labels).to(device[0])

            opt.zero_grad()

            if amp_enable:
                with th.cuda.amp.autocast():
                    out = base(graph, inp) if is_base_gcn else base(inp)
                    if is_base_gcn:
                        out = classifier(graph, out)[b_nodes] if is_classifier_gcn else classifier(out[b_nodes])
                    else:
                        out = classifier(out)
                    loss = lossF(out, outp)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
            else:
                out = base(graph, inp) if is_base_gcn else base(inp)
                if is_base_gcn:
                    out = classifier(graph, out)[b_nodes] if is_classifier_gcn else classifier(out[b_nodes])
                else:
                    out = classifier(out)
                loss = lossF(out, outp)

                loss.backward()
                opt.step()