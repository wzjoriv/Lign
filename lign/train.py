import torch as th
import torch.nn as nn
import torch.nn.functional as F

from lign.utils import functions as fn

def semi_superv(
            models, graphs, labels, opt, 
            tags = ('x', 'label'), device = (th.device('cpu'), None), lossF = nn.CrossEntropyLoss(), epochs=1000, subgraph_size = 200
        ):
    
    pass

def unsuperv(
            models, graphs, labels, opt, 
            tags = ('x', 'label'), device = (th.device('cpu'), None), lossF = nn.CrossEntropyLoss(), epochs=1000, subgraph_size = 200
        ):
    
    pass

def superv(
            models, graphs, labels, opt, 
            tags = ('x', 'label'), device = (th.device('cpu'), None), lossF = nn.CrossEntropyLoss(), epochs=1000, subgraph_size = 200
        ):
    
    base, classifier = models
    graph, t_graph = graphs
    tag_in, tag_out = tags

    scaler = device[1]
    amp_enable = device[1] != None

    is_base_gcn = fn.has_gcn(base)
    is_classifier_gcn = fn.has_gcn(classifier)

    with th.no_grad(): # get nodes that are part of the current label subset
        nodes = fn.filter_tags(tag_out, labels, t_graph)
    
    nodes_len = len(nodes)

    # training
    base.train()
    classifier.train()
    for i in range(epochs):

        opt.zero_grad()

        nodes = fn.randomize_tensor(nodes)
        for batch in range(0, nodes_len, subgraph_size):
            with th.no_grad():
                sub = graph.subgraph(nodes[batch:min(nodes_len, batch + subgraph_size)])

                inp = sub.get_parent_data(tag_in).to(device[0])
                outp = fn.onehot_encoding(sub.get_parent_data(tag_out), labels).to(device[0])

            opt.zero_grad()

            if amp_enable:
                with th.cuda.amp.autocast():
                    out = base(sub, inp) if is_base_gcn else base(inp)
                    out = classifier(sub, out) if is_classifier_gcn else classifier(out)
                    loss = lossF(out, outp)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
            else:
                out = base(sub, inp) if is_base_gcn else base(inp)
                out = classifier(sub, out) if is_classifier_gcn else classifier(out)
                loss = lossF(out, outp)

                loss.backward()
                opt.step()