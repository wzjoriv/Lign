import torch
from torchvision import datasets

import pandas as pd
import os

class DatasetNotFound(Exception):

    def __init__(self, dataset, location):
        super().__init__("Dataset(" + dataset + ") not found at location: " + location)

def onehot_encoding(data, labels):
    labels = (data) * 0 # onehot encoding
    for lab in range(1, len(labels)):
        labels |= (data == labels[lab]) * lab

    return labels

def mnist_to_lign(path, transforms = None, train = True):
    from lign.graph import GraphDataset

    dataset =  datasets.MNIST(path, train=train)
    graph = GraphDataset()

    digits = []
    labels = []
    graph.add(len(dataset)) # add n nodes
    for img, lab in dataset:
        """out = {
                "data": {},
                "edges": set()
            }"""

        if transforms:
            img = transforms(img)

        digits.append(img)
        labels.append(lab)
        #graph.add(out)
    
    if(torch.is_tensor(digits[0])):
        digits = torch.stack(digits)
        labels = torch.LongTensor(labels)

    graph.set_data('x', digits)
    graph.set_data('labels', labels)

    return graph

def cifar_to_lign(path, transforms = None, train = True):
    from lign.graph import GraphDataset

    dataset =  datasets.CIFAR100(path, train=train)
    graph = GraphDataset()

    imgs = []
    labels = []
    
    graph.add(len(dataset)) # add n nodes
    for img, lab in dataset:
        """out = {
                "data": {},
                "edges": set()
            }"""

        if transforms:
            img = transforms(img)

        imgs.append(img)
        labels.append(lab)
        #graph.add(out)
    
    if(torch.is_tensor(imgs[0])):
        imgs = torch.stack(imgs)
        labels = torch.LongTensor(labels)

    graph.set_data('x', imgs)
    graph.set_data('labels', labels)

    return graph

def cora_to_lign(path, train = True, split = 0.8):
    from lign.graph import GraphDataset
    graph = GraphDataset()

    try:
        cora_cont =  pd.read_csv(os.path.join(path, "cora.content"), sep="\t", header=None)
        cora_cite =  pd.read_csv(os.path.join(path, "cora.cites"), sep="\t", header=None)
    except:
        raise DatasetNotFound("CORA", path)
    
    
    n = len(cora_cont[0])
    graph.add(n) # add n empty nodes

    marker = [1, 1433] # where data is seperated in the csv
    unq_labels = cora_cont[marker[0] + 1].unique()

    labels = onehot_encoding(cora_cont[marker[0] + 1], unq_labels) # onehot encoding

    graph.set_data("id", torch.tensor(cora_cont[0].values))
    graph.set_data("x", torch.tensor(cora_cont.loc[:, marker[0]:marker[1]].values))
    graph.set_data("labels", torch.LongTensor(labels.values))

    edge_parents = cora_cite.groupby(0)
    parents = edge_parents.groups.keys()

    for key in parents:
        p_node = graph.filter(lambda x: x == key, "id")

        childrens = edge_parents.get_group(key)[1].values
        c_nodes = list(cora_cont.loc[cora_cont[0].isin(childrens)].index.values)
        graph.add_edge(p_node, c_nodes)

    n = len(cora_cont[0])
    split = int(n * split)
    subnodes_train = list(range(split))  # training nodes
    subnodes_test = list(range(split, n)) # testing nodes

    return graph, graph.subgraph(nodes=subnodes_train), graph.subgraph(nodes=subnodes_test)
