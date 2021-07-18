import torch
from torchvision import datasets

import pandas as pd
import os

from lign.utils.functions import onehot_encode

def mnist_to_lign(path, transforms = None, split = 0.0):
    from lign.graph import Graph

    try:
        d_train = datasets.MNIST(path, train=True)
        d_test = datasets.MNIST(path, train=False)
        len_train = len(d_train)
        len_test = len(d_test)

        dataset =  d_train + d_test
    except:
        raise FileNotFoundError(f"Error loading MNIST from location: {path}")

    if (type(split) == int):
        split = float(split) / (len_train + len_test)
    
    if (split >= 1.0 or split <= 0.0):
        split = float(len_train) / (len_train + len_test)

    graph = Graph()

    graph.add(len(dataset)) # add n_{train} and n_{validate} nodes

    digits = []
    labels = []
    for img, lab in dataset:
        digits.append(img)
        labels.append(lab)
    
    digits = torch.stack(digits)
    if transforms:
        digits = transforms(digits)
    labels = torch.LongTensor(labels)

    graph.set_data('x', digits)
    graph.set_data('labels', labels)
    
    n = len(graph)
    split = int(n * split)
    subnodes_train = list(range(split))  # training nodes
    subnodes_test = list(range(split, n)) # testing nodes

    graph_train = graph.subgraph(nodes=subnodes_train, get_data=True, get_edges=True)
    graph_test = graph.subgraph(nodes=subnodes_test, get_data=True, get_edges=True)

    return graph, graph_train, graph_test

def cifar_to_lign(path, transforms = None, split = 0.0):
    from lign.graph import Graph

    try:

        d_train = datasets.CIFAR100(path, train=True)
        d_test = datasets.CIFAR100(path, train=False)
        len_train = len(d_train)
        len_test = len(d_test)

        dataset =  d_train + d_test
    except:
        raise FileNotFoundError(f"Error loading CIFAR from location: {path}")

    if (type(split) == int):
        split = float(split) / (len_train + len_test)
    
    if (split >= 1.0 or split <= 0.0):
        split = float(len_train) / (len_train + len_test)

    graph = Graph()
    
    graph.add(len(dataset))
    
    imgs = []
    labels = []
    for img, lab in dataset:
        imgs.append(img)
        labels.append(lab)
    
    imgs = torch.stack(imgs)
    if transforms:
        imgs = transforms(imgs)
    labels = torch.LongTensor(labels)

    n = len(graph)
    split = int(n * split)
    subnodes_train = list(range(split))  # training nodes
    subnodes_test = list(range(split, n)) # testing nodes

    graph_train = graph.subgraph(nodes=subnodes_train, get_data=True, get_edges=True)
    graph_test = graph.subgraph(nodes=subnodes_test, get_data=True, get_edges=True)

    return graph, graph_train, graph_test

def cora_to_lign(path, split = 0.0):
    from lign.graph import Graph
    graph = Graph()

    if (type(split) == int or split >= 1.0 or split <= 0.0):
        split = 0.8

    try:
        cora_cont =  pd.read_csv(os.path.join(path, "cora.content"), sep="\t", header=None)
        cora_cite =  pd.read_csv(os.path.join(path, "cora.cites"), sep="\t", header=None)
    except:
        raise FileNotFoundError(f"Error loading CORA from location: {path}")
    
    
    n = len(cora_cont[0])
    graph.add(n) # add n empty nodes

    marker = [1, 1433] # where data is seperated in the csv
    unq_labels = list(cora_cont[marker[1] + 1].unique())

    labels = onehot_encode(cora_cont[marker[1] + 1].values, unq_labels) # onehot encoding

    graph.set_data("id", torch.tensor(cora_cont[0].values))
    graph.set_data("x", torch.tensor(cora_cont.loc[:, marker[0]:marker[1]].values))
    graph.set_data("labels", torch.LongTensor(labels))

    edge_parents = cora_cite.groupby(0)
    parents = edge_parents.groups.keys()

    for key in parents:
        p_node = graph.filter(lambda x: x == key, "id")[0]

        childrens = edge_parents.get_group(key)[1].values
        c_nodes = list(cora_cont.loc[cora_cont[0].isin(childrens)].index.values)
        graph.add_edge(p_node, c_nodes)

    n = len(cora_cont[0])
    split = int(n * split)
    subnodes_train = list(range(split))  # training nodes
    subnodes_test = list(range(split, n)) # testing nodes

    graph_train = graph.subgraph(nodes=subnodes_train, get_data=True, get_edges=True)
    graph_test = graph.subgraph(nodes=subnodes_test, get_data=True, get_edges=True)

    return graph, graph_train, graph_test


def dataset_to_lign(format, **locations):
    """
    formats cheat sheet:
        (format[, folder/file1, folder/file2])                  ## size of data type in format must be the same as the number of directories/files

        syntax:
            - = addition entries in the data field
            (NAME) = give data the name NAME in the data field
            [##] = optional
                csv: [column1, column2, 3, [0_9]]               ##  Indicate index or column name to retrieve; multiple columns are merges as one

        data type:
            imgs = images folder
            tensor
            csv = csv file
            imgs_url = file of list of images url

        example:
            format = "(imgs('x')['data_[0_9]*.png'], csv('label')['file.csv/column2'])"
            'data/', 'labels.txt'
    """
    raise NotImplementedError("Not implemented. May come in a future update")