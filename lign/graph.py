import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os.path
from .utils import io

"""
    node = {
        "data": {},
        "edges": set()
    }

    dataset = {
        "count": 0,
        "data": {},
        "edges": [],
        "__temp__": []
    }
    
"""

class GraphDataset(Dataset):
    def __init__(self, fl="", workers=1):
        self.dataset = None
        self.workers = workers

        self.__files__ = {}

        if not len(fl):
            fl = os.path.join(os.path.dirname(__file__),
                              "utils", "defaults", "graph.lign")
            self.__files__["file"] = os.path.join("data", "graph.lign")
            self.__files__["folder"] = os.path.dirname(self.__files__["file"])
        else:
            self.__files__["file"] = fl
            self.__files__["folder"] = os.path.dirname(self.__files__["file"])

        self.dataset = io.unpickle(fl)

        if "count" not in self.dataset and "data" not in self.dataset and \
                "edges" not in self.dataset and "__temp__" not in self.dataset:
            raise FileNotFoundError

    def __len__(self):
        return self.dataset["count"]

    def __getitem__(self, indx):
        node = {
            "data": {}
        }

        for key in self.dataset["data"].keys():
            node["data"][key] = self.dataset["data"][key][indx]

        node["edges"] = self.dataset["edges"][indx]

        return node

    def get_data(self, data, nodes=[]):
        ls = io.to_iter(nodes)
        if not len(ls):
            return self.dataset["data"][data]
        else:
            if torch.is_tensor(self.dataset["data"][data]):
                return self.dataset["data"][data][nodes]
            else:
                return [self.dataset["data"][data][nd] for nd in ls]

    def set_data(self, data, features, nodes=[]):
        ls = io.to_iter(nodes)
        if not len(ls):
            self.dataset["data"][data] = features
        else:
            if torch.is_tensor(self.dataset["data"][data]):
                self.dataset["data"][data][nodes] = features
            else:
                for indx, nd in enumerate(ls):
                    self.dataset["data"][data][nd] = features[indx]

    def add_edge(self, node, edges):
        edges = io.to_iter(edges)
        self.dataset["edges"][node].update(edges)

    def get_edge(self, node):
        return self.dataset["edges"][node]

    def remove_edge(self, node, edges):
        edges = io.to_iter(edges)
        self.dataset["edges"][node].difference_update(edges)

    def __add_data__(self, data, obj):
        if torch.is_tensor(obj):
            if data not in self.dataset["data"]:
                self.dataset["data"][data] = obj.unsqueeze(0)
            else:
                self.dataset["data"][data] = torch.cat(
                    (self.dataset["data"][data], obj.unsqueeze(0)))
        else:
            if data not in self.dataset["data"]:
                self.dataset["data"][data] = [obj]
            else:
                self.dataset["data"][data].append(data)

    def __copy_node__(self, node1):
        node = {
            "data": {}
        }

        for key, value in node1.items():
            if torch.is_tensor(value):
                node[key] = value.detach().clone()
            elif io.is_primitve(value):
                node[key] = value
            else:
                node[key] = value.copy()

        return node

    def pop_data(self, data):
        return self.dataset["data"].pop(data, None)

    def add(self, nodes):
        nodes = io.to_iter(nodes)
        for nd in nodes:
            nd["edges"] = nd["edges"]
            nd["edges"].add(self.dataset["count"])

            for key in nd["data"].keys():
                self.__add_data__(key, nd["data"][key])

            self.dataset["edges"].append(nd["edges"])

            self.dataset["__temp__"].append([])
            self.dataset["count"] += 1

    def subgraph(self, nodes):  # returns isolated graph
        nodes = io.to_iter(nodes)
        subgraph = SubGraph(self, nodes)
        return subgraph

    # pulls others' data from nodes that it points to into it's temp
    def pull(self, nodes=[], func=None, data='x', reset_buffer=True):
        nodes = io.to_iter(nodes)

        if not len(nodes):
            self.push(nodes, func, data, reset_buffer)
        else:
            nodes = set(nodes)
            lis = range(self.dataset["count"])

            for node in lis:
                nd = self.__getitem__(node)
                tw = nodes.intersection(nd["edges"])

                for el in tw:
                    self.dataset["__temp__"][el].append(nd)

            if func:
                out = []
                for node in nodes:
                    out.append(func(self.dataset["__temp__"][node][data]))

                for indx, node in enumerate(nodes):
                    self.dataset["data"][data][node] = out[indx]

        self.reset_temp() if reset_buffer else print("Temporaty buffer was not reset")

    # pushes its data to nodes that it points to into nodes's temp
    def push(self, func=None, data='x', nodes=[], reset_buffer=True):
        nodes = io.to_iter(nodes)

        if not len(nodes):
            nodes = range(self.dataset["count"])

        for node in nodes:
            nd = self.__getitem__(node)
            for edge in nd["edges"]:
                self.dataset["__temp__"][edge].append(nd)

        if func:
            out = []
            for node in nodes:
                out.append(func(self.dataset["__temp__"][node][data]))

            for indx, node in enumerate(nodes):
                self.dataset["data"][data][node] = out[indx]

        self.reset_temp() if reset_buffer else print("Temporaty buffer was not reset")

    def apply(self, func, data, nodes=[]):
        nodes = io.to_iter(nodes)

        if not len(nodes):
            if(issubclass(func.__class__, nn.Module)):
                self.dataset["data"][data] = func(self.dataset["data"][data])
                return
            nodes = range(self.dataset["count"])

        if(issubclass(func.__class__, nn.Module)):
            self.dataset["data"][data][nodes] = func(
                self.dataset["data"][data][nodes])
        else:
            for indx, node in enumerate(nodes):
                self.dataset["data"][data][node] = func(
                    self.dataset["data"][data][indx])

    def reset_temp(self, nodes=[]):  # clear collected data from other nodes
        nodes = io.to_iter(nodes)

        if not len(nodes):
            nodes = range(self.dataset["count"])

        self.dataset["__temp__"] = [[] for i in nodes]

    def filter(self, funcs, data):  # returns nodes' index that pass at least one of the filters
        funs = io.to_iter(funcs)

        out = funs[0](self.dataset["data"][data])

        for fun in funs[1:]:
            out |= fun(self.dataset["data"][data])

        return torch.nonzero(out, as_tuple=False)

    def save(self, fl=""):
        if not len(fl):
            fl = self.__files__["file"]

        io.pickle(self.dataset, fl)


class SubGraph(GraphDataset):  # creates a isolated graph from the dataset (i.e. changes made here might not be written back to parents unless data is a reference). Meant to be more efficient if only processing a few nodes from the dataset
    def __init__(self, graph_dataset, nodes):
        super().__init__()

        self.parent = graph_dataset
        self.nodes = io.to_iter(nodes)

    def add_parent_node(self, nodes):
        nodes = io.to_iter(nodes)
        
        mutual_data = set(self.dataset["data"].keys())
        mutual_data = mutual_data.intersection(self.parent.dataset["data"].keys())

        for node in nodes:
            mutual_edges = self.parent.get_edge(node).intersection(self.nodes)
            edges = [self.nodes.index(ed) for ed in mutual_edges]

            out = {"data": {},
                   "edges": set(edges)}

            for mut in mutual_data:
                out["data"][mut] = self.parent.get_data(mut, nodes=node)[0]

            self.add(out)
            self.nodes.append(node)

    def get_parent_node(self, nodes):
        nodes = io.to_iter(nodes)
        return [self.parent[self.nodes[node]] for node in nodes]

    def get_parent_data(self, data):
        p_data = self.parent.get_data(data, nodes=self.nodes)

        self.set_data(data, p_data)
        return p_data

    def get_parent_index(self, nodes=[]):
        nodes = io.to_iter(nodes)

        if len(nodes) == len(self.nodes) or not len(nodes):
            return self.nodes

        return [self.nodes[i] for i in nodes]


"""
formats cheat sheet:
    (format[, folder/file1, folder/file2])                  ## size of data type in format must be the same as the number of directories/files

    syntax:
        - = addition entries in the data field
        (NAME) = give data the name NAME in the data field
        [##] = optional
            csv: [column1, column2, 3, [0_9]]               ##  Indicate index or column name to retrieve; multiple columns are merges as one

    data type:
        imgs = images folder                                ### Heavy lign graph suggested for large images
        csv = csv file
        imgs_url = file of list of images url                ### Heavy lign graph suggested

    example:
        format = ('imgs(x)-csv(label)[column2]', 'data/', 'labels.txt')
"""


def data_to_dataset(format, out_path):
    pass
    return out_path
