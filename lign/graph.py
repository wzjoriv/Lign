import os.path
import warnings

import torch
from torch import nn
from torch.utils.data import Dataset

from lign.utils import io

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

        if not len(fl):
            fl = os.path.join(os.path.dirname(__file__),
                              "utils", "defaults", "graph.lign")
            self._file_ = os.path.join("data", "graph.lign")
        else:
            self._file_ = fl

        self.dataset = io.unpickle(fl)

        if "count" not in self.dataset and "data" not in self.dataset and \
                "edges" not in self.dataset and "__temp__" not in self.dataset:
            raise FileNotFoundError(f".lign file not found at location: {self._file_}")

    def __len__(self):
        return self.dataset["count"]

    def __getitem__(self, indx):

        if indx < 0:
            if -indx > len(self):
                raise IndexError(
                    "absolute value of index should not exceed dataset length")
            indx = len(self) + indx

        node = {
            "data": {}
        }

        for key in self.dataset["data"].keys():
            node["data"][key] = self.dataset["data"][key][indx]

        node["edges"] = self.dataset["edges"][indx]

        return node

    def get_properties(self):
        return io.to_iter(self.dataset["data"].keys())

    def get_data(self, data, nodes=[]):
        ls = io.to_iter(nodes)
        if not len(ls):
            return self.dataset["data"][data]
        else:
            if torch.is_tensor(self.dataset["data"][data]):
                return self.dataset["data"][data][ls]
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
        if not data in self.dataset["data"].keys():
            raise ValueError(
                f"{data} is not one of the properties of the graph dataset")

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

    def add(self, nodes=None, add_self=True):

        if not nodes:
            nodes = {
                "data": {},
                "edges": set()
            }
        elif type(nodes) is int:
            nodes = [{
                "data": {},
                "edges": set()
            } for i in range(nodes)]

        nodes = io.to_iter(nodes)
        for nd in nodes:
            if add_self:
                nd["edges"].add(len(self))

            mutual_data = set(self.dataset["data"].keys())
            mutual_data = mutual_data.intersection(nd["data"].keys())

            if len(mutual_data) != len(self.dataset["data"].keys()):
                node_keys = nd["data"].keys()
                raise ValueError(
                    f"The data in the node is not the same as in the dataset. Node Data:\n\t{node_keys}\nDataset data:\n\t{self.get_properties()}")

            for key in nd["data"].keys():
                self.__add_data__(key, nd["data"][key])

            self.dataset["edges"].append(nd["edges"])

            self.dataset["__temp__"].append([])
            self.dataset["count"] += 1

    def subgraph(self, nodes, get_data=False, get_edges=False):  # returns isolated graph
        nodes = io.to_iter(nodes)
        subgraph = SubGraph(self, nodes, get_data, get_edges)
        return subgraph

    # pulls others' data from nodes that it points to into it's temp
    def pull(self, func=None, data=None, nodes=[], reset_buffer=True):
        nodes = io.to_iter(nodes)

        if not len(nodes):
            self.push(func=func, data=data, nodes=nodes,
                      reset_buffer=reset_buffer)
        else:
            nodes = set(nodes)
            lis = range(len(self))

            for node in lis:
                nd = self[node]
                tw = nodes.intersection(nd["edges"])

                for el in tw:
                    self.dataset["__temp__"][el].append(nd)

            if func:
                if data:
                    for node in nodes:
                        out = [self.dataset["__temp__"][node][i]["data"][data]
                               for i in range(len(self.dataset["__temp__"][node]))]
                        out = func(out)
                        self.dataset["data"][data][node] = out
                else:
                    out = [func(self.dataset["__temp__"][node])
                           for node in nodes]
                    data_keys = out[0]["data"].keys()

                    for indx, node in enumerate(nodes):
                        for key in data_keys:
                            self.dataset["data"][key][node] = out[indx]["data"][key]

                        self.add_edge(node, out[indx]["edges"])

        if reset_buffer:
            self.reset_temp()
        else:
            warnings.warn("Temporary buffer was not reset")

    # pushes its data to nodes that it points to into nodes' temp
    def push(self, func=None, data=None, nodes=[], reset_buffer=True):
        nodes = io.to_iter(nodes)

        if not len(nodes):
            nodes = range(len(self))

        for node in nodes:
            nd = self[node]
            for edge in nd["edges"]:
                self.dataset["__temp__"][edge].append(nd)

        if func:
            if data:
                for node in nodes:
                    out = [self.dataset["__temp__"][node][i]["data"][data]
                           for i in range(len(self.dataset["__temp__"][node]))]
                    out = func(out)
                    self.dataset["data"][data][node] = out
            else:
                out = [func(self.dataset["__temp__"][node]) for node in nodes]
                data_keys = out[0]["data"].keys()

                for indx, node in enumerate(nodes):
                    for key in data_keys:
                        self.dataset["data"][key][node] = out[indx]["data"][key]

                    self.add_edge(node, out[indx]["edges"])

        if reset_buffer:
            self.reset_temp()
        else:
            warnings.warn("Temporary buffer was not reset")

    def apply(self, func, data, nodes=[]):
        nodes = io.to_iter(nodes)

        if not len(nodes):
            if(issubclass(func.__class__, nn.Module)):
                self.dataset["data"][data] = func(self.dataset["data"][data])
                return
            nodes = range(len(self))

        if(issubclass(func.__class__, nn.Module)):
            self.dataset["data"][data][nodes] = func(
                self.dataset["data"][data][nodes])
        else:
            for node in nodes:
                self.dataset["data"][data][node] = func(
                    self.dataset["data"][data][node])

    def reset_temp(self, nodes=[]):  # clear collected data from other nodes
        nodes = io.to_iter(nodes)

        if not len(nodes):
            nodes = range(len(self))

        self.dataset["__temp__"] = [[] for i in nodes]

    def filter(self, filters, data):  # returns nodes' index that pass at least one of the filters
        filters = io.to_iter(filters)

        if not len(filters):
            raise ValueError("Filters must at least have one filter")

        out = filters[0](self.dataset["data"][data])

        for fun in filters[1:]:
            out |= fun(self.dataset["data"][data])

        return torch.nonzero(out, as_tuple=False).squeeze()

    def save(self, fl=""):
        if not len(fl):
            fl = self._file_

        io.pickle(self.dataset, fl)


class SubGraph(GraphDataset):  # creates a isolated graph from the dataset (i.e. changes made here might not be written back to parents unless data is a reference). Meant to be more efficient if only processing a few nodes from the dataset
    def __init__(self, graph_dataset, nodes, get_data=False, get_edges=False):
        super().__init__()

        self.parent = graph_dataset
        self.p_nodes = io.to_iter(nodes)
        self.i_nodes = list(range(len(self.p_nodes)))

        self.add(len(self.p_nodes))

        if get_data:
            self.get_all_parent_data()

        if get_edges:
            self.get_all_parent_edges()

    def peek_children_index(self):
        return self.i_nodes

    def peek_parent_node(self, nodes):
        nodes = io.to_iter(nodes)
        return [self.parent[self.p_nodes[node]] for node in nodes]

    def get_parent_node(self, nodes, get_data=False, get_edges=False):
        nodes = io.to_iter(nodes)

        mutual_data = set(self.dataset["data"].keys())
        mutual_data = mutual_data.intersection(
            self.parent.dataset["data"].keys())

        if len(mutual_data) != len(self.get_properties()):
            raise LookupError(
                f"Parent graph and sub graph do not have the same properties: \nSubgraph:\n\t{self.get_properties()}\nMutual data:\n\t{mutual_data}")

        for node in nodes:

            out = {"data": {},
                   "edges": set()}

            for mut in mutual_data:
                out["data"][mut] = self.parent.get_data(mut, nodes=node)[0]

            self.i_nodes.append(len(self))
            self.p_nodes.append(node)
            self.add(out)

        if get_data:
            self.get_all_parent_data()

        if get_edges:
            self.get_all_parent_edges()

        return self.peek_parent_node(nodes)

    def peek_parent_data(self, data):
        return self.parent.get_data(data, nodes=self.p_nodes)

    def get_parent_data(self, data):
        p_data = self.peek_parent_data(data)

        self.set_data(data, p_data)
        return p_data

    def get_all_parent_data(self):

        all_v = self.parent.get_properties()

        for data in all_v:
            self.get_parent_data(data)
    
    def peek_parent_edges(self):
        return [self.parent.get_edge(node) for node in self.p_nodes]

    def get_parent_edges(self, nodes):
        nodes = io.to_iter(nodes)

        edges = []

        for node in nodes:
            mutual_edges = self.parent.get_edge(
                node).intersection(self.p_nodes)
            edge = [self.p_nodes.index(ed) for ed in mutual_edges]
            edges.append(edge)

            self.dataset["edges"][node] = edge

        return edges

    def get_all_parent_edges(self):

        edges = self.get_parent_edges(self.p_nodes)

        return edges

    def peek_parent_index(self, nodes=[]):
        nodes = io.to_iter(nodes)

        if len(nodes) == len(self.p_nodes) or not len(nodes):
            return self.p_nodes

        return [self.p_nodes[i] for i in nodes]
