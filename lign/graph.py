from __future__ import annotations
import os.path
import warnings
from typing import Callable, Union, Optional

import torch
from torch import nn
from torch.utils.data import Dataset

from lign.utils import io
from lign.utils.g_types import Tensor, T, List, Tuple, Set, Dict

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


def node(data: dict = {}, edges: Union[Set[int], Tuple[int], List[int]] = set()) -> Node:
    return Node(data, edges)

class Node():
    def __init__(self, data: dict = {}, edges: Union[Set[int], Tuple[int], List[int]] = set()) -> None:

        self.data = data
        self.edges = set(edges)

    def __str__(self) -> str:
        return "node(" + str(self.to_dict()) + ")"

    def __repr__(self):
        return str(self)

    def copy(self) -> Node:
        out = Node()
        out.data = self.data.copy()
        out.edges = self.data.copy()
        return out

    def to_dict(self) -> Dict:
        return {
            "data": self.data,
            "edges": self.edges
        }

class Graph(Dataset):

    def __init__(self, fl: str = "", workers: int = 1) -> None:

        self.dataset = None
        self.workers = workers

        if not len(fl):
            fl = os.path.join(os.path.dirname(__file__),
                              "utils", "defaults", "graph.lign")
            self._file_ = os.path.join("data", "graph.lign")
        else:
            self._file_ = fl

        try:
            self.dataset = io.unpickle(fl)
            if "count" not in self.dataset and "data" not in self.dataset and \
                    "edges" not in self.dataset and "__temp__" not in self.dataset:
                raise
        except Exception:
            raise FileNotFoundError(
                f".lign file not found at location: {self._file_}")

    def __str__(self) -> str:
        return "graph(" + str(self.dataset) + ")"

    def __setitem__(
            self, 
            indx: Union[List[int], slice, str, int, Tuple[int]], 
            features: Union[int, Node, List[Node], set, List[set], Tensor, List[T]]
            ) -> None:

        tp = type(indx)

        if (tp is str):
            self.set_data(indx, features)
        elif (tp is tuple and type(indx[0]) is int):
            self.add_edges(indx, features)
        else:
            raise TypeError("Input is invalid. Only str or Tuple[int] is accepted")

    def __repr__(self):
        return str(self)

    def __len__(self) -> int:
        return self.dataset["count"]

    def __getitem__(
            self, 
            indx: Union[List[int], slice, str, int, Tuple[int]]
            ) -> Union[Node, List[Node], set, List[set], Tensor, List[T]]:

        tp = type(indx)

        if (tp is int or tp is slice or (tp is list and type(indx[0]) is int)):
            return self.get_nodes(indx)
        elif (tp is str):
            return self.get_data(indx)
        elif (tp is tuple and type(indx[0]) is int):
            return self.get_edges(indx)

        raise TypeError("Input is invalid. Only List[int], slice, str, int or Tuple[int] is accepted")
    
    def get_nodes(self, indx:Union[slice, int, List[int]]) -> Node:

        indx = range(len(self))[indx] if type(indx) == slice else io.to_iter(indx)
        nodes = []

        for i in indx:
            if i < 0:
                if -i > len(self):
                    raise IndexError(
                        "absolute value of index should not exceed dataset length")
                ind = len(self) + i
            else:
                ind = i

            node = Node()
            dts = {}

            for key in self.dataset["data"].keys():
                dts[key] = self.dataset["data"][key][ind]

            node.data = dts
            node.edges = self.dataset["edges"][ind]
            nodes.append(node)

        return nodes if len(nodes) > 1 else nodes[0]

    def get_properties(self) -> List[str]:
        return list(self.dataset["data"].keys())

    def get_data(self, data: str, nodes=[]) -> Union[Tensor, List[T]]:
        ls = io.to_iter(nodes)

        if (data not in self.get_properties()):
            raise KeyError(f"{data} is not a property of the graph")

        if not len(ls):
            return self.dataset["data"][data]
        else:
            if torch.is_tensor(self.dataset["data"][data]):
                return self.dataset["data"][data][ls]
            else:
                return [self.dataset["data"][data][nd] for nd in ls] if len(ls) > 1 else self.dataset["data"][data][ls[0]]

    def set_data(self, data: str, features: Union[Tensor, List[T]], nodes: Union[int, List[int]] = []) -> None:
        nodes = io.to_iter(nodes)

        if (len(features) > len(self)):
            raise ValueError("There are more features then there are nodes in the graph")
        elif (len(nodes) > len(self)):
            raise IndexError("More node indexes were given than number of nodes in the graph")
        elif (not len(nodes) and len(features) != len(self)):
            raise ValueError("The number of features must match the number of nodes currently in the graph")

        if not len(nodes):
            self.dataset["data"][data] = features
        else:
            if torch.is_tensor(self.dataset["data"][data]):
                self.dataset["data"][data][nodes] = features
            else:
                for indx, nd in enumerate(nodes):
                    self.dataset["data"][data][nd] = features[indx]

    def add_edges(self, nodes: Union[int, List[int]], edges: Union[int, List[int]]) -> None:
        nodes = io.to_iter(nodes)
        edges = io.to_iter(edges)

        for node in nodes:
            self.dataset["edges"][node].update(edges)

    def get_edges(self, nodes: Union[int, List[int], Set[int], Tuple[int]]) -> Union[set, List[Set]]:

        nodes = io.to_iter(nodes)
        edges = []

        for node in nodes:
            edges.append(self.dataset["edges"][node])

        return edges if len(nodes) > 1 else edges[0]

    def remove_edges(self, nodes: int, edges: Union[int, List[int]]) -> None:

        nodes = io.to_iter(nodes)
        edges = io.to_iter(edges)

        for node in nodes:
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

    def pop_data(self, data: str) -> Union[Tensor, List[T], None]:
        return self.dataset["data"].pop(data, None)

    def add(self, nodes: Optional[Union[int, Node, List[Node]]] = None, add_edges: bool = True) -> None:

        if not nodes:
            nodes = Node()
        elif type(nodes) is int:
            nodes = [Node() for i in range(nodes)]

        nodes = io.to_iter(nodes)
        for nd in nodes:
            if add_edges:
                nd.edges.add(len(self))

            mutual_data = set(self.dataset["data"].keys())
            mutual_data = mutual_data.intersection(nd.data.keys())

            if len(mutual_data) != len(self.dataset["data"].keys()):
                node_keys = list(nd.data.keys())
                raise ValueError(
                    f"The data in the node is not the same as in the dataset. Node Data:\n\t{node_keys}\nDataset data:\n\t{self.get_properties()}")

            for key in nd.data.keys():
                self.__add_data__(key, nd.data[key])

            self.dataset["edges"].append(nd.edges)

            self.dataset["__temp__"].append([])
            self.dataset["count"] += 1

    # returns isolated graph
    def subgraph(self, nodes: Union[int, List[int]], get_data: bool = False, get_edges: bool = False) -> SubGraph:
        nodes = io.to_iter(nodes)
        subgraph = SubGraph(self, nodes, get_data, get_edges)
        return subgraph

    # pulls others' data from nodes that it points to into it's temp
    def pull(
        self,
        func: Optional[Union[Callable[[Union[Tensor, List[T]]], Union[Tensor, T]], nn.Module]] = None,
        data: Optional[str] = None,
        nodes: Union[int, List[int]] = [],
        reset_buffer: bool = True
    ) -> None:

        nodes = io.to_iter(nodes)

        if not len(nodes):
            self.push(func=func, data=data, nodes=nodes,
                      reset_buffer=reset_buffer)
        else:
            nodes = set(nodes)
            lis = range(len(self))

            for node in lis:
                nd = self[node]
                tw = nodes.intersection(nd.edges)

                for el in tw:
                    self.dataset["__temp__"][el].append(nd)

            if func:
                if data:
                    for node in nodes:
                        out = [self.dataset["__temp__"][node][i].data[data]
                               for i in range(len(self.dataset["__temp__"][node]))]
                        out = func(out)
                        self.dataset["data"][data][node] = out
                else:
                    out = [func(self.dataset["__temp__"][node])
                           for node in nodes]

                    for indx, node in enumerate(nodes):

                        mutual_data = set(self.dataset["data"].keys())
                        mutual_data = mutual_data.intersection(
                            node.data.keys())

                        if len(mutual_data) != len(self.dataset["data"].keys()):
                            node_keys = nd.data.keys()
                            raise ValueError(
                                f"The data in the node is not the same as in the dataset. Node Data:\n\t{node_keys}\nDataset data:\n\t{self.get_properties()}")

                        data_keys = node.data.keys()

                        for key in data_keys:
                            self.dataset["data"][key][node] = out[indx].data[key]

                        self.add_edge(node, out[indx].edges)

        if reset_buffer:
            self.reset_temp()
        else:
            warnings.warn("Temporary buffer was not reset")

    # pushes its data to nodes that it points to into nodes' temp
    def push(self, func: Optional[Union[Callable[[Union[Tensor, List[T]]], Union[Tensor, T]], nn.Module]] = None, data: Optional[str] = None, nodes: Union[int, List[int]] = [], reset_buffer: bool = True) -> None:
        nodes = io.to_iter(nodes)

        if not len(nodes):
            nodes = range(len(self))

        for node in nodes:
            nd = self[node]
            for edge in nd.edges:
                self.dataset["__temp__"][edge].append(nd)

        if func:
            if data:
                for node in nodes:
                    out = [self.dataset["__temp__"][node][i].data[data]
                           for i in range(len(self.dataset["__temp__"][node]))]
                    out = func(out)
                    self.dataset["data"][data][node] = out
            else:
                out = [func(self.dataset["__temp__"][node]) for node in nodes]

                for indx, node in enumerate(nodes):

                    mutual_data = set(self.dataset["data"].keys())
                    mutual_data = mutual_data.intersection(node.data.keys())

                    if len(mutual_data) != len(self.dataset["data"].keys()):
                        node_keys = nd.data.keys()
                        raise ValueError(
                            f"The data in the node is not the same as in the dataset. Node Data:\n\t{node_keys}\nDataset data:\n\t{self.get_properties()}")

                    data_keys = node.data.keys()

                    for key in data_keys:
                        self.dataset["data"][key][node] = out[indx].data[key]

                    self.add_edge(node, out[indx].edges)

        if reset_buffer:
            self.reset_temp()
        else:
            warnings.warn("Temporary buffer was not reset")

    def apply(self, func: Optional[Callable[[Union[Tensor, List[T]]], Union[Tensor, T]]], data: str, nodes: Union[int, List[int]] = []) -> None:
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

    # clear collected data from other nodes
    def reset_temp(self, nodes: Union[int, List[int]] = []) -> None:
        nodes = io.to_iter(nodes)

        if not len(nodes):
            nodes = range(len(self))

        self.dataset["__temp__"] = [[] for i in nodes]

    # returns nodes' index that pass at least one of the filters
    def filter(self, filters: Union[Callable[[Tensor], bool], List[Callable[[Tensor], bool]]], data: str) -> Tensor:
        filters = io.to_iter(filters)

        if not len(filters):
            raise ValueError("Filters must at least have one filter")

        out = filters[0](self.dataset["data"][data])

        for fun in filters[1:]:
            out |= fun(self.dataset["data"][data])

        return torch.nonzero(out, as_tuple=False).squeeze()

    def save(self, fl: str = "") -> None:
        if not len(fl):
            fl = self._file_
        else:
            self._file_ = fl

        io.pickle(self.dataset, fl)


class SubGraph(Graph):  # creates a isolated graph from the dataset (i.e. changes made here might not be written back to parents unless data is a reference). Meant to be more efficient if only processing a few nodes from the dataset
    def __init__(self, graph_dataset: Graph, nodes: Union[List[int], int], get_data: bool = False, get_edges: bool = False) -> None:
        super().__init__()

        self.parent = graph_dataset
        self.p_nodes = io.to_iter(nodes)
        self.i_nodes = list(range(len(self.p_nodes)))

        self.add(len(self.p_nodes))

        if get_data:
            self.get_all_parent_data()

        if get_edges:
            self.get_all_parent_edges()

    def peek_children_index(self) -> List[int]:
        return self.i_nodes

    def peek_parent_node(self, nodes: Union[List[int], int]) -> List[Node]:
        nodes = io.to_iter(nodes)
        return [self.parent[self.p_nodes[node]] for node in nodes]

    def get_parent_node(self, nodes: Union[List[int], int], get_data: bool = False, get_edges: bool = False) -> List[Node]:
        nodes = io.to_iter(nodes)

        mutual_data = set(self.dataset["data"].keys())
        mutual_data = mutual_data.intersection(
            self.parent.dataset["data"].keys())

        if len(mutual_data) != len(self.get_properties()):
            raise LookupError(
                f"Parent graph and sub graph do not have the same properties: \nSub graph:\n\t{self.get_properties()}\nMutual data:\n\t{mutual_data}")

        for node in nodes:

            out = Node()

            for mut in mutual_data:
                out.data[mut] = self.parent.get_data(mut, nodes=node)[0]

            self.i_nodes.append(len(self))
            self.p_nodes.append(node)
            self.add(out)

        if get_data:
            self.get_all_parent_data()

        if get_edges:
            self.get_all_parent_edges()

        return self.peek_parent_node(nodes)

    def peek_parent_data(self, data: str) -> Union[Tensor, List[T]]:
        return self.parent.get_data(data, nodes=self.p_nodes)

    def get_parent_data(self, data: str) -> Union[Tensor, List[T]]:
        p_data = self.peek_parent_data(data)

        self.set_data(data, p_data)
        return p_data

    def get_all_parent_data(self) -> None:

        all_v = self.parent.get_properties()

        for data in all_v:
            self.get_parent_data(data)

    def peek_parent_edges(self) -> List[List[int]]:
        return [self.parent.get_edges(node) for node in self.p_nodes]

    def get_parent_edges(self, nodes: Union[List[int], int]) -> List[List[int]]:
        nodes = io.to_iter(nodes)

        edges = []

        for node in nodes:
            mutual_edges = self.parent.get_edges(
                node).intersection(self.p_nodes)
            edge = [self.p_nodes.index(ed) for ed in mutual_edges]
            edges.append(edge)

            self.dataset["edges"][node] = edge

        return edges

    def get_all_parent_edges(self) -> None:
        return self.get_parent_edges(self.p_nodes)

    def peek_parent_index(self, nodes: Union[int, List[int]] = []) -> List[int]:
        nodes = io.to_iter(nodes)

        if len(nodes) == len(self.p_nodes) or not len(nodes):
            return self.p_nodes

        return [self.p_nodes[i] for i in nodes]
