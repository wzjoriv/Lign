import torch
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
    def __init__(self, fl="", workers = 1):
        self.dataset = None
        self.workers = workers

        self.__files__ = {}

        if not len(fl):
            fl = os.path.join(os.path.dirname(__file__), "utils", "defaults","graph.lign")
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

        for key in self.dataset.keys():
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

    def __add_data__(self, obj, data):
        if torch.is_tensor(data):
            obj = torch.cat(obj, data.unqueeze(0))
        else:
            obj.append(data)

    def __copy_node__(self, node1):
        node = {
            "data": {}
        }

        for key, value in node1.items():
            if torch.is_tensor(value):
                node[key] = value.detach().clone()
            elif io.is_primitive(value):
                node[key] = value
            else:
                node[key] = value.copy()
        
        return node

    def del_data(self, data):
        self.dataset["data"].pop(data, None)

    def add(self, nodes):
        nodes = io.to_iter(nodes)

        for nd in nodes:
            nd["edges"] = nd["edges"]
            nd["edges"].add(self.dataset["count"])

            for key in self.dataset["data"].keys():
                self.__add_data__(self.dataset["data"][key], node["data"][key])

            self.dataset["edges"].append(nd["edges"])
                
            self.dataset["__temp__"].append([])
            self.dataset["count"] += 1

    def subgraph(self, nodes): #returns isolated graph
        nodes = io.to_iter(nodes)
        subgraph = SubGraph(self, nodes)
        return subgraph

    def pull(self, nodes=[], func = None, data = 'x'): #pulls others' data from nodes that it points to into it's temp
        temp = self.dataset["__temp__"]
        nodes = io.to_iter(nodes)

        if not len(nodes):
            push(self, func = func)
        else:
            nodes = set(nodes)
            lis = range(self.database["count"])

            for node in lis:
                tw = nodes.intersection(nd["edges"])

                if len(tw):
                    nd = self.__getitem__(node)
                    for el in tw:
                        self.dataset["__temp__"][el].append(nd)

            if func:
                out = []
                for node in nodes:
                    out.append(func(self.dataset["__temp__"][node][data]))

                for indx, node in enumerate(nodes):
                    self.dataset["data"][data][node] = out[indx]
        
        self.dataset["__temp__"] = temp

    def push(self, nodes=[], func = None, data = 'x'): #pushes its data to nodes that it points to into nodes's temp
        
        temp = self.dataset["__temp__"]

        if not len(nodes):
            nodes = range(self.database["count"])

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
        
        self.dataset["__temp__"] = temp

    def apply(self, func, nodes=[]):

        if not len(nodes):
            nodes = range(self.database["count"])

        for node in nodes:
            func(self.__getitem__(node))

    def reset_temp(self, nodes=[]): #clear collected data from other nodes
        if len(nodes):
            for nd in nodes:
                self.dataset["__temp__"][nd] = []
        else:
            self.dataset["__temp__"] = [[] for i in range(self.dataset["count"])]

    def filter(self, func): #returns nodes that pass the filter
        pass

    def save(self, fl=""):
        if not len(fl):
            fl = self.__files__["file"]

        io.pickle(self.dataset, fl)

class SubGraph(): #creates a isolated graph from the dataset. Meant to be more efficient if only changing a few nodes from the dataset
    def __init__(self, graph_dataset, nodes):
        self.dataset = graph_dataset
        self.count = len(nodes)
        self.__temp__ = [[] for i in range(self.count)]
        self.__nodes__ = self.dataset[nodes]                             ### Need to fix

        self.__linked_nodes__ = self.__nodes__
        self.__nodes__ = [self.dataset[i].copy() for i in nodes]   ### Need to fix
    
    def __len__(self):
        return self.dataset["count"]

    def __getitem__(self, indx):
        pass

    def from_dataset(self, nodes): ### Add nodes prom dataset
        pass

    def pull(self): #pulls others' data from nodes that it points to into it's temp
        pass

    def push(self): #pushes its data to nodes that it points to into nodes's temp
        pass

    def apply(func, nodes=[]):
        pass

    def reset_temp(): #clear collected data from other nodes
        self.__temp__ = [[] for i in range(self.count)]

    def filter(func): #returns nodes that pass the filter
        pass



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

def data_to_dataset(format, out_path, heavy = False):
    pass
    return out_path