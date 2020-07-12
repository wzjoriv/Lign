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
        "heavy": False,
        "data": [],
        "edges": [],
        "__temp__": []
    }
    
"""

class GraphDataset(Dataset):
    def __init__(self, fl="", workers = 1, heavy=False):
        self.dataset = None
        self.workers = workers

        self.__files__ = {}

        if not len(fl):
            fl = os.path.join(os.path.dirname(__file__), "utils", "defaults","graph.lign")
            self.__files__["file"] = "data\graph.lign"
            self.__files__["folder"] = os.path.dirname(self.__files__["file"])

        else:
            self.__files__["file"] = fl
            self.__files__["folder"] = os.path.dirname(self.__files__["file"])

        self.dataset = io.unpickle(fl)

        if not len(fl):
            self.dataset["heavy"] = heavy

        if self.dataset["heavy"]:
            self.__files__["data"] = os.path.join(self.__files__["folder"], ".data_LIGN", "")
            self.__files__["edges"] = os.path.join(self.__files__["folder"], ".edges_LIGN", "")

        if "count" not in self.dataset and "data" not in self.dataset and \
            "edges" not in self.dataset and "__temp__" not in self.dataset and \
            "heavy" not in self.dataset:
            raise FileNotFoundError
    
    def __len__(self):
        return self.dataset["count"]

    def __getitem__(self, indx):
        out = {}

        if self.dataset["heavy"]:
            out["data"] = io.unpickle(self.dataset["data"][indx])
            out["edges"] = io.unpickle(self.dataset["edges"][indx])
            
        else:
            out["data"] = self.dataset["data"][indx]
            out["edges"] = self.dataset["edges"][indx]

        return out

    def get_data(data, nodes=[]):
        if not len(nodes):
            nodes = range(self.database["count"])

        return [nd["data"][data] for nd in nodes]

    def set_data(data, features, nodes=[]):
        if not len(nodes):
            nodes = range(self.database["count"])

        for nd in nodes:
            nd["data"][data] = features[nd]

    def add(self, nodes):
        tp = type(nodes)

        if tp != list or tp != tuple or tp != set:
            nodes = [nodes]

        for nd in nodes:
            nd["edges"] = set(nd["edges"])
            nd["edges"].add(self.dataset["count"])

            if self.dataset["heavy"]:
                fl = str(self.dataset["count"]) + ".lign.dt"

                out = os.path.join(self.__files__["data"], fl)
                self.dataset["data"].append(out)
                io.pickle(nd["data"], out)

                out = os.path.join(self.__files__["edges"], fl)
                self.dataset["edges"].append(out)
                io.pickle(nd["data"], out)
            else:
                self.dataset["data"].append(nd["data"])
                self.dataset["edges"].append(nd["edges"])
                
            self.dataset["__temp__"].append([])
            self.dataset["count"] += 1

    def subgraph(self, nodes, linked = False): #returns graph instance
        subgraph = SubGraph(self, nodes, edges, linked)
        return subgraph

    def pull(self, nodes=[], func = None): #pulls others' data from nodes that it points to into it's temp
        
        temp = self.dataset["__temp__"]

        if not len(nodes):
            push(self, func = func)
        else:
            nodes = set(nodes)
            lis = range(self.database["count"])

            for node in lis:
                nd = self.__getitem__(node)

                for el in nodes.intersection(nd["edges"]):
                    self.dataset["__temp__"][el].append(nd)

            if func:
                for node in nodes:
                    func(self.__getitem__(node), self.dataset["__temp__"][edge])
        
        self.dataset["__temp__"] = temp
        del temp

    def push(self, nodes=[], func = None): #pushes its data to nodes that it points to into nodes's temp
        
        temp = self.dataset["__temp__"]

        if not len(nodes):
            nodes = range(self.database["count"])

        for node in nodes:
            nd = self.__getitem__(node)
            for edge in nd["edges"]:
                self.dataset["__temp__"][edge].append(nd)

        if func:
            for node in nodes:
                func(self.__getitem__(node), self.dataset["__temp__"][node])
        
        self.dataset["__temp__"] = temp
        del temp

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
        folder = os.path.dirname(fl)

        if self.dataset["heavy"]:
            io.move_dir(self.__files__["data"], os.path.join(folder, ".data_LIGN", ""))
            io.move_dir(self.__files__["edges"], os.path.join(folder, ".edges_LIGN", ""))

class SubGraph(): #creates a isolated graph from the dataset. Meant to be more efficient if only changing a few nodes from the dataset
    def __init__(self, graph_dataset, nodes, linked = True):
        self.dataset = graph_dataset
        self.count = len(nodes)
        self.__temp__ = [[] for i in range(self.count)]
        self.__nodes__ = self.dataset[nodes]                             ### Need to fix

        if not linked: 
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

    def to_dataset(): #if not linked
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